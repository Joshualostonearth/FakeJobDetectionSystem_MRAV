"""
Voice Tools — Shared components for voice-based assistants.
============================================================

Provides reusable building blocks:
  - EdgeTTSSpeaker : Text-to-Speech via Edge-TTS  (speak)
  - OllamaChat     : LLM conversation via Ollama  (think)
  - WhisperListener: Speech-to-Text via WhisperLive (listen)
"""

import asyncio
import glob
import io
import os
import shutil
import sys
import threading
import time

# Ensure FFmpeg/FFprobe binaries are located and added to PATH automatically
if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        matches = glob.glob(os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "*FFmpeg*", "**", "ffmpeg.exe"), recursive=True)
        if matches:
            bin_dir = os.path.dirname(matches[0])
            os.environ["PATH"] = bin_dir + os.path.pathsep + os.environ["PATH"]

import edge_tts
import numpy as np
import pyaudio
from pydub import AudioSegment
from whisper_live.client import TranscriptionClient


# ──────────────────────────────────────────────
# Configuration defaults
# ──────────────────────────────────────────────

WHISPER_HOST = "127.0.0.1"
WHISPER_PORT = 9090
WHISPER_MODEL = "small"
WHISPER_LANG = "en"

TTS_VOICE = "en-US-AriaNeural"


# ──────────────────────────────────────────────
# EdgeTTSSpeaker — Text-to-Speech via Edge-TTS
# ──────────────────────────────────────────────

class EdgeTTSSpeaker:
    """Streams Edge-TTS audio to the speaker in real-time via PyAudio.

    Usage:
        speaker = EdgeTTSSpeaker()
        speaker.speak("Hello, how are you?")   # blocks until playback done
        speaker.shutdown()
    """

    def __init__(self, voice: str = TTS_VOICE):
        self.voice = voice
        # Dedicated event loop for async edge-tts calls
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def speak(self, text: str):
        """Speak the given text. Blocks until playback finishes."""
        if not text.strip():
            return
        future = asyncio.run_coroutine_threadsafe(self._speak_async(text), self._loop)
        future.result()  # block until done

    async def _speak_async(self, text: str):
        """Fetch TTS audio and play it through PyAudio."""
        communicate = edge_tts.Communicate(text, self.voice)

        # Collect all audio bytes from the stream
        audio_bytes = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.write(chunk["data"])

        audio_bytes.seek(0)
        if audio_bytes.getbuffer().nbytes == 0:
            return

        # Decode MP3 → raw PCM using pydub
        try:
            audio_segment = AudioSegment.from_mp3(audio_bytes)
        except Exception as e:
            print(f"\n  [TTS ERROR] Failed to decode audio: {e}")
            return

        # Convert to a standard format for PyAudio
        audio_segment = audio_segment.set_channels(1).set_frame_rate(24000).set_sample_width(2)
        raw_pcm = audio_segment.raw_data

        # Play through PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
        )
        try:
            chunk_size = 4096
            for i in range(0, len(raw_pcm), chunk_size):
                stream.write(raw_pcm[i : i + chunk_size])
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def shutdown(self):
        """Stop the background event loop."""
        self._loop.call_soon_threadsafe(self._loop.stop)


# ──────────────────────────────────────────────
# OllamaChat — LLM conversation via Ollama
# ──────────────────────────────────────────────

class OllamaChat:
    """Manages a multi-turn conversation with an Ollama model.

    Usage:
        chat = OllamaChat(model="ministral-3:latest")
        for token in chat.chat("What is Python?"):
            print(token, end="")
    """

    def __init__(self, model: str = "ministral-3:latest", system_prompt: str = ""):
        self.model = model
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, user_message: str):
        """
        Send a user message and yield assistant response tokens as they stream in.
        The full assistant response is appended to history after streaming completes.
        """
        self.messages.append({"role": "user", "content": user_message})

        full_response = []
        try:
            import ollama
            stream = ollama.chat(
                model=self.model,
                messages=self.messages,
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                full_response.append(token)
                yield token
        finally:
            self.messages.append({"role": "assistant", "content": "".join(full_response)})

    def generate(self, user_message: str) -> str:
        """Send a message and return the complete response as a single string."""
        return "".join(self.chat(user_message))

    def clear(self):
        """Reset conversation history (keeps system prompt)."""
        self.messages = [self.messages[0]] if self.messages and self.messages[0]["role"] == "system" else []


# ──────────────────────────────────────────────
# WhisperListener — On-demand Speech-to-Text
# ──────────────────────────────────────────────

class WhisperListener:
    """Blocking speech-to-text using WhisperLive.

    Unlike the continuous listening in the original VoiceAssistant,
    this class provides an on-demand `listen()` method that:
    - Starts accumulating transcriptions when called
    - Returns the transcript after silence_seconds of quiet
    - Returns "" if timeout_seconds elapses with no speech

    Usage:
        listener = WhisperListener()
        listener.connect()
        text = listener.listen(silence_seconds=2, timeout_seconds=10)
        listener.disconnect()
    """

    def __init__(
        self,
        host: str = WHISPER_HOST,
        port: int = WHISPER_PORT,
        model: str = WHISPER_MODEL,
        lang: str = WHISPER_LANG,
    ):
        self.host = host
        self.port = port
        self.model = model
        self.lang = lang

        # Internal state
        self._client = None
        self._record_thread = None
        self._connected = False

        # Transcription accumulator (guarded by _lock)
        self._lock = threading.Lock()
        self._committed_texts: list[str] = []
        self._partial_text: str = ""
        self._last_activity_time: float = 0.0
        self._listening = False  # gate: ignore callbacks when not listening

        # Segment offset: WhisperLive sends ALL accumulated segments in each
        # callback, so we need to track how many belong to previous listen() calls
        self._segment_offset: int = 0

        # Connection verification
        self._got_callback = threading.Event()

    def connect(self):
        """Connect to the WhisperLive server and start the recording thread."""
        if self._connected:
            return

        print("  [LISTENER] Connecting to WhisperLive server...")

        self._client = TranscriptionClient(
            host=self.host,
            port=self.port,
            lang=self.lang,
            model=self.model,
            use_vad=True,
            log_transcription=False,
            transcription_callback=self._on_transcription,
        )

        # Start the recording thread (TranscriptionClient.__call__ blocks)
        self._record_thread = threading.Thread(target=self._start_client, daemon=True)
        self._record_thread.start()

        # Wait for the connection to actually work (up to 10 seconds)
        # We verify by temporarily enabling listening and checking for callbacks
        self._listening = True
        connected = self._got_callback.wait(timeout=10)
        self._listening = False

        if connected:
            self._connected = True
            # Reset any segments that came in during connection check
            self._segment_offset = 0
            with self._lock:
                self._committed_texts = []
                self._partial_text = ""
                self._last_activity_time = 0.0
            print("  [LISTENER] ✓ Connected to WhisperLive server.")
        else:
            print("  [LISTENER] ✗ WARNING: Could not verify WhisperLive connection.")
            print("  [LISTENER]   Make sure 'python server.py' is running in another terminal.")
            self._connected = True  # Still set to true so the caller can proceed

    def _start_client(self):
        """Run the TranscriptionClient. Runs in a daemon thread."""
        try:
            self._client()
        except Exception as e:
            print(f"\n  [LISTENER ERROR] TranscriptionClient failed: {e}")

    def disconnect(self):
        """Disconnect from the WhisperLive server."""
        self._connected = False
        self._listening = False

    def _on_transcription(self, text: str, segments: list):
        """Callback from whisper-live. Only accumulates when _listening is True."""
        # Signal that we got at least one callback (connection is alive)
        if not self._got_callback.is_set():
            self._got_callback.set()

        if not self._listening:
            return

        with self._lock:
            # Separate committed (finalized) segments from partial (in-progress)
            committed = []
            partial = ""
            for seg in segments:
                if seg.get("completed", False):
                    committed.append(seg["text"].strip())
                else:
                    partial = seg["text"].strip()

            # Only look at segments AFTER our offset (new segments for this listen call)
            new_committed = committed[self._segment_offset:]

            if new_committed:
                self._committed_texts = new_committed
                self._last_activity_time = time.time()

            # Partial text always reflects the current in-progress segment
            old_partial = self._partial_text
            self._partial_text = partial

            # If partial text appeared or changed, mark activity
            # (someone is actively speaking even if no segment committed yet)
            if partial and partial != old_partial:
                self._last_activity_time = time.time()

            # Show live transcription
            full = " ".join(self._committed_texts)
            if partial:
                full = f"{full} {partial}".strip() if full else partial
            if full:
                sys.stdout.write(f"\r\033[K  🎧  {full}")
                sys.stdout.flush()

    def listen(self, silence_seconds: float = 2.0, timeout_seconds: float = 10.0) -> str:
        """
        Listen for speech and return the transcript.

        Returns the accumulated transcript once the speaker goes silent for
        `silence_seconds`. Returns "" if `timeout_seconds` elapses with no
        speech detected at all.
        """
        if not self._connected:
            print("  [LISTENER] ERROR: Not connected. Call connect() first.")
            return ""

        # Reset state for this listen call (but keep segment_offset!)
        with self._lock:
            self._committed_texts = []
            self._partial_text = ""
            self._last_activity_time = 0.0

        self._listening = True
        start_time = time.time()

        try:
            while True:
                elapsed = time.time() - start_time

                with self._lock:
                    has_speech = bool(self._committed_texts or self._partial_text)
                    last_activity = self._last_activity_time

                # Timeout: no speech detected at all
                if not has_speech and elapsed >= timeout_seconds:
                    print("\r\033[K  [LISTENER] Timeout — no speech detected.")
                    return ""

                # Silence detection: speech was heard, now quiet
                if has_speech and last_activity > 0:
                    silence_duration = time.time() - last_activity
                    if silence_duration >= silence_seconds:
                        break

                # Overall hard timeout (even if speech was heard)
                if elapsed >= timeout_seconds + silence_seconds:
                    break

                time.sleep(0.1)

        finally:
            self._listening = False

        # Collect the result
        with self._lock:
            full = " ".join(self._committed_texts)
            if self._partial_text:
                full = f"{full} {self._partial_text}".strip() if full else self._partial_text

            # Advance the offset so the next listen() call skips these segments
            self._segment_offset += len(self._committed_texts)

            self._committed_texts = []
            self._partial_text = ""

        result = full.strip()
        if result:
            print()  # newline after the live transcript
        return result

