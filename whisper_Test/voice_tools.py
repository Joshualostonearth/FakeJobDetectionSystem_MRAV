"""
Voice Tools — Shared components for voice-based assistants.
============================================================

Provides reusable building blocks:
  - EdgeTTSSpeaker       : Text-to-Speech via Edge-TTS — plays through speaker (speak)
  - EdgeTTSFile          : Text-to-Speech via Edge-TTS — returns audio bytes (for WebRTC bridge)
  - OllamaChat           : LLM conversation via Ollama  (think)
  - WhisperListener      : Speech-to-Text via WhisperLive — live mic (listen)
  - WhisperFileTranscriber: Speech-to-Text via WhisperLive — from audio file (for WebRTC bridge)
"""

import asyncio
import io
import sys
import threading
import time

import edge_tts
import numpy as np
import pyaudio
from pydub import AudioSegment
from whisper_live.client import TranscriptionClient


# ──────────────────────────────────────────────
# Configuration defaults
# ──────────────────────────────────────────────

WHISPER_HOST = "localhost"
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
# EdgeTTSFile — Text-to-Speech to bytes/file
# ──────────────────────────────────────────────

class EdgeTTSFile:
    """Generate Edge-TTS audio to an in-memory buffer (no speaker playback).

    Unlike EdgeTTSSpeaker which plays through PyAudio, this class returns
    raw MP3 bytes for sending over HTTP (used by the WebRTC call bridge).

    Usage:
        tts = EdgeTTSFile()
        mp3_bytes = tts.synthesize("Hello, how are you?")
        # mp3_bytes is a bytes object containing the MP3 audio
    """

    def __init__(self, voice: str = TTS_VOICE):
        self.voice = voice
        # Dedicated event loop for async edge-tts calls
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def synthesize(self, text: str) -> bytes:
        """Generate TTS audio and return MP3 bytes. Blocks until complete."""
        if not text.strip():
            return b""
        future = asyncio.run_coroutine_threadsafe(self._synthesize_async(text), self._loop)
        return future.result()

    async def _synthesize_async(self, text: str) -> bytes:
        """Fetch TTS audio and return as MP3 bytes (no playback)."""
        communicate = edge_tts.Communicate(text, self.voice)

        audio_bytes = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.write(chunk["data"])

        return audio_bytes.getvalue()

    def synthesize_to_wav(self, text: str, sample_rate: int = 24000) -> bytes:
        """Generate TTS audio and return WAV bytes (PCM 16-bit mono).

        Useful when the consumer expects WAV format instead of MP3.
        """
        mp3_bytes = self.synthesize(text)
        if not mp3_bytes:
            return b""

        # Decode MP3 → raw PCM using pydub
        audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        audio_segment = audio_segment.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)

        # Export as WAV to bytes
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        return wav_buffer.getvalue()

    def shutdown(self):
        """Stop the background event loop."""
        self._loop.call_soon_threadsafe(self._loop.stop)

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
            try:
                stream = ollama.chat(
                    model=self.model,
                    messages=self.messages,
                    stream=True,
                )
                for chunk in stream:
                    token = chunk["message"]["content"]
                    full_response.append(token)
                    yield token
            except Exception as e:
                print(f"  [LLM MOCK] Error from Ollama: {e}. Yielding mock response.")
                mock_text = "This is a mocked response because the Ollama model is unavailable. Are you available for a phone screen?"
                # Simulate streaming
                for word in mock_text.split():
                    token = word + " "
                    full_response.append(token)
                    time.sleep(0.1)
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

    def flush(self, wait: float = 0.5):
        """Discard any segments accumulated while not listening (e.g., TTS bleed).

        Call this after TTS playback finishes and before listen() to ensure
        the mic-captured question audio is skipped.

        Args:
            wait: Seconds to wait for in-flight segments to finalize before
                  advancing the offset.
        """
        # Briefly enable listening so pending callbacks are processed
        self._listening = True
        time.sleep(wait)
        self._listening = False

        with self._lock:
            # Advance offset past everything accumulated (including from TTS bleed)
            self._segment_offset += len(self._committed_texts)
            self._committed_texts = []
            self._partial_text = ""
            self._last_activity_time = 0.0

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


# ──────────────────────────────────────────────
# WhisperFileTranscriber — STT from audio files
# ──────────────────────────────────────────────

class WhisperFileTranscriber:
    """Transcribe audio files by streaming bytes to the WhisperLive server.

    Instead of live mic input, reads an audio file (or bytes), converts it
    to 16kHz mono PCM, and streams the chunks to WhisperLive as if they
    were coming from a microphone.

    Usage:
        transcriber = WhisperFileTranscriber()
        text = transcriber.transcribe_file("recording.webm")
        # or
        text = transcriber.transcribe_bytes(audio_bytes, format="webm")
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

    def transcribe_bytes(self, audio_bytes: bytes, format: str = "webm") -> str:
        """Transcribe audio from raw bytes.

        Args:
            audio_bytes: Raw audio file bytes (any format pydub supports).
            format: Audio format hint for pydub (e.g. 'webm', 'mp3', 'wav').

        Returns:
            Transcript string. Empty string if no speech detected.
        """
        if not audio_bytes:
            return ""

        print(f"  [FILE-STT] Transcribing {len(audio_bytes)} bytes ({format})...")

        # Convert to 16kHz mono PCM (the format WhisperLive expects)
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
            audio_segment = audio_segment.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            pcm_data = audio_segment.raw_data
        except Exception as e:
            print(f"  [FILE-STT] Audio conversion failed: {e}")
            return ""

        duration_sec = len(pcm_data) / (16000 * 2)  # 16kHz, 16-bit = 2 bytes/sample
        print(f"  [FILE-STT] Audio duration: {duration_sec:.1f}s")

        # Stream PCM to WhisperLive and collect transcript
        transcript = self._stream_to_whisper(pcm_data)

        print(f"  [FILE-STT] Transcript: \"{transcript[:80]}{'...' if len(transcript) > 80 else ''}\"")
        return transcript

    def transcribe_file(self, file_path: str) -> str:
        """Transcribe an audio file from disk.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcript string.
        """
        import os
        ext = os.path.splitext(file_path)[1].lstrip(".")
        with open(file_path, "rb") as f:
            return self.transcribe_bytes(f.read(), format=ext or "webm")

    def _stream_to_whisper(self, pcm_data: bytes) -> str:
        """Stream PCM audio to WhisperLive server and return transcript.

        Uses TranscriptionClient with a custom approach: instead of mic
        recording, we directly feed PCM chunks via the WebSocket.
        """
        result_text = []
        done_event = threading.Event()
        got_segments = threading.Event()

        def on_transcription(text: str, segments: list):
            """Callback from WhisperLive with transcription results."""
            got_segments.set()
            # Collect all completed segments
            committed = []
            for seg in segments:
                if seg.get("completed", False):
                    committed.append(seg["text"].strip())
            if committed:
                result_text.clear()
                result_text.extend(committed)

        # Create a TranscriptionClient
        client = TranscriptionClient(
            host=self.host,
            port=self.port,
            lang=self.lang,
            model=self.model,
            use_vad=True,
            log_transcription=False,
            transcription_callback=on_transcription,
        )

        # We need to feed audio data through the client's WebSocket
        # TranscriptionClient normally records from mic — we'll use its
        # internal client to send data directly
        def run_client():
            try:
                client()
            except Exception as e:
                print(f"  [FILE-STT] Client error: {e}")
            finally:
                done_event.set()

        client_thread = threading.Thread(target=run_client, daemon=True)
        client_thread.start()

        # Wait for connection to establish
        time.sleep(1.0)

        # Stream PCM chunks through the WebSocket
        # WhisperLive expects chunks of raw audio at the recording rate
        chunk_size = 4096  # bytes per chunk
        chunk_interval = chunk_size / (16000 * 2)  # seconds per chunk at 16kHz 16-bit

        try:
            ws_client = client.client
            if ws_client and hasattr(ws_client, 'client_socket') and ws_client.client_socket:
                for i in range(0, len(pcm_data), chunk_size):
                    chunk = pcm_data[i:i + chunk_size]
                    try:
                        ws_client.client_socket.send(chunk, opcode=0x2)  # binary
                    except Exception:
                        break
                    # Pace the sending to match real-time (but 4x faster)
                    time.sleep(chunk_interval * 0.25)

                # Send a small silence buffer to trigger final segment
                silence = b'\x00' * (16000 * 2)  # 1 second of silence
                try:
                    ws_client.client_socket.send(silence, opcode=0x2)
                except Exception:
                    pass

                # Wait for final transcription
                time.sleep(2.0)

        except Exception as e:
            print(f"  [FILE-STT] Streaming error: {e}")

        # Collect results
        transcript = " ".join(result_text).strip()

        # Clean up - disconnect the client
        try:
            if ws_client and hasattr(ws_client, 'client_socket') and ws_client.client_socket:
                ws_client.client_socket.close()
        except Exception:
            pass

        return transcript

