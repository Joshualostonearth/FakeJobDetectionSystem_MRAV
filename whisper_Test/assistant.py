"""
Real-Time Voice Assistant
=========================

Pipeline:  Microphone → WhisperLive (STT) → Ollama Ministral 3B (LLM) → Edge-TTS (TTS) → Speaker

Usage:
    1. Start the WhisperLive server in another terminal:   python server.py
    2. Make sure Ollama is running:                        ollama serve
    3. Run this script:                                    python assistant.py

Press Ctrl+C to stop.
"""

import re
import sys
import threading
import time
from enum import Enum, auto

from whisper_live.client import TranscriptionClient

from voice_tools import EdgeTTSSpeaker, OllamaChat


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

WHISPER_HOST = "127.0.0.1"
WHISPER_PORT = 9090
WHISPER_MODEL = "small"
WHISPER_LANG = "en"

OLLAMA_MODEL = "qwen3-vl:latest"
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep your responses concise and conversational "
    "since they will be spoken aloud. Avoid using markdown, bullet points, numbered lists, "
    "or special formatting — just use plain, natural sentences."
)

TTS_VOICE = "en-US-AriaNeural"  # Edge-TTS default voice

SILENCE_TIMEOUT = 2.0  # seconds of silence before treating speech as complete


# ──────────────────────────────────────────────
# State
# ──────────────────────────────────────────────

class State(Enum):
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


# ──────────────────────────────────────────────
# Sentence splitter for streaming TTS
# ──────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split text into sentences at . ! ? boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


# ──────────────────────────────────────────────
# VoiceAssistant — Main orchestrator
# ──────────────────────────────────────────────

class VoiceAssistant:
    """
    State machine that orchestrates the full pipeline:
    LISTENING → PROCESSING → SPEAKING → LISTENING ...
    """

    def __init__(self):
        self.state = State.LISTENING
        self.llm = OllamaChat(model=OLLAMA_MODEL, system_prompt=SYSTEM_PROMPT)
        self.speaker = EdgeTTSSpeaker(voice=TTS_VOICE)

        # Transcription state
        self._committed_texts: list[str] = []
        self._partial_text: str = ""
        self._last_commit_time: float = 0.0
        self._lock = threading.Lock()

    def _on_transcription(self, text: str, segments: list):
        """
        Callback from whisper-live. Called on EVERY transcript update
        (both partial and committed segments).
        """
        if self.state != State.LISTENING:
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

            if committed:
                # Only add new committed segments
                new_count = len(committed) - len(self._committed_texts)
                if new_count > 0:
                    self._committed_texts = committed
                    self._last_commit_time = time.time()

            self._partial_text = partial

            # Display live transcription
            full = " ".join(self._committed_texts)
            if partial:
                full = f"{full} {partial}".strip() if full else partial

            sys.stdout.write(f"\r\033[K  🎤  {full}")
            sys.stdout.flush()

    def _get_accumulated_text(self) -> str:
        """Return all accumulated text and clear the buffer."""
        with self._lock:
            full = " ".join(self._committed_texts)
            if self._partial_text:
                full = f"{full} {self._partial_text}".strip() if full else self._partial_text
            self._committed_texts = []
            self._partial_text = ""
            self._last_commit_time = 0.0
            return full.strip()

    def _silence_exceeded(self) -> bool:
        """Check if enough silence has passed since the last committed segment."""
        with self._lock:
            if not self._committed_texts and not self._partial_text:
                return False
            if self._last_commit_time == 0.0:
                return False
            return (time.time() - self._last_commit_time) >= SILENCE_TIMEOUT

    def run(self):
        """Main loop. Start this after the WhisperLive server is running."""

        print("=" * 60)
        print("    Voice Assistant")
        print("=" * 60)
        print(f"  Model   : {OLLAMA_MODEL}")
        print(f"  Whisper  : {WHISPER_MODEL}")
        print(f"  Voice    : {TTS_VOICE}")
        print("=" * 60)
        print("  Connecting to WhisperLive server...")
        print()

        # Create the whisper-live transcription client
        client = TranscriptionClient(
            host=WHISPER_HOST,
            port=WHISPER_PORT,
            lang=WHISPER_LANG,
            model=WHISPER_MODEL,
            use_vad=True,
            log_transcription=False,  # We handle display ourselves
            transcription_callback=self._on_transcription,
        )

        # Start recording in a background thread
        # TranscriptionClient.__call__() blocks while recording
        record_thread = threading.Thread(target=lambda: client(), daemon=True)
        record_thread.start()

        # Give it a moment to connect
        time.sleep(2)
        print("   Connected! Start speaking...\n")

        try:
            while True:
                if self.state == State.LISTENING:
                    # Check if user stopped speaking
                    if self._silence_exceeded():
                        user_text = self._get_accumulated_text()
                        if user_text:
                            self.state = State.PROCESSING
                            print()  # newline after the transcription line
                            self._process_and_speak(user_text)
                    time.sleep(0.1)  # small poll interval

        except KeyboardInterrupt:
            print("\n\n   Shutting down...")
            self.speaker.shutdown()
            print("  Done.")

    def _process_and_speak(self, user_text: str):
        """Send user text to LLM, stream response, speak each sentence."""

        print(f"\n   You: {user_text}")
        print(f"   Thinking...", end="", flush=True)

        # Collect LLM response, buffering by sentence
        self.state = State.PROCESSING
        full_response = ""
        sentence_buffer = ""
        sentences_to_speak: list[str] = []

        for token in self.llm.chat(user_text):
            sentence_buffer += token
            full_response += token

            # Check if we have a complete sentence
            parts = re.split(r'(?<=[.!?])\s+', sentence_buffer, maxsplit=1)
            if len(parts) > 1:
                # We have a complete sentence + remainder
                sentences_to_speak.append(parts[0])
                sentence_buffer = parts[1]

        # Don't forget the last chunk (may not end with punctuation)
        if sentence_buffer.strip():
            sentences_to_speak.append(sentence_buffer.strip())

        # Display the full response
        print(f"\r\033[K   Assistant: {full_response}")

        # Speak each sentence
        self.state = State.SPEAKING
        print("   Speaking...", end="", flush=True)

        for sentence in sentences_to_speak:
            self.speaker.speak(sentence)

        print("\r\033[K   Done speaking.")
        print()

        # Back to listening
        self.state = State.LISTENING
        print("   Listening... (speak now)\n")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
