"""
WhisperLive Transcription Server.

Start this first in a separate terminal:
    python server.py

Then run the voice assistant in another terminal:
    python assistant.py
"""

from whisper_live.server import TranscriptionServer

HOST = "0.0.0.0"
PORT = 9090

if __name__ == "__main__":
    print(f"Starting WhisperLive server on {HOST}:{PORT} ...")
    print("(First run will download model weights — this can take a minute.)")
    server = TranscriptionServer()
    server.run(HOST, PORT)
