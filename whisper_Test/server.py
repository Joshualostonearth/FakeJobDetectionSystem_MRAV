"""
WhisperLive Transcription Server.

Start this first in a separate terminal:
    python server.py

Then run the voice assistant in another terminal:
    python assistant.py
"""

import glob
import os
import shutil

# Ensure FFmpeg/FFprobe binaries are located and added to PATH automatically
if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        matches = glob.glob(os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "*FFmpeg*", "**", "ffmpeg.exe"), recursive=True)
        if matches:
            bin_dir = os.path.dirname(matches[0])
            os.environ["PATH"] = bin_dir + os.path.pathsep + os.environ["PATH"]

from whisper_live.server import TranscriptionServer

HOST = "0.0.0.0"
PORT = 9090

if __name__ == "__main__":
    print(f"Starting WhisperLive server on {HOST}:{PORT} ...")
    print("(First run will download model weights — this can take a minute.)")
    server = TranscriptionServer()
    server.run(HOST, PORT)
