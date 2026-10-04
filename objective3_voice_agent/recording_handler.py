"""
recording_handler.py — Step 4: Wait for and download the Twilio call recording.

Polls the shared pending_recordings dict (populated by the Twilio webhook in audio_server.py)
and downloads the WAV file with authenticated HTTP.
"""

import logging
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv(override=False)

logger = logging.getLogger(__name__)


def wait_for_recording(
    call_sid: str,
    pending_recordings: dict,
    timeout_seconds: int = 90,
) -> Optional[str]:
    """
    Poll the pending_recordings dict until the Twilio webhook delivers a recording URL.

    Args:
        call_sid:           Twilio Call SID to wait for
        pending_recordings: Shared dict populated by POST /webhook/call-complete
        timeout_seconds:    Maximum wait time before returning None

    Returns:
        Local path to the downloaded WAV file, or None if timed out.
    """
    start = time.time()
    logger.info(f"[RecordingHandler] Waiting for recording (call_sid={call_sid}, timeout={timeout_seconds}s)")

    while time.time() - start < timeout_seconds:
        if call_sid in pending_recordings:
            recording_url = pending_recordings.pop(call_sid)
            logger.info(f"[RecordingHandler] Recording URL received: {recording_url}")
            return download_recording(recording_url, call_sid)
        time.sleep(3)

    logger.warning(f"[RecordingHandler] Timed out waiting for recording (call_sid={call_sid})")
    return None


def download_recording(recording_url: str, call_sid: str = "") -> str:
    """
    Download the Twilio recording WAV file using Basic Auth.

    Args:
        recording_url: Twilio recording resource URL (without extension)
        call_sid:      Used to generate a unique local filename

    Returns:
        Local path where the WAV was saved.
    """
    import pathlib
    pathlib.Path("temp/recordings").mkdir(parents=True, exist_ok=True)

    wav_url = recording_url.rstrip("/") + ".wav"
    logger.debug(f"[RecordingHandler] Downloading: {wav_url}")

    response = requests.get(
        wav_url,
        auth=(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN"),
        ),
        timeout=30,
    )
    response.raise_for_status()

    # Use call_sid for filename if available, otherwise derive from URL
    filename = f"{call_sid}.wav" if call_sid else recording_url.split("/")[-1] + ".wav"
    local_path = f"temp/recordings/{filename}"

    with open(local_path, "wb") as f:
        f.write(response.content)

    size_kb = len(response.content) / 1024
    logger.info(f"[RecordingHandler] Saved {local_path} ({size_kb:.1f} KB)")
    return local_path
