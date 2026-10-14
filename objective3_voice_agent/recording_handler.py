"""
recording_handler.py — Step 4: Retrieve the completed call transcript from WebRTC bridge.

After the WebRTC call session completes, this module retrieves the
conversation transcript and responses from the call bridge instead of
downloading a Twilio recording.

The bridge (whisper_Test/call_bridge.py) already handles STT via WhisperLive
in real-time during the call. This module polls for session completion and
retrieves the final transcript + response data.
"""

import json
import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Default bridge URL — override via config["webrtc_bridge_url"]
DEFAULT_BRIDGE_URL = "http://localhost:5000"


def _bridge_url(config: dict) -> str:
    """Return the bridge base URL from config or default."""
    return config.get("webrtc_bridge_url", DEFAULT_BRIDGE_URL)


def wait_for_recording(
    session_id: str,
    config: dict,
    timeout_seconds: int = 300,
    poll_interval: int = 5,
) -> Optional[dict]:
    """
    Poll the WebRTC bridge until the call session completes and return
    the conversation transcript.

    In the WebRTC flow, there is no separate recording file to download.
    The bridge handles real-time STT and records all Q&A responses directly.
    This function polls the bridge status and returns the transcript data
    once the session is complete.

    Args:
        session_id:      Bridge session identifier
        config:          voice_agent_config dict with webrtc_bridge_url
        timeout_seconds: Maximum wait time before returning None
        poll_interval:   Seconds between polls

    Returns:
        Dict with transcript data, or None if timed out or errored.
        {
            "transcript": "Full conversation transcript text",
            "word_count": 150,
            "responses": [ { "question": "...", "answer": "...", ... }, ... ],
            "session_complete": True
        }
    """
    bridge = _bridge_url(config)
    start = time.time()
    logger.info(
        f"[RecordingHandler] Waiting for WebRTC session to complete "
        f"(session_id={session_id}, timeout={timeout_seconds}s)"
    )

    while time.time() - start < timeout_seconds:
        try:
            response = requests.get(f"{bridge}/bridge/status", timeout=10)
            response.raise_for_status()
            status = response.json()

            if not status.get("session_active"):
                # Session ended — either completed or never started
                logger.info(
                    f"[RecordingHandler] Session no longer active "
                    f"(session_id={session_id})"
                )
                # Try to retrieve the final response file
                return _retrieve_transcript(session_id, config)

            state = status.get("state", "unknown")
            q_idx = status.get("question_index", "?")
            total = status.get("total_questions", "?")

            logger.debug(
                f"[RecordingHandler] Session {session_id}: "
                f"state={state}, progress={q_idx}/{total}"
            )

            if state == "complete":
                logger.info(
                    f"[RecordingHandler] ✅ Session completed "
                    f"(session_id={session_id})"
                )
                return _retrieve_transcript(session_id, config)

        except requests.ConnectionError:
            logger.warning(
                "[RecordingHandler] Cannot connect to bridge — "
                "is call_bridge.py running?"
            )
        except Exception as e:
            logger.warning(f"[RecordingHandler] Error polling bridge: {e}")

        time.sleep(poll_interval)

    logger.warning(
        f"[RecordingHandler] Timed out waiting for session "
        f"(session_id={session_id})"
    )
    return None


def _retrieve_transcript(session_id: str, config: dict) -> Optional[dict]:
    """
    Load the saved response JSON from the whisper_Test/responses/ directory.

    The call bridge saves the full Q&A transcript to a JSON file in the
    responses/ directory. This function reads it and returns a normalized
    transcript dict.

    Args:
        session_id: The session identifier (used to find the response file)
        config:     voice_agent_config dict

    Returns:
        Dict with transcript, word_count, responses, and session_complete
    """
    import pathlib

    # The response file is saved by the bridge in whisper_Test/responses/
    responses_dir = pathlib.Path(__file__).parent.parent / "whisper_Test" / "responses"

    # Session ID is the filename stem (e.g., "acme_tech_solutions_data_analyst_intern")
    response_file = responses_dir / f"{session_id}.json"

    if not response_file.exists():
        logger.warning(
            f"[RecordingHandler] Response file not found: {response_file}"
        )
        return None

    try:
        with open(response_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        responses = data.get("responses", [])

        # Build full transcript from Q&A pairs
        transcript_parts = []
        for r in responses:
            q = r.get("question", "")
            a = r.get("answer", "")
            status = r.get("status", "pending")

            if status == "answered" and a:
                transcript_parts.append(f"Q: {q}")
                transcript_parts.append(f"A: {a}")

        full_transcript = "\n".join(transcript_parts)
        word_count = len(full_transcript.split()) if full_transcript else 0

        result = {
            "transcript": full_transcript,
            "word_count": word_count,
            "responses": responses,
            "session_complete": True,
            "source": "webrtc_bridge",
        }

        logger.info(
            f"[RecordingHandler] ✅ Transcript loaded: "
            f"{word_count} words, {len(responses)} Q&A pairs"
        )

        return result

    except Exception as e:
        logger.error(f"[RecordingHandler] Error reading response file: {e}")
        return None
