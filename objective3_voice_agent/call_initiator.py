"""
call_initiator.py — Step 3b: Initiate and track a WebRTC verification call.

Communicates with the Flask call_bridge (whisper_Test/call_bridge.py) which
wraps the AI caller logic (question generation, STT, TTS) into REST endpoints.

The WebRTC Node.js server (webrtc-call/server/server.js) handles the actual
peer-to-peer audio connection and relays audio to/from the bridge.

Endpoints used:
    POST http://localhost:5000/bridge/start-session
    POST http://localhost:5000/bridge/speak-first
    GET  http://localhost:5000/bridge/status
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Default bridge URL — override via config["webrtc_bridge_url"]
DEFAULT_BRIDGE_URL = "http://localhost:5000"


def _bridge_url(config: dict) -> str:
    """Return the bridge base URL from config or default."""
    return config.get("webrtc_bridge_url", DEFAULT_BRIDGE_URL)


def initiate_verification_call(
    entity: dict,
    config: dict,
    timeout: int = 30,
) -> dict:
    """
    Start a WebRTC verification session via the call bridge.

    Sends entity data (company name, job title, etc.) to the bridge which
    generates verification questions via Ollama LLM and prepares the session.

    Args:
        entity:   Job offer entity dict — must have: company_name, role/job_role
        config:   voice_agent_config dict (webrtc_bridge_url, timeouts)
        timeout:  HTTP request timeout in seconds

    Returns:
        Bridge session response dict containing:
            - success: bool
            - session_id: str
            - questions: list of question dicts
            - total_questions: int
            - meta: dict with company, job, etc.
    """
    bridge = _bridge_url(config)
    url = f"{bridge}/bridge/start-session"

    # Map entity fields to bridge expected format
    payload = {
        "company_name": entity.get("company_name", ""),
        "job_title": entity.get("role", "") or entity.get("job_role", ""),
        "job_type": entity.get("job_type", "unknown"),
        "offer_details": {
            "location": entity.get("location", ""),
            "recruiter_name": entity.get("recruiter_name", ""),
        },
        "suspicion_reasons": entity.get("suspicion_reasons", []),
    }

    logger.info(f"[CallInitiator] Starting WebRTC session → {url}")
    logger.info(f"[CallInitiator] Entity: {payload['company_name']} — {payload['job_title']}")

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            session_id = result.get("session_id", "unknown")
            total_q = result.get("total_questions", 0)
            logger.info(f"[CallInitiator] ✅ Session started: {session_id} ({total_q} questions)")
        else:
            logger.error(f"[CallInitiator] Bridge returned error: {result.get('error')}")

        return result

    except requests.ConnectionError:
        logger.error(
            f"[CallInitiator] Cannot connect to call bridge at {bridge}. "
            "Make sure call_bridge.py is running: python whisper_Test/call_bridge.py"
        )
        return {"success": False, "error": f"Cannot connect to call bridge at {bridge}"}

    except Exception as e:
        logger.error(f"[CallInitiator] Error starting session: {e}")
        return {"success": False, "error": str(e)}


def trigger_greeting(config: dict, timeout: int = 30) -> dict:
    """
    Trigger the Q1 greeting (called after the WebRTC call is connected).

    The bridge generates TTS audio for the first question and returns it.

    Args:
        config:  voice_agent_config dict
        timeout: HTTP request timeout in seconds

    Returns:
        Bridge response dict with agent_text, tts_audio_b64, etc.
    """
    bridge = _bridge_url(config)
    url = f"{bridge}/bridge/speak-first"

    logger.info("[CallInitiator] Triggering Q1 greeting...")

    try:
        response = requests.post(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"[CallInitiator] Greeting error: {e}")
        return {"success": False, "error": str(e)}


def get_session_status(config: dict) -> dict:
    """
    Check current bridge session status.

    Returns:
        Session state dict (bridge status, question index, etc.)
    """
    bridge = _bridge_url(config)
    url = f"{bridge}/bridge/status"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"[CallInitiator] Status check error: {e}")
        return {"bridge": "error", "session_active": False, "error": str(e)}


def wait_for_session_completion(
    config: dict,
    timeout: int = 300,
    poll_interval: int = 5,
) -> str:
    """
    Poll the bridge every few seconds until the session completes.

    The session completes when all verification questions have been
    answered through the WebRTC call.

    Args:
        config:        voice_agent_config dict
        timeout:       Maximum seconds to wait before returning "timeout"
        poll_interval: Seconds between status polls

    Returns:
        Final session state string: "complete", "active", "error", or "timeout"
    """
    start = time.time()

    while time.time() - start < timeout:
        try:
            status = get_session_status(config)

            if not status.get("session_active"):
                logger.info("[CallInitiator] No active session — may have completed or not started.")
                return "no_session"

            state = status.get("state", "unknown")
            q_idx = status.get("question_index", "?")
            total = status.get("total_questions", "?")

            logger.debug(f"[CallInitiator] Session state: {state} (Q{q_idx}/{total})")

            if state == "complete":
                logger.info("[CallInitiator] ✅ Session completed — all questions answered.")
                return "complete"

        except Exception as e:
            logger.warning(f"[CallInitiator] Error polling session status: {e}")

        time.sleep(poll_interval)

    logger.warning(f"[CallInitiator] Session did not complete within {timeout}s")
    return "timeout"
