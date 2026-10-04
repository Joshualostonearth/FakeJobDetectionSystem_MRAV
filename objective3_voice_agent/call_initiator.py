"""
call_initiator.py — Step 3b: Initiate and track the Twilio verification call.

Creates the call with TwiML: Play MP3 → Record response → Say goodbye → Hangup.
Polls Twilio for terminal call status after initiating.
"""

import logging
import os
import time

from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv(override=False)

logger = logging.getLogger(__name__)

TWIML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
    <Record
        maxLength="60"
        timeout="5"
        recordingStatusCallback="{webhook_url}"
        recordingStatusCallbackMethod="POST"
    />
    <Say>Thank you for your response. Goodbye.</Say>
    <Hangup/>
</Response>"""


def _get_client() -> Client:
    """Build and return an authenticated Twilio REST client."""
    return Client(
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN"),
    )


def initiate_verification_call(
    to_number: str,
    twiml_url: str,
    webhook_url: str,
    config: dict,
) -> str:
    """
    Place a Twilio call that plays the verification audio and records the response.

    Args:
        to_number:   Destination phone number in E.164 format
        twiml_url:   Public URL to the TwiML instructions
        webhook_url: Twilio will POST recording details here when recording is ready
        config:      voice_agent_config dict (timeouts)

    Returns:
        Twilio Call SID string
    """
    client = _get_client()

    call = client.calls.create(
        url=twiml_url,
        to=to_number,
        from_=os.getenv("TWILIO_PHONE_NUMBER"),
        status_callback=webhook_url,
        status_callback_method="POST",
    )

    logger.info(f"[CallInitiator] Call placed → SID={call.sid}, to={to_number}")
    return call.sid


def wait_for_call_completion(
    call_sid: str,
    client: Client = None,
    timeout: int = 60,
) -> str:
    """
    Poll Twilio every 5 seconds until the call reaches a terminal status.

    Terminal statuses: completed, failed, busy, no-answer, canceled

    Args:
        call_sid: Twilio Call SID
        client:   Twilio Client (creates a new one if not provided)
        timeout:  Maximum seconds to wait before returning "timeout"

    Returns:
        Final call status string
    """
    if client is None:
        client = _get_client()

    terminal = {"completed", "failed", "busy", "no-answer", "canceled"}
    start = time.time()

    while time.time() - start < timeout:
        try:
            call = client.calls(call_sid).fetch()
            logger.debug(f"[CallInitiator] Call {call_sid} status: {call.status}")
            if call.status in terminal:
                logger.info(f"[CallInitiator] Call {call_sid} finished: {call.status}")
                return call.status
        except Exception as e:
            logger.warning(f"[CallInitiator] Error polling call status: {e}")

        time.sleep(5)

    logger.warning(f"[CallInitiator] Call {call_sid} did not complete within {timeout}s")
    return "timeout"
