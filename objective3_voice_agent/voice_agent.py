"""
voice_agent.py — Master Orchestrator for Objective 3.

Public API:
    run_voice_verification(claim_id, entity, config, event_push) -> dict

This is the only file the main pipeline needs to import from this module.
All other files are internal implementation details.

Uses WebRTC + call_bridge for the voice call instead of Twilio. The call is
simulated: no phone number is dialled — whoever answers on the WebRTC page
(http://localhost:3000) plays the company.
The call_bridge (whisper_Test/call_bridge.py) handles:
  - Session creation + question generation (Ollama)
  - Real-time STT (WhisperLive) during the WebRTC call
  - TTS (Edge-TTS) for agent voice output
  - Answer processing + response recording

__main__ block: starts FastAPI server, then keeps process alive.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

import yaml
from dotenv import load_dotenv

from consensus_handoff import push_voice_signals
from env_check import check_env, check_ollama
from fact_extractor import extract_facts
from recording_handler import wait_for_recording
from voice_schemas import PipelineResult, SourcedNumber, VoiceVerificationResult
from script_generator import generate_verification_audio
from transcriber import transcribe_recording

load_dotenv(override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load voice_agent_config.yaml from the config/ directory."""
    config_path = Path(__file__).parent / "config" / "voice_agent_config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["ollama_model"] = os.getenv("OLLAMA_MODEL", config.get("ollama_model", "ministral-3"))
    return config


# ──────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────

def run_voice_verification(
    claim_id: str,
    entity: dict,
    config: dict,
    event_push: Optional[Callable[[str, int], None]] = None,
) -> dict:
    """
    End-to-end voice verification pipeline using WebRTC.

    Args:
        claim_id:     Unique identifier for this verification run
        entity:       Job offer entity dict — must have: company_name, role, location
                      Optional: recruiter_name
                      The 'phone' field is deliberately ignored.
        config:       voice_agent_config dict (from load_config())
        event_push:   Optional callback(text, step_number) for SSE progress events.

    Returns:
        PipelineResult serialised as a dict.
    """
    def push(text: str, step: int = None):
        logger.info(f"[Obj3 Step {step}] {text}")
        if event_push:
            event_push(text, step)

    result = PipelineResult(claim_id=claim_id, pipeline_completed=False)
    audio_path = Path(__file__).parent / "temp" / "audio" / f"{claim_id}_verification.mp3"

    try:
        # ── Step 1: Simulated call target ──────────────────────────────
        # No number is dialled; the company is played by whoever answers
        # on the WebRTC page.
        sourced = SourcedNumber(
            phone_number="",
            source="webrtc_simulation",
            company_name_matched=entity.get("company_name", ""),
            confidence=1.0,
        )
        push("Simulated call — answer it on the WebRTC page (http://localhost:3000).", step=1)

        # ── Step 2: Generate verification audio ────────────────────────
        push("Generating verification script and synthesising audio...", step=2)
        asyncio.run(generate_verification_audio(entity, str(audio_path), config))
        push("Audio ready.", step=2)

        # ── Step 3: Initiate WebRTC call via bridge ────────────────────
        push(f"Starting WebRTC verification session...", step=3)

        from call_initiator import initiate_verification_call, wait_for_session_completion

        session_result = initiate_verification_call(entity, config)

        if not session_result.get("success"):
            error_msg = session_result.get("error", "Unknown error starting session")
            push(f"WebRTC session failed: {error_msg}", step=3)
            result.skip_reason = "webrtc_session_failed"
            # Still push zero-signal to Obj2 so it knows the call was attempted
            push_voice_signals(
                claim_id,
                VoiceVerificationResult(
                    company_reached=False,
                    company_denied_role=False,
                    suspicious_response=False,
                    response_summary=f"WebRTC session failed: {error_msg}",
                    confidence=0.0,
                ),
                sourced,
            )
            return result.model_dump()

        session_id = session_result.get("session_id", "unknown")
        total_q = session_result.get("total_questions", 0)
        push(f"Session started: {session_id} ({total_q} questions). Waiting for call...", step=3)

        # ── Step 4: Wait for session completion ────────────────────────
        push("Waiting for WebRTC call to complete (answer the call in the WebRTC dashboard)...", step=4)

        transcript_data = wait_for_recording(
            session_id=session_id,
            config=config,
            timeout_seconds=config.get("recording_timeout", 300),
        )

        if transcript_data is None:
            push("Session not completed (timeout or no call).", step=4)
            result.skip_reason = "session_not_completed"
            # Still push zero-signal to Obj2
            push_voice_signals(
                claim_id,
                VoiceVerificationResult(
                    company_reached=False,
                    company_denied_role=False,
                    suspicious_response=False,
                    response_summary="WebRTC session did not complete",
                    confidence=0.0,
                ),
                sourced,
            )
            return result.model_dump()

        push(f"Call completed: {transcript_data['word_count']} words recorded", step=4)

        # ── Step 5: Process transcript ─────────────────────────────────
        push("Processing call transcript...", step=5)

        # The WebRTC bridge already produces transcripts in real-time
        # Use the combined Q&A transcript directly
        transcript_text = transcript_data.get("transcript", "")
        result.transcript = transcript_text

        push(
            f"Transcript ({transcript_data['word_count']} words): "
            f"\"{transcript_text[:80]}{'...' if len(transcript_text) > 80 else ''}\"",
            step=5,
        )

        # ── Step 6: Extract facts ──────────────────────────────────────
        push("Extracting verification signals via LLM...", step=6)
        facts = extract_facts(transcript_text, entity, config)
        push(
            f"Extraction done — company_reached={facts.company_reached}, "
            f"confidence={facts.confidence}",
            step=6,
        )

        # ── Step 7: Push to consensus engine ───────────────────────────
        push("Pushing signals to consensus engine...", step=7)
        signals = push_voice_signals(claim_id, facts, sourced)
        push("Pipeline complete. ✅", step=7)

        result.pipeline_completed = True
        result.voice_signals = signals
        return result.model_dump()

    except Exception as e:
        logger.error(f"[Obj3] Unhandled error in pipeline: {e}", exc_info=True)
        result.error = str(e)
        if event_push:
            event_push(f"Pipeline error: {e}", None)
        return result.model_dump()

    finally:
        # Always clean up temp files — whether pipeline succeeded or failed
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 0a — env validation
    check_env()

    # Step 0b — Ollama check
    check_ollama()

    # Load config
    config = load_config()

    # Import here (after env check passes)
    from audio_server import set_config, start_server

    # Inject config into audio_server
    set_config(config)

    # Start FastAPI server in background thread
    start_server(port=config["server_port"])

    print("\n" + "=" * 60)
    print("  ✅  Objective 3 — Voice Verification Agent Ready (WebRTC)")
    print("=" * 60)
    print(f"  Dashboard : http://localhost:{config['server_port']}/")
    print(f"  SSE stream: http://localhost:{config['server_port']}/stream/<call_id>")
    print("=" * 60)
    print("\n  ⚠️  Make sure the following are also running:")
    print("    1. WebRTC bridge:  python whisper_Test/call_bridge.py  (port 5000)")
    print("    2. WebRTC server:  cd webrtc-call/server && node server.js  (port 3000)")
    print("\n  Open the dashboard to run a live test call.\n")
    print("  Press Ctrl+C to stop.\n")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Obj3] Shutting down.")
