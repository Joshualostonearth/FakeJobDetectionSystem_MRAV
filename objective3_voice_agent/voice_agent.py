"""
voice_agent.py — Master Orchestrator for Objective 3.

Public API:
    run_voice_verification(claim_id, entity, config, override_phone, event_push) -> dict

This is the only file the main pipeline needs to import from this module.
All other files are internal implementation details.

__main__ block: starts FastAPI server + Ngrok, then keeps process alive.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import yaml
from dotenv import load_dotenv

from consensus_handoff import push_voice_signals
from env_check import check_env, check_ollama
from fact_extractor import extract_facts
from recording_handler import wait_for_recording
from schemas import PipelineResult, SourcedNumber, VoiceVerificationResult
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
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────

def run_voice_verification(
    claim_id: str,
    entity: dict,
    config: dict,
    override_phone: Optional[str] = None,
    event_push: Optional[Callable[[str, int], None]] = None,
) -> dict:
    """
    End-to-end voice verification pipeline.

    Args:
        claim_id:     Unique identifier for this verification run
        entity:       Job offer entity dict — must have: company_name, role, location
                      Optional: recruiter_name
                      The 'phone' field is deliberately ignored.
        config:       voice_agent_config dict (from load_config())
        override_phone: Developer test override — only accepted from the dashboard.
                        Skips number sourcing when set.
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
    wav_path_used: Optional[str] = None

    try:
        # ── Step 1: Source phone number ────────────────────────────────
        push("Sourcing company phone number...", step=1)

        if override_phone:
            push(f"Using dashboard override number: {override_phone}", step=1)
            sourced = SourcedNumber(
                phone_number=override_phone,
                source="dashboard_override",
                company_name_matched=entity.get("company_name", ""),
                confidence=1.0,
            )
        else:
            from number_sourcer import source_official_number
            sourced = source_official_number(
                entity.get("company_name", ""),
                entity.get("location", ""),
                config,
            )
            if sourced is None:
                push("No official number found — skipping call.", step=1)
                result.skip_reason = "no_official_number_found"
                return result.model_dump()

        result.sourced_number = sourced.phone_number
        push(f"Number sourced: {sourced.phone_number} (via {sourced.source})", step=1)

        # ── Step 2: Generate verification audio ────────────────────────
        push("Generating verification script and synthesising audio...", step=2)
        asyncio.run(generate_verification_audio(entity, str(audio_path), config))
        push("Audio ready.", step=2)

        # ── Step 3: Initiate Twilio call ───────────────────────────────
        push(f"Dialling {sourced.phone_number}...", step=3)

        from audio_server import pending_recordings
        from call_initiator import initiate_verification_call

        public_base = config.get("public_base_url", "")
        twiml_url = f"{public_base}/twiml/{claim_id}"
        webhook_url = f"{public_base}/webhook/call-complete"

        call_sid = initiate_verification_call(
            sourced.phone_number, twiml_url, webhook_url, config
        )
        push(f"Call placed. SID={call_sid}. Waiting for answer...", step=3)

        # ── Step 4: Wait for recording ─────────────────────────────────
        push("Waiting for recording (speak your response after the tone)...", step=4)
        wav_path_used = wait_for_recording(
            call_sid,
            pending_recordings,
            timeout_seconds=config.get("recording_timeout", 90),
        )

        if wav_path_used is None:
            push("Recording not received (no answer or timeout).", step=4)
            result.skip_reason = "recording_not_received"
            # Still push zero-signal to Obj2 so it knows the call was attempted
            push_voice_signals(
                claim_id,
                VoiceVerificationResult(
                    company_reached=False,
                    company_denied_role=False,
                    suspicious_response=False,
                    response_summary="No answer or recording not received",
                    confidence=0.0,
                ),
                sourced,
            )
            return result.model_dump()

        push(f"Recording downloaded: {wav_path_used}", step=4)

        # ── Step 5: Transcribe ─────────────────────────────────────────
        push("Transcribing recording...", step=5)
        transcription = transcribe_recording(wav_path_used, config)
        result.transcript = transcription["transcript"]
        push(
            f"Transcript ({transcription['word_count']} words): "
            f"\"{transcription['transcript'][:80]}{'...' if len(transcription['transcript']) > 80 else ''}\"",
            step=5,
        )

        # ── Step 6: Extract facts ──────────────────────────────────────
        push("Extracting verification signals via LLM...", step=6)
        facts = extract_facts(transcription["transcript"], entity, config)
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
        for path in [audio_path, wav_path_used]:
            if path:
                Path(path).unlink(missing_ok=True)


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
    from audio_server import set_config, start_ngrok, start_server

    # Inject config into audio_server
    set_config(config)

    # Start FastAPI server in background thread
    start_server(port=config["server_port"])

    # Start Ngrok tunnel
    public_url = start_ngrok(config["server_port"], config)
    config["public_base_url"] = public_url

    print("\n" + "=" * 60)
    print("  ✅  Objective 3 — Voice Verification Agent Ready")
    print("=" * 60)
    print(f"  Dashboard : {public_url}/")
    print(f"  Webhook   : {public_url}/webhook/call-complete")
    print(f"  SSE stream: {public_url}/stream/<call_id>")
    print("=" * 60)
    print("\n  Open the dashboard to run a live test call.\n")
    print("  Press Ctrl+C to stop.\n")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Obj3] Shutting down.")
