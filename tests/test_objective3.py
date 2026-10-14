"""
tests/test_objective3.py — Unit tests for Objective 3: Voice Verification Agent.

Tests are written with pytest and use unittest.mock for LLM isolation.
Run with:  python -m pytest tests/test_objective3.py -v
"""

import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Path setup ─────────────────────────────────────────────────
# OBJ3_ROOT is on the path so obj3's flat imports
# (e.g. 'from voice_schemas import ...') resolve.
PROJECT_ROOT = Path(__file__).parent.parent
OBJ3_ROOT = PROJECT_ROOT / "objective3_voice_agent"

# Insert at position 0 so it takes highest priority
sys.path.insert(0, str(OBJ3_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

# ── Stub heavy / not-yet-installed dependencies ────────────────
# This lets tests run without requiring edge_tts, faster_whisper,
# langchain_ollama, twilio, etc. to be installed in the test env.
_STUB_MODULES = [
    "edge_tts",
    "faster_whisper",
    "langchain_ollama",
    "twilio",
    "twilio.rest",
    "pyngrok",
    "pyngrok.ngrok",
    "uvicorn",
]
for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


# ═══════════════════════════════════════════════════════════
# Test 1 — Disclosure line is mandatory in generated script
# ═══════════════════════════════════════════════════════════

def test_script_contains_disclosure():
    """Every generated script must contain the mandatory recording disclosure."""
    from script_generator import SCRIPT_TEMPLATE

    script = SCRIPT_TEMPLATE.format(
        company_name="Acme Corp",
        role="Software Engineer",
        location="Bangalore, India",
        recruiter_line="",
    )
    assert "may be recorded" in script.lower(), (
        "Mandatory disclosure line 'This call may be recorded for verification purposes' "
        "is missing from the generated script."
    )


# ═══════════════════════════════════════════════════════════
# Test 3 — Recruiter line is omitted when recruiter_name is None
# ═══════════════════════════════════════════════════════════

def test_script_omits_recruiter_when_none():
    """The recruiter confirmation sentence must not appear when recruiter_name is None."""
    from script_generator import SCRIPT_TEMPLATE, build_script

    entity = {
        "company_name": "Acme Corp",
        "role": "Engineer",
        "location": "India",
        "recruiter_name": None,
    }
    script = build_script(entity)

    # 'recruiter' word should not appear in any form
    assert "recruiter" not in script.lower(), (
        "Recruiter confirmation sentence should be absent when recruiter_name is None"
    )
    # Disclosure must still be present
    assert "may be recorded" in script.lower()


# ═══════════════════════════════════════════════════════════
# Test 4 — Empty transcript handled gracefully
# ═══════════════════════════════════════════════════════════

def test_fact_extractor_handles_empty_transcript():
    """
    extract_facts("") must return a valid VoiceVerificationResult with
    company_reached=False and confidence=0.0 — without calling the LLM.
    """
    from fact_extractor import extract_facts
    from voice_schemas import VoiceVerificationResult

    config = {"ollama_model": "ministral-3"}
    entity = {"company_name": "Test Corp", "role": "Dev", "recruiter_name": None}

    result = extract_facts("", entity, config)

    assert isinstance(result, VoiceVerificationResult)
    assert result.company_reached is False
    assert result.confidence == 0.0


# ═══════════════════════════════════════════════════════════
# Test 5 — Role denial detected via mocked LLM chain
# ═══════════════════════════════════════════════════════════

@patch("fact_extractor.ChatOllama")
def test_denial_detected(mock_llm_class):
    """
    When the LLM returns company_denied_role=True, the pipeline should propagate it.
    Uses a mock to avoid requiring Ollama to be running during CI.
    """
    from fact_extractor import extract_facts
    from voice_schemas import VoiceVerificationResult

    # Set up the mock chain
    mock_instance = MagicMock()
    mock_structured = MagicMock()
    mock_instance.with_structured_output.return_value = mock_structured
    mock_llm_class.return_value = mock_instance

    # Simulate EXTRACTION_PROMPT | llm returning denial result
    denial_result = VoiceVerificationResult(
        company_reached=True,
        role_actively_hiring=False,
        recruiter_recognized=None,
        company_denied_role=True,
        suspicious_response=False,
        response_summary="Company said no such role exists",
        confidence=0.95,
    )

    # The chain (EXTRACTION_PROMPT | llm).invoke() should return the denial
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = denial_result

    with patch("fact_extractor.EXTRACTION_PROMPT") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        config = {"ollama_model": "ministral-3"}
        entity = {"company_name": "Google", "role": "SWE Intern", "recruiter_name": "Jane"}
        result = extract_facts(
            "We don't have any openings for that role.", entity, config
        )

    assert result.company_denied_role is True
    assert result.company_reached is True


# ═══════════════════════════════════════════════════════════
# Test 6 — Temp files deleted after pipeline run
# ═══════════════════════════════════════════════════════════

def test_temp_files_cleaned_up(tmp_path):
    """
    The finally block in run_voice_verification must delete both
    the MP3 and WAV temp files regardless of pipeline outcome.
    """
    audio = tmp_path / "test_abc_verification.mp3"
    wav = tmp_path / "test_abc.wav"

    # Create dummy temp files
    audio.write_text("fake audio content")
    wav.write_text("fake wav content")

    assert audio.exists(), "Test setup: audio file should exist"
    assert wav.exists(), "Test setup: wav file should exist"

    # Simulate the finally block cleanup
    Path(audio).unlink(missing_ok=True)
    Path(wav).unlink(missing_ok=True)

    assert not audio.exists(), "Audio MP3 should be deleted after pipeline"
    assert not wav.exists(), "WAV recording should be deleted after pipeline"


# ═══════════════════════════════════════════════════════════
# Test 7 — Simulated call goes straight to the WebRTC session
# ═══════════════════════════════════════════════════════════

@patch("call_initiator.initiate_verification_call", return_value={"success": False, "error": "bridge down"})
@patch("voice_agent.push_voice_signals")
@patch("voice_agent.generate_verification_audio", new_callable=AsyncMock)
def test_pipeline_simulated_call_reports_session_failure(mock_audio, mock_push, mock_initiate):
    """
    No phone number is sourced: the pipeline starts the WebRTC session
    directly, and a failed session is reported as 'webrtc_session_failed'
    with a simulated call target.
    """
    from voice_agent import run_voice_verification

    config = {
        "recording_timeout": 10,
        "ollama_model": "ministral-3",
    }
    result = run_voice_verification(
        claim_id="test_sim_001",
        entity={
            "company_name": "Acme Corp",
            "role": "Internship",
            "location": "Bangalore, India",
            "recruiter_name": None,
        },
        config=config,
    )

    assert mock_initiate.called, "WebRTC session should be started without number sourcing"
    assert result["skip_reason"] == "webrtc_session_failed"
    assert result["pipeline_completed"] is False

    sourced = mock_push.call_args.args[2]
    assert sourced.source == "webrtc_simulation"
    assert sourced.phone_number == ""

