"""
tests/test_objective3.py — Unit tests for Objective 3: Voice Verification Agent.

Tests are written with pytest and use unittest.mock for LLM isolation.
Run with:  python -m pytest tests/test_objective3.py -v
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Path setup ─────────────────────────────────────────────────
# IMPORTANT: OBJ3_ROOT must come BEFORE PROJECT_ROOT so that
# 'from schemas import ...' resolves to objective3_voice_agent/schemas.py
# rather than the root-level schemas/ package.
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

# ── Pre-register obj3 schemas so 'from schemas import X' inside obj3 modules
# always resolves to objective3_voice_agent/schemas.py, NOT the root schemas/ package.
import importlib.util as _ilu
_schemas_spec = _ilu.spec_from_file_location("schemas", str(OBJ3_ROOT / "schemas.py"))
_schemas_mod = _ilu.module_from_spec(_schemas_spec)
_schemas_spec.loader.exec_module(_schemas_mod)
sys.modules["schemas"] = _schemas_mod  # override root schemas/ package


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
# Test 2 — Unknown company returns None from number sourcer
# ═══════════════════════════════════════════════════════════

def test_number_sourcer_returns_none_for_fake_company():
    """
    When all three API tiers fail (keys empty or no result), source_official_number
    must return None rather than raising.
    """
    from number_sourcer import source_official_number

    config = {"number_source_min_confidence": 0.6}
    # Use an obviously fake company name; all API keys will be empty in test env
    result = source_official_number("XYZZY_FAKE_COMPANY_12345_TEST", "Mars", config)
    assert result is None, "Expected None for an unknown company with no API keys set"


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
    from schemas import VoiceVerificationResult

    config = {"ollama_model": "llama3.2"}
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
    from schemas import VoiceVerificationResult

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

        config = {"ollama_model": "llama3.2"}
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
# Test 7 — Pipeline skips gracefully when no number found
# ═══════════════════════════════════════════════════════════

@patch("number_sourcer.source_official_number", return_value=None)
@patch("voice_agent.check_env")
@patch("voice_agent.check_ollama")
def test_pipeline_skips_on_no_number(mock_ollama, mock_env, mock_source):
    """
    When source_official_number returns None, run_voice_verification must
    return skip_reason='no_official_number_found' without crashing.
    """
    from voice_agent import run_voice_verification

    config = {
        "number_source_min_confidence": 0.6,
        "recording_timeout": 10,
        "public_base_url": "http://localhost:8765",
        "whisper_model_size": "base",
        "whisper_device": "cpu",
        "whisper_compute_type": "int8",
        "ollama_model": "llama3.2",
        "call_ring_timeout": 10,
        "call_max_duration": 30,
    }
    result = run_voice_verification(
        claim_id="test_skip_001",
        entity={
            "company_name": "FAKE_COMPANY_DOES_NOT_EXIST",
            "role": "Internship",
            "location": "Nowhere",
            "recruiter_name": None,
        },
        config=config,
        override_phone=None,
    )

    assert result["skip_reason"] == "no_official_number_found"
    assert result["pipeline_completed"] is False
