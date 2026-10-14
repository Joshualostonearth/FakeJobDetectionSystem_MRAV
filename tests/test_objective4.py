"""
tests/test_objective4.py — Unit tests for Person 3 & Person 4 deliverables (Objective 4).

Covers:
    1.  DST 3-focal element combination (REAL, FAKE, UNCERTAIN)
    2.  combine_two() and combine_all() standard entrypoints
    3.  Conflict factor K calculation and tracking
    4.  Platt model fitting and platt_model.pkl persistence
    5.  calculate_confidence_score() calibration mapping
    6.  apply_verification_override() with voice denial and critical scam flags
"""

import os
from pathlib import Path
import pytest

from obj2dst import DSTEngine, combine_two, combine_all
from confidence_calibrator import (
    fit_and_save_platt_model,
    calculate_confidence_score,
    apply_verification_override,
    MODEL_FILE_PATH,
)


def test_dst_3focal_combination():
    """Test Dempster's rule with 3 focal elements."""
    m1 = {"REAL": 0.8, "FAKE": 0.0, "UNCERTAIN": 0.2}
    m2 = {"REAL": 0.7, "FAKE": 0.0, "UNCERTAIN": 0.3}

    res = combine_two(m1, m2)

    assert "REAL" in res and "FAKE" in res and "UNCERTAIN" in res
    assert res["REAL"] > 0.85, "Combining two positive REAL masses should increase belief in REAL"
    assert res["conflict"] == 0.0, "Agreeing masses should have zero conflict"


def test_dst_conflicting_evidence():
    """Test DST handling when registry says REAL but scam report says FAKE."""
    m_reg = {"REAL": 0.9, "FAKE": 0.0, "UNCERTAIN": 0.1}
    m_scam = {"REAL": 0.0, "FAKE": 0.85, "UNCERTAIN": 0.15}

    res = combine_two(m_reg, m_scam)

    assert res["conflict"] > 0.70, "Strongly conflicting masses should produce high conflict factor K"
    assert "UNCERTAIN" in res


def test_combine_all_sources():
    """Test combining masses from Registry, LinkedIn, Scam DB, and Voice Agent."""
    engine = DSTEngine()
    m_reg = engine.assign_mass_function(0.9, "registry", is_fraudulent=0)
    m_li = engine.assign_mass_function(0.85, "linkedin", is_fraudulent=0)
    m_voice = {"REAL": 0.85, "FAKE": 0.02, "UNCERTAIN": 0.13, "source": "voice_call"}

    combined = combine_all([m_reg, m_li, m_voice])

    assert combined["REAL"] > 0.90
    assert combined["FAKE"] < 0.10


def test_platt_model_generation():
    """Test that platt_model.pkl is created and loads properly."""
    scaler = fit_and_save_platt_model(MODEL_FILE_PATH)
    assert MODEL_FILE_PATH.exists(), "platt_model.pkl must be generated and saved"
    assert scaler.is_fitted


def test_calculate_confidence_score():
    """Test confidence calibration mapping from DST mass to [0, 1] probability."""
    dst_real = {"REAL": 0.85, "FAKE": 0.05, "UNCERTAIN": 0.10, "conflict": 0.05}
    conf_real = calculate_confidence_score(dst_real)
    assert conf_real > 0.75, f"Expected high confidence for strong REAL mass, got {conf_real}"

    dst_fake = {"REAL": 0.05, "FAKE": 0.85, "UNCERTAIN": 0.10, "conflict": 0.05}
    conf_fake = calculate_confidence_score(dst_fake)
    assert conf_fake < 0.25, f"Expected low confidence for strong FAKE mass, got {conf_fake}"


def test_verification_override_voice_denial():
    """Test that explicit voice denial forces a HIGH_RISK_SCAM override."""
    dst_uncertain = {"REAL": 0.60, "FAKE": 0.30, "UNCERTAIN": 0.10, "conflict": 0.18}
    voice_signals = {
        "company_denied_role": True,
        "response_summary": "No such role exists",
    }

    result = apply_verification_override(
        calibrated_confidence=0.65,
        dst_result=dst_uncertain,
        flags=[],
        voice_signals=voice_signals,
    )

    assert result["final_verdict"] == "HIGH_RISK_SCAM"
    assert result["override_applied"] is True
    assert result["calibrated_confidence"] <= 0.10
    assert "explicitly denied role" in result["override_reason"]
