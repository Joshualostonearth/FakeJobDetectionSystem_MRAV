"""
tests/test_dst.py — Dempster-Shafer combination (obj2dst.py) and the
Obj2 → Obj3 import path used by voice_verification_bridge.py.

Run with:
    python -m pytest tests/test_dst.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
OBJ3_ROOT = PROJECT_ROOT / "objective3_voice_agent"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(1, str(OBJ3_ROOT))

from obj2dst import DSTEngine
from voice_to_mass import voice_transcript_to_mass


@pytest.fixture
def dst():
    return DSTEngine()


def _legit_doc(dst):
    return dst.assign_mass_function(0.7, "legitimate_job_record", 0, metadata_quality=0.5)


def assert_mass_sums_to_one(mass):
    total = mass["REAL"] + mass["FAKE"] + mass["UNCERTAIN"]
    assert total == pytest.approx(1.0), f"masses sum to {total}"


def test_weak_evidence_goes_to_uncertain_not_opposite(dst):
    """A weak legitimate record must not produce FAKE mass."""
    mass = _legit_doc(dst)
    assert mass["FAKE"] == 0.0
    assert mass["REAL"] > 0.0
    assert_mass_sums_to_one(mass)


def test_agreeing_legit_evidence_leans_real(dst):
    combined = dst.combine_all_evidence([_legit_doc(dst)] * 3)
    assert_mass_sums_to_one(combined)
    assert combined["REAL"] > combined["FAKE"]
    assert "Real" in dst.get_belief_strength(combined)


def test_inconclusive_call_does_not_change_belief(dst):
    """A fully uncertain voice mass is the identity element of Dempster's rule."""
    docs = [_legit_doc(dst)] * 3
    no_answer = voice_transcript_to_mass(
        {"company_reached": False, "source_confidence": 0.9, "extraction_confidence": 0.0}
    )
    before = dst.combine_all_evidence(docs)
    after = dst.combine_all_evidence(docs + [no_answer])
    for key in ("REAL", "FAKE", "UNCERTAIN"):
        assert after[key] == pytest.approx(before[key], abs=1e-6)


def test_confirming_call_strengthens_real(dst):
    docs = [_legit_doc(dst)] * 3
    good_call = voice_transcript_to_mass({
        "company_reached": True,
        "role_actively_hiring": True,
        "recruiter_recognized": True,
        "company_denied_role": False,
        "suspicious_response": False,
        "source_confidence": 0.9,
        "extraction_confidence": 0.9,
    })
    before = dst.combine_all_evidence(docs)
    after = dst.combine_all_evidence(docs + [good_call])
    assert after["REAL"] > before["REAL"]
    assert_mass_sums_to_one(after)


def test_denial_call_moves_scam_evidence_to_fake(dst):
    scam = dst.assign_mass_function(0.7, "scam_report", 1, metadata_quality=0.3)
    denial = voice_transcript_to_mass({
        "company_reached": True,
        "company_denied_role": True,
        "suspicious_response": True,
        "source_confidence": 0.9,
        "extraction_confidence": 0.9,
    })
    combined = dst.combine_all_evidence([scam, scam, denial])
    assert combined["FAKE"] > combined["REAL"]
    assert "Fake" in dst.get_belief_strength(combined)


def test_two_key_masses_still_supported(dst):
    """Legacy masses without UNCERTAIN are treated as having none left over."""
    combined = dst.dempster_combine({"REAL": 0.8, "FAKE": 0.2}, {"REAL": 0.6, "FAKE": 0.4})
    assert combined["UNCERTAIN"] == pytest.approx(0.0)
    assert combined["REAL"] + combined["FAKE"] == pytest.approx(1.0)


def test_total_conflict_is_uncertain(dst):
    combined = dst.dempster_combine({"REAL": 1.0, "FAKE": 0.0}, {"REAL": 0.0, "FAKE": 1.0})
    assert combined["UNCERTAIN"] == 1.0
    assert combined["conflict"] == 1.0


def test_no_evidence_is_uncertain(dst):
    combined = dst.combine_all_evidence([])
    assert combined["UNCERTAIN"] == 1.0
    assert dst.get_belief_strength(combined).startswith("UNCERTAIN")


def test_objective3_importable_after_root_schemas():
    """voice_verification_bridge imports root schemas/ first; Obj3 must still import."""
    import schemas.credibility_schema  # noqa: F401  (root package, as the bridge does)
    from objective3_voice_agent import run_voice_verification, PipelineResult  # noqa: F401
