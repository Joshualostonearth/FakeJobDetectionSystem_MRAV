"""
tests/test_voice_to_mass.py — Comprehensive tests for voice_transcript_to_mass().

Covers:
    1.  All-confirmations scenario (clearly REAL)
    2.  All-red-flags scenario (clearly FAKE)
    3.  Mixed signals (Uncertain territory)
    4.  Empty / None / garbage input (graceful degradation)
    5.  Confidence dampening (low confidence → pulled toward Uncertain)
    6.  Edge cases: partial signals, boundary values, single-signal dicts
    7.  Adapter: voice_result_to_analysis() round-trip
    8.  DST compatibility: output is directly usable by DSTEngine.dempster_combine()
    9.  Mass axiom validation: REAL + FAKE + UNCERTAIN ≈ 1.0

Run with:
    python -m pytest tests/test_voice_to_mass.py -v
"""

import sys
from pathlib import Path

import pytest

# ── Path setup ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
OBJ3_ROOT = PROJECT_ROOT / "objective3_voice_agent"
sys.path.insert(0, str(OBJ3_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

from voice_to_mass import (
    voice_transcript_to_mass,
    voice_result_to_analysis,
    SIGNAL_WEIGHTS,
    MAX_FOCAL_MASS,
    CONFIDENCE_FLOOR,
)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def assert_mass_sums_to_one(mass: dict, tol: float = 1e-4):
    """Every mass function must satisfy the axiom: sum of all masses = 1."""
    total = mass["REAL"] + mass["FAKE"] + mass["UNCERTAIN"]
    assert abs(total - 1.0) < tol, (
        f"Mass axiom violated: REAL={mass['REAL']:.6f} + FAKE={mass['FAKE']:.6f} "
        f"+ UNCERTAIN={mass['UNCERTAIN']:.6f} = {total:.6f} (expected 1.0)"
    )


def assert_non_negative(mass: dict):
    """All mass values must be ≥ 0."""
    for key in ("REAL", "FAKE", "UNCERTAIN"):
        assert mass[key] >= 0.0, f"Negative mass: {key}={mass[key]}"


# ═══════════════════════════════════════════════════════════════
# Test 1 — Clearly REAL: all confirmations, high confidence
# ═══════════════════════════════════════════════════════════════

def test_all_confirmations_yields_real():
    """
    Scenario: Company picked up, confirmed the role, recognised the
    recruiter, no denial, no suspicion.  High confidence all around.
    → Should produce a strongly Real-leaning mass.
    """
    analysis = {
        "company_reached":       True,
        "role_actively_hiring":  True,
        "recruiter_recognized":  True,
        "company_denied_role":   False,
        "suspicious_response":   False,
        "source_confidence":     0.95,
        "extraction_confidence": 0.90,
        "number_source":         "google_places",
        "response_summary":      "Yes, we are hiring for that role.",
    }

    mass = voice_transcript_to_mass(analysis)

    assert_mass_sums_to_one(mass)
    assert_non_negative(mass)
    assert mass["REAL"] > mass["FAKE"], "Expected REAL > FAKE for all-confirmations"
    assert mass["REAL"] > 0.30, f"Expected significant REAL mass, got {mass['REAL']}"
    assert mass["net_score"] > 0, "Net score should be positive"
    assert mass["source"] == "voice_call"

    print(f"\n[Test 1] All confirmations → REAL={mass['REAL']:.4f}, "
          f"FAKE={mass['FAKE']:.4f}, UNCERTAIN={mass['UNCERTAIN']:.4f}")


# ═══════════════════════════════════════════════════════════════
# Test 2 — Clearly FAKE: denial + suspicious + not reached
# ═══════════════════════════════════════════════════════════════

def test_all_red_flags_yields_fake():
    """
    Scenario: Company unreachable, explicit denial, suspicious response.
    → Should produce a strongly Fake-leaning mass.
    """
    analysis = {
        "company_reached":       False,
        "role_actively_hiring":  False,
        "recruiter_recognized":  False,
        "company_denied_role":   True,
        "suspicious_response":   True,
        "source_confidence":     0.80,
        "extraction_confidence": 0.85,
        "number_source":         "serpapi",
        "response_summary":      "No such role. Don't call again.",
    }

    mass = voice_transcript_to_mass(analysis)

    assert_mass_sums_to_one(mass)
    assert_non_negative(mass)
    assert mass["FAKE"] > mass["REAL"], "Expected FAKE > REAL for all red flags"
    assert mass["FAKE"] > 0.30, f"Expected significant FAKE mass, got {mass['FAKE']}"
    assert mass["net_score"] < 0, "Net score should be negative"

    print(f"\n[Test 2] All red flags → REAL={mass['REAL']:.4f}, "
          f"FAKE={mass['FAKE']:.4f}, UNCERTAIN={mass['UNCERTAIN']:.4f}")


# ═══════════════════════════════════════════════════════════════
# Test 3 — Mixed signals → Uncertain territory
# ═══════════════════════════════════════════════════════════════

def test_mixed_signals_yields_uncertain():
    """
    Scenario: Company picked up (good), but denied the role (bad).
    The conflicting signals should push the mass toward Uncertain.
    """
    analysis = {
        "company_reached":       True,     # +0.25
        "role_actively_hiring":  None,     # unknown
        "recruiter_recognized":  None,     # unknown
        "company_denied_role":   True,     # -0.40
        "suspicious_response":   False,    # mildly reassuring
        "source_confidence":     0.80,
        "extraction_confidence": 0.85,
    }

    mass = voice_transcript_to_mass(analysis)

    assert_mass_sums_to_one(mass)
    assert_non_negative(mass)
    # Net score should be negative (denial outweighs reaching company)
    assert mass["net_score"] < 0, "Denial should outweigh reaching the company"
    # But uncertainty should be significant because signals conflict
    assert mass["UNCERTAIN"] > 0.30, f"Expected high UNCERTAIN, got {mass['UNCERTAIN']}"

    print(f"\n[Test 3] Mixed → REAL={mass['REAL']:.4f}, "
          f"FAKE={mass['FAKE']:.4f}, UNCERTAIN={mass['UNCERTAIN']:.4f}")


# ═══════════════════════════════════════════════════════════════
# Test 4 — Empty input → fully Uncertain
# ═══════════════════════════════════════════════════════════════

def test_empty_input_yields_uncertain():
    """Empty dict, None, and garbage inputs should all produce full uncertainty."""
    for bad_input in [None, {}, [], "garbage", 42]:
        mass = voice_transcript_to_mass(bad_input)

        assert mass["REAL"] == 0.0
        assert mass["FAKE"] == 0.0
        assert mass["UNCERTAIN"] == 1.0
        assert mass["source"] == "voice_call"
        assert mass["strength"] == 0.0

    print("\n[Test 4] Empty/invalid inputs → UNCERTAIN=1.0 ✓")


# ═══════════════════════════════════════════════════════════════
# Test 5 — Low confidence dampening
# ═══════════════════════════════════════════════════════════════

def test_low_confidence_dampens_mass():
    """
    Same strong confirmations, but with very low confidence.
    → Should produce near-Uncertain mass despite positive signals.
    """
    high_conf_analysis = {
        "company_reached":       True,
        "role_actively_hiring":  True,
        "recruiter_recognized":  True,
        "company_denied_role":   False,
        "suspicious_response":   False,
        "source_confidence":     0.95,
        "extraction_confidence": 0.90,
    }

    low_conf_analysis = {
        **high_conf_analysis,
        "source_confidence":     0.10,
        "extraction_confidence": 0.10,
    }

    mass_high = voice_transcript_to_mass(high_conf_analysis)
    mass_low = voice_transcript_to_mass(low_conf_analysis)

    assert_mass_sums_to_one(mass_high)
    assert_mass_sums_to_one(mass_low)

    # Low confidence should produce less REAL mass
    assert mass_low["REAL"] < mass_high["REAL"], \
        "Low confidence should produce less REAL mass"
    # Low confidence should produce more UNCERTAIN mass
    assert mass_low["UNCERTAIN"] > mass_high["UNCERTAIN"], \
        "Low confidence should increase UNCERTAIN"

    print(f"\n[Test 5] High conf REAL={mass_high['REAL']:.4f}, "
          f"Low conf REAL={mass_low['REAL']:.4f}")


# ═══════════════════════════════════════════════════════════════
# Test 6 — Below confidence floor → near-zero mass
# ═══════════════════════════════════════════════════════════════

def test_below_confidence_floor():
    """
    When combined confidence < CONFIDENCE_FLOOR, mass should be
    nearly all Uncertain with only a tiny directional nudge.
    """
    analysis = {
        "company_reached":       True,
        "role_actively_hiring":  True,
        "source_confidence":     0.01,
        "extraction_confidence": 0.01,
    }

    mass = voice_transcript_to_mass(analysis)

    assert_mass_sums_to_one(mass)
    assert mass["UNCERTAIN"] > 0.95, \
        f"Below confidence floor, UNCERTAIN should dominate. Got {mass['UNCERTAIN']}"
    assert mass.get("low_confidence_warning") is True, \
        "Should flag low_confidence_warning"

    print(f"\n[Test 6] Below floor → UNCERTAIN={mass['UNCERTAIN']:.4f} ✓")


# ═══════════════════════════════════════════════════════════════
# Test 7 — Single signal: only company_reached=True
# ═══════════════════════════════════════════════════════════════

def test_single_signal_company_reached():
    """
    Only one signal present. Should produce a mild REAL lean.
    """
    analysis = {
        "company_reached": True,
        "source_confidence": 0.80,
        "extraction_confidence": 0.80,
    }

    mass = voice_transcript_to_mass(analysis)

    assert_mass_sums_to_one(mass)
    assert mass["REAL"] > 0, "company_reached=True should produce some REAL mass"
    assert mass["REAL"] < 0.40, "Single signal shouldn't produce overwhelming REAL"
    assert mass["UNCERTAIN"] > 0.50, "Should still be substantially uncertain"

    print(f"\n[Test 7] Single signal → REAL={mass['REAL']:.4f}, "
          f"UNCERTAIN={mass['UNCERTAIN']:.4f}")


# ═══════════════════════════════════════════════════════════════
# Test 8 — Single red flag: only company_denied_role=True
# ═══════════════════════════════════════════════════════════════

def test_single_red_flag_denial():
    """
    Only a denial signal. Should produce a moderate FAKE lean.
    Denial has the highest weight (-0.40) among red flags.
    """
    analysis = {
        "company_denied_role": True,
        "source_confidence": 0.80,
        "extraction_confidence": 0.85,
    }

    mass = voice_transcript_to_mass(analysis)

    assert_mass_sums_to_one(mass)
    assert mass["FAKE"] > mass["REAL"], "Denial should push toward FAKE"
    assert mass["net_score"] < 0

    print(f"\n[Test 8] Denial only → FAKE={mass['FAKE']:.4f}, "
          f"REAL={mass['REAL']:.4f}")


# ═══════════════════════════════════════════════════════════════
# Test 9 — Company not reached (False) triggers absence weight
# ═══════════════════════════════════════════════════════════════

def test_company_not_reached_absence_penalty():
    """
    company_reached=False should invoke the absence weight (-0.10),
    acting as a mild red flag.
    """
    analysis = {
        "company_reached": False,
        "source_confidence": 0.80,
        "extraction_confidence": 0.80,
    }

    mass = voice_transcript_to_mass(analysis)

    assert_mass_sums_to_one(mass)
    assert mass["FAKE"] > mass["REAL"], "Not reaching company should lean FAKE"
    assert mass["details"]["company_reached"] < 0, "Absence weight should be negative"

    print(f"\n[Test 9] Not reached → FAKE={mass['FAKE']:.4f}, "
          f"detail={mass['details']['company_reached']}")


# ═══════════════════════════════════════════════════════════════
# Test 10 — Mass axiom holds across all scenarios
# ═══════════════════════════════════════════════════════════════

def test_mass_axiom_across_scenarios():
    """
    REAL + FAKE + UNCERTAIN must equal 1.0 across a battery of
    different analysis configurations.
    """
    scenarios = [
        # (name, analysis_dict)
        ("all_true", {
            "company_reached": True, "role_actively_hiring": True,
            "recruiter_recognized": True, "company_denied_role": False,
            "suspicious_response": False, "source_confidence": 1.0,
            "extraction_confidence": 1.0,
        }),
        ("all_false", {
            "company_reached": False, "role_actively_hiring": False,
            "recruiter_recognized": False, "company_denied_role": True,
            "suspicious_response": True, "source_confidence": 1.0,
            "extraction_confidence": 1.0,
        }),
        ("all_none", {
            "company_reached": None, "role_actively_hiring": None,
            "recruiter_recognized": None, "company_denied_role": None,
            "suspicious_response": None, "source_confidence": 0.5,
            "extraction_confidence": 0.5,
        }),
        ("zero_confidence", {
            "company_reached": True, "source_confidence": 0.0,
            "extraction_confidence": 0.0,
        }),
        ("max_confidence", {
            "company_reached": True, "source_confidence": 1.0,
            "extraction_confidence": 1.0,
        }),
        ("partial_signals", {
            "company_reached": True, "company_denied_role": True,
            "source_confidence": 0.6, "extraction_confidence": 0.7,
        }),
    ]

    for name, analysis in scenarios:
        mass = voice_transcript_to_mass(analysis)
        assert_mass_sums_to_one(mass)
        assert_non_negative(mass)

    print(f"\n[Test 10] Mass axiom verified across {len(scenarios)} scenarios ✓")


# ═══════════════════════════════════════════════════════════════
# Test 11 — Output is DST-compatible with obj2dst.DSTEngine
# ═══════════════════════════════════════════════════════════════

def test_dst_compatibility():
    """
    The mass dict returned by voice_transcript_to_mass() must be directly
    consumable by DSTEngine.dempster_combine() from obj2dst.py.

    This means it must have "REAL" and "FAKE" keys with float values.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from obj2dst import DSTEngine

    analysis = {
        "company_reached": True,
        "role_actively_hiring": True,
        "source_confidence": 0.90,
        "extraction_confidence": 0.85,
    }

    voice_mass = voice_transcript_to_mass(analysis)

    # A registry mass from obj2dst
    dst = DSTEngine()
    registry_mass = dst.assign_mass_function(
        similarity_score=0.85,
        source_type="registry",
        is_fraudulent=0,
        metadata_quality=0.8,
    )

    # Combining should not raise
    combined = dst.dempster_combine(voice_mass, registry_mass)

    assert "REAL" in combined
    assert "FAKE" in combined
    assert combined["REAL"] + combined["FAKE"] <= 1.0 + 1e-6

    print(f"\n[Test 11] DST combine OK → REAL={combined['REAL']:.4f}, "
          f"FAKE={combined['FAKE']:.4f}, conflict={combined.get('conflict', 0):.4f}")


# ═══════════════════════════════════════════════════════════════
# Test 12 — Adapter: voice_result_to_analysis() round-trip
# ═══════════════════════════════════════════════════════════════

def test_adapter_round_trip():
    """
    voice_result_to_analysis() should convert a VoiceVerificationResult-like
    object into a dict that voice_transcript_to_mass() accepts.
    """
    # Mock a VoiceVerificationResult-like object
    class MockResult:
        company_reached = True
        role_actively_hiring = True
        recruiter_recognized = None
        company_denied_role = False
        suspicious_response = False
        confidence = 0.88
        response_summary = "Yes, we have that position open."

    analysis = voice_result_to_analysis(
        MockResult(),
        source_confidence=0.92,
        number_source="google_places",
    )

    # All expected keys should be present
    assert analysis["company_reached"] is True
    assert analysis["role_actively_hiring"] is True
    assert analysis["source_confidence"] == 0.92
    assert analysis["extraction_confidence"] == 0.88
    assert analysis["number_source"] == "google_places"

    # Should produce a valid mass
    mass = voice_transcript_to_mass(analysis)
    assert_mass_sums_to_one(mass)
    assert mass["REAL"] > mass["FAKE"]

    print(f"\n[Test 12] Adapter round-trip → REAL={mass['REAL']:.4f} ✓")


# ═══════════════════════════════════════════════════════════════
# Test 13 — Monotonicity: more confirmations → more REAL mass
# ═══════════════════════════════════════════════════════════════

def test_monotonicity_confirmations():
    """
    Adding more confirmations (keeping everything else equal) should
    monotonically increase REAL mass.
    """
    base = {
        "company_denied_role": False,
        "suspicious_response": False,
        "source_confidence": 0.85,
        "extraction_confidence": 0.85,
    }

    # Level 0: nothing confirmed
    level_0 = {**base}
    # Level 1: company reached
    level_1 = {**base, "company_reached": True}
    # Level 2: + role confirmed
    level_2 = {**base, "company_reached": True, "role_actively_hiring": True}
    # Level 3: + recruiter recognised
    level_3 = {**base, "company_reached": True, "role_actively_hiring": True,
               "recruiter_recognized": True}

    masses = [voice_transcript_to_mass(lvl) for lvl in [level_0, level_1, level_2, level_3]]

    for i in range(len(masses) - 1):
        assert masses[i + 1]["REAL"] >= masses[i]["REAL"], (
            f"Monotonicity violated: level {i} REAL={masses[i]['REAL']:.4f} > "
            f"level {i+1} REAL={masses[i+1]['REAL']:.4f}"
        )

    real_vals = ' → '.join(f"{m['REAL']:.4f}" for m in masses)
    print(f"\n[Test 13] Monotonicity: {real_vals} ✓")


# ═══════════════════════════════════════════════════════════════
# Test 14 — Monotonicity: more red flags → more FAKE mass
# ═══════════════════════════════════════════════════════════════

def test_monotonicity_red_flags():
    """
    Adding more red flags should monotonically increase FAKE mass.
    """
    base = {
        "company_reached": False,
        "source_confidence": 0.85,
        "extraction_confidence": 0.85,
    }

    # Level 0: just not reached
    level_0 = {**base}
    # Level 1: + suspicious
    level_1 = {**base, "suspicious_response": True}
    # Level 2: + denial
    level_2 = {**base, "suspicious_response": True, "company_denied_role": True}

    masses = [voice_transcript_to_mass(lvl) for lvl in [level_0, level_1, level_2]]

    for i in range(len(masses) - 1):
        assert masses[i + 1]["FAKE"] >= masses[i]["FAKE"], (
            f"Monotonicity violated: level {i} FAKE={masses[i]['FAKE']:.4f} > "
            f"level {i+1} FAKE={masses[i+1]['FAKE']:.4f}"
        )

    fake_vals = ' → '.join(f"{m['FAKE']:.4f}" for m in masses)
    print(f"\n[Test 14] Red flag monotonicity: {fake_vals} ✓")


# ═══════════════════════════════════════════════════════════════
# Test 15 — Realistic scenario: legitimate Google offer
# ═══════════════════════════════════════════════════════════════

def test_realistic_google_legitimate():
    """
    Simulates calling Google's official number.
    Receptionist confirms the SWE role and knows the recruiter.
    → Strongly Real.
    """
    analysis = {
        "company_reached":       True,
        "role_actively_hiring":  True,
        "recruiter_recognized":  True,
        "company_denied_role":   False,
        "suspicious_response":   False,
        "source_confidence":     0.95,   # Google Places API → high
        "extraction_confidence": 0.92,
        "number_source":         "google_places",
        "response_summary":      "Google receptionist confirmed SWE hiring. "
                                 "Recruiter Amy Chen is in HR team.",
    }

    mass = voice_transcript_to_mass(analysis)
    assert_mass_sums_to_one(mass)
    assert mass["REAL"] > 0.40, f"Google confirmed → strong REAL expected, got {mass['REAL']}"
    assert mass["FAKE"] < 0.20

    print(f"\n[Test 15] Google legit → REAL={mass['REAL']:.4f}, FAKE={mass['FAKE']:.4f}")


# ═══════════════════════════════════════════════════════════════
# Test 16 — Realistic scenario: scam impersonation
# ═══════════════════════════════════════════════════════════════

def test_realistic_scam_impersonation():
    """
    Simulates calling the official company number.
    Company says they have no such role and no such recruiter.
    → Strongly Fake.
    """
    analysis = {
        "company_reached":       True,    # We did reach the real company
        "role_actively_hiring":  False,
        "recruiter_recognized":  False,
        "company_denied_role":   True,    # They deny the role
        "suspicious_response":   False,   # Company was polite, just said no
        "source_confidence":     0.90,
        "extraction_confidence": 0.88,
        "number_source":         "opencorporates",
        "response_summary":      "We have no open positions for Data Analyst. "
                                 "Nobody named John Smith works here.",
    }

    mass = voice_transcript_to_mass(analysis)
    assert_mass_sums_to_one(mass)
    assert mass["FAKE"] > mass["REAL"], "Denied role → more FAKE than REAL"

    print(f"\n[Test 16] Scam impersonation → REAL={mass['REAL']:.4f}, "
          f"FAKE={mass['FAKE']:.4f}")


# ═══════════════════════════════════════════════════════════════
# Test 17 — Realistic scenario: no answer (inconclusive)
# ═══════════════════════════════════════════════════════════════

def test_realistic_no_answer():
    """
    Call placed but nobody picked up.  Transcript is essentially empty.
    → Should be mostly Uncertain with a slight FAKE lean.
    """
    analysis = {
        "company_reached":       False,
        "role_actively_hiring":  None,
        "recruiter_recognized":  None,
        "company_denied_role":   False,
        "suspicious_response":   False,
        "source_confidence":     0.85,
        "extraction_confidence": 0.20,   # Low — empty transcript
        "number_source":         "serpapi",
        "response_summary":      "No answer or recording not received",
    }

    mass = voice_transcript_to_mass(analysis)
    assert_mass_sums_to_one(mass)
    assert mass["UNCERTAIN"] > 0.60, \
        f"No answer → high uncertainty expected, got {mass['UNCERTAIN']}"

    print(f"\n[Test 17] No answer → UNCERTAIN={mass['UNCERTAIN']:.4f}")


# ═══════════════════════════════════════════════════════════════
# Test 18 — Details breakdown is populated
# ═══════════════════════════════════════════════════════════════

def test_details_breakdown_populated():
    """
    The 'details' dict in the output should contain per-signal
    contribution values for all known signals.
    """
    analysis = {
        "company_reached": True,
        "role_actively_hiring": True,
        "company_denied_role": False,
        "suspicious_response": False,
        "source_confidence": 0.80,
        "extraction_confidence": 0.80,
    }

    mass = voice_transcript_to_mass(analysis)

    assert "details" in mass
    details = mass["details"]

    # All signal keys from SIGNAL_WEIGHTS should appear
    for key in SIGNAL_WEIGHTS:
        assert key in details, f"Missing signal '{key}' in details breakdown"

    # company_reached=True should have positive contribution
    assert details["company_reached"] > 0
    # company_denied_role=False (red flag being false) → small positive
    assert details["company_denied_role"] >= 0

    print(f"\n[Test 18] Details: {details}")


# ═══════════════════════════════════════════════════════════════
# Test 19 — Confidence values are clamped to [0, 1]
# ═══════════════════════════════════════════════════════════════

def test_out_of_range_confidence_clamped():
    """
    Confidence values > 1.0 or < 0.0 should be clamped gracefully.
    """
    analysis = {
        "company_reached": True,
        "source_confidence": 5.0,       # way above 1.0
        "extraction_confidence": -2.0,  # below 0.0
    }

    # Should not raise
    mass = voice_transcript_to_mass(analysis)
    assert_mass_sums_to_one(mass)
    # The combined confidence should be clamped
    assert 0.0 <= mass["combined_confidence"] <= 1.0

    print(f"\n[Test 19] Clamped confidence={mass['combined_confidence']:.4f} ✓")


# ═══════════════════════════════════════════════════════════════
# Test 20 — REAL mass never exceeds MAX_FOCAL_MASS
# ═══════════════════════════════════════════════════════════════

def test_real_mass_bounded():
    """
    Even with maximum signals and confidence, REAL mass should not exceed
    MAX_FOCAL_MASS (leaving room for irreducible uncertainty).
    """
    analysis = {
        "company_reached": True,
        "role_actively_hiring": True,
        "recruiter_recognized": True,
        "company_denied_role": False,
        "suspicious_response": False,
        "source_confidence": 1.0,
        "extraction_confidence": 1.0,
    }

    mass = voice_transcript_to_mass(analysis)
    assert mass["REAL"] <= MAX_FOCAL_MASS, \
        f"REAL mass {mass['REAL']} exceeds MAX_FOCAL_MASS {MAX_FOCAL_MASS}"

    print(f"\n[Test 20] Max REAL bounded: {mass['REAL']:.4f} <= {MAX_FOCAL_MASS}")
