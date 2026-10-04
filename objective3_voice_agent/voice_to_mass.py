"""
voice_to_mass.py — Voice-to-Mass Conversion (Joshua's Deliverable)

Converts the structured voice verification analysis (from Edwin's
analyze_transcript / fact_extractor) into a Dempster-Shafer mass
function:  m({Real}), m({Fake}), m({Uncertain}).

The output dict is directly consumable by Allan's combine_all() /
combine_two() in obj2dst.py.

Public API:
    voice_transcript_to_mass(analysis: dict) -> dict

Design notes:
    1.  Each boolean signal contributes a POSITIVE (confirmation) or
        NEGATIVE (red-flag) score.  Weights are tuned to reflect how
        much each signal shifts belief toward Real or Fake.
    2.  The raw net-score is mapped into one of three regimes:
            net > 0  → Real-leaning mass
            net < 0  → Fake-leaning mass
            net == 0 → maximally Uncertain
    3.  Both the source_confidence (how trustworthy the phone number
        source was) and extraction_confidence (how confident the LLM
        was in its parsing) act as *global dampeners* — if either is
        low, the overall mass assignment is pulled toward Uncertain.
    4.  The function never raises.  Missing / None keys are treated as
        "no evidence" and contribute 0 to the score.
"""

from typing import Dict, Optional


# ──────────────────────────────────────────────────────────
# Signal weights — tunable knobs
# ──────────────────────────────────────────────────────────
# Positive weights = confirmations that push toward REAL
# Negative weights = red flags that push toward FAKE

SIGNAL_WEIGHTS: Dict[str, float] = {
    # ---- Confirmations (positive = evidence of REAL) ----
    "company_reached":       +0.25,   # Company picked up the phone
    "role_actively_hiring":  +0.30,   # They confirmed the role exists
    "recruiter_recognized":  +0.20,   # They recognise the recruiter name

    # ---- Red flags (negative = evidence of FAKE) --------
    "company_denied_role":   -0.40,   # Explicit denial — strongest red flag
    "suspicious_response":   -0.30,   # Evasive / hostile / scam-like behaviour
}

# Signals whose *absence* (False) is itself weak evidence of the opposite
ABSENCE_WEIGHTS: Dict[str, float] = {
    "company_reached":  -0.10,   # Nobody picked up → mild red flag
}

# Confidence floor: below this, we refuse to make a strong claim
CONFIDENCE_FLOOR: float = 0.15

# Maximum mass that can be assigned to REAL or FAKE (rest goes to Uncertain)
MAX_FOCAL_MASS: float = 0.95

# Source type label used in the mass dict (matches obj2dst.py convention)
SOURCE_LABEL: str = "voice_call"


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def voice_transcript_to_mass(analysis: Dict) -> Dict[str, float]:
    """
    Convert a structured voice verification analysis dict into a
    Dempster-Shafer mass function.

    Args:
        analysis: Dict produced by Edwin's analyze_transcript() or the
                  consensus_handoff.push_voice_signals() output.

                  Expected keys (all optional — missing = no evidence):
                      company_reached:        bool
                      role_actively_hiring:    bool | None
                      recruiter_recognized:    bool | None
                      company_denied_role:     bool
                      suspicious_response:     bool
                      source_confidence:       float  (0-1)
                      extraction_confidence:   float  (0-1)
                      number_source:           str    (informational)
                      response_summary:        str    (informational)

    Returns:
        Dict with keys:
            REAL:       float   — belief mass assigned to {Real}
            FAKE:       float   — belief mass assigned to {Fake}
            UNCERTAIN:  float   — belief mass assigned to {Real, Fake} (ignorance)
            source:     str     — always "voice_call"
            strength:   float   — overall evidence strength (0-1)
            net_score:  float   — raw signed score before mass mapping
            details:    dict    — per-signal contribution breakdown
    """
    if not analysis or not isinstance(analysis, dict):
        return _empty_mass("empty_or_invalid_analysis")

    # ── Step 1: Compute raw net score from boolean signals ──────────
    net_score = 0.0
    details: Dict[str, float] = {}

    for signal_key, weight in SIGNAL_WEIGHTS.items():
        value = analysis.get(signal_key)

        if value is True:
            # Signal is present and affirmative
            contribution = weight
        elif value is False and signal_key in ABSENCE_WEIGHTS:
            # Signal is explicitly False — use the absence weight
            contribution = ABSENCE_WEIGHTS[signal_key]
        elif value is False and weight < 0:
            # A red-flag signal being False is mildly reassuring
            contribution = abs(weight) * 0.1
        elif value is None:
            # Unknown / not mentioned — no contribution
            contribution = 0.0
        else:
            contribution = 0.0

        net_score += contribution
        details[signal_key] = round(contribution, 4)

    # ── Step 2: Apply confidence dampening ──────────────────────────
    source_conf = _clamp(analysis.get("source_confidence", 0.5))
    extract_conf = _clamp(analysis.get("extraction_confidence", 0.5))

    # Combined confidence: geometric mean gives a harsher penalty when
    # either confidence is low
    combined_confidence = (source_conf * extract_conf) ** 0.5

    # If confidence is below the floor, collapse everything to uncertain
    if combined_confidence < CONFIDENCE_FLOOR:
        return _low_confidence_mass(net_score, combined_confidence, details)

    # Dampen the net score by combined confidence
    dampened_score = net_score * combined_confidence

    # ── Step 3: Map dampened score → mass function ──────────────────
    mass = _score_to_mass(dampened_score, combined_confidence)

    mass["source"] = SOURCE_LABEL
    mass["strength"] = round(combined_confidence * abs(dampened_score), 4)
    mass["net_score"] = round(net_score, 4)
    mass["dampened_score"] = round(dampened_score, 4)
    mass["combined_confidence"] = round(combined_confidence, 4)
    mass["details"] = details

    return mass


# ──────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────

def _clamp(value, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a numeric value to [lo, hi], treating None as the midpoint."""
    if value is None:
        return (lo + hi) / 2.0
    try:
        return max(lo, min(float(value), hi))
    except (TypeError, ValueError):
        return (lo + hi) / 2.0


def _score_to_mass(dampened_score: float, confidence: float) -> Dict[str, float]:
    """
    Map a dampened net-score into {REAL, FAKE, UNCERTAIN} masses.

    Logic:
        - Positive dampened_score → more mass on REAL
        - Negative dampened_score → more mass on FAKE
        - Magnitude determines how much mass is taken from UNCERTAIN
        - confidence controls how far from 50/50 we're willing to go
    """
    # Scale factor: maps the raw score (typically -0.5 to +0.5) into a
    # 0-to-MAX_FOCAL_MASS range via a smooth sigmoid-like curve
    magnitude = min(abs(dampened_score) * 2.0, 1.0)  # normalise to 0-1
    focal_mass = magnitude * MAX_FOCAL_MASS * confidence

    # Ensure masses are within valid range
    focal_mass = min(focal_mass, MAX_FOCAL_MASS)
    uncertain_mass = 1.0 - focal_mass

    if dampened_score > 0:
        # Real-leaning
        # Split: most of focal_mass goes to REAL, a small residual to FAKE
        real_share = 0.85
        return {
            "REAL": round(focal_mass * real_share, 6),
            "FAKE": round(focal_mass * (1.0 - real_share), 6),
            "UNCERTAIN": round(uncertain_mass, 6),
        }
    elif dampened_score < 0:
        # Fake-leaning
        fake_share = 0.85
        return {
            "REAL": round(focal_mass * (1.0 - fake_share), 6),
            "FAKE": round(focal_mass * fake_share, 6),
            "UNCERTAIN": round(uncertain_mass, 6),
        }
    else:
        # Perfectly neutral — maximum uncertainty
        return {
            "REAL": 0.0,
            "FAKE": 0.0,
            "UNCERTAIN": 1.0,
        }


def _empty_mass(reason: str) -> Dict[str, float]:
    """Return a fully uncertain mass when no analysis is available."""
    return {
        "REAL": 0.0,
        "FAKE": 0.0,
        "UNCERTAIN": 1.0,
        "source": SOURCE_LABEL,
        "strength": 0.0,
        "net_score": 0.0,
        "dampened_score": 0.0,
        "combined_confidence": 0.0,
        "details": {},
        "reason": reason,
    }


def _low_confidence_mass(
    net_score: float,
    confidence: float,
    details: Dict[str, float],
) -> Dict[str, float]:
    """
    Return a near-uncertain mass when combined confidence is below the floor.
    We still record the direction (net_score) but assign almost no focal mass.
    """
    # Give a tiny nudge in the direction of the net score
    tiny = confidence * 0.1
    if net_score > 0:
        real_mass = tiny
        fake_mass = 0.0
    elif net_score < 0:
        real_mass = 0.0
        fake_mass = tiny
    else:
        real_mass = 0.0
        fake_mass = 0.0

    return {
        "REAL": round(real_mass, 6),
        "FAKE": round(fake_mass, 6),
        "UNCERTAIN": round(1.0 - real_mass - fake_mass, 6),
        "source": SOURCE_LABEL,
        "strength": round(confidence * abs(net_score) * 0.1, 4),
        "net_score": round(net_score, 4),
        "dampened_score": round(net_score * confidence, 4),
        "combined_confidence": round(confidence, 4),
        "details": details,
        "low_confidence_warning": True,
    }


# ──────────────────────────────────────────────────────────
# Convenience: convert VoiceVerificationResult → analysis dict
# ──────────────────────────────────────────────────────────

def voice_result_to_analysis(
    result,
    source_confidence: float = 0.8,
    number_source: str = "unknown",
) -> Dict:
    """
    Adapter that converts a VoiceVerificationResult pydantic model (from
    Edwin's fact_extractor) into the flat analysis dict that
    voice_transcript_to_mass() expects.

    Usage:
        from fact_extractor import extract_facts
        from voice_to_mass import voice_result_to_analysis, voice_transcript_to_mass

        facts = extract_facts(transcript, entity, config)
        analysis = voice_result_to_analysis(facts, source_confidence=0.9)
        mass = voice_transcript_to_mass(analysis)
    """
    return {
        "company_reached":       getattr(result, "company_reached", False),
        "role_actively_hiring":  getattr(result, "role_actively_hiring", None),
        "recruiter_recognized":  getattr(result, "recruiter_recognized", None),
        "company_denied_role":   getattr(result, "company_denied_role", False),
        "suspicious_response":   getattr(result, "suspicious_response", False),
        "source_confidence":     source_confidence,
        "extraction_confidence": getattr(result, "confidence", 0.5),
        "number_source":         number_source,
        "response_summary":      getattr(result, "response_summary", ""),
    }
