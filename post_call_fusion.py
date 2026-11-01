"""
post_call_fusion.py - combine M4's text score with the phone-call evidence (replaces the old
run_objective4 path for the UI).

    m_text  = prob_mass(p_fake_M4, 0.9)          # 0.9 = the repo's old CLASSIFIER_RELIABILITY
    m_voice = voice_transcript_to_mass(signals)  # objective3_voice_agent/voice_to_mass.py
    m       = Dempster's rule (mravplus_infer.combine_all)
    BetP(FAKE) = m(FAKE) + m(U)/2 ; confidence real = 1 - BetP(FAKE)

then the repo's voice overrides and verdict bands
(confidence_calibrator.apply_verification_override: company denied the role -> HIGH_RISK_SCAM,
hostile/scam-like response -> HIGH RISK; >= 0.75 VERIFIED_REAL, <= 0.25 HIGH_RISK_SCAM,
<= 0.60 SUSPICIOUS_FAKE, else UNCERTAIN_NEEDS_VERIFICATION).

NOT VALIDATED: there is no labelled phone-call dataset, so the voice mass, its reliability
and these bands have never been evaluated. Only M4's text score is validated.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

from mravplus_infer import combine_all, prob_mass

TEXT_RELIABILITY: float = 0.9
UNVALIDATED_NOTE = ("The post-call verdict (voice evidence fusion and its bands) is NOT validated: there is "
                    "no labelled phone-call dataset. Only M4's text score is validated.")

_VOICE_TO_MASS = Path(__file__).resolve().parent / "objective3_voice_agent" / "voice_to_mass.py"
_vtm = None


def _voice_to_mass_module():
    """Load voice_to_mass.py by path: importing the objective3_voice_agent package would pull in
    the whole voice stack (edge-tts, whisper, langchain), which the fusion itself does not need."""
    global _vtm
    if _vtm is None:
        spec = importlib.util.spec_from_file_location("_voice_to_mass", _VOICE_TO_MASS)
        _vtm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_vtm)
    return _vtm


def fuse_text_and_voice(p_fake: Optional[float], voice_signals: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    from confidence_calibrator import apply_verification_override

    masses, parts = [], {}
    if p_fake is not None:
        m_text = prob_mass(float(p_fake), TEXT_RELIABILITY)
        masses.append(m_text)
        parts["text_m4"] = {"REAL": float(m_text[0]), "FAKE": float(m_text[1]), "UNCERTAIN": float(m_text[2])}
    voice_mass = None
    if voice_signals:
        voice_mass = _voice_to_mass_module().voice_transcript_to_mass(voice_signals)
        m_voice = [voice_mass.get("REAL", 0.0), voice_mass.get("FAKE", 0.0), voice_mass.get("UNCERTAIN", 1.0)]
        masses.append(m_voice)
        parts["voice"] = {"REAL": float(m_voice[0]), "FAKE": float(m_voice[1]), "UNCERTAIN": float(m_voice[2])}

    if masses:
        m, K = combine_all(masses)
    else:
        m, K = [0.0, 0.0, 1.0], 0.0
    combined = {"REAL": float(m[0]), "FAKE": float(m[1]), "UNCERTAIN": float(m[2]), "conflict": float(K)}
    betp_fake = combined["FAKE"] + combined["UNCERTAIN"] / 2
    fv = apply_verification_override(
        calibrated_confidence=round(1.0 - betp_fake, 4),
        dst_result=combined,
        flags=[],                 # M4 has no Objective 2 flags; only the voice overrides apply
        voice_signals=voice_signals or None,
    )
    return {
        "source_masses": parts,
        "voice_mass": voice_mass,
        "dst_combined": combined,
        "betp_fake": float(betp_fake),
        "final_verification": fv,
        "validated": False,
        "note": UNVALIDATED_NOTE,
    }
