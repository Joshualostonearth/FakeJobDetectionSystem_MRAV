"""
confidence_calibrator.py — Person 4 Deliverable: Platt Scaling & Override Framework.

Outputs calibrated confidence scores and applies verification overrides for Objective 4.

Public API:
    - fit_and_save_platt_model(model_path) -> PlattScaler
    - calculate_confidence_score(dst_result: dict) -> float
    - apply_verification_override(calibrated_confidence: float, dst_result: dict, flags: list, voice_signals: dict = None) -> dict
"""

import math
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_FILE_PATH = Path(__file__).parent / "platt_model.pkl"


class PlattScaler:
    """
    Platt Scaling model to calibrate raw DST mass belief margins s = m(REAL) - m(FAKE)
    into true posterior probabilities P(Real | Evidence).
    """

    def __init__(self):
        self.model: Optional[LogisticRegression] = None
        self.is_fitted: bool = False

    def fit(self, X_scores: np.ndarray, y_labels: np.ndarray):
        """Fit Platt logistic regression on validation belief margins X_scores [-1.0, +1.0]."""
        X = np.asarray(X_scores).reshape(-1, 1)
        y = np.asarray(y_labels)

        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(X, y)
        self.model = lr
        self.is_fitted = True

    def predict_proba(self, score: float) -> float:
        """Predict calibrated probability P(Real) given a belief score s = m(REAL) - m(FAKE)."""
        if not self.is_fitted or self.model is None:
            # Fallback sigmoid if not fitted
            return 1.0 / (1.0 + math.exp(-3.0 * score))

        X_input = np.array([[score]])
        prob_real = self.model.predict_proba(X_input)[0][1]
        return float(prob_real)

    def save(self, filepath: Path = MODEL_FILE_PATH):
        """Save fitted Platt model to disk as a pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: Path = MODEL_FILE_PATH) -> bool:
        """Load fitted Platt model from disk if it exists."""
        if not filepath.exists():
            return False
        try:
            with open(filepath, "rb") as f:
                self.model = pickle.load(f)
            self.is_fitted = True
            return True
        except Exception:
            return False


# Final-verdict bands on the (override-adjusted) calibrated P(Real)
VERIFIED_REAL_MIN_CONFIDENCE: float = 0.75   # >= → VERIFIED_REAL
SCAM_MAX_CONFIDENCE: float = 0.60            # <= → SUSPICIOUS_FAKE (is_scam); middle of the stable range on the validation split
HIGH_RISK_MAX_CONFIDENCE: float = 0.25       # <= → HIGH_RISK_SCAM
# Confidence given to an offer that no retrieved record confirms (UNCERTAIN band, not scam)
UNVERIFIED_MAX_CONFIDENCE: float = 0.65

# Global singleton instance
_platt_scaler = PlattScaler()


def generate_synthetic_validation_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic validation calibration data representing real and fake job offer profiles.
    Used to fit the Platt model if platt_model.pkl is missing.
    """
    np.random.seed(42)

    # 100 Legitimate job offers (Real): belief margin s = m(REAL) - m(FAKE) ~ N(0.5, 0.25)
    real_scores = np.random.normal(loc=0.55, scale=0.20, size=100)
    real_labels = np.ones(100, dtype=int)

    # 100 Fake/Scam job offers (Fake): belief margin s = m(REAL) - m(FAKE) ~ N(-0.55, 0.20)
    fake_scores = np.random.normal(loc=-0.55, scale=0.20, size=100)
    fake_labels = np.zeros(100, dtype=int)

    # 50 Uncertain offers near 0.0
    uncertain_scores = np.random.uniform(low=-0.2, high=0.2, size=50)
    uncertain_labels = np.random.choice([0, 1], size=50, p=[0.5, 0.5])

    X = np.concatenate([real_scores, fake_scores, uncertain_scores])
    X = np.clip(X, -1.0, 1.0)
    y = np.concatenate([real_labels, fake_labels, uncertain_labels])

    return X, y


def fit_and_save_platt_model(
    filepath: Path = MODEL_FILE_PATH,
    scores: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> PlattScaler:
    """
    Fits the Platt model and saves platt_model.pkl.

    Pass real validation belief margins s = m(REAL) - m(FAKE) and labels
    (1 = real, 0 = fake), e.g. from evaluate_test_split.py fit-platt. Without them it
    falls back to the synthetic profiles, which are only a placeholder.
    """
    if scores is None or labels is None:
        scores, labels = generate_synthetic_validation_data()
    scaler = PlattScaler()
    scaler.fit(scores, labels)
    scaler.save(filepath)
    return scaler


def ensure_platt_model_loaded():
    """Ensure the global Platt scaler model is loaded or fitted and saved."""
    if not _platt_scaler.is_fitted:
        if not _platt_scaler.load(MODEL_FILE_PATH):
            fit_and_save_platt_model(MODEL_FILE_PATH)
            _platt_scaler.load(MODEL_FILE_PATH)


def calculate_confidence_score(dst_result: Dict[str, float]) -> float:
    """
    Person 4 Deliverable: Calculates calibrated confidence score P(Real) from combined DST result.

    Args:
        dst_result: Combined DST mass dictionary containing REAL, FAKE, UNCERTAIN, conflict

    Returns:
        float: Calibrated probability P(Real) between 0.0 and 1.0
    """
    ensure_platt_model_loaded()

    real_mass = dst_result.get("REAL", 0.0)
    fake_mass = dst_result.get("FAKE", 0.0)
    uncertain_mass = dst_result.get("UNCERTAIN", 0.0)
    conflict = dst_result.get("conflict", 0.0)

    # Raw belief margin s = m(REAL) - m(FAKE) in [-1.0, 1.0]
    score_margin = real_mass - fake_mass

    # Pass through Platt model
    calibrated_prob = _platt_scaler.predict_proba(score_margin)

    # If conflict factor K is high (> 0.4), temper the confidence toward 0.5 (uncertainty)
    if conflict > 0.4:
        dampener = min((conflict - 0.4) * 0.8, 0.4)
        if calibrated_prob > 0.5:
            calibrated_prob -= dampener
        else:
            calibrated_prob += dampener

    return round(float(np.clip(calibrated_prob, 0.0, 1.0)), 4)


def apply_verification_override(
    calibrated_confidence: float,
    dst_result: Dict[str, float],
    flags: List[Dict],
    voice_signals: Optional[Dict] = None,
    alignment_hypotheses: Optional[List[str]] = None,
) -> Dict:
    """
    Person 4 Deliverable: Applies safety & verification overrides to produce final verdict.

    Args:
        calibrated_confidence: Calibrated P(Real) from calculate_confidence_score()
        dst_result: Combined DST mass dict
        flags: List of CredibilityFlags / warning strings from Obj 2
        voice_signals: Optional structured voice signals from Obj 3
        alignment_hypotheses: M-RAV hypotheses ("Confirms"/"Contradicts"/"Neutral") of the
            retained documents; omitted or empty when the LLM alignment did not run

    Returns:
        Dict with final_verdict, calibrated_confidence, is_scam, override_applied, override_reason
    """
    override_applied = False
    override_reason = None
    final_conf = calibrated_confidence

    # 1. Voice Agent Overrides (Highest Authority)
    if voice_signals:
        if voice_signals.get("company_denied_role") is True:
            override_applied = True
            override_reason = "CRITICAL OVERRIDE: Official company representative explicitly denied role existence during phone verification."
            final_conf = 0.02
        elif voice_signals.get("suspicious_response") is True:
            override_applied = True
            override_reason = "HIGH RISK OVERRIDE: Hostile/scam-like response detected during official voice verification."
            final_conf = min(final_conf, 0.15)

    # 2. Critical Scam Flag Overrides from Objective 2
    if not override_applied and flags:
        # Only flags marked CRITICAL count. Matching on words such as "SCAM" anywhere in
        # the text let WARNING flags (e.g. an LLM explanation mentioning scams) force a verdict.
        critical_flags = [
            f for f in flags
            if (isinstance(f, dict) and f.get("severity") == "CRITICAL") or
               (hasattr(f, "severity") and f.severity == "CRITICAL") or
               (isinstance(f, str) and f.strip().upper().startswith("CRITICAL"))
        ]
        if critical_flags:
            override_applied = True
            first = critical_flags[0]
            if isinstance(first, dict):
                flag_msg = first.get("message")
            else:
                flag_msg = getattr(first, "message", str(first))
            override_reason = f"CRITICAL EVIDENCE OVERRIDE: {flag_msg}"
            final_conf = min(final_conf, 0.20)

    # 3. No retrieved record confirms the offer and most contradict it: nothing verifies the
    # claimed company, so the offer cannot be VERIFIED_REAL without phone verification.
    if not override_applied and not voice_signals and alignment_hypotheses:
        n_contradicts = alignment_hypotheses.count("Contradicts")
        if "Confirms" not in alignment_hypotheses and n_contradicts * 2 >= len(alignment_hypotheses):
            if final_conf >= VERIFIED_REAL_MIN_CONFIDENCE:
                override_applied = True
                override_reason = (
                    f"UNVERIFIED OVERRIDE: {n_contradicts} of {len(alignment_hypotheses)} retrieved "
                    "record(s) contradict the offer and none confirm it; phone verification needed."
                )
                final_conf = UNVERIFIED_MAX_CONFIDENCE

    # 4. Determine Final Verdict Recommendation
    if final_conf >= VERIFIED_REAL_MIN_CONFIDENCE:
        verdict = "VERIFIED_REAL"
        is_scam = False
    elif final_conf <= HIGH_RISK_MAX_CONFIDENCE:
        verdict = "HIGH_RISK_SCAM"
        is_scam = True
    elif final_conf <= SCAM_MAX_CONFIDENCE:
        verdict = "SUSPICIOUS_FAKE"
        is_scam = True
    else:
        verdict = "UNCERTAIN_NEEDS_VERIFICATION"
        is_scam = False

    return {
        "final_verdict": verdict,
        "calibrated_confidence": final_conf,
        "is_scam": is_scam,
        "override_applied": override_applied,
        "override_reason": override_reason,
        "dst_masses": dst_result,
        "conflict_factor_k": dst_result.get("conflict", 0.0),
    }


# Auto-fit model on import if platt_model.pkl does not exist yet
if __name__ != "__main__":
    ensure_platt_model_loaded()
