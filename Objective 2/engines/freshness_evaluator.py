"""
Freshness & Temporal Evaluator — Step 3 of the Credibility Filtration Layer.

Uses a **plateau scale (logistic sigmoid)** to score document recency.
Unlike exponential decay (which penalizes from day 1), the sigmoid holds
near 1.0 for fresh documents, drops sharply around a configurable midpoint,
and floors out for stale documents.

Formula:
    F_score = max(floor, 1 / (1 + e^(k * (t - t_mid))))

Where:
    t     = document age in months
    t_mid = midpoint age where score = 0.5 (env: FRESHNESS_T_MID, default 12.0)
    k     = steepness of drop-off        (env: FRESHNESS_K, default 0.5)
    floor = minimum score                 (hardcoded 0.05)

Behavior (with defaults):
    0–6 months    → ~0.95–1.0  (plateau, no penalty)
    6–12 months   → 0.95–0.50  (gentle decline)
    12–18 months  → 0.50–0.05  (sharp drop-off)
    18+ months    → ~0.05      (floor, stale)
"""

import os
import math
from datetime import date, datetime
from typing import Optional

from dateutil.parser import parse as dateutil_parse
from dotenv import load_dotenv

from schemas.input_schema import RetrivedDocMetadata

load_dotenv()

# ──────────────────────────────────────────────────────────
# Constants & environment-driven parameters
# ──────────────────────────────────────────────────────────

# Sigmoid midpoint: document age (months) at which score = 0.5
T_MID: float = float(os.getenv("FRESHNESS_T_MID", "12.0"))

# Sigmoid steepness: higher = sharper transition
K: float = float(os.getenv("FRESHNESS_K", "0.5"))

# Floor: minimum score to avoid absolute zero
FLOOR: float = 0.05

# Penalty score when no timestamp is available
MISSING_DATE_SCORE: float = 0.3

# Threshold (days) for flagging very new registrations
NEW_REGISTRATION_THRESHOLD_DAYS: int = 30

# Average days per month for conversion
DAYS_PER_MONTH: float = 30.44


def _sigmoid_freshness(age_months: float) -> float:
    """
    Compute the plateau-scale freshness score using a logistic sigmoid.

    Args:
        age_months: Document age in months (non-negative).

    Returns:
        Score between FLOOR and 1.0.
    """
    raw = 1.0 / (1.0 + math.exp(K * (age_months - T_MID)))
    return max(FLOOR, raw)


def calculate_freshness(
    metadata: RetrivedDocMetadata,
    reference_date: Optional[date] = None,
) -> tuple[float, list[str]]:
    """
    Calculate the freshness score and anomaly flags for a retrieved document.

    Reads FRESHNESS_T_MID and FRESHNESS_K from environment variables internally.

    Args:
        metadata: The document's metadata containing `last_updated` timestamp.
        reference_date: The date to measure age against (defaults to today).

    Returns:
        A tuple of (freshness_score, list_of_anomaly_flags).
        - freshness_score: float between 0.0 and 1.0
        - anomaly_flags: list of string descriptions for any detected anomalies
    """
    flags: list[str] = []

    if reference_date is None:
        reference_date = date.today()

    # Handle missing timestamp
    if not metadata.last_updated:
        flags.append("INFO: No last_updated timestamp available — applying penalty score")
        return MISSING_DATE_SCORE, flags

    # Parse the ISO timestamp
    try:
        parsed_date = dateutil_parse(metadata.last_updated).date()
    except (ValueError, TypeError):
        flags.append(
            f"WARNING: Could not parse last_updated '{metadata.last_updated}' — applying penalty score"
        )
        return MISSING_DATE_SCORE, flags

    # Compute age
    age_days = (reference_date - parsed_date).days

    # Flag very new registrations (potential shell company / lookalike domain)
    if age_days < NEW_REGISTRATION_THRESHOLD_DAYS and metadata.source_type in (
        "registry",
        "government_registry",
    ):
        flags.append(
            f"FRESHNESS_ANOMALY: Recently registered entity "
            f"(age: {age_days} days, source: {metadata.source_type})"
        )

    # Handle future dates gracefully
    if age_days < 0:
        flags.append(
            f"WARNING: Document date is in the future ({metadata.last_updated})"
        )
        return 1.0, flags  # Treat future-dated as maximally fresh

    # Compute plateau-scale sigmoid score
    age_months = age_days / DAYS_PER_MONTH
    score = _sigmoid_freshness(age_months)

    return score, flags
