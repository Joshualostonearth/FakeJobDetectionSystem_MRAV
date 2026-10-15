"""
Source Authority Evaluator — Step 2 of the Credibility Filtration Layer.

Assigns a numeric authority weight (0.0 to 1.0) based on the source's
origin type and credibility tier. Official government registries score
highest; unverified community sources score lowest.

Tier Hierarchy:
    Tier 1 (1.00) — Official registries (government, SEC, federal databases)
    Tier 2 (0.75) — Verified professional platforms (LinkedIn verified pages)
    Tier 3 (0.50) — Third-party aggregators (Glassdoor, Crunchbase, job boards)
    Tier 4 (0.30) — Unverified / crowdsourced (Reddit, scam forums, community)
    Fallback (0.20) — Unknown or unrecognized sources
"""

from schemas.input_schema import RetrivedDocMetadata


# ──────────────────────────────────────────────────────────
# Authority weight lookup tables
# ──────────────────────────────────────────────────────────

# Primary mapping: source_type → weight
SOURCE_TYPE_WEIGHTS: dict[str, float] = {
    "registry": 1.0,
    "government_registry": 1.0,
    "linkedin": 0.75,
    "linkedin_verified": 0.75,
    "third_party_aggregator": 0.50,
    "glassdoor": 0.50,
    "crunchbase": 0.50,
    "job_board": 0.50,
    "legitimate_job_record": 0.50,  # EMSCAD labelled posting (job-board listing)
    "scam_report": 0.30,
    "scam_report_forum": 0.30,
    "reddit": 0.30,
    "community": 0.30,
}

# Secondary mapping: credibility_tier → weight (used as fallback)
TIER_WEIGHTS: dict[str, float] = {
    "official": 1.0,
    "high": 0.75,
    "third_party": 0.50,
    "community": 0.30,
}

# Tertiary mapping: numeric authority_tier (set by ingest_emscad.py) → weight
AUTHORITY_TIER_WEIGHTS: dict[int, float] = {
    1: 1.0,
    2: 0.75,
    3: 0.50,
    4: 0.30,
}

# Default weight for completely unknown sources
DEFAULT_WEIGHT: float = 0.20


def calculate_authority(metadata: RetrivedDocMetadata) -> float:
    """
    Calculate the authority score for a retrieved document based on its
    source type and credibility tier metadata.

    The function first checks the `source_type` field against known
    source types. If no match is found, it falls back to the numeric
    `authority_tier`, then to the `credibility_tier` field. If none
    matches, returns DEFAULT_WEIGHT.

    Args:
        metadata: The document's metadata containing source_type and
                  credibility_tier fields.

    Returns:
        A float between 0.0 and 1.0 representing source authority.
    """
    # Try primary lookup by source_type
    source_type = metadata.source_type.lower().strip()
    if source_type in SOURCE_TYPE_WEIGHTS:
        return SOURCE_TYPE_WEIGHTS[source_type]

    # Fallback to the numeric authority tier
    if metadata.authority_tier in AUTHORITY_TIER_WEIGHTS:
        return AUTHORITY_TIER_WEIGHTS[metadata.authority_tier]

    # Fallback to credibility_tier
    tier = (metadata.credibility_tier or "").lower().strip()
    if tier in TIER_WEIGHTS:
        return TIER_WEIGHTS[tier]

    return DEFAULT_WEIGHT
