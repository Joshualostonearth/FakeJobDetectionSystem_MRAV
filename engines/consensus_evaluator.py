"""
Cross-Source Consensus Evaluator — Step 4 of the Credibility Filtration Layer.

Evaluates whether entity details claimed in the offer letter match across
all retrieved evidence sources. Uses fuzzy string matching (RapidFuzz),
domain comparison (tldextract), and email provider detection.

Consensus Score Formula:
    Consensus = w1 * S_name_match + w2 * I_domain_match
              + w3 * I_email_trust + w4 * S_location_match

Weights (sum to 1.0):
    w1 = 0.30  (company name)
    w2 = 0.30  (domain)
    w3 = 0.20  (email trust)
    w4 = 0.20  (location)

A field that is missing on either side (e.g. EMSCAD records carry no company
name or domain) cannot confirm or contradict the offer, so it contributes the
neutral value NEUTRAL_SCORE instead of counting as a mismatch.
"""

from typing import Optional

from rapidfuzz import fuzz
import tldextract

from schemas.input_schema import JobOfferEntities, VectorDocument
from schemas.credibility_schema import EntityConsensusScore


# ──────────────────────────────────────────────────────────
# Consensus weights (must sum to 1.0)
# ──────────────────────────────────────────────────────────
W_NAME: float = 0.30
W_DOMAIN: float = 0.30
W_EMAIL: float = 0.20
W_LOCATION: float = 0.20

# Contribution of a field that cannot be compared (missing on either side)
NEUTRAL_SCORE: float = 0.5

# Free email providers that should not be trusted for corporate claims
FREE_EMAIL_PROVIDERS: set[str] = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "aol.com",
    "protonmail.com",
    "mail.com",
    "yandex.com",
    "zoho.com",
    "icloud.com",
    "live.com",
    "msn.com",
    "gmx.com",
    "inbox.com",
    "fastmail.com",
}


def _extract_root_domain(domain_str: Optional[str]) -> Optional[str]:
    """
    Extract the registered root domain (e.g. 'apextech.com')
    from a domain or URL string.
    """
    if not domain_str:
        return None
    extracted = tldextract.extract(domain_str)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    return None


def _extract_email_domain(email_str: Optional[str]) -> Optional[str]:
    """Extract the domain part from an email address."""
    if not email_str or "@" not in email_str:
        return None
    return email_str.split("@")[-1].strip().lower()


def _compute_name_match(
    claimed_name: str, doc_name: Optional[str]
) -> float:
    """
    Compute fuzzy match score between claimed company name and
    document's company name using RapidFuzz token_sort_ratio.

    Returns a score between 0.0 and 1.0.
    """
    if not doc_name:
        return 0.0
    return fuzz.token_sort_ratio(
        claimed_name.lower().strip(),
        doc_name.lower().strip(),
    ) / 100.0


def _compute_domain_match(
    claimed_domain: Optional[str], doc_domain: Optional[str]
) -> bool:
    """
    Compare root domains after tldextract normalization.
    Returns True if they match exactly.
    """
    root_claimed = _extract_root_domain(claimed_domain)
    root_doc = _extract_root_domain(doc_domain)

    if root_claimed is None or root_doc is None:
        return False

    return root_claimed == root_doc


def _compute_email_trust(
    email_or_phone: Optional[str], claimed_domain: Optional[str]
) -> float:
    """
    Evaluate the trustworthiness of the contact email.

    Returns:
        1.0  — email domain matches the claimed corporate domain
        0.5  — email uses a non-free domain that doesn't match (could be legitimate subsidiary)
        0.0  — email uses a free provider (gmail, yahoo, etc.)
    """
    email_domain = _extract_email_domain(email_or_phone)

    if email_domain is None:
        # No email provided — neutral, give benefit of the doubt
        return 0.5

    # Check for free email providers
    if email_domain in FREE_EMAIL_PROVIDERS:
        return 0.0

    # Check if email domain matches claimed corporate domain
    root_email = _extract_root_domain(email_domain)
    root_claimed = _extract_root_domain(claimed_domain)

    if root_email and root_claimed and root_email == root_claimed:
        return 1.0

    # Non-free domain but doesn't match claimed domain
    return 0.5


def _compute_location_match(
    claimed_location: Optional[str], doc_content: str
) -> bool:
    """
    Check if the claimed location appears in the document content
    using fuzzy partial matching.

    Returns True if partial_ratio >= 70 (generous threshold for
    geographical name variations like "NY" vs "New York").
    """
    if not claimed_location:
        return False

    score = fuzz.partial_ratio(
        claimed_location.lower().strip(),
        doc_content.lower(),
    )
    return score >= 70


def calculate_consensus(
    entities: JobOfferEntities,
    doc: VectorDocument,
) -> tuple[float, EntityConsensusScore]:
    """
    Calculate the cross-source consensus score between the offer's
    claimed entities and a retrieved evidence document.

    Args:
        entities: Extracted entities from the job offer letter.
        doc: A retrieved evidence document from FAISS.

    Returns:
        A tuple of (consensus_score, entity_consensus_breakdown).
        - consensus_score: float between 0.0 and 1.0
        - entity_consensus_breakdown: EntityConsensusScore with per-field details
    """
    # Company name match
    name_score = _compute_name_match(
        entities.company_name, doc.metadata.company_name
    )

    # Domain match
    domain_match = _compute_domain_match(
        entities.domain_or_website, doc.metadata.domain
    )

    # Email trust
    email_trust = _compute_email_trust(
        entities.email_or_phone, entities.domain_or_website
    )

    # Location match
    location_match = _compute_location_match(
        entities.location, doc.content
    )

    # Build the breakdown
    breakdown = EntityConsensusScore(
        company_name_match_score=name_score,
        domain_match=domain_match,
        email_domain_trust=email_trust,
        location_match=location_match,
    )

    # Fields missing on either side are neutral, not mismatches
    name_known = bool((entities.company_name or "").strip() and doc.metadata.company_name)
    domain_known = (
        _extract_root_domain(entities.domain_or_website) is not None
        and _extract_root_domain(doc.metadata.domain) is not None
    )
    location_known = bool((entities.location or "").strip())

    # Compute weighted consensus score
    consensus = (
        W_NAME * (name_score if name_known else NEUTRAL_SCORE)
        + W_DOMAIN * ((1.0 if domain_match else 0.0) if domain_known else NEUTRAL_SCORE)
        + W_EMAIL * email_trust
        + W_LOCATION * ((1.0 if location_match else 0.0) if location_known else NEUTRAL_SCORE)
    )

    return consensus, breakdown
