"""
Credibility Filtration Engine — Top-level orchestrator for Objective 2.

Wires together the four sub-engines (Authority, Freshness, Consensus,
M-RAV Alignment) to produce a unified credibility evaluation for every
document in the retrieved evidence bundle.

Pipeline per document:
    1. calculate_authority()     → authority score
    2. calculate_freshness()     → freshness score + anomaly flags
    3. calculate_consensus()     → consensus score + entity breakdown
    4. generate_alignment_triple() → M-RAV (hypothesis, explanation, score)
    5. C_idx = 0.4 * authority + 0.2 * freshness + 0.4 * consensus

Credibility Tiers:
    C_idx < 0.3         → Discard  (filtered out)
    0.3 ≤ C_idx < 0.5   → Low Confidence (kept but flagged)
    C_idx ≥ 0.5         → Trusted

Routing:
    - PROCEED_TO_VOICE_VERIFICATION: When critical flags are detected or
      overall trust is below the voice-verification threshold (< 0.6).
    - PROCEED_TO_CONFIDENCE_FRAMEWORK: When the evidence is sufficient and
      no critical signals warrant phone-based verification.
"""

from typing import Optional

from schemas.input_schema import (
    JobOfferEntities,
    VectorDocument,
    RetrievedEvidenceBundle,
    Objective2InputPayload,
)
from schemas.credibility_schema import (
    DocumentCredibility,
    MRAVTriple,
    CredibilityFlag,
    CredibilityEvaluation,
    EvaluatedDocument,
    EntityConsensusScore,
    Objective2OutputPayload,
)
from engines.authority_evaluator import calculate_authority
from engines.freshness_evaluator import calculate_freshness
from engines.consensus_evaluator import calculate_consensus
from engines.alignment_chain import generate_alignment_triple


# ──────────────────────────────────────────────────────────
# Credibility Index weights
# ──────────────────────────────────────────────────────────
ALPHA: float = 0.4  # Authority weight
BETA: float = 0.2   # Freshness weight
GAMMA: float = 0.4  # Consensus weight

# Tier thresholds
DISCARD_THRESHOLD: float = 0.3
TRUSTED_THRESHOLD: float = 0.5

# Voice verification routing threshold:
# If overall trust < this AND critical flags exist → route to Obj3
VOICE_VERIFICATION_TRUST_THRESHOLD: float = 0.6


def _classify_tier(c_idx: float) -> str:
    """Classify a credibility index into one of three tiers."""
    if c_idx < DISCARD_THRESHOLD:
        return "discard"
    elif c_idx < TRUSTED_THRESHOLD:
        return "low_confidence"
    else:
        return "trusted"


def _generate_flags(
    entities: JobOfferEntities,
    doc: VectorDocument,
    consensus_breakdown: EntityConsensusScore,
    freshness_flags: list[str],
    authority_score: float,
    consensus_score: float,
    c_idx: float,
    mrav_triple: MRAVTriple,
) -> list[CredibilityFlag]:
    """
    Generate structured credibility flags based on evaluation results.
    Flags are generated even for discarded documents — they carry
    important signal about the overall evaluation.
    """
    flags: list[CredibilityFlag] = []

    # Forward freshness anomaly flags
    for f_flag in freshness_flags:
        if "ANOMALY" in f_flag.upper():
            flags.append(CredibilityFlag(severity="WARNING", message=f_flag))
        elif "WARNING" in f_flag.upper():
            flags.append(CredibilityFlag(severity="WARNING", message=f_flag))
        else:
            flags.append(CredibilityFlag(severity="INFO", message=f_flag))

    # Domain mismatch
    if not consensus_breakdown.domain_match and entities.domain_or_website and doc.metadata.domain:
        flags.append(
            CredibilityFlag(
                severity="CRITICAL",
                message=(
                    f"Domain mismatch between offer ({entities.domain_or_website}) "
                    f"and {doc.metadata.source_type} record ({doc.metadata.domain})"
                ),
            )
        )

    # Free email provider
    if consensus_breakdown.email_domain_trust == 0.0:
        flags.append(
            CredibilityFlag(
                severity="WARNING",
                message=f"Recruiter email uses public/free domain ({entities.email_or_phone})",
            )
        )

    # Company name mismatch
    if consensus_breakdown.company_name_match_score < 0.8 and doc.metadata.company_name:
        flags.append(
            CredibilityFlag(
                severity="WARNING",
                message=(
                    f"Company name weak match: claimed '{entities.company_name}' "
                    f"vs record '{doc.metadata.company_name}' "
                    f"(score: {consensus_breakdown.company_name_match_score:.2f})"
                ),
            )
        )

    # EMSCAD Scam Pattern Trigger
    if doc.metadata.fraudulent == 1 or doc.metadata.source_type in ("scam_report", "scam_report_forum"):
        flags.append(
            CredibilityFlag(
                severity="CRITICAL",
                message=f"High-risk SCAM pattern detected from EMSCAD records (doc: {doc.doc_id})",
            )
        )

    # EMSCAD Legitimate Baseline Comparison
    if doc.metadata.fraudulent == 0 or doc.metadata.source_type == "legitimate_job_record":
        if mrav_triple.hypothesis == "Contradicts":
            flags.append(
                CredibilityFlag(
                    severity="WARNING",
                    message=f"Offer contradicts EMSCAD legitimate baseline: {mrav_triple.explanation} (doc: {doc.doc_id})",
                )
            )

    # Low credibility tier
    if c_idx < DISCARD_THRESHOLD:
        flags.append(
            CredibilityFlag(
                severity="INFO",
                message=f"Document {doc.doc_id} discarded (C_idx={c_idx:.3f} < {DISCARD_THRESHOLD})",
            )
        )
    elif c_idx < TRUSTED_THRESHOLD:
        flags.append(
            CredibilityFlag(
                severity="INFO",
                message=f"Document {doc.doc_id} classified as low confidence (C_idx={c_idx:.3f})",
            )
        )

    return flags


def _evaluate_single_document(
    entities: JobOfferEntities,
    doc: VectorDocument,
) -> tuple[EvaluatedDocument, list[CredibilityFlag]]:
    """
    Run the full evaluation pipeline on a single document.

    Returns:
        Tuple of (EvaluatedDocument, list of CredibilityFlags).
    """
    # Step 1: Authority
    authority = calculate_authority(doc.metadata)

    # Step 2: Freshness
    freshness, freshness_flags = calculate_freshness(doc.metadata)

    # Step 3: Consensus
    consensus, consensus_breakdown = calculate_consensus(entities, doc)

    # Step 4: M-RAV Alignment (LLM call)
    mrav_triple = generate_alignment_triple(entities, doc)

    # Step 5: Unified Credibility Index
    c_idx = (ALPHA * authority) + (BETA * freshness) + (GAMMA * consensus)
    c_idx = round(c_idx, 4)
    tier = _classify_tier(c_idx)

    # Generate flags
    flags = _generate_flags(
        entities=entities,
        doc=doc,
        consensus_breakdown=consensus_breakdown,
        freshness_flags=freshness_flags,
        authority_score=authority,
        consensus_score=consensus,
        c_idx=c_idx,
        mrav_triple=mrav_triple,
    )

    evaluated = EvaluatedDocument(
        source=doc.metadata.source_url or doc.doc_id,
        source_type=doc.metadata.source_type,
        authority=round(authority, 4),
        freshness=round(freshness, 4),
        consensus=round(consensus, 4),
        credibility_index=c_idx,
        credibility_tier=tier,
        consensus_breakdown=consensus_breakdown,
        mrav_triple=mrav_triple,
    )

    return evaluated, flags


def _determine_routing(
    overall_trust: float,
    flags: list[CredibilityFlag],
) -> str:
    """
    Determine whether the offer should be routed to voice verification
    or directly to the confidence framework.

    Routes to voice verification when:
      - Overall trust is below the threshold (< 0.6), OR
      - Any CRITICAL flags were raised (scam patterns, domain mismatches)

    Returns:
        Routing decision string.
    """
    has_critical_flags = any(f.severity == "CRITICAL" for f in flags)

    if has_critical_flags or overall_trust < VOICE_VERIFICATION_TRUST_THRESHOLD:
        return "PROCEED_TO_VOICE_VERIFICATION"
    else:
        return "PROCEED_TO_CONFIDENCE_FRAMEWORK"


def evaluate_credibility(payload: Objective2InputPayload) -> Objective2OutputPayload:
    """
    Main entrypoint for Objective 2: Credibility Filtration Layer.

    Evaluates every document in the retrieved evidence bundle and produces
    a unified credibility assessment with per-document scores, flags,
    and a routing decision.

    Routing logic:
        - If critical flags are detected or overall trust < 0.6,
          routes to PROCEED_TO_VOICE_VERIFICATION (Objective 3).
        - Otherwise routes to PROCEED_TO_CONFIDENCE_FRAMEWORK.

    Args:
        payload: The Objective2InputPayload containing extracted offer
                 entities and the retrieved evidence bundle.

    Returns:
        Objective2OutputPayload with overall trust score, evaluated
        evidence (split into retained and discarded), flags, and
        routing decision.
    """
    entities = payload.extracted_entities
    evidence = payload.retrieved_evidence

    # Collect all documents from all evidence categories
    all_docs: list[VectorDocument] = (
        evidence.legitimate_job_records
        + evidence.registry_records
        + evidence.linkedin_data
        + evidence.scam_reports
    )

    evaluated_docs: list[EvaluatedDocument] = []
    discarded_docs: list[EvaluatedDocument] = []
    all_flags: list[CredibilityFlag] = []

    print(f"[CredibilityEngine] Evaluating {len(all_docs)} documents for offer {payload.offer_id}")

    for i, doc in enumerate(all_docs):
        print(f"  [{i + 1}/{len(all_docs)}] Evaluating doc '{doc.doc_id}' (source: {doc.metadata.source_type})...")
        evaluated, flags = _evaluate_single_document(entities, doc)
        all_flags.extend(flags)

        if evaluated.credibility_tier == "discard":
            discarded_docs.append(evaluated)
        else:
            evaluated_docs.append(evaluated)

        print(
            f"    -> Authority={evaluated.authority:.2f}, "
            f"Freshness={evaluated.freshness:.2f}, "
            f"Consensus={evaluated.consensus:.2f}, "
            f"C_idx={evaluated.credibility_index:.3f} "
            f"[{evaluated.credibility_tier.upper()}]"
        )

    # Compute overall trust score as weighted average of non-discarded docs
    if evaluated_docs:
        overall_trust = sum(d.credibility_index for d in evaluated_docs) / len(evaluated_docs)
    else:
        overall_trust = 0.0
        all_flags.append(
            CredibilityFlag(
                severity="CRITICAL",
                message="All evidence documents were discarded — no trusted evidence found",
            )
        )

    overall_trust = round(overall_trust, 4)

    # Deduplicate flags by message
    seen_messages: set[str] = set()
    unique_flags: list[CredibilityFlag] = []
    for flag in all_flags:
        if flag.message not in seen_messages:
            unique_flags.append(flag)
            seen_messages.add(flag.message)

    # Determine routing decision based on trust score and flags
    routing = _determine_routing(overall_trust, unique_flags)

    print(f"[CredibilityEngine] Done. Overall trust: {overall_trust:.3f}, "
          f"Retained: {len(evaluated_docs)}, Discarded: {len(discarded_docs)}, "
          f"Flags: {len(unique_flags)}, Routing: {routing}")

    return Objective2OutputPayload(
        offer_id=payload.offer_id,
        credibility_evaluation=CredibilityEvaluation(
            overall_trust_score=overall_trust,
            flags_triggered=unique_flags,
        ),
        evaluated_evidence=evaluated_docs,
        discarded_evidence=discarded_docs,
        routing_decision=routing,
    )

