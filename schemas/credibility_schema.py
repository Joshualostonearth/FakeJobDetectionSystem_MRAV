"""
Output and intermediate schemas for Objective 2: Credibility Filtration Layer.

These models capture the results of the credibility evaluation pipeline —
per-document scores, M-RAV alignment triples, flags, and the final
output payload that is handed off to Objective 4 (Confidence Framework).
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ==========================================
# 1. INTERMEDIATE SCORING MODELS
# ==========================================
class EntityConsensusScore(BaseModel):
    """Per-field breakdown of how well the offer claims match a retrieved document."""

    company_name_match_score: float = Field(
        ..., ge=0.0, le=1.0, description="Fuzzy match score (RapidFuzz token_sort_ratio / 100)"
    )
    domain_match: bool = Field(
        ..., description="True if offer domain matches the document's registered domain"
    )
    email_domain_trust: float = Field(
        ..., ge=0.0, le=1.0,
        description="1.0 = custom corporate domain, 0.0 = free provider (@gmail)"
    )
    location_match: bool = Field(
        ..., description="True if job location aligns with registered presence"
    )


class MRAVTriple(BaseModel):
    """(Hypothesis, Explanation, Consistency Score) triple from M-RAV LLM alignment."""

    hypothesis: str = Field(
        ..., description="'Confirms', 'Contradicts', or 'Neutral'"
    )
    explanation: str = Field(
        ..., description="Verbalized reasoning about field-level matches/mismatches"
    )
    consistency_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="0.0 = total contradiction, 1.0 = exact match"
    )


# ==========================================
# 2. PER-DOCUMENT CREDIBILITY RESULT
# ==========================================
class DocumentCredibility(BaseModel):
    """Synthesized credibility scores for a single retrieved document."""

    doc_id: str
    source_type: str = Field(
        ..., description="'registry', 'linkedin', 'scam_report'"
    )
    authority_score: float = Field(
        ..., ge=0.0, le=1.0, description="Tier weight based on source origin"
    )
    freshness_score: float = Field(
        ..., ge=0.0, le=1.0, description="Recency weight from plateau sigmoid"
    )
    consensus_score: float = Field(
        ..., ge=0.0, le=1.0, description="Agreement weight across fields"
    )
    overall_credibility_index: float = Field(
        ..., ge=0.0, le=1.0, description="C_idx = 0.4*authority + 0.2*freshness + 0.4*consensus"
    )
    credibility_tier: Literal["discard", "low_confidence", "trusted"] = Field(
        ..., description="Classification based on C_idx thresholds"
    )


# ==========================================
# 3. FLAGS & EVALUATION SUMMARY
# ==========================================
class CredibilityFlag(BaseModel):
    """A structured flag raised during credibility evaluation."""

    severity: Literal["CRITICAL", "WARNING", "INFO"] = Field(
        ..., description="Flag severity level"
    )
    message: str = Field(
        ..., description="Human-readable description of the issue"
    )


class EvaluatedDocument(BaseModel):
    """Complete evaluation result for a single document, including M-RAV triple."""

    source: str
    source_type: str
    authority: float
    freshness: float
    consensus: float
    credibility_index: float
    credibility_tier: Literal["discard", "low_confidence", "trusted"]
    consensus_breakdown: EntityConsensusScore
    mrav_triple: MRAVTriple


class CredibilityEvaluation(BaseModel):
    """Aggregated credibility result across all documents."""

    overall_trust_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weighted average C_idx of non-discarded documents"
    )
    flags_triggered: List[CredibilityFlag] = Field(
        default_factory=list, description="All flags aggregated from evaluation"
    )


# ==========================================
# 4. OBJECTIVE 2 OUTPUT PAYLOAD
# ==========================================
class Objective2OutputPayload(BaseModel):
    """
    Final deliverable from Objective 2, handed off to Objective 4.

    Contains the overall trust assessment, per-document evaluated evidence,
    triggered flags, and the routing decision.
    """

    offer_id: str
    credibility_evaluation: CredibilityEvaluation
    evaluated_evidence: List[EvaluatedDocument] = Field(
        default_factory=list,
        description="Per-document evaluation results (excludes discarded documents)"
    )
    discarded_evidence: List[EvaluatedDocument] = Field(
        default_factory=list,
        description="Documents filtered out due to C_idx < 0.3"
    )
    routing_decision: Literal["PROCEED_TO_CONFIDENCE_FRAMEWORK"] = Field(
        default="PROCEED_TO_CONFIDENCE_FRAMEWORK",
        description="Always routes to confidence framework (voice verification out of scope)"
    )
