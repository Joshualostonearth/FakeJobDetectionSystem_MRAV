"""
Input schemas for Objective 2: Credibility Filtration Layer.

These models define the contract between Objective 1 (FAISS Retrieval)
and Objective 2 (Credibility Evaluation). The Objective2InputPayload
is the single entrypoint that the credibility engine consumes.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ==========================================
# 1. ENTITY SCHEMA (From User's Offer)
# ==========================================
class JobOfferEntities(BaseModel):
    company_name: str = Field(..., description="Name extracted from offer letter")
    job_role: str = Field(..., description="Target role title")
    recruiter_name: Optional[str] = None
    domain_or_website: Optional[str] = Field(
        None, description="Apex domain claimed in offer, e.g. 'apextech.com'"
    )
    email_or_phone: Optional[str] = Field(
        None, description="Contact email/phone from offer"
    )
    salary_or_stipend: Optional[str] = None
    location: Optional[str] = None


# ==========================================
# 2. FAISS VECTOR RETRIEVAL SCHEMA
# ==========================================
class RetrivedDocMetadata(BaseModel):
    source_type: str = Field(
        ..., description="'legitimate_job_record', 'scam_report', or other source types"
    )
    job_id: Optional[str] = None
    fraudulent: Optional[int] = None
    authority_tier: Optional[int] = None
    title: Optional[str] = None
    industry: Optional[str] = None
    location: Optional[str] = None
    has_logo: Optional[int] = None
    telecommuting: Optional[int] = None
    
    company_name: Optional[str] = None
    domain: Optional[str] = None
    credibility_tier: Optional[str] = Field(
        "community",
        description="'official', 'high', 'third_party', 'community'",
    )
    last_updated: Optional[str] = Field(
        None, description="ISO timestamp (YYYY-MM-DD) for freshness calculation"
    )
    source_url: Optional[str] = None


class VectorDocument(BaseModel):
    doc_id: str
    content: str = Field(..., description="Raw chunk text retrieved from FAISS")
    similarity_score: float = Field(
        ..., ge=0.0, description="FAISS cosine/L2 similarity score"
    )
    metadata: RetrivedDocMetadata


class RetrievedEvidenceBundle(BaseModel):
    legitimate_job_records: List[VectorDocument] = []
    registry_records: List[VectorDocument] = []
    linkedin_data: List[VectorDocument] = []
    scam_reports: List[VectorDocument] = []


# ==========================================
# 3. OBJECTIVE 2 INPUT ENTRYPOINT PAYLOAD
# ==========================================
class Objective2InputPayload(BaseModel):
    offer_id: str
    extracted_entities: JobOfferEntities
    retrieved_evidence: RetrievedEvidenceBundle
