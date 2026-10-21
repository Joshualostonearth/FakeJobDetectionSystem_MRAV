"""
schemas.py — All Pydantic models and dataclasses for Objective 3.

Every other module in objective3_voice_agent imports from here.
No model definitions exist anywhere else.
"""

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────
# 1. Sourced Phone Number
# ──────────────────────────────────────────────────────────

@dataclass
class SourcedNumber:
    """A phone number independently sourced from an official API — never from the job posting."""
    phone_number: str               # E.164 format: +91XXXXXXXXXX
    source: str                     # "google_places" | "opencorporates" | "serpapi" | "dashboard_override"
    company_name_matched: str       # Company name as returned by the source
    confidence: float               # 0.0 to 1.0


# ──────────────────────────────────────────────────────────
# 2. Voice Verification Result (LLM-extracted signals)
# ──────────────────────────────────────────────────────────

class VoiceVerificationResult(BaseModel):
    company_reached: bool = Field(
        description="True if a human or IVR at the company answered"
    )
    role_actively_hiring: Optional[bool] = Field(
        default=None,
        description="True if company confirmed the role is open. None if unclear."
    )
    recruiter_recognized: Optional[bool] = Field(
        default=None,
        description="True if the recruiter name was recognized. None if not mentioned."
    )
    company_denied_role: bool = Field(
        default=False,
        description="True if the company explicitly said no such role exists"
    )
    suspicious_response: bool = Field(
        default=False,
        description="True if the response was evasive, hostile, or scam-like"
    )
    response_summary: str = Field(
        description="One-sentence plain English summary of what was said"
    )
    confidence: float = Field(
        description="Extraction confidence 0.0 to 1.0"
    )


# ──────────────────────────────────────────────────────────
# 3. Pipeline Result (returned by run_voice_verification)
# ──────────────────────────────────────────────────────────

class PipelineResult(BaseModel):
    claim_id: str
    pipeline_completed: bool
    skip_reason: Optional[str] = None       # e.g. "no_official_number_found", "recording_not_received"
    voice_signals: Optional[dict] = None    # Populated on success
    error: Optional[str] = None             # Populated on unhandled exception
    transcript: Optional[str] = None
    sourced_number: Optional[str] = None
