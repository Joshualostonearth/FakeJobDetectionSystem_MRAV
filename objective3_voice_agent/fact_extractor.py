"""
fact_extractor.py — Step 6: Extract structured boolean signals from the call transcript.

Uses langchain-ollama ChatOllama with .with_structured_output() to enforce the
VoiceVerificationResult Pydantic schema. Handles empty transcripts and LLM errors
gracefully — never crashes the pipeline.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from schemas import VoiceVerificationResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Extraction prompt
# ──────────────────────────────────────────────────────────

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a structured data extractor. Extract boolean verification signals "
        "from a phone call transcript. Be conservative — if something is unclear, use null. "
        "Respond only with the structured output. No explanation.",
    ),
    (
        "human",
        "Company: {company_name}\n"
        "Role: {role}\n"
        "Recruiter: {recruiter_name}\n\n"
        "Transcript:\n{transcript}\n\n"
        "Extract the verification signals.",
    ),
])


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def extract_facts(
    transcript: str,
    entity: dict,
    config: dict,
) -> VoiceVerificationResult:
    """
    Extract structured boolean signals from the call transcript using ChatOllama.

    Args:
        transcript: Raw transcribed text from the call recording
        entity:     Job offer entity dict (company_name, role, recruiter_name)
        config:     voice_agent_config dict (ollama_model)

    Returns:
        VoiceVerificationResult — always returns a valid instance, never raises.
    """
    # Fast-path: empty transcript — skip LLM entirely
    if not transcript or not transcript.strip():
        logger.warning("[FactExtractor] Empty transcript — returning zero-confidence result")
        return VoiceVerificationResult(
            company_reached=False,
            company_denied_role=False,
            suspicious_response=False,
            response_summary="Empty or missing transcript — no response recorded",
            confidence=0.0,
        )

    try:
        llm = ChatOllama(
            model=config.get("ollama_model", "llama3.2"),
            temperature=0,
        ).with_structured_output(VoiceVerificationResult)

        chain = EXTRACTION_PROMPT | llm

        result: VoiceVerificationResult = chain.invoke({
            "company_name": entity.get("company_name", "Unknown"),
            "role": entity.get("role", entity.get("job_role", "Unknown")),
            "recruiter_name": entity.get("recruiter_name") or "Not specified",
            "transcript": transcript,
        })

        logger.info(
            f"[FactExtractor] Extraction complete — "
            f"company_reached={result.company_reached}, "
            f"confidence={result.confidence}"
        )
        return result

    except Exception as e:
        # Never crash the pipeline on extraction failure
        error_msg = str(e)[:120]
        logger.error(f"[FactExtractor] Extraction failed: {error_msg}")
        return VoiceVerificationResult(
            company_reached=False,
            company_denied_role=False,
            suspicious_response=False,
            response_summary=f"Signal extraction failed: {error_msg}",
            confidence=0.0,
        )
