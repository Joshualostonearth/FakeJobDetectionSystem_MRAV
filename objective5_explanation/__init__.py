# objective5_explanation — Grounded LLM explanation of the Real / Fake / Unable to Verify verdict
# See explainer.py for the public API.
from .explainer import (
    explain_confidence_result,
    explain_evidence_sheet,
    explain_voice_result,
    get_llm_client,
)
from .evidence_builder import build_from_confidence_result, build_from_voice_result
from .schemas import EvidenceSheet, ExplanationReport

__all__ = [
    "explain_confidence_result",
    "explain_voice_result",
    "explain_evidence_sheet",
    "get_llm_client",
    "build_from_confidence_result",
    "build_from_voice_result",
    "EvidenceSheet",
    "ExplanationReport",
]
