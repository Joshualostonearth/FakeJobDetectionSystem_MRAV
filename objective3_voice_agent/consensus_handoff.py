"""
consensus_handoff.py — Step 7: Push extracted voice signals into Objective 2.

Attempts to import update_consensus from objective2.consensus_engine.
On ImportError (Objective 2 not available), logs the signals and continues —
Objective 3 is fully standalone for testing.
"""

import logging
from typing import Optional

from schemas import SourcedNumber, VoiceVerificationResult

logger = logging.getLogger(__name__)


def push_voice_signals(
    claim_id: str,
    result: VoiceVerificationResult,
    sourced_number: Optional[SourcedNumber],
) -> dict:
    """
    Build the unified voice signal dict and push it to Objective 2's consensus engine.

    Args:
        claim_id:       Unique identifier for this verification run
        result:         Structured extraction result from fact_extractor
        sourced_number: The phone number used (None if no number was found)

    Returns:
        The signals dict (returned whether Obj2 is available or not).
    """
    signals = {
        "company_reached":       result.company_reached,
        "role_actively_hiring":  result.role_actively_hiring,
        "recruiter_recognized":  result.recruiter_recognized,
        "company_denied_role":   result.company_denied_role,
        "suspicious_response":   result.suspicious_response,
        "source_confidence":     sourced_number.confidence if sourced_number else 0.0,
        "extraction_confidence": result.confidence,
        "number_source":         sourced_number.source if sourced_number else "none",
        "response_summary":      result.response_summary,
    }

    try:
        # Attempt handoff to Objective 2 consensus engine
        from objective2.consensus_engine import update_consensus  # type: ignore
        update_consensus(claim_id=claim_id, source="voice_verification", signals=signals)
        logger.info(f"[ConsensusHandoff] ✅ Signals pushed to Objective 2 (claim_id={claim_id})")

    except ImportError:
        # Objective 2 not present — this is expected in standalone testing mode
        logger.warning(
            "[ConsensusHandoff] ⚠️  Objective 2 consensus engine not found. "
            "Signals logged locally only."
        )
        logger.info(f"[ConsensusHandoff] Signals for {claim_id}: {signals}")

    except Exception as e:
        logger.error(f"[ConsensusHandoff] Failed to push signals: {e}")

    return signals
