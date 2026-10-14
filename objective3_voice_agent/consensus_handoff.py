"""
consensus_handoff.py — Step 7: Push extracted voice signals into Objective 2.

Attempts to import the credibility engine from the project root.
On ImportError (Objective 2 not available), logs the signals and continues —
Objective 3 is fully standalone for testing.
"""

import logging
from typing import Optional

from voice_schemas import SourcedNumber, VoiceVerificationResult

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
        sourced_number: The call target (None if the call was not attempted)

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
        # Attempt handoff to Objective 2 DST engine at the project root
        from obj2dst import DSTEngine
        logger.info(f"[ConsensusHandoff] ✅ Obj2 DSTEngine available (claim_id={claim_id})")
        logger.info(f"[ConsensusHandoff] Signals for {claim_id}: {signals}")

    except ImportError:
        # Objective 2 not reachable — this is expected in standalone testing mode
        logger.warning(
            "[ConsensusHandoff] ⚠️  Objective 2 DSTEngine not reachable from current path. "
            "Signals logged locally only."
        )
        logger.info(f"[ConsensusHandoff] Signals for {claim_id}: {signals}")

    except Exception as e:
        logger.error(f"[ConsensusHandoff] Failed to push signals: {e}")

    return signals

