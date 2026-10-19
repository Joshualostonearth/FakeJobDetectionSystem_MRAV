"""
voice_verification_bridge.py — Glue module connecting Objective 2 → Objective 3 → DST.

This module is the integration point between:
  - Objective 2 (Credibility Filtration Engine) output
  - Objective 3 (Voice Verification Agent) pipeline
  - Dempster-Shafer Theory (DST) evidence combination

Flow:
    1. Receives Obj2's output payload + extracted entities
    2. Checks routing_decision — if PROCEED_TO_VOICE_VERIFICATION, runs Obj3
    3. Converts Obj3's voice signals into DS mass via voice_to_mass
    4. Combines voice mass with Obj2's document-based evidence via DSTEngine
    5. Returns a unified final verdict with combined belief scores

Usage:
    from voice_verification_bridge import run_full_verification

    final_result = run_full_verification(
        obj2_output=obj2_result,
        entities=entities_dict,
        voice_config=voice_agent_config,  # optional
    )

    # From the Streamlit UI (simple Obj2 filter, raw FAISS evidence):
    from voice_verification_bridge import verify_offer_by_phone

    final_result = verify_offer_by_phone(
        offer_id="UI_1234",
        entities=entities_dict,
        retrieved_evidence=evidence_list,  # pipeline_bridge format
        suspicion_reasons=flags,
    )
"""

import logging
import os
import re
import urllib.request
import uuid
from typing import Callable, Optional

from obj2dst import DSTEngine
from schemas.credibility_schema import Objective2OutputPayload

logger = logging.getLogger(__name__)

# Longest slice of the offer text given to the question generator (local LLM context)
MAX_OFFER_EXCERPT_CHARS: int = 1500


def infer_job_type(job_role: str, offer_text: str) -> str:
    """'internship' if the role or offer text mentions an internship, otherwise 'job'."""
    return "internship" if re.search(r"\bintern", f"{job_role} {offer_text}", re.IGNORECASE) else "job"


def offer_excerpt(offer_text: str) -> str:
    """Offer text without merge source tags, whitespace collapsed, cut to MAX_OFFER_EXCERPT_CHARS."""
    text = re.sub(r"\[SOURCE: [A-Z_]+\]", " ", offer_text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_OFFER_EXCERPT_CHARS]



def run_full_verification(
    obj2_output: Objective2OutputPayload,
    entities: dict,
    voice_config: Optional[dict] = None,
) -> dict:
    """
    End-to-end bridge: Objective 2 → Objective 3 → DST combination.

    If Obj2 routes to voice verification, this function:
      1. Runs the voice agent pipeline (Obj3)
      2. Converts voice signals to DS mass functions
      3. Combines them with Obj2's document-based evidence
      4. Returns a unified final verdict

    If Obj2 routes to the confidence framework (no voice needed),
    this function returns the Obj2 results as-is with DS-combined
    document evidence only.

    Args:
        obj2_output:    The Objective2OutputPayload from credibility_engine
        entities:       Extracted job offer entities dict (company_name, job_role, etc.)
        voice_config:   Optional voice_agent_config dict. If None, loads default.

    Returns:
        Dict containing:
            - obj2_trust_score: Original Obj2 trust score
            - routing_decision: What Obj2 decided
            - voice_verification: Obj3 pipeline result (or None if skipped)
            - dst_combined: DST-combined belief masses from all evidence
            - final_verdict: Human-readable verdict string
            - final_belief_real: Combined belief in REAL (0-1)
            - final_belief_fake: Combined belief in FAKE (0-1)
            - final_belief_uncertain: Combined mass left uncommitted (0-1)
            - flags: All flags from Obj2 + any voice-derived flags
    """
    print("\n" + "=" * 60)
    print("🔗 VOICE VERIFICATION BRIDGE — Obj2 → Obj3 → DST")
    print("=" * 60)

    result = {
        "offer_id": obj2_output.offer_id,
        "obj2_trust_score": obj2_output.credibility_evaluation.overall_trust_score,
        "routing_decision": obj2_output.routing_decision,
        "obj2_flags": [f.model_dump() for f in obj2_output.credibility_evaluation.flags_triggered],
        "voice_verification": None,
        "voice_mass": None,
        "dst_combined": None,
        "final_verdict": None,
        "final_belief_real": None,
        "final_belief_fake": None,
        "final_belief_uncertain": None,
    }

    dst = DSTEngine()

    # ── Step 1: Build DS masses from Obj2's evaluated evidence ──────────
    print("\n--- Bridge Step 1: Building DS masses from Obj2 evidence ---")
    obj2_masses = []
    for doc in obj2_output.evaluated_evidence:
        # Map credibility tier to a rough is_fraudulent signal for DST
        # Documents from scam sources get is_fraudulent=1, others get 0
        is_fraud = 1 if doc.source_type in ("scam_report", "scam_report_forum") else 0

        mass = dst.assign_mass_function(
            similarity_score=doc.credibility_index,
            source_type=doc.source_type,
            is_fraudulent=is_fraud,
            metadata_quality=doc.authority,
        )
        obj2_masses.append(mass)
        print(f"  📄 {doc.source_type}: REAL={mass['REAL']:.3f}, FAKE={mass['FAKE']:.3f}")

    # ── Step 2: Voice verification (conditional) ───────────────────────
    if obj2_output.routing_decision == "PROCEED_TO_VOICE_VERIFICATION":
        print("\n--- Bridge Step 2: Running Voice Verification (Objective 3) ---")
        print(f"  🎯 Routing decision: {obj2_output.routing_decision}")
        _run_voice_step(
            result, obj2_masses, entities, voice_config,
            suspicion_reasons=[f.message for f in obj2_output.credibility_evaluation.flags_triggered],
        )
    else:
        print(f"\n--- Bridge Step 2: Skipping Voice Verification ---")
        print(f"  📋 Routing decision: {obj2_output.routing_decision}")
        print(f"  ✅ Evidence is sufficient — no voice verification needed.")

    # ── Step 3: Combine all masses via DST ─────────────────────────────
    _combine_masses(dst, obj2_masses, result)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔗 BRIDGE RESULT SUMMARY")
    print("=" * 60)
    print(f"  Offer ID:       {result['offer_id']}")
    print(f"  Obj2 Trust:     {result['obj2_trust_score']}")
    print(f"  Routing:        {result['routing_decision']}")
    print(f"  Voice Used:     {'Yes' if result['voice_verification'] else 'No'}")
    print(f"  DST Belief:     REAL={result['final_belief_real']}, FAKE={result['final_belief_fake']}")
    print(f"  Final Verdict:  {result['final_verdict']}")
    print("=" * 60)

    return result


# ──────────────────────────────────────────────────────────
# Shared steps
# ──────────────────────────────────────────────────────────

def _run_voice_step(
    result: dict,
    masses: list,
    entities: dict,
    voice_config: Optional[dict],
    suspicion_reasons: Optional[list] = None,
    offer_text: str = "",
    on_progress: Optional[Callable[[str, int], None]] = None,
) -> None:
    """
    Run Objective 3 and append its DS mass to `masses`.

    Fills result["voice_verification"] and result["voice_mass"]. Any failure
    is logged and leaves `masses` untouched, so the caller still combines the
    document evidence.
    """
    try:
        from objective3_voice_agent import (
            run_voice_verification,
            load_config as load_voice_config,
            voice_transcript_to_mass,
        )

        # Load voice config if not provided
        if voice_config is None:
            try:
                voice_config = load_voice_config()
            except Exception as e:
                logger.warning(f"[Bridge] Could not load voice config: {e}")
                voice_config = {
                    "ollama_model": os.getenv("OLLAMA_MODEL", "ministral-3"),
                    "server_port": 8765,
                    "recording_timeout": 600,
                }

        # Generate a claim ID
        claim_id = f"{result['offer_id']}_voice_{uuid.uuid4().hex[:8]}"

        # Build entity dict for Obj3 from Obj2's entities
        voice_entity = {
            # Extraction leaves missing fields as None; Obj3 expects strings
            "company_name": entities.get("company_name") or "",
            "role": entities.get("job_role") or "",
            "job_role": entities.get("job_role") or "",
            "location": entities.get("location") or "",
            "recruiter_name": entities.get("recruiter_name") or "",
            "domain_or_website": entities.get("domain_or_website") or "",
            "recruiter_email": entities.get("recruiter_email") or "",
            "job_type": infer_job_type(entities.get("job_role") or "", offer_text),
            "offer_text": offer_excerpt(offer_text),
            "suspicion_reasons": list(suspicion_reasons or []),
        }

        # Run the voice verification pipeline
        voice_result = run_voice_verification(
            claim_id=claim_id,
            entity=voice_entity,
            config=voice_config,
            event_push=on_progress,
        )

        result["voice_verification"] = voice_result

        # Convert voice signals to DS mass
        if voice_result.get("voice_signals"):
            print("\n--- Bridge Step 2b: Converting voice signals to DS mass ---")
            voice_mass = voice_transcript_to_mass(voice_result["voice_signals"])
            result["voice_mass"] = voice_mass

            # Convert to DSTEngine-compatible format
            voice_dst_mass = {
                "REAL": voice_mass.get("REAL", 0.0),
                "FAKE": voice_mass.get("FAKE", 0.0),
                "UNCERTAIN": voice_mass.get("UNCERTAIN", 1.0),
                "source": "voice_call",
                "strength": voice_mass.get("strength", 0.0),
            }
            masses.append(voice_dst_mass)

            print(f"  🎤 Voice mass: REAL={voice_dst_mass['REAL']:.3f}, "
                  f"FAKE={voice_dst_mass['FAKE']:.3f}, "
                  f"strength={voice_dst_mass['strength']:.3f}")
        else:
            skip_reason = voice_result.get("skip_reason") or voice_result.get("error") or "unknown"
            print(f"\n  ⚠️  Voice pipeline did not produce signals (reason: {skip_reason})")

    except ImportError as e:
        logger.warning(f"[Bridge] Objective 3 not available: {e}")
        print(f"\n  ⚠️  Objective 3 import failed: {e}")
        print("  Continuing with Obj2 evidence only.")
        result["voice_error"] = f"Objective 3 import failed: {e}"

    except Exception as e:
        logger.error(f"[Bridge] Voice verification error: {e}", exc_info=True)
        print(f"\n  ❌ Voice verification error: {e}")
        print("  Continuing with Obj2 evidence only.")
        result["voice_error"] = str(e)


def _combine_masses(dst: DSTEngine, masses: list, result: dict) -> None:
    """Combine all masses via Dempster's rule and fill the final_* fields of `result`."""
    print("\n--- Bridge Step 3: Combining all evidence via Dempster-Shafer ---")

    if masses:
        combined = dst.combine_all_evidence(masses)
        result["dst_combined"] = combined
        result["final_belief_real"] = round(combined.get("REAL", 0.0), 4)
        result["final_belief_fake"] = round(combined.get("FAKE", 0.0), 4)
        result["final_belief_uncertain"] = round(combined.get("UNCERTAIN", 1.0), 4)

        belief_desc = dst.get_belief_strength(combined)
        conflict_desc = dst.get_conflict_info(combined)
        result["final_verdict"] = belief_desc

        print(f"  📊 Combined REAL: {result['final_belief_real']:.4f}")
        print(f"  📊 Combined FAKE: {result['final_belief_fake']:.4f}")
        print(f"  📊 Uncertain:     {result['final_belief_uncertain']:.4f}")
        print(f"  📊 Conflict:      {combined.get('conflict', 0):.4f}")
        print(f"  🏷️  Verdict:       {belief_desc}")
        print(f"  ⚖️  Conflict:      {conflict_desc}")
    else:
        result["final_belief_real"] = 0.0
        result["final_belief_fake"] = 0.0
        result["final_belief_uncertain"] = 1.0
        result["final_verdict"] = "UNCERTAIN - No evidence to combine"
        print("  ⚠️  No evidence masses to combine.")


# ──────────────────────────────────────────────────────────
# UI entry point (simple Obj2 filter → Obj3 → DST)
# ──────────────────────────────────────────────────────────

def distance_to_similarity(distance: float) -> float:
    """
    Convert a FAISS IndexFlatL2 score into cosine similarity (0-1).

    all-MiniLM-L6-v2 embeddings are unit-length and IndexFlatL2 returns the
    squared L2 distance, so cos = 1 - d / 2.
    """
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


def check_voice_services(voice_config: Optional[dict] = None, timeout: float = 2.0) -> Optional[str]:
    """
    Check that the call bridge needed by Objective 3 is reachable.

    Returns None if it is, otherwise a human-readable message describing
    what to start.
    """
    if voice_config is None:
        try:
            from objective3_voice_agent import load_config as load_voice_config
            voice_config = load_voice_config()
        except Exception:
            voice_config = {}

    bridge = voice_config.get("webrtc_bridge_url", "http://localhost:5000").rstrip("/")
    try:
        with urllib.request.urlopen(f"{bridge}/bridge/status", timeout=timeout):
            return None
    except Exception:
        return (
            f"The call bridge is not reachable at {bridge}. Start it with "
            "'python whisper_Test/call_bridge.py' and the WebRTC server with "
            "'node webrtc-call/server/server.js', then open http://localhost:3000 "
            "to answer the verification call."
        )


def verify_offer_by_phone(
    offer_id: str,
    entities: dict,
    retrieved_evidence: list,
    suspicion_reasons: Optional[list] = None,
    voice_config: Optional[dict] = None,
    scam_probability: Optional[float] = None,
    offer_text: str = "",
    on_progress: Optional[Callable[[str, int], None]] = None,
    obj2_credibility: Optional[dict] = None,
) -> dict:
    """
    Run voice verification for an offer analysed by the Streamlit UI and
    produce the final verdict with the call's evidence included.

    With obj2_credibility (the Objective 2 result of the analysis), the verdict
    comes from Objective 4 (run_objective4): Obj2's retained evidence, the text
    classifier and the voice signals combined via DST, Platt-calibrated, with the
    voice and Obj2 flag overrides. Without it, the raw FAISS evidence is combined
    via DST only.

    Args:
        offer_id:           Identifier for this offer (used in the claim ID)
        entities:           Extracted entities dict (models.JobOfferEntities.model_dump())
        retrieved_evidence: pipeline_bridge evidence list, each item
                            {"page_content", "metadata", "distance"}
        suspicion_reasons:  Flags from the Obj2 filter, passed to question generation
        voice_config:       Optional voice_agent_config dict. If None, loads default.
        scam_probability:   Text classifier P(scam), combined as another mass if given
        offer_text:         The analysed offer text, used to tailor the call questions
        on_progress:        Optional callback(text, step) for Objective 3 progress
        obj2_credibility:   Objective2OutputPayload dump from run_full_pipeline (its
                            "obj2_credibility" field); not re-evaluated

    Returns:
        Dict with the same final_* / voice_* fields as run_full_verification(), plus
        calibrated_confidence, override_reason and final_verification when
        obj2_credibility is given.
    """
    print("\n" + "=" * 60)
    print("🔗 VOICE VERIFICATION BRIDGE — UI → Obj3 → DST")
    print("=" * 60)

    result = {
        "offer_id": offer_id,
        "voice_verification": None,
        "voice_mass": None,
        "voice_error": None,
        "dst_combined": None,
        "final_verdict": None,
        "final_belief_real": None,
        "final_belief_fake": None,
        "final_belief_uncertain": None,
    }

    if obj2_credibility is not None:
        return _verify_with_objective4(
            result, obj2_credibility, entities, voice_config, suspicion_reasons,
            scam_probability, offer_text, on_progress,
        )

    dst = DSTEngine()

    print("\n--- Bridge Step 1: Building DS masses from FAISS evidence ---")
    masses = []
    for ev in retrieved_evidence:
        meta = ev.get("metadata", {}) or {}
        source_type = meta.get("source_type", "unknown")
        mass = dst.assign_mass_function(
            similarity_score=distance_to_similarity(float(ev.get("distance", 2.0))),
            source_type=source_type,
            is_fraudulent=1 if meta.get("fraudulent") == 1 or source_type == "scam_report" else 0,
        )
        masses.append(mass)
        print(f"  📄 {source_type}: REAL={mass['REAL']:.3f}, FAKE={mass['FAKE']:.3f}")

    if scam_probability is not None:
        from full_pipeline_orchestrator import CLASSIFIER_RELIABILITY
        mass = dst.classifier_mass(scam_probability, CLASSIFIER_RELIABILITY)
        masses.append(mass)
        print(f"  🧮 text_classifier: REAL={mass['REAL']:.3f}, FAKE={mass['FAKE']:.3f}")

    print("\n--- Bridge Step 2: Running Voice Verification (Objective 3) ---")
    _run_voice_step(result, masses, entities, voice_config, suspicion_reasons, offer_text, on_progress)

    _combine_masses(dst, masses, result)
    return result


def _verify_with_objective4(
    result: dict,
    obj2_credibility: dict,
    entities: dict,
    voice_config: Optional[dict],
    suspicion_reasons: Optional[list],
    scam_probability: Optional[float],
    offer_text: str,
    on_progress: Optional[Callable[[str, int], None]],
) -> dict:
    """Run the call, then Objective 4 on the Obj2 result plus the call's voice signals."""
    from full_pipeline_orchestrator import run_objective4

    print("\n--- Bridge Step 1: Running Voice Verification (Objective 3) ---")
    # The voice mass is built again inside run_objective4 from the voice signals
    _run_voice_step(result, [], entities, voice_config, suspicion_reasons, offer_text, on_progress)
    voice_signals = (result["voice_verification"] or {}).get("voice_signals") or None

    print("\n--- Bridge Step 2: Objective 4 on the Objective 2 result + voice evidence ---")
    # The call was placed by the UI's risk decision, so the voice evidence is used
    # regardless of Obj2's routing_decision.
    obj4 = run_objective4(
        Objective2OutputPayload.model_validate(obj2_credibility),
        scam_probability=scam_probability,
        voice_analysis=voice_signals,
    )
    combined = obj4["dst_combined_masses"]
    fv = obj4["final_verification"]

    result["dst_combined"] = combined
    result["final_belief_real"] = round(combined.get("REAL", 0.0), 4)
    result["final_belief_fake"] = round(combined.get("FAKE", 0.0), 4)
    result["final_belief_uncertain"] = round(combined.get("UNCERTAIN", 1.0), 4)
    result["final_verdict"] = fv["final_verdict"]
    result["calibrated_confidence"] = fv["calibrated_confidence"]
    result["override_reason"] = fv["override_reason"]
    result["final_verification"] = fv
    return result
