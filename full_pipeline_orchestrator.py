"""
full_pipeline_orchestrator.py — Master Orchestrator connecting Objectives 1, 2, 3, and 4 end-to-end.

Pipeline Flow:
    1. Objective 1: Input job offer payload / entities & evidence retrieval
    2. Objective 2: Credibility Filtration Layer (evaluate_credibility)
    3. Objective 3: Voice Verification & Voice-to-Mass conversion (voice_transcript_to_mass)
    4. Objective 4: DST Evidence Mass Combination (combine_all) + Platt Scaling Calibration (calculate_confidence_score) + Final Override Verification (apply_verification_override)
"""

from typing import Dict, Any, Optional

from schemas.input_schema import Objective2InputPayload
from schemas.credibility_schema import Objective2OutputPayload
from credibility_engine import evaluate_credibility
from obj2dst import DSTEngine, combine_all
from confidence_calibrator import calculate_confidence_score, apply_verification_override
from objective3_voice_agent.voice_to_mass import voice_transcript_to_mass, voice_result_to_analysis


# How much the text classifier is trusted as a DST source (rest of its mass is ignorance)
CLASSIFIER_RELIABILITY: float = 0.9


def run_objective4(
    credibility_output: Objective2OutputPayload,
    scam_probability: Optional[float] = None,
    offer_text: Optional[str] = None,
    voice_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Objective 4 on an Objective 2 result: DST combination of the retained evidence,
    text classifier and (optional) voice evidence, Platt calibration, and the override.

    Used by run_full_pipeline before the call, and by the phone verification bridge
    after the call with the same Objective 2 result plus the call's voice signals.

    Returns:
        Dict with scam_probability, dst_combined_masses and final_verification.
    """
    retained_docs = credibility_output.evaluated_evidence
    flags = credibility_output.credibility_evaluation.flags_triggered

    # ── Step 3: Convert Retained Evidence into DST Masses (Person 3) ─────
    dst_engine = DSTEngine()
    evidence_masses = []

    for doc in retained_docs:
        is_scam = 1 if (doc.source_type == "scam_report" or getattr(doc, "fraudulent", 0) == 1) else 0
        if not is_scam and doc.mrav_triple.hypothesis == "Contradicts":
            # A legitimate record that the offer contradicts describes some other posting;
            # it is no support for this offer being real, so it contributes only ignorance.
            evidence_masses.append({"REAL": 0.0, "FAKE": 0.0, "UNCERTAIN": 1.0,
                                    "source": doc.source_type, "strength": 0.0})
            continue
        mass = dst_engine.assign_mass_function(
            similarity_score=doc.credibility_index,
            source_type=doc.source_type,
            is_fraudulent=is_scam,
            metadata_quality=doc.authority,
        )
        evidence_masses.append(mass)

    # ── Step 3b: Objective 1b — Supervised text classifier evidence ─────
    if scam_probability is None and offer_text:
        try:
            from scam_classifier import predict_scam_probability
            scam_probability = predict_scam_probability(offer_text)
        except FileNotFoundError as e:
            print(f"\n[Classifier Skipped] {e}")
    if scam_probability is not None:
        classifier_mass = dst_engine.classifier_mass(scam_probability, CLASSIFIER_RELIABILITY)
        evidence_masses.append(classifier_mass)
        print(f"\n[Classifier] P(scam)={scam_probability:.4f} -> REAL={classifier_mass['REAL']:.4f}, FAKE={classifier_mass['FAKE']:.4f}")

    # ── Step 4: Objective 3 — Voice Mass Integration (Person 2) ────────
    if voice_analysis:
        print("\n[Obj 3 Voice Integration] Processing voice verification transcript analysis...")
        voice_mass = voice_transcript_to_mass(voice_analysis)
        evidence_masses.append(voice_mass)
        print(f"  -> Voice Mass: REAL={voice_mass['REAL']:.4f}, FAKE={voice_mass['FAKE']:.4f}, UNCERTAIN={voice_mass['UNCERTAIN']:.4f}")
    else:
        voice_mass = None

    # ── Step 5: Objective 4 — DST Evidence Combination (Person 3) ───────
    combined_dst = combine_all(evidence_masses)
    print(f"\n[Obj 4 DST Combination] Combined Masses: REAL={combined_dst['REAL']:.4f}, FAKE={combined_dst['FAKE']:.4f}, UNCERTAIN={combined_dst['UNCERTAIN']:.4f} (Conflict K={combined_dst['conflict']:.4f})")

    # ── Step 6: Objective 4 — Platt Scaling Calibration (Person 4) ─────
    calibrated_prob = calculate_confidence_score(combined_dst)
    print(f"[Obj 4 Platt Scaling] Calibrated P(Real) Confidence: {calibrated_prob:.4f}")

    # ── Step 7: Objective 4 — Safety Override & Final Verdict (Person 4) ─
    final_verdict_payload = apply_verification_override(
        calibrated_confidence=calibrated_prob,
        dst_result=combined_dst,
        flags=flags,
        voice_signals=voice_analysis,
        alignment_hypotheses=[d.mrav_triple.hypothesis for d in retained_docs
                              if d.mrav_triple.explanation != "NOT_RUN"],
    )

    print("\n" + "=" * 70)
    print("FINAL UNIFIED SYSTEM VERDICT REPORT")
    print("=" * 70)
    print(f"VERDICT              : {final_verdict_payload['final_verdict']}")
    print(f"CALIBRATED CONFIDENCE: {final_verdict_payload['calibrated_confidence']:.2%}")
    print(f"IS SCAM              : {final_verdict_payload['is_scam']}")
    print(f"CONFLICT FACTOR (K)  : {final_verdict_payload['conflict_factor_k']:.4f}")
    if final_verdict_payload["override_applied"]:
        print(f"OVERRIDE APPLIED     : YES -> {final_verdict_payload['override_reason']}")
    print("=" * 70)

    return {
        "scam_probability": scam_probability,
        "dst_combined_masses": combined_dst,
        "final_verification": final_verdict_payload,
    }


def run_full_pipeline(
    raw_payload: Dict[str, Any],
    voice_analysis_override: Optional[Dict[str, Any]] = None,
    offer_text: Optional[str] = None,
    scam_probability: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executes the end-to-end multi-objective Fake Job Detection & Verification pipeline.

    Args:
        raw_payload: Dictionary conforming to Objective2InputPayload schema
        voice_analysis_override: Optional voice analysis dict to simulate or incorporate Objective 3 voice results
        offer_text: Optional raw offer text; scored by the supervised text classifier (scam_classifier.py)
        scam_probability: Optional precomputed classifier P(scam); takes precedence over offer_text

    Returns:
        Complete unified report dictionary with overall verdict, calibrated confidence score, DST masses, and flags.
    """
    print("=" * 70)
    print("STARTING FULL PIPELINE VERIFICATION (OBJECTIVES 1 -> 2 -> 3 -> 4)")
    print("=" * 70)

    # ── Step 1 & 2: Objective 1 & 2 — Credibility Filtration Layer ─────
    payload = Objective2InputPayload(**raw_payload)
    credibility_output = evaluate_credibility(payload)

    print(f"\n[Obj 2 Complete] Evaluated {len(credibility_output.evaluated_evidence)} trusted documents. Overall trust: {credibility_output.credibility_evaluation.overall_trust_score:.3f}")

    # ── Step 3-7: Objective 3 voice gate + Objective 4 ────────────────
    # Obj2's routing decides whether voice evidence is used at all
    voice_routed = credibility_output.routing_decision == "PROCEED_TO_VOICE_VERIFICATION"
    if voice_analysis_override and not voice_routed:
        print(f"\n[Obj 3 Skipped] Routing decision is {credibility_output.routing_decision}; ignoring voice analysis.")
        voice_analysis_override = None

    obj4 = run_objective4(credibility_output, scam_probability, offer_text, voice_analysis_override)

    return {
        "offer_id": payload.offer_id,
        "scam_probability": obj4["scam_probability"],
        "obj2_credibility": credibility_output.model_dump(mode="json"),
        "dst_combined_masses": obj4["dst_combined_masses"],
        "final_verification": obj4["final_verification"],
    }


if __name__ == "__main__":
    from main import get_demo_payload

    demo = get_demo_payload()

    # Simulate voice verification red flag
    simulated_voice_result = {
        "company_reached": True,
        "company_denied_role": False,
        "suspicious_response": True,
        "source_confidence": 0.90,
        "extraction_confidence": 0.85,
    }

    run_full_pipeline(demo, simulated_voice_result)
