"""
full_pipeline_orchestrator.py — Master Orchestrator connecting Objectives 1, 2, 3, and 4 end-to-end.

Pipeline Flow:
    1. Objective 1: Input job offer payload / entities & evidence retrieval
    2. Objective 2: Credibility Filtration Layer (evaluate_credibility) + DST evidence
       combination (combine_all) + DST sufficiency score (score_dst_sufficiency)
    3. Objective 3: Voice Verification & Voice-to-Mass conversion (voice_transcript_to_mass),
       only when the Objective 2 DST score is NOT_SUFFICIENT_INFORMATION
    4. Objective 4: Platt Scaling Calibration (calculate_confidence_score) + Final Override Verification (apply_verification_override)
    5. Objective 5: Grounded LLM explanation of the verdict (explain_confidence_result), when explain=True
"""

from typing import Dict, Any, Optional, Callable

from schemas.input_schema import Objective2InputPayload
from credibility_engine import evaluate_credibility
from obj2dst import DSTEngine, combine_all, score_dst_sufficiency, ROUTE_TO_VOICE_AGENT
from confidence_calibrator import calculate_confidence_score, apply_verification_override
from objective3_voice_agent.voice_to_mass import voice_transcript_to_mass, voice_result_to_analysis
from objective5_explanation import explain_confidence_result


def run_full_pipeline(
    raw_payload: Dict[str, Any],
    voice_analysis_override: Optional[Dict[str, Any]] = None,
    voice_agent_fn: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    explain: bool = False,
) -> Dict[str, Any]:
    """
    Executes the end-to-end multi-objective Fake Job Detection & Verification pipeline.

    Args:
        raw_payload: Dictionary conforming to Objective2InputPayload schema
        voice_analysis_override: Optional voice analysis dict to simulate or incorporate Objective 3 voice results
        voice_agent_fn: Optional callable that runs the Objective 3 voice agent for raw_payload and returns
            a voice analysis dict (or None). Only called when Objective 2 routes to the voice agent.
        explain: When True, runs Objective 5 and adds an "explanation" report to the result.

    Returns:
        Complete unified report dictionary with overall verdict, calibrated confidence score, DST masses, and flags.
    """
    print("=" * 70)
    print("STARTING FULL PIPELINE VERIFICATION (OBJECTIVES 1 -> 2 -> 3 -> 4)")
    print("=" * 70)

    # ── Step 1 & 2: Objective 1 & 2 — Credibility Filtration Layer ─────
    payload = Objective2InputPayload(**raw_payload)
    credibility_output = evaluate_credibility(payload)
    retained_docs = credibility_output.evaluated_evidence
    flags = credibility_output.credibility_evaluation.flags_triggered

    print(f"\n[Obj 2 Complete] Evaluated {len(retained_docs)} trusted documents. Overall trust: {credibility_output.credibility_evaluation.overall_trust_score:.3f}")

    # ── Step 3: Convert Retained Evidence into DST Masses (Person 3) ─────
    dst_engine = DSTEngine()
    evidence_masses = []

    for doc in retained_docs:
        is_scam = 1 if (doc.source_type == "scam_report" or getattr(doc, "fraudulent", 0) == 1) else 0
        mass = dst_engine.assign_mass_function(
            similarity_score=doc.credibility_index,
            source_type=doc.source_type,
            is_fraudulent=is_scam,
            metadata_quality=doc.authority,
            domain_mismatch=doc.domain_mismatch,
        )
        evidence_masses.append(mass)

    # ── Step 4: Objective 2 — DST Evidence Combination & Sufficiency Score (Person 3) ─
    evidence_dst = combine_all(evidence_masses)
    obj2_dst = score_dst_sufficiency(evidence_dst)
    credibility_output.routing_decision = obj2_dst["routing_decision"]
    print(f"\n[Obj 2 DST] Masses: REAL={evidence_dst['REAL']:.4f}, FAKE={evidence_dst['FAKE']:.4f}, UNCERTAIN={evidence_dst['UNCERTAIN']:.4f} (Conflict K={evidence_dst['conflict']:.4f})")
    print(f"[Obj 2 DST] Score={obj2_dst['dst_score']:.4f} -> {obj2_dst['dst_status']} -> {obj2_dst['routing_decision']} ({obj2_dst['reason']})")

    # ── Step 5: Objective 3 — Voice Verification, only if evidence is insufficient (Person 2) ─
    voice_analysis = None
    voice_mass = None
    combined_dst = evidence_dst

    if obj2_dst["routing_decision"] == ROUTE_TO_VOICE_AGENT:
        voice_analysis = voice_analysis_override
        if voice_analysis is None and voice_agent_fn is not None:
            voice_analysis = voice_agent_fn(raw_payload)

        if voice_analysis:
            print("\n[Obj 3 Voice Integration] Processing voice verification transcript analysis...")
            voice_mass = voice_transcript_to_mass(voice_analysis)
            combined_dst = combine_all(evidence_masses + [voice_mass])
            voice_status = "COMPLETED"
            print(f"  -> Voice Mass: REAL={voice_mass['REAL']:.4f}, FAKE={voice_mass['FAKE']:.4f}, UNCERTAIN={voice_mass['UNCERTAIN']:.4f}")
            print(f"  -> Combined with voice: REAL={combined_dst['REAL']:.4f}, FAKE={combined_dst['FAKE']:.4f}, UNCERTAIN={combined_dst['UNCERTAIN']:.4f} (Conflict K={combined_dst['conflict']:.4f})")
        else:
            voice_status = "REQUIRED_NOT_RUN"
            print("\n[Obj 3 Voice Integration] Voice verification required but no voice result available. Verdict below is provisional.")
    else:
        voice_status = "SKIPPED"
        print("\n[Obj 3 Voice Integration] Skipped: evidence sufficient, going directly to Objective 4.")

    # ── Step 6: Objective 4 — Platt Scaling Calibration (Person 4) ─────
    calibrated_prob = calculate_confidence_score(combined_dst)
    print(f"[Obj 4 Platt Scaling] Calibrated P(Real) Confidence: {calibrated_prob:.4f}")

    # ── Step 7: Objective 4 — Safety Override & Final Verdict (Person 4) ─
    final_verdict_payload = apply_verification_override(
        calibrated_confidence=calibrated_prob,
        dst_result=combined_dst,
        flags=flags,
        voice_signals=voice_analysis,
    )
    final_verdict_payload["provisional"] = voice_status == "REQUIRED_NOT_RUN"

    print("\n" + "=" * 70)
    print("FINAL UNIFIED SYSTEM VERDICT REPORT")
    print("=" * 70)
    print(f"VERDICT              : {final_verdict_payload['final_verdict']}")
    print(f"CALIBRATED CONFIDENCE: {final_verdict_payload['calibrated_confidence']:.2%}")
    print(f"IS SCAM              : {final_verdict_payload['is_scam']}")
    print(f"CONFLICT FACTOR (K)  : {final_verdict_payload['conflict_factor_k']:.4f}")
    print(f"VOICE VERIFICATION   : {voice_status}")
    if final_verdict_payload["override_applied"]:
        print(f"OVERRIDE APPLIED     : YES -> {final_verdict_payload['override_reason']}")
    print("=" * 70)

    result = {
        "offer_id": payload.offer_id,
        "obj2_credibility": credibility_output.model_dump(mode="json"),
        "obj2_dst": {"evidence_masses": evidence_dst, **obj2_dst},
        "voice_verification": {"status": voice_status, "voice_mass": voice_mass},
        "dst_combined_masses": combined_dst,
        "final_verification": final_verdict_payload,
    }

    # ── Step 8: Objective 5 — Grounded Explanation of the Verdict ─────
    if explain:
        report = explain_confidence_result(
            result,
            entities=payload.extracted_entities.model_dump(),
        )
        print("\n" + report.to_text())
        result["explanation"] = report.model_dump(mode="json")

    return result


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

    run_full_pipeline(demo, simulated_voice_result, explain=True)
