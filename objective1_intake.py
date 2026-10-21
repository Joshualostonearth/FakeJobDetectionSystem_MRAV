"""
objective1_intake.py — Connects Objective 1 to Objective 2 and the full pipeline.

Pipeline Flow:
    1. OCR on the offer image (ocr_engine) merged with any raw text
    2. Entity extraction (models.extract_entities_from_text)
    3. FAISS evidence retrieval (search_engine.EvidenceRetriever)
    4. Conversion into the Objective2InputPayload contract
    5. Hand-off to run_full_pipeline (Objectives 2 -> 3 -> 4)

Public API:
    build_objective2_payload(text, image_path, ...) -> dict
    verify_offer_from_input(text, image_path, ...) -> dict

Usage:
    python objective1_intake.py --text "offer text"
    python objective1_intake.py --image richmond_offer.png
"""

import argparse
import json
import uuid
from typing import Any, Callable, Dict, Optional

EVIDENCE_CATEGORIES = ("legitimate_job_records", "registry_records", "linkedin_data", "scam_reports")


def _empty_evidence() -> Dict[str, list]:
    return {category: [] for category in EVIDENCE_CATEGORIES}


def _to_objective2_entities(entities) -> Dict[str, Any]:
    """Map Objective 1's JobOfferEntities onto Objective 2's entity schema."""
    return {
        "company_name": entities.company_name or "",
        "job_role": entities.job_role or "",
        "recruiter_name": None,
        "domain_or_website": entities.domain_or_website,
        "email_or_phone": entities.recruiter_email,
        "salary_or_stipend": None,
        "location": entities.location,
    }


def _to_objective2_evidence(bundle: Dict[str, Any]) -> Dict[str, list]:
    """Keep only a successful, categorised evidence bundle; otherwise no evidence."""
    retrieved = bundle.get("retrieved_evidence")
    if bundle.get("status") != "SUCCESS" or not isinstance(retrieved, dict):
        return _empty_evidence()
    return {category: list(retrieved.get(category, [])) for category in EVIDENCE_CATEGORIES}


def build_objective2_payload(
    text: str = "",
    image_path: str = "",
    offer_id: Optional[str] = None,
    retriever=None,
    ocr_reader=None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Runs Objective 1 on a raw offer and returns a dict conforming to Objective2InputPayload,
    plus an "objective1" block describing what was extracted and retrieved.
    """
    from models import extract_entities_from_text

    ocr_text = ""
    if image_path:
        from ocr_engine import init_ocr_reader, extract_text_from_image
        reader = ocr_reader or init_ocr_reader()
        ocr_text = extract_text_from_image(reader, image_path)

    from ocr_engine import merge_text_and_ocr
    merged_text = merge_text_and_ocr(text, ocr_text)

    entities = extract_entities_from_text(merged_text)

    if retriever is None:
        from search_engine import EvidenceRetriever
        retriever = EvidenceRetriever()

    # Without a loaded index the retriever returns a placeholder record that simply
    # repeats the offer's own claims, so treat that case as "no evidence".
    if getattr(retriever, "vector_db", None) is None:
        retrieval_status = "NO_INDEX"
        evidence = _empty_evidence()
    else:
        bundle = retriever.retrieve_evidence_bundle(entities, top_k=top_k)
        retrieval_status = bundle.get("status", "FAILED")
        evidence = _to_objective2_evidence(bundle)

    return {
        "offer_id": offer_id or f"OFFER_{uuid.uuid4().hex[:8].upper()}",
        "extracted_entities": _to_objective2_entities(entities),
        "retrieved_evidence": evidence,
        "objective1": {
            "merged_text": merged_text,
            "retrieval_status": retrieval_status,
            "documents_retrieved": sum(len(docs) for docs in evidence.values()),
        },
    }


def verify_offer_from_input(
    text: str = "",
    image_path: str = "",
    offer_id: Optional[str] = None,
    voice_analysis_override: Optional[Dict[str, Any]] = None,
    voice_agent_fn: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    retriever=None,
    ocr_reader=None,
    top_k: int = 3,
    explain: bool = True,
) -> Dict[str, Any]:
    """Runs Objectives 1 -> 2 -> 3 -> 4 -> 5 on a raw offer (text and/or image)."""
    from full_pipeline_orchestrator import run_full_pipeline

    payload = build_objective2_payload(
        text=text,
        image_path=image_path,
        offer_id=offer_id,
        retriever=retriever,
        ocr_reader=ocr_reader,
        top_k=top_k,
    )
    objective1 = payload.pop("objective1")

    result = run_full_pipeline(payload, voice_analysis_override, voice_agent_fn, explain=explain)
    result["objective1"] = {**objective1, "extracted_entities": payload["extracted_entities"]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a job offer from raw text and/or an image")
    parser.add_argument("--text", type=str, default="", help="Raw offer text")
    parser.add_argument("--image", type=str, default="", help="Path to an offer letter image")
    args = parser.parse_args()

    if not args.text and not args.image:
        parser.error("provide --text and/or --image")

    report = verify_offer_from_input(text=args.text, image_path=args.image)
    print(json.dumps(
        {
            "offer_id": report["offer_id"],
            "objective1": report["objective1"],
            "obj2_dst": report["obj2_dst"],
            "voice_verification": report["voice_verification"],
            "final_verification": report["final_verification"],
            "explanation": report.get("explanation"),
        },
        indent=2,
        default=str,
    ))
