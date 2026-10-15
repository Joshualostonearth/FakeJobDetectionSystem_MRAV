"""
Pipeline integration bridge connecting the Streamlit User Interface
to the backend fake job detection pipeline (OCR, Entity Extraction, FAISS, Credibility Engine).
"""

import os
import warnings

# Suppress internal developer warnings from transformers and PyTorch
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import transformers
    transformers.logging.set_verbosity_error()
except (ImportError, AttributeError):
    pass

import tempfile
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ocr_engine import init_ocr_reader, extract_text_from_image, merge_text_and_ocr
from models import extract_entities_from_text, JobOfferEntities
from search_engine import EvidenceRetriever
from scam_classifier import predict_scam_probability
from engines.credibility_engine import OLLAMA_MODEL
from full_pipeline_orchestrator import run_full_pipeline
from voice_verification_bridge import check_voice_services, verify_offer_by_phone

# run_full_pipeline final verdict -> risk level shown in the UI (and used for the call decision)
VERDICT_TO_RISK_LEVEL = {
    "HIGH_RISK_SCAM": "HIGH RISK",
    "SUSPICIOUS_FAKE": "HIGH RISK",
    "UNCERTAIN_NEEDS_VERIFICATION": "MEDIUM RISK",
    "VERIFIED_REAL": "LOW RISK",
}

# Text classifier P(scam) at which its score is listed among the warning flags
CLASSIFIER_FLAG_CRITICAL: float = 0.45
CLASSIFIER_FLAG_WARNING: float = 0.25

# Longest flag text shown in the UI (alignment flags carry full LLM explanations)
MAX_FLAG_CHARS: int = 300

# Risk levels that warrant phone verification
VOICE_VERIFICATION_RISK_LEVELS = ("HIGH RISK", "MEDIUM RISK")
# Risk levels that start the verification call without waiting for the user
AUTO_CALL_RISK_LEVELS = ("HIGH RISK",)


def check_ollama_service(host: str = "http://localhost:11434", timeout: float = 2.0) -> bool:
    """Checks if the local Ollama daemon is reachable."""
    try:
        url = f"{host.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "FakeJobDetection/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def validate_user_inputs(image_file: Any, user_text: str) -> None:
    """Validates that at least one valid input is supplied."""
    has_text = bool(user_text and user_text.strip())
    has_image = image_file is not None

    if not has_text and not has_image:
        raise ValueError(
            "Please provide an offer letter image, job description text, or both to analyze."
        )


def handle_ui_submission(
    image_file: Any,
    user_text: str,
    reader: Any = None,
    retriever: Optional[EvidenceRetriever] = None,
) -> Dict[str, Any]:
    """
    Handles processing of UI inputs (image, text, or both) through the multi-stage pipeline.

    :param image_file: Uploaded file object (Streamlit UploadedFile), filepath (str), or bytes
    :param user_text: Multiline raw text from job offer / email
    :param reader: Optional cached EasyOCR reader instance
    :param retriever: Optional cached EvidenceRetriever instance
    :return: Structured result dictionary for UI presentation
    """
    # 1. Validate inputs
    validate_user_inputs(image_file, user_text)

    # 2. Lazy-init or use provided cached reader/retriever
    if reader is None:
        reader = init_ocr_reader(lang_list=["en"], gpu=False)

    if retriever is None:
        retriever = EvidenceRetriever(index_folder="faiss_index")

    # Verify FAISS index presence
    if retriever.vector_db is None:
        raise RuntimeError(
            "FAISS vector database index could not be loaded from 'faiss_index/'. "
            "Please ensure 'build_vector_db.py' has been run to generate the local index."
        )

    # 3. Process image via OCR if supplied
    ocr_text = ""
    temp_path: Optional[str] = None

    try:
        if image_file is not None:
            if isinstance(image_file, str):
                if not os.path.exists(image_file):
                    raise FileNotFoundError(f"Provided image path does not exist: {image_file}")
                ocr_text = extract_text_from_image(reader, image_file)
            else:
                # Determine file extension safely
                ext = ".png"
                orig_name = getattr(image_file, "name", "")
                if orig_name:
                    _, guessed_ext = os.path.splitext(orig_name)
                    if guessed_ext.lower() in [".png", ".jpg", ".jpeg"]:
                        ext = guessed_ext.lower()

                # Write to temp file safely
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    if hasattr(image_file, "getvalue"):
                        tmp.write(image_file.getvalue())
                    elif hasattr(image_file, "read"):
                        data = image_file.read()
                        tmp.write(data)
                        if hasattr(image_file, "seek"):
                            image_file.seek(0)
                    elif isinstance(image_file, (bytes, bytearray)):
                        tmp.write(image_file)
                    else:
                        raise ValueError("Unsupported image file object format.")
                    temp_path = tmp.name

                ocr_text = extract_text_from_image(reader, temp_path)
                if ocr_text is None:
                    ocr_text = ""
    finally:
        # Ensure temporary file cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    ocr_preview = ocr_text[:300].strip() if ocr_text else ""

    # 4. Combine inputs
    merged_text = merge_text_and_ocr(user_text, ocr_text)
    if not merged_text.strip():
        raise ValueError(
            "Could not extract any readable text from the provided image, "
            "and no text input was entered. Please provide clear text or an image with legible text."
        )

    # 5. Check Ollama availability before calling extraction
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if not check_ollama_service(ollama_host):
        raise RuntimeError(
            f"Ollama service is unreachable at {ollama_host}. "
            "Please ensure the Ollama background daemon is running ('ollama serve') "
            f"and the '{OLLAMA_MODEL}' model has been downloaded ('ollama pull {OLLAMA_MODEL}')."
        )

    # 6. Entity Extraction (Objective 1a)
    try:
        entities: JobOfferEntities = extract_entities_from_text(merged_text)
    except Exception as e:
        raise RuntimeError(f"Entity extraction failed: {str(e)}")

    result = analyze_extracted_offer(merged_text, entities, retriever)
    result["ocr_preview"] = ocr_preview
    return result


def retrieve_offer_evidence(
    merged_text: str,
    entities: JobOfferEntities,
    retriever: EvidenceRetriever,
) -> List[Dict[str, Any]]:
    """
    Step 7-8 of handle_ui_submission: FAISS evidence retrieval, formatted as
    [{"page_content", "metadata", "distance"}], closest match first.
    """
    # The index stores full postings, so search with the full text
    try:
        bundle = retriever.retrieve_evidence_bundle(entities, top_k=3, query_text=merged_text)
        if bundle.get("status") == "FAILED":
            raise RuntimeError("FAISS similarity search returned FAILED status.")
    except Exception as e:
        raise RuntimeError(f"Evidence retrieval failed: {str(e)}")

    retrieved_evidence: List[Dict[str, Any]] = []
    bundle_evidence = bundle.get("retrieved_evidence", {})
    for category in ["legitimate_job_records", "scam_reports", "registry_records", "linkedin_data"]:
        for item in bundle_evidence.get(category, []):
            retrieved_evidence.append({
                "page_content": item.get("content", ""),
                "metadata": item.get("metadata", {}),
                "distance": float(item.get("similarity_score", 0.0)),
            })

    # Closest match first (lower L2 distance = more similar)
    retrieved_evidence.sort(key=lambda doc: doc["distance"])
    return retrieved_evidence


def build_objective2_payload(offer_id: str, entities: Dict[str, Any], retrieved_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map extracted entities and FAISS evidence onto the Objective2InputPayload schema."""
    from voice_verification_bridge import distance_to_similarity

    bundle = {"legitimate_job_records": [], "registry_records": [], "linkedin_data": [], "scam_reports": []}
    for ev in retrieved_evidence:
        meta = ev.get("metadata", {}) or {}
        source_type = meta.get("source_type", "")
        doc = {
            "doc_id": f"ROW{meta.get('row_id', '')}",
            "content": ev.get("page_content", ""),
            # VectorDocument.similarity_score is "higher = closer"; FAISS returns squared L2 distance
            "similarity_score": distance_to_similarity(float(ev.get("distance", 2.0))),
            "metadata": meta,
        }
        key = {"legitimate_job_record": "legitimate_job_records", "scam_report": "scam_reports",
               "linkedin_verified": "linkedin_data"}.get(source_type, "registry_records")
        bundle[key].append(doc)

    return {
        "offer_id": offer_id,
        "extracted_entities": {
            "company_name": entities.get("company_name") or "",
            "job_role": entities.get("job_role") or "",
            "recruiter_name": entities.get("recruiter_name"),
            "domain_or_website": entities.get("domain_or_website"),
            "email_or_phone": entities.get("recruiter_email"),
            "salary_or_stipend": None,
            "location": entities.get("location"),
        },
        "retrieved_evidence": bundle,
    }


def _ui_flags(scam_probability: Optional[float], full_result: Dict[str, Any]) -> List[str]:
    """Warning/critical flags for the UI: classifier score, Objective 2 flags and any override."""
    flags: List[str] = []
    if scam_probability is not None:
        if scam_probability >= CLASSIFIER_FLAG_CRITICAL:
            flags.append(f"CRITICAL: Text classifier rates the posting as likely fraudulent (P(scam)={scam_probability:.2f})")
        elif scam_probability >= CLASSIFIER_FLAG_WARNING:
            flags.append(f"WARNING: Text classifier finds scam-like wording (P(scam)={scam_probability:.2f})")

    for flag in full_result["obj2_credibility"]["credibility_evaluation"]["flags_triggered"]:
        if flag["severity"] in ("CRITICAL", "WARNING"):
            text = f"{flag['severity']}: {flag['message']}"
            flags.append(text if len(text) <= MAX_FLAG_CHARS else text[:MAX_FLAG_CHARS].rstrip() + "…")

    reason = full_result["final_verification"].get("override_reason")
    if reason:
        flags.append(reason)
    return list(dict.fromkeys(flags))


def _ui_explanation(scam_probability: Optional[float], full_result: Dict[str, Any]) -> str:
    """One-paragraph summary of how the final verdict was reached."""
    fv = full_result["final_verification"]
    dst = full_result["dst_combined_masses"]
    cred = full_result["obj2_credibility"]
    parts = [
        f"Final verdict: {fv['final_verdict'].replace('_', ' ')} "
        f"(calibrated confidence the offer is real: {fv['calibrated_confidence']:.0%}).",
    ]
    if scam_probability is not None:
        parts.append(f"Text classifier P(scam) = {scam_probability:.2f}.")
    parts.append(
        f"Evidence: {len(cred['evaluated_evidence'])} retrieved record(s) kept, "
        f"{len(cred['discarded_evidence'])} discarded; combined belief REAL {dst.get('REAL', 0.0):.2f}, "
        f"FAKE {dst.get('FAKE', 0.0):.2f}, uncertain {dst.get('UNCERTAIN', 1.0):.2f} "
        f"(conflict K = {dst.get('conflict', 0.0):.2f})."
    )
    if fv.get("override_reason"):
        parts.append(fv["override_reason"])
    return " ".join(parts)


def analyze_extracted_offer(
    merged_text: str,
    entities: JobOfferEntities,
    retriever: EvidenceRetriever,
) -> Dict[str, Any]:
    """
    Steps 7-10 of handle_ui_submission for already-extracted entities: evidence
    retrieval, text classifier, and the full Objective 2 -> 4 pipeline
    (run_full_pipeline: credibility filtration, DST, Platt calibration, override).
    """
    entities_dict = entities.model_dump()
    offer_id = f"UI_{uuid.uuid4().hex[:8]}"

    # 7-8. Evidence retrieval (Objective 1b)
    retrieved_evidence = retrieve_offer_evidence(merged_text, entities, retriever)

    # 8b. Supervised text classifier (Objective 1b); optional if the model has not been trained
    try:
        scam_probability = predict_scam_probability(merged_text)
    except FileNotFoundError as e:
        print(f"[Warning] {e}")
        scam_probability = None

    # 9. Objectives 2 -> 4 (no voice evidence yet; the call happens afterwards)
    try:
        payload = build_objective2_payload(offer_id, entities_dict, retrieved_evidence)
        full_result = run_full_pipeline(payload, None, scam_probability=scam_probability)
    except Exception as e:
        raise RuntimeError(f"Credibility evaluation failed: {str(e)}")

    fv = full_result["final_verification"]
    risk_level = VERDICT_TO_RISK_LEVEL.get(fv["final_verdict"], "MEDIUM RISK")

    # 10. Assemble complete UI payload
    return {
        "offer_id": offer_id,
        "trust_score": round(float(fv["calibrated_confidence"]) * 100.0, 1),
        "scam_probability": scam_probability,
        "risk_level": risk_level,
        "final_verdict": fv["final_verdict"],
        "voice_recommended": risk_level in VOICE_VERIFICATION_RISK_LEVELS,
        "auto_call": risk_level in AUTO_CALL_RISK_LEVELS,
        "flags": _ui_flags(scam_probability, full_result),
        "explanation": _ui_explanation(scam_probability, full_result),
        "extracted_entities": entities_dict,
        "retrieved_evidence": retrieved_evidence,
        "full_pipeline": full_result,
    }



def handle_phone_verification(
    analysis_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Runs Objective 3 voice verification for an offer already analysed by
    handle_ui_submission() and combines it with the FAISS evidence via DST.

    Blocks until the WebRTC call completes or times out.

    :param analysis_result: The dict returned by handle_ui_submission()
    :return: verify_offer_by_phone() result dict
    """
    service_error = check_voice_services()
    if service_error:
        raise RuntimeError(service_error)

    return verify_offer_by_phone(
        offer_id=analysis_result["offer_id"],
        entities=analysis_result["extracted_entities"],
        retrieved_evidence=analysis_result["retrieved_evidence"],
        suspicion_reasons=analysis_result.get("flags", []),
        scam_probability=analysis_result.get("scam_probability"),
    )
