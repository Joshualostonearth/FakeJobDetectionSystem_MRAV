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
from engines.credibility_engine import Run_Objective_2_Filter, OLLAMA_MODEL
from voice_verification_bridge import check_voice_services, verify_offer_by_phone

# Risk levels from Run_Objective_2_Filter that warrant phone verification
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


def analyze_extracted_offer(
    merged_text: str,
    entities: JobOfferEntities,
    retriever: EvidenceRetriever,
) -> Dict[str, Any]:
    """
    Steps 7-10 of handle_ui_submission: evidence retrieval, text classifier,
    Objective 2 filter and the UI payload, for already-extracted entities.
    """
    entities_dict = entities.model_dump()

    # 7. Evidence Retrieval (Objective 1b) — the index stores full postings, so search with the full text
    try:
        bundle = retriever.retrieve_evidence_bundle(entities, top_k=3, query_text=merged_text)
        if bundle.get("status") == "FAILED":
            raise RuntimeError("FAISS similarity search returned FAILED status.")
    except Exception as e:
        raise RuntimeError(f"Evidence retrieval failed: {str(e)}")

    # 8. Format retrieved evidence for Objective 2 consumption
    retrieved_evidence: List[Dict[str, Any]] = []
    bundle_evidence = bundle.get("retrieved_evidence", {})

    for category in ["legitimate_job_records", "scam_reports", "registry_records", "linkedin_data"]:
        docs = bundle_evidence.get(category, [])
        for item in docs:
            retrieved_evidence.append({
                "page_content": item.get("content", ""),
                "metadata": item.get("metadata", {}),
                "distance": float(item.get("similarity_score", 0.0)),
            })

    # Closest match first (lower L2 distance = more similar)
    retrieved_evidence.sort(key=lambda doc: doc["distance"])

    # 8b. Supervised text classifier (Objective 1b); optional if the model has not been trained
    try:
        scam_probability = predict_scam_probability(merged_text)
    except FileNotFoundError as e:
        print(f"[Warning] {e}")
        scam_probability = None

    # 9. Credibility Evaluation (Objective 2)
    try:
        results_obj2 = Run_Objective_2_Filter(entities_dict, retrieved_evidence, scam_probability=scam_probability)
    except Exception as e:
        raise RuntimeError(f"Credibility evaluation failed: {str(e)}")

    risk_level = str(results_obj2.get("risk_level", "UNKNOWN"))

    # 10. Assemble complete UI payload
    return {
        "offer_id": f"UI_{uuid.uuid4().hex[:8]}",
        "trust_score": float(results_obj2.get("final_trust_score", 0.0)),
        "scam_probability": scam_probability,
        "risk_level": risk_level,
        "voice_recommended": risk_level in VOICE_VERIFICATION_RISK_LEVELS,
        "auto_call": risk_level in AUTO_CALL_RISK_LEVELS,
        "flags": list(results_obj2.get("flags_raised", [])),
        "explanation": str(results_obj2.get("mrav_explanation", "No explanation available.")),
        "extracted_entities": entities_dict,
        "retrieved_evidence": retrieved_evidence,
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
