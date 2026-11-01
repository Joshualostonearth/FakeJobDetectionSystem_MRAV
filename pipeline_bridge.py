"""
Pipeline bridge between the Streamlit UI and the M4 detector (no LLM in the classifier).

    image / text -> OCR (EasyOCR) -> strip [SOURCE: ...] tags -> regex entities
                 -> M4Detector.predict -> XAI (Objective 5) -> result dict

Image -> verdict needs neither Ollama nor the voice services. The old LLM/FAISS/DST
bridge is in legacy/pipeline_bridge_llm.py.

Verdict mapping (M4 zone -> UI):
    HIGH, or REVIEW with p_fake >= 0.5  -> FAKE,     HIGH RISK
    REVIEW with p_fake < 0.5            -> NOT FAKE, MEDIUM RISK (borderline, resolved by DistilBERT)
    LOW                                 -> NOT FAKE, LOW RISK
    trust score = p_real * 100 ("Calibrated confidence: Real")
"""

import os
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning)

import json
import tempfile
import urllib.request
import uuid
from typing import Any, Callable, Dict, List, Optional

from entity_extractor import enrich_with_llm, extract_entities, has_job_offer, strip_source_tags

# Post-call verdict (Objective 4 bands, reused for the voice fusion) -> risk level shown in the UI
VERDICT_TO_RISK_LEVEL = {
    "HIGH_RISK_SCAM": "HIGH RISK",
    "SUSPICIOUS_FAKE": "HIGH RISK",
    "UNCERTAIN_NEEDS_VERIFICATION": "MEDIUM RISK",
    "VERIFIED_REAL": "LOW RISK",
}

# Risk levels that warrant phone verification
VOICE_VERIFICATION_RISK_LEVELS = ("HIGH RISK", "MEDIUM RISK")
# Risk levels that start the verification call without waiting for the user
AUTO_CALL_RISK_LEVELS = ("HIGH RISK",)

M4_LIMITATION = (
    "M4 was trained and tested on structured EMSCAD job postings. Poster / OCR text is unstructured, "
    "so the validated test metrics apply to EMSCAD-style input, not to arbitrary images."
)

_detector = None


def get_detector():
    """Process-wide M4Detector (the bundle loads once; MiniLM / DistilBERT load lazily)."""
    global _detector
    if _detector is None:
        from mravplus_infer import M4Detector
        _detector = M4Detector()
    return _detector


def m4_verdict(m4: Dict[str, Any]) -> Dict[str, str]:
    zone, p_fake = m4["zone"], m4["p_fake"]
    if zone == "HIGH" or (zone == "REVIEW" and p_fake >= 0.5):
        return {"verdict": "FAKE", "risk_level": "HIGH RISK"}
    if zone == "REVIEW":
        return {"verdict": "NOT FAKE", "risk_level": "MEDIUM RISK"}
    return {"verdict": "NOT FAKE", "risk_level": "LOW RISK"}


def check_ollama_service(host: Optional[str] = None, timeout: float = 2.0) -> bool:
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", headers={"User-Agent": "FakeJobDetection/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def check_voice_services() -> Optional[str]:
    """None if the call bridge is reachable, else a message saying what to start."""
    from voice_verification_bridge import check_voice_services as _check
    return _check()


def validate_user_inputs(image_file: Any, user_text: str) -> None:
    if not (user_text and user_text.strip()) and image_file is None:
        raise ValueError("Please provide an offer letter image, job description text, or both to analyze.")


def _ocr_image(image_file: Any, reader: Any) -> str:
    from ocr_engine import extract_text_from_image, init_ocr_reader

    if reader is None:
        reader = init_ocr_reader(lang_list=["en"], gpu=False)
    if reader is None:
        raise RuntimeError("EasyOCR could not be initialised. Run 'python scripts/download_models.py' "
                           "once with internet access, or paste the offer text instead.")
    if isinstance(image_file, (str, os.PathLike)):
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Provided image path does not exist: {image_file}")
        return extract_text_from_image(reader, str(image_file)) or ""

    ext = os.path.splitext(getattr(image_file, "name", "") or "")[1].lower()
    ext = ext if ext in (".png", ".jpg", ".jpeg") else ".png"
    if hasattr(image_file, "getvalue"):
        data = image_file.getvalue()
    elif hasattr(image_file, "read"):
        data = image_file.read()
        if hasattr(image_file, "seek"):
            image_file.seek(0)
    elif isinstance(image_file, (bytes, bytearray)):
        data = bytes(image_file)
    else:
        raise ValueError("Unsupported image file object format.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(data)
        tmp.close()
        return extract_text_from_image(reader, tmp.name) or ""
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def _flags(m4: Dict[str, Any], verdict: Dict[str, str]) -> List[str]:
    flags: List[str] = []
    if m4["zone"] == "HIGH":
        flags.append(f"CRITICAL: M4 rates the posting as fraudulent (calibrated P(fraud) = {m4['p_fake']:.2f}).")
    elif m4["zone"] == "REVIEW":
        flags.append(
            f"WARNING: Borderline posting (cheap P(fraud) = {m4['p_cheap']:.2f}); DistilBERT was consulted and "
            f"the final P(fraud) is {m4['p_fake']:.2f}. Independent verification is recommended."
        )
    share = m4["neighbour_fraud_share"]
    if share >= 0.6:
        flags.append(f"WARNING: {round(share * len(m4['neighbours']))} of the {len(m4['neighbours'])} most "
                     "similar known postings were fraudulent.")
    if m4["conflict"] >= 0.3:
        flags.append(f"WARNING: The evidence sources disagree (DST conflict K = {m4['conflict']:.2f}).")
    flags += [f"NOTE: {w}" for w in m4["warnings"]]
    return flags


def handle_ui_submission(
    image_file: Any,
    user_text: str,
    reader: Any = None,
    detector: Any = None,
    fields: Optional[Dict[str, Any]] = None,
    with_xai: bool = True,
) -> Dict[str, Any]:
    """
    Analyse an uploaded image and/or pasted text with M4.

    :param image_file: Streamlit UploadedFile, file path or bytes (optional)
    :param user_text: pasted offer text (optional)
    :param reader: cached EasyOCR reader (created on demand if None)
    :param detector: cached M4Detector (process-wide one if None)
    :param fields: optional structured fields (title, location, employment_type, has_company_logo, ...)
    :param with_xai: attach the Objective 5 explanation under result["xai"]
    """
    from ocr_engine import merge_text_and_ocr

    validate_user_inputs(image_file, user_text)

    ocr_text = _ocr_image(image_file, reader) if image_file is not None else ""
    if image_file is not None and not ocr_text.strip() and not (user_text and user_text.strip()):
        raise ValueError("Could not read any text from the image. Upload a sharper, well-lit image "
                         "or paste the offer text.")

    merged_text = merge_text_and_ocr(user_text, ocr_text)
    scoring_text = strip_source_tags(merged_text)
    if not scoring_text.strip():
        raise ValueError("No readable text was found in the input. Please provide clear text or an image "
                         "with legible text.")

    entities = extract_entities(scoring_text, fields)
    if not has_job_offer(entities):
        # Ported from integrated2 (a3c42c3): an event poster or random photo has no offer to verify
        raise ValueError(
            "No job offer was found in the input: neither a company name nor a job role could be "
            "identified, so there is nothing to verify. Upload the offer letter or job posting, "
            "or paste its text."
        )

    detector = detector or get_detector()
    m4 = detector.predict(scoring_text, fields)
    verdict = m4_verdict(m4)

    xai = None
    if with_xai:
        from xai import explain
        xai = explain(detector, scoring_text, fields, m4)

    m4_public = {k: v for k, v in m4.items() if not k.startswith("_")}
    return {
        "offer_id": f"UI_{uuid.uuid4().hex[:8]}",
        "model": "M4 (MRAV+)",
        "final_verdict": verdict["verdict"],
        "risk_level": verdict["risk_level"],
        "trust_score": round(m4["p_real"] * 100.0, 1),
        "p_fake": m4["p_fake"],
        "p_real": m4["p_real"],
        "scam_probability": m4["p_fake"],
        "zone": m4["zone"],
        "voice_recommended": verdict["risk_level"] in VOICE_VERIFICATION_RISK_LEVELS,
        "auto_call": verdict["risk_level"] in AUTO_CALL_RISK_LEVELS,
        "flags": _flags(m4, verdict),
        "explanation": xai["summary"]["text"] if xai else "",
        "extracted_entities": entities,
        "fields": dict(fields or {}),
        "offer_text": merged_text,
        "scoring_text": scoring_text,
        "ocr_text": ocr_text,
        "ocr_preview": ocr_text[:300].strip(),
        "neighbours": m4["neighbours"],
        "m4": m4_public,
        "xai": xai,
        "limitations": M4_LIMITATION,
    }


def handle_phone_verification(
    analysis_result: Dict[str, Any],
    on_progress: Optional[Callable[[str, int], None]] = None,
    voice_config: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Run the Objective 3 call for an analysed offer, then fuse M4's p_fake with the call's
    voice mass (post_call_fusion; NOT validated). Blocks until the call completes or times out.
    """
    from post_call_fusion import fuse_text_and_voice
    from voice_verification_bridge import _run_voice_step

    service_error = check_voice_services()
    if service_error:
        raise RuntimeError(service_error)

    entities = dict(analysis_result.get("extracted_entities") or {})
    offer_text = analysis_result.get("offer_text", "")
    llm_note = None
    needed = ("company_name", "job_role", "recruiter_name", "recruiter_email", "domain_or_website")
    if any(not entities.get(k) for k in needed) and check_ollama_service():
        enriched = enrich_with_llm(entities, offer_text)
        entities = enriched["entities"]
        llm_note = enriched["error"] or (f"LLM filled: {', '.join(enriched['filled'])}" if enriched["filled"] else None)

    result: Dict[str, Any] = {
        "offer_id": analysis_result["offer_id"],
        "voice_verification": None,
        "voice_mass": None,
        "voice_error": None,
        "entities_used": entities,
        "entity_llm_note": llm_note,
    }
    _run_voice_step(result, [], entities, voice_config, analysis_result.get("flags", []), offer_text, on_progress)
    signals = (result["voice_verification"] or {}).get("voice_signals") or None

    fused = fuse_text_and_voice(analysis_result.get("p_fake"), signals)
    fv, combined = fused["final_verification"], fused["dst_combined"]
    result.update({
        "dst_combined": combined,
        "source_masses": fused["source_masses"],
        "final_belief_real": round(combined["REAL"], 4),
        "final_belief_fake": round(combined["FAKE"], 4),
        "final_belief_uncertain": round(combined["UNCERTAIN"], 4),
        "final_verdict": fv["final_verdict"],
        "calibrated_confidence": fv["calibrated_confidence"],
        "override_reason": fv["override_reason"],
        "final_verification": fv,
        "validated": False,
        "validation_note": fused["note"],
    })
    if fused["voice_mass"] is not None:
        result["voice_mass"] = fused["voice_mass"]
    return result


if __name__ == "__main__":  # quick manual check: python pipeline_bridge.py <image-or-text-file>
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "richmond_offer.png"
    if arg.lower().endswith((".png", ".jpg", ".jpeg")):
        out = handle_ui_submission(arg, "")
    else:
        out = handle_ui_submission(None, open(arg, encoding="utf-8").read())
    print(json.dumps({k: out[k] for k in ("final_verdict", "risk_level", "trust_score", "zone", "flags",
                                          "extracted_entities", "explanation")}, indent=2))
