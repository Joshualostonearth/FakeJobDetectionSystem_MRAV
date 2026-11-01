"""
Headless UI test of app.py with Streamlit's AppTest (no browser).

  - the app loads with the M4 model
  - pasted text -> Analyze -> verdict card + Explanation (Objective 5) card, no exceptions
  - an image result (richmond_offer.png through the bridge) renders
  - a mocked post-call result renders in the phone-verification section

AppTest cannot drive st.file_uploader, so the image case injects the bridge's result for
richmond_offer.png into the session; tests/test_e2e_image.py covers upload -> OCR -> M4.

    python -m pytest tests/test_app_ui.py -v
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mravplus_infer as m4  # noqa: E402

if m4.find_model_file(m4.BUNDLE_FILENAME) is None:
    pytest.skip("models/mravplus_bundle.joblib missing", allow_module_level=True)

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(PROJECT_ROOT / "app.py")
TIMEOUT = 600
SCAM = ("Position: Data Entry Clerk (Work From Home)\nCompany: Global Star Solutions Ltd\n"
        "Earn $750 per week with no experience required. Pay a $49 registration fee for your training kit. "
        "Send your bank details to hr.globalstar@gmail.com.")


@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens there
    os.chdir(PROJECT_ROOT)


def _text(at):
    return "\n".join(m.value for m in at.markdown)


def test_app_loads():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert "Model files missing" not in _text(at)


def test_text_analysis_renders_verdict_and_xai():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.text_area[0].input(SCAM)
    at.button[0].click().run()  # "Analyze Job Offer"
    assert not at.exception, at.exception
    page = _text(at)
    assert "M4 Detection Result" in page
    assert "FAKE" in page
    assert "Explanation (Objective 5)" in page
    assert "How the evidence sources were combined" in page
    assert at.session_state.analysis_result["xai"]["m1"]["identity_error"] < 1e-6


def test_image_result_and_mocked_call_render():
    import pipeline_bridge as pb
    from post_call_fusion import fuse_text_and_voice

    res = pb.handle_ui_submission(str(PROJECT_ROOT / "richmond_offer.png"), "")
    signals = {"company_reached": True, "role_actively_hiring": True, "recruiter_recognized": False,
               "company_denied_role": False, "suspicious_response": False, "source_confidence": 1.0,
               "extraction_confidence": 0.9, "response_summary": "Mocked call: HR confirmed the opening."}
    fused = fuse_text_and_voice(res["p_fake"], signals)
    fv, comb = fused["final_verification"], fused["dst_combined"]
    voice = {"offer_id": res["offer_id"], "voice_verification": {"voice_signals": signals}, "voice_error": None,
             "final_verdict": fv["final_verdict"], "final_belief_real": comb["REAL"], "final_belief_fake": comb["FAKE"],
             "final_belief_uncertain": comb["UNCERTAIN"], "calibrated_confidence": fv["calibrated_confidence"],
             "override_reason": fv["override_reason"], "validated": False}

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.session_state.analysis_result = res
    at.session_state.voice_result = voice
    at.session_state.auto_call_pending = False
    at.run()
    assert not at.exception, at.exception
    page = _text(at)
    assert "M4 Detection Result" in page
    assert "COMBINED VERDICT (UNVALIDATED)" in page
    assert "Mocked call: HR confirmed the opening." in page
    assert "Calibrated P(Real)" in page
    assert "Extracted OCR Text Preview" in [e.label for e in at.expander] or res["ocr_preview"] == ""
