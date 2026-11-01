"""
Tests for the M4 pipeline bridge, the regex entity extractor and the post-call fusion.
No Ollama and no voice services are needed (the call is mocked).

    python -m pytest tests/test_m4_bridge.py -v
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mravplus_infer as m4  # noqa: E402
from entity_extractor import extract_entities, has_job_offer, strip_source_tags  # noqa: E402
from post_call_fusion import fuse_text_and_voice  # noqa: E402

HAVE_MODELS = m4.find_model_file(m4.BUNDLE_FILENAME) is not None

SCAM_TEXT = """[SOURCE: RAW_TEXT]
Position: Data Entry Clerk (Work From Home)
Company: Global Star Solutions Ltd
Earn $750 per week with no experience required. To start immediately, pay a $49 registration fee
for your training kit. Send your bank account details to hr.globalstar@gmail.com or WhatsApp
+1 (555) 201-7788. Regards,
Daniel Brooks"""

CONFIRMING = {"company_reached": True, "role_actively_hiring": True, "recruiter_recognized": True,
              "company_denied_role": False, "suspicious_response": False, "source_confidence": 1.0,
              "extraction_confidence": 0.9, "response_summary": "HR confirmed the role."}
DENIED = {**CONFIRMING, "role_actively_hiring": False, "company_denied_role": True,
          "response_summary": "HR said no such role exists."}


# ── entity extraction (regex) ─────────────────────────────

def test_regex_entities():
    e = extract_entities(SCAM_TEXT)
    assert e["company_name"] == "Global Star Solutions Ltd"
    assert e["job_role"].startswith("Data Entry Clerk")
    assert e["recruiter_email"] == "hr.globalstar@gmail.com"
    assert e["recruiter_phone"] and "555" in e["recruiter_phone"]
    assert e["domain_or_website"] is None          # gmail is free mail, not a company domain
    assert e["recruiter_name"] == "Daniel Brooks"
    assert "[SOURCE" not in strip_source_tags(SCAM_TEXT)


def test_fields_override_and_no_offer():
    e = extract_entities("Security And Hacking Conference 2026", {"title": "Event volunteer"})
    assert e["job_role"] == "Event volunteer"
    assert not has_job_offer(extract_entities("Security And Hacking Conference 2026"))


def test_richmond_style_letter():
    text = ("RICHMOND COLLEGE\nDear Applicant, we are pleased to offer you the position of Administrative Officer. "
            "Contact hr@richmond.edu. Website: www.richmond.edu")
    e = extract_entities(text)
    assert e["company_name"] == "RICHMOND COLLEGE"
    assert e["job_role"] == "Administrative Officer"
    assert e["domain_or_website"] == "www.richmond.edu"


# ── post-call fusion (NOT validated; tests check the arithmetic and overrides) ──

def test_fusion_without_voice_is_text_only():
    out = fuse_text_and_voice(0.2, None)
    m = out["dst_combined"]
    assert abs(m["REAL"] - 0.9 * 0.8) < 1e-9 and abs(m["FAKE"] - 0.9 * 0.2) < 1e-9
    assert abs(out["final_verification"]["calibrated_confidence"] - round(1 - (0.18 + 0.1 / 2), 4)) < 1e-9
    assert out["validated"] is False


def test_fusion_confirming_call_raises_confidence():
    base = fuse_text_and_voice(0.4, None)["final_verification"]["calibrated_confidence"]
    after = fuse_text_and_voice(0.4, CONFIRMING)["final_verification"]["calibrated_confidence"]
    assert after > base


def test_fusion_denied_role_override():
    fv = fuse_text_and_voice(0.01, DENIED)["final_verification"]
    assert fv["final_verdict"] == "HIGH_RISK_SCAM"
    assert fv["override_applied"] and "denied" in fv["override_reason"]


def test_fusion_suspicious_response_override():
    fv = fuse_text_and_voice(0.05, {**CONFIRMING, "suspicious_response": True})["final_verification"]
    assert fv["calibrated_confidence"] <= 0.15
    assert fv["final_verdict"] == "HIGH_RISK_SCAM"


def test_phone_verification_with_mocked_call(monkeypatch):
    import pipeline_bridge
    import voice_verification_bridge

    def fake_voice_step(result, masses, entities, voice_config, suspicion_reasons=None, offer_text="", on_progress=None):
        result["voice_verification"] = {"pipeline_completed": True, "voice_signals": CONFIRMING}

    monkeypatch.setattr(pipeline_bridge, "check_voice_services", lambda: None)
    monkeypatch.setattr(pipeline_bridge, "check_ollama_service", lambda *a, **k: False)
    monkeypatch.setattr(voice_verification_bridge, "_run_voice_step", fake_voice_step)
    res = pipeline_bridge.handle_phone_verification(
        {"offer_id": "UI_t", "p_fake": 0.3, "extracted_entities": extract_entities(SCAM_TEXT),
         "flags": ["F1"], "offer_text": SCAM_TEXT})
    for key in ("final_verdict", "final_belief_real", "final_belief_fake", "final_belief_uncertain",
                "calibrated_confidence", "override_reason", "voice_verification"):
        assert key in res
    assert res["voice_verification"]["voice_signals"]["response_summary"] == "HR confirmed the role."
    assert res["validated"] is False
    assert res["final_verdict"] in pipeline_bridge.VERDICT_TO_RISK_LEVEL


# ── M4 verdict mapping and the full text path ─────────────

def test_verdict_mapping():
    from pipeline_bridge import m4_verdict
    assert m4_verdict({"zone": "HIGH", "p_fake": 0.9}) == {"verdict": "FAKE", "risk_level": "HIGH RISK"}
    assert m4_verdict({"zone": "REVIEW", "p_fake": 0.6}) == {"verdict": "FAKE", "risk_level": "HIGH RISK"}
    assert m4_verdict({"zone": "REVIEW", "p_fake": 0.3}) == {"verdict": "NOT FAKE", "risk_level": "MEDIUM RISK"}
    assert m4_verdict({"zone": "LOW", "p_fake": 0.01}) == {"verdict": "NOT FAKE", "risk_level": "LOW RISK"}


@pytest.mark.skipif(not HAVE_MODELS, reason="models/mravplus_bundle.joblib missing")
def test_text_submission_end_to_end():
    import pipeline_bridge
    res = pipeline_bridge.handle_ui_submission(None, SCAM_TEXT)
    assert res["final_verdict"] in ("FAKE", "NOT FAKE")
    assert 0 <= res["trust_score"] <= 100 and abs(res["trust_score"] - res["p_real"] * 100) < 0.051
    assert res["xai"]["m1"]["identity_error"] < 1e-6
    assert len(res["neighbours"]) == 5
    assert "[SOURCE" not in res["scoring_text"]
    assert res["auto_call"] == (res["risk_level"] == "HIGH RISK")


@pytest.mark.skipif(not HAVE_MODELS, reason="models/mravplus_bundle.joblib missing")
def test_no_job_offer_is_rejected():
    import pipeline_bridge
    with pytest.raises(ValueError, match="No job offer"):
        pipeline_bridge.handle_ui_submission(None, "Security And Hacking Conference 2026")


@pytest.mark.skipif(not HAVE_MODELS, reason="models/mravplus_bundle.joblib missing")
def test_bridge_imports_no_llm_or_faiss_modules():
    code = (
        "import sys, json; sys.path.insert(0, '.');"
        "import pipeline_bridge as pb;"
        f"pb.handle_ui_submission(None, {SCAM_TEXT!r});"
        "banned = ('langchain', 'langchain_core', 'langchain_ollama', 'ollama', 'faiss', 'engines',"
        " 'credibility_engine', 'search_engine', 'scam_classifier', 'models');"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules} & set(banned))))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_title_line_fallback_for_posters():
    # all-caps or unlisted role words: the poster's first line is taken as the role when the text is job-like
    for title in ("ADMINISTRATIVE & OFFICE ASSISTANT", "Journeyman Electrician", "Appointment setter", "Mensajero"):
        e = extract_entities(f"{title}\nLocation: US, TX, Houston\nJob description: we are hiring. Requirements: none.")
        assert e["job_role"] == title, e
    # an event poster has no job vocabulary and stays without a role
    assert extract_entities("Security And Hacking Conference 2026\nKeynotes and workshops")["job_role"] is None


def test_signoff_title_is_not_a_name():
    text = ("Richmond College\nDear Ms Waxler, we promote your job status. Salary raise of $2,000 per month.\n"
            "Please let the HR Department know at hr@richmond.edu:\nSincerely\nAdministrative Officer\nRichmond College")
    e = extract_entities(text)
    assert e["company_name"] == "Richmond College"
    assert e["job_role"] == "Administrative Officer"
    assert e["recruiter_name"] is None


def test_ocr_offer_letter_entities():
    # EasyOCR output of a WhatsApp-compressed GAO Tek internship letter (logo split over two
    # lines, "Sincerely ." with a stray dot, blank "Signed by:" followed by the candidate's name)
    from ocr_engine import sanitize_ocr_text
    text = sanitize_ocr_text(
        "Gao\nTek Inc.\nAMember of GAO Group\nInternship Offer_Letter\nDear Asim ,\n"
        "We are pleased to offer you an intemship at GAO Tek Inc Your internship is scheduled\n"
        "Sincerely .\nFlora Abraham\nHR Manager\nlu @gaotek com\nGAO Tek Inc\nwww GAOTek com\n"
        "agree with the terms and condition outlined above and in the agreement:\n"
        "Signed by:\nAiAlud\nPosition applied: Tech Support")
    e = extract_entities(text)
    assert e["company_name"] == "GAO Tek Inc"
    assert e["recruiter_name"] == "Flora Abraham"
    assert e["recruiter_email"] == "lu@gaotek.com"
    assert e["domain_or_website"] == "www.GAOTek.com"
    assert e["job_role"] == "Tech Support"


def test_label_value_stays_on_its_line():
    e = extract_entities("Offer from Acme Ltd\nSigned by:\nJane Candidate\nRecruiter: Tom Lee")
    assert e["recruiter_name"] == "Tom Lee"


def test_red_flags_rules():
    from red_flags import find_red_flags
    fee = find_red_flags('To "CONFIRM" your enrolment, you need to pay 1594/- the Internship fees online')
    assert [f["id"] for f in fee] == ["upfront_fee"]
    assert find_red_flags("There is no registration fee. Pay is $15/hr. We offer gift cards as perks.") == []
    chat = find_red_flags("Please contact our manager on WhatsApp to start today.")
    assert [f["id"] for f in chat] == ["off_platform_chat"]
    mail = find_red_flags("Department of Computer Science, UCLA", {"recruiter_email": "dckstr9@gmail.com"})
    assert [f["id"] for f in mail] == ["free_mail_institution"]


def test_red_flags_raise_risk_one_step():
    import pipeline_bridge as pb
    text = ("Software Engineer II - Backend (Python). Acme Analytics, Austin, TX. Full-time.\n"
            "We are looking for a backend engineer to build and maintain data pipelines and REST services in\n"
            "Python and PostgreSQL. Requirements: 3+ years of professional experience, familiarity with AWS,\n"
            "code review and testing practices. Benefits: health, dental and vision insurance, 401(k) match,\n"
            "20 days PTO. Apply through our careers page at careers.acmeanalytics.com; interviews are held\n"
            "with the engineering team over two rounds.")
    clean = pb.handle_ui_submission(None, text, with_xai=False)
    assert clean["risk_level"] == clean["m4_risk_level"] == "LOW RISK" and clean["red_flags"] == []
    flagged = pb.handle_ui_submission(None, text + " Contact our HR on WhatsApp to proceed.", with_xai=False)
    assert flagged["m4_risk_level"] == "LOW RISK"
    assert flagged["risk_level"] == "MEDIUM RISK" and flagged["voice_recommended"]
    assert flagged["final_verdict"] == "NOT FAKE"  # verdict and P(fraud) are M4's
    assert any(f.startswith("RED FLAG:") for f in flagged["flags"])
    # a borderline (MEDIUM) offer with a red flag becomes HIGH RISK and starts the call; verdict unchanged
    # EasyOCR text of a real pay-to-join internship letter (Annexure A, Rs 1,594 fee)
    letter = ("Annexure\nA\nstructures can be bifurcated as under:\nBasic\nRs:\n22,500/- Per Month\n"
              "Performance Incentives\nRs:\nAs per work performance\nInternet Allowances\nRs:\ns00/- Per Month_\n"
              "Gross Salary\nRs:\n23,000/- Per Month:\nTotal In-hand\nRs. 23,000/- Per Month\n"
              "You will be assigned a client and project upon onboarding: A stipend will be provided\n"
              "upon successful completion of the assigned project\n"
              "To \"CONFIRM\" your enrolment; YOu need to pay 1594/- the Internship fees\n"
              "online through the following Jink Internship Eorm\nthrough the payment method shared:\n"
              "Upon successful payment confirmation, a confirmation email will be sent\n"
              "immediately Login credentials and complete onboarding details, including\n"
              "the platform link, internship session access, and mentor information, will\n"
              "be shared by End of\nFlexible\nJoining Date: You may choose your joining date as per your\n"
              "convenience\nSalary\nPay\nsalary\nDay:")
    r = pb.handle_ui_submission(None, letter, with_xai=False)
    assert r["m4_risk_level"] == "MEDIUM RISK", r["m4_risk_level"]
    assert r["risk_level"] == "HIGH RISK" and r["auto_call"]
    assert r["final_verdict"] == "NOT FAKE"
