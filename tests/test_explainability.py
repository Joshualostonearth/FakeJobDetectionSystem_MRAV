"""Objective 5: explainability.py explains the pipeline's verdict from its own numbers."""

import re
from types import SimpleNamespace
from unittest.mock import patch

import explainability
import pipeline_bridge
from confidence_calibrator import calculate_confidence_score
from explainability import (
    explain_phone_verification, explain_verdict, is_grounded, overclaims, summarize_with_llm,
)
from models import JobOfferEntities
from obj2dst import combine_all
from tests.test_app_full_pipeline import NEUTRAL, _StubRetriever


def _analyze(prob):
    with patch("credibility_engine.generate_alignment_triple", return_value=NEUTRAL), \
         patch.object(pipeline_bridge, "predict_scam_probability", return_value=prob):
        return pipeline_bridge.analyze_extracted_offer(
            "Driver job in Houston", JobOfferEntities(company_name="Acme", job_role="Driver"), _StubRetriever())


def test_ui_result_carries_the_explanation():
    res = _analyze(0.99)
    vexp = res["verdict_explanation"]
    assert res["explanation"] == vexp["summary"]
    assert len(vexp["factors"]) == len(res["full_pipeline"]["evidence_masses"])


def test_scam_like_text_is_attributed_to_the_classifier():
    vexp = _analyze(0.99)["verdict_explanation"]
    classifier = next(f for f in vexp["factors"] if f["source"] == "text_classifier")
    assert classifier["effect"] < 0
    assert vexp["factors"][0] is classifier
    assert "Pointing to a scam: Text classifier" in vexp["summary"]


def test_record_factor_names_the_retrieved_record():
    vexp = _analyze(0.01)["verdict_explanation"]
    record = next(f for f in vexp["factors"] if f["source"] == "legitimate_job_record")
    # The stub record has no title, so only the source type is shown
    assert record["label"] == "Legitimate job record"
    assert record["detail"].startswith("Credibility")


def test_effect_is_leave_one_out_change_in_confidence():
    masses = [{"REAL": 0.6, "FAKE": 0.0, "UNCERTAIN": 0.4, "source": "legitimate_job_record", "doc_index": 0},
              {"REAL": 0.09, "FAKE": 0.81, "UNCERTAIN": 0.1, "source": "text_classifier"}]
    combined = combine_all(masses)
    full_result = {
        "obj2_credibility": {
            "evaluated_evidence": [{"source": "ROW1", "credibility_index": 0.7, "credibility_tier": "trusted",
                                    "mrav_triple": {"hypothesis": "Neutral", "explanation": "x"}}],
            "credibility_evaluation": {"flags_triggered": []},
        },
        "evidence_masses": masses,
        "dst_combined_masses": combined,
        "final_verification": {"final_verdict": "SUSPICIOUS_FAKE", "override_applied": False,
                               "calibrated_confidence": calculate_confidence_score(combined)},
    }
    vexp = explain_verdict(full_result, scam_probability=0.9)
    base = calculate_confidence_score(combined)
    by_source = {f["source"]: f for f in vexp["factors"]}
    assert by_source["text_classifier"]["effect"] == round(base - calculate_confidence_score(combine_all(masses[:1])), 4)
    assert by_source["legitimate_job_record"]["effect"] == round(base - calculate_confidence_score(combine_all(masses[1:])), 4)


def test_grounding_check_rejects_invented_numbers():
    facts = ["Final verdict: Verified Real, with calibrated confidence that the offer is real of 0.82."]
    assert is_grounded("The offer looks real, with 82% confidence.", facts)
    assert is_grounded("The offer looks real (confidence 0.82).", facts)
    assert not is_grounded("The offer looks real, with 95% confidence.", facts)
    # 99.7% must not pass as a rounded 1%
    assert not is_grounded("This is a scam with 99.7% certainty.", ["Confidence that the offer is real: 1%."])


def test_llm_summary_falls_back_when_ungrounded_or_unavailable():
    vexp = _analyze(0.99)["verdict_explanation"]

    class FakeLLM:
        def __init__(self, text=None, **kwargs):
            self.text = text

        def invoke(self, prompt):
            if self.text is None:
                raise ConnectionError("ollama down")
            return SimpleNamespace(content=self.text)

    with patch("langchain_ollama.ChatOllama", lambda **kw: FakeLLM(None)):
        assert summarize_with_llm(vexp) is None
    with patch("langchain_ollama.ChatOllama", lambda **kw: FakeLLM("This is a scam with 99.7% certainty.")):
        assert summarize_with_llm(vexp) is None
    with patch("langchain_ollama.ChatOllama", lambda **kw: FakeLLM("The wording of the offer points to a scam.")):
        assert summarize_with_llm(vexp) == "The wording of the offer points to a scam."


def test_phone_explanation_lists_call_signals():
    vres = {
        "voice_mass": {"REAL": 0.0, "FAKE": 0.5, "UNCERTAIN": 0.5, "combined_confidence": 0.8,
                       "details": {"company_reached": 0.25, "company_denied_role": -0.4, "suspicious_response": 0.0}},
        "final_belief_real": 0.1, "final_belief_fake": 0.7, "final_belief_uncertain": 0.2,
    }
    pexp = explain_phone_verification(vres)
    assert [s["label"] for s in pexp["signals"]] == [
        explainability.VOICE_SIGNAL_LABELS["company_denied_role"],
        explainability.VOICE_SIGNAL_LABELS["company_reached"],
    ]
    assert any("belief fake 0.70" in f for f in pexp["facts"])
    assert explain_phone_verification({})["facts"] == [
        "The call produced no usable signals, so the verdict uses database evidence only."]


def test_llm_facts_carry_only_the_confidence_number():
    vexp = _analyze(0.99)["verdict_explanation"]
    numbers = {n for fact in vexp["llm_facts"] for n in re.findall(r"\d+(?:\.\d+)?", fact)}
    assert numbers == {f"{vexp['confidence'] * 100:.0f}"}
    assert any("Text classifier strongly points to a scam" in f for f in vexp["llm_facts"])


def test_flags_for_the_llm_drop_mrav_reasoning_and_doc_ids():
    flag = ("WARNING: Offer contradicts EMSCAD legitimate baseline: Claimed Role: *Warehouse Associate*\n"
            "   - Location: Houston vs Hong Kong ... (doc: ROW17808)")
    assert explainability._short_flag(flag) == "WARNING: Offer contradicts EMSCAD legitimate baseline"
    assert explainability._short_flag("WARNING: Similar known fraudulent listing (doc: ROW5507)") == \
        "WARNING: Similar known fraudulent listing"
    assert len(explainability._short_flag("WARNING: " + "word " * 100)) <= explainability.MAX_LLM_FLAG_CHARS + 1


def test_llm_summary_markdown_is_stripped():
    vexp = _analyze(0.99)["verdict_explanation"]
    reply = SimpleNamespace(content="The offer is a **high-risk scam**.")
    with patch("langchain_ollama.ChatOllama", lambda **kw: SimpleNamespace(invoke=lambda prompt: reply)):
        assert summarize_with_llm(vexp) == "The offer is a high-risk scam."


def test_llm_facts_name_the_main_reason_first():
    scam = _analyze(0.99)["verdict_explanation"]["llm_facts"]
    assert scam[1] == "Main reason: Text classifier strongly points to a scam."
    assert all(not f.startswith("Main reason") for f in scam[2:])

    masses = [{"REAL": 0.9, "FAKE": 0.0, "UNCERTAIN": 0.1, "source": "text_classifier"}]
    full_result = {
        "obj2_credibility": {"evaluated_evidence": [], "credibility_evaluation": {"flags_triggered": [
            {"severity": "WARNING", "message": "Recruiter email uses public/free domain (x@gmail.com)"}]}},
        "evidence_masses": masses,
        "dst_combined_masses": combine_all(masses),
        "final_verification": {"final_verdict": "UNCERTAIN_NEEDS_VERIFICATION", "calibrated_confidence": 0.65,
                               "override_applied": True, "override_reason": "UNVERIFIED OVERRIDE: none confirm it"},
    }
    facts = explain_verdict(full_result, scam_probability=0.0)["llm_facts"]
    # The override set the result, so it is the main reason and the classifier is ranked below it
    assert facts[1] == "Main reason: a safety rule changed the result: UNVERIFIED OVERRIDE: none confirm it"
    assert facts[2].startswith("Less important evidence: Text classifier")
    assert "Warning noted, it did not change the score: Recruiter email uses public/free domain (x@gmail.com)" in facts


def test_overclaims_finds_strength_words_not_in_the_facts():
    facts = ["Main reason: Text classifier strongly points to a scam.", "Similar fraudulent listing found."]
    assert overclaims("The main reason is the text classifier, which is a strong signal.", facts) == []
    assert overclaims("The strongest red flag was a nearly identical listing.", facts) == ["identical", "strongest"]


def test_llm_summary_rejected_when_it_overclaims_or_omits_the_verdict():
    vexp = _analyze(0.99)["verdict_explanation"]
    for reply_text in ("This is a scam; the listing is identical to a known fraud.",  # overclaims
                       "The wording of the offer looks suspicious."):                  # no verdict
        reply = SimpleNamespace(content=reply_text)
        with patch("langchain_ollama.ChatOllama", lambda **kw: SimpleNamespace(invoke=lambda prompt: reply)):
            assert summarize_with_llm(vexp) is None


def test_llm_summary_is_retried_with_the_rejection_reason():
    vexp = _analyze(0.99)["verdict_explanation"]
    replies = iter(["The strongest sign is a scam pattern.", "The offer is a high-risk scam."])
    prompts = []

    def invoke(prompt):
        prompts.append(prompt)
        return SimpleNamespace(content=next(replies))

    with patch("langchain_ollama.ChatOllama", lambda **kw: SimpleNamespace(invoke=invoke)):
        assert summarize_with_llm(vexp) == "The offer is a high-risk scam."
    assert len(prompts) == 2
    assert "rejected: it used the words strongest" in prompts[1]
