"""
tests/test_objective5.py — Unit tests for Objective 5: Explanation Module.

Covers:
    1.  Verdict mapping from both input paths (Objective 4 confidence, Objective 3 voice yes/no)
    2.  Evidence sheet grounding: source credibility, DST masses, conflict K, flags, voice findings
    3.  Contradiction detection between Real- and Fake-leaning evidence
    4.  Template fallback when no API key / LLM is available
    5.  LLM output grounding checks: unknown fact IDs and invented numbers are rejected
    6.  Fallback to template when the LLM call fails

No network calls are made: the LLM is replaced by a fake client.

Run with:
    python -m pytest tests/test_objective5.py -v
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from objective5_explanation import (  # noqa: E402
    build_from_confidence_result,
    build_from_voice_result,
    explain_confidence_result,
    explain_evidence_sheet,
    explain_voice_result,
)


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

def _doc(source_type, c_idx, hypothesis, domain_mismatch=False):
    return {
        "source": f"https://example.com/{source_type}",
        "source_type": source_type,
        "authority": 0.9,
        "freshness": 0.8,
        "consensus": 0.7,
        "credibility_index": c_idx,
        "credibility_tier": "trusted",
        "consensus_breakdown": {},
        "mrav_triple": {"hypothesis": hypothesis, "explanation": "field check", "consistency_score": 0.6},
        "domain_mismatch": domain_mismatch,
    }


@pytest.fixture
def scam_pipeline_result():
    """Shape of run_full_pipeline() output for a conflicting, overridden scam offer."""
    return {
        "offer_id": "OFFER_T1",
        "obj2_credibility": {
            "evaluated_evidence": [
                _doc("registry", 0.82, "Confirms"),
                _doc("scam_report", 0.71, "Confirms"),
            ],
            "discarded_evidence": [_doc("linkedin", 0.21, "Neutral")],
            "credibility_evaluation": {
                "overall_trust_score": 0.76,
                "flags_triggered": [
                    {"severity": "CRITICAL", "message": "Domain mismatch between offer (apextech-jobs.org) and registry record (apextech.com)"},
                    {"severity": "INFO", "message": "Document LI_001 discarded"},
                ],
            },
        },
        "obj2_dst": {"dst_status": "NOT_SUFFICIENT_INFORMATION", "reason": "Sources disagree (conflict K=0.452 > 0.4)"},
        "voice_verification": {
            "status": "COMPLETED",
            "voice_mass": {"REAL": 0.03, "FAKE": 0.17, "UNCERTAIN": 0.8, "details": {"suspicious_response": -0.3, "company_reached": 0.25}},
        },
        "dst_combined_masses": {"REAL": 0.31, "FAKE": 0.52, "UNCERTAIN": 0.17, "conflict": 0.452, "conflict_history": [{"step": 1, "conflict_K": 0.452}]},
        "final_verification": {
            "final_verdict": "HIGH_RISK_SCAM",
            "calibrated_confidence": 0.15,
            "is_scam": True,
            "override_applied": True,
            "override_reason": "HIGH RISK OVERRIDE: Hostile/scam-like response detected during official voice verification.",
            "conflict_factor_k": 0.452,
            "provisional": False,
        },
    }


class FakeLLM:
    """Mimics openai.OpenAI().chat.completions.create() and returns a fixed JSON reply."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        content = json.dumps(self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    """Make sure tests never pick up a real key from the environment."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ──────────────────────────────────────────────────────────
# 1. Verdict mapping
# ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("VERIFIED_REAL", "Real"),
    ("HIGH_RISK_SCAM", "Fake"),
    ("SUSPICIOUS_FAKE", "Fake"),
    ("UNCERTAIN_NEEDS_VERIFICATION", "Unable to Verify"),
])
def test_confidence_path_verdict_mapping(scam_pipeline_result, raw, expected):
    scam_pipeline_result["final_verification"]["final_verdict"] = raw
    sheet = build_from_confidence_result(scam_pipeline_result)
    assert sheet.verdict == expected
    assert sheet.decision_path == "CONFIDENCE_PATH"


@pytest.mark.parametrize("decision,expected", [
    ("yes", "Real"), (True, "Real"), ("no", "Fake"), (False, "Fake"), ("unknown", "Unable to Verify"),
])
def test_voice_path_explicit_decision(decision, expected):
    sheet = build_from_voice_result({"company_reached": True}, voice_decision=decision)
    assert sheet.verdict == expected
    assert sheet.decision_path == "VOICE_PATH"


def test_voice_path_derived_decision():
    assert build_from_voice_result({"company_denied_role": True}).verdict == "Fake"
    assert build_from_voice_result({"company_reached": True, "role_actively_hiring": True}).verdict == "Real"
    assert build_from_voice_result({"company_reached": False}).verdict == "Unable to Verify"


# ──────────────────────────────────────────────────────────
# 2-3. Evidence sheet contents and contradictions
# ──────────────────────────────────────────────────────────

def test_confidence_sheet_contains_pipeline_values(scam_pipeline_result):
    sheet = build_from_confidence_result(scam_pipeline_result)
    all_text = " ".join(f.text for f in sheet.facts)

    assert "credibility index 0.820" in all_text          # source credibility score
    assert "m(Fake) = 0.520" in all_text                  # DST masses
    assert "Conflict factor K = 0.452" in all_text        # conflict value
    assert "suspicious_response" in all_text              # voice-call finding
    assert "Safety override applied" in all_text          # Objective 4 override
    assert "discarded" in all_text                         # filtered evidence is disclosed
    assert not any("LI_001" in f.text for f in sheet.facts if f.category == "flag")  # INFO flags excluded
    assert sheet.confidence == pytest.approx(0.15)


def test_contradiction_between_registry_and_scam_report(scam_pipeline_result):
    sheet = build_from_confidence_result(scam_pipeline_result)
    assert len(sheet.contradictions) == 1
    ids = sheet.contradictions[0].fact_ids
    registry = next(f for f in sheet.facts if f.text.startswith("registry"))
    scam = next(f for f in sheet.facts if f.text.startswith("scam_report"))
    assert registry.fact_id in ids and scam.fact_id in ids
    assert "K1" in ids  # conflict K is above the moderate threshold


def test_no_contradiction_when_evidence_agrees():
    sheet = build_from_voice_result({"company_reached": True, "role_actively_hiring": True, "recruiter_recognized": True})
    assert sheet.contradictions == []


# ──────────────────────────────────────────────────────────
# 4. Template fallback
# ──────────────────────────────────────────────────────────

def test_template_used_without_api_key(scam_pipeline_result):
    report = explain_confidence_result(scam_pipeline_result)
    assert report.generated_by == "template"
    assert report.verdict == "Fake"
    assert report.key_evidence and all(p.cites for p in report.key_evidence)
    assert "Do not pay" in report.recommendation
    text = report.to_text()
    assert "FAKE" in text and "Conflict factor K = 0.452" in text


def test_template_voice_report():
    report = explain_voice_result({"company_denied_role": True, "company_reached": True}, offer_id="V1")
    assert report.verdict == "Fake"
    assert report.generated_by == "template"
    assert "denied" in report.summary


# ──────────────────────────────────────────────────────────
# 5. LLM grounding checks
# ──────────────────────────────────────────────────────────

def test_grounded_llm_output_is_accepted(scam_pipeline_result):
    sheet = build_from_confidence_result(scam_pipeline_result)
    scam = next(f for f in sheet.facts if f.text.startswith("scam_report"))
    llm = FakeLLM({
        "summary": "The offer is judged fake because a scam report matches it and P(Real) is 0.150.",
        "summary_cites": [scam.fact_id, "Q1"],
        "key_evidence": [{"text": "Sources disagree strongly, with K = 0.452.", "cites": ["K1"]}],
        "contradictions": [],
    })
    report = explain_evidence_sheet(sheet, client=(llm, "test-model", "groq"))

    assert report.generated_by == "llm"
    assert report.model == "groq:test-model"
    assert report.verdict == "Fake"                 # verdict always comes from the pipeline
    assert report.contradictions                    # pipeline contradictions kept when LLM omits them
    assert report.grounding_issues == []
    assert "EVIDENCE SHEET" in llm.calls[0]["messages"][1]["content"]


def test_ungrounded_llm_statements_are_rejected(scam_pipeline_result):
    sheet = build_from_confidence_result(scam_pipeline_result)
    llm = FakeLLM({
        "summary": "Confidence is only 15%, so the offer is fake.",
        "summary_cites": ["Q1"],
        "key_evidence": [
            {"text": "Conflict K was 0.452.", "cites": ["K1"]},
            {"text": "The company has 37 complaints online.", "cites": ["S1"]},   # invented number
            {"text": "The recruiter is blacklisted.", "cites": ["Z9"]},           # unknown fact ID
            {"text": "The salary is unrealistic.", "cites": []},                  # no citation
        ],
        "contradictions": [],
    })
    report = explain_evidence_sheet(sheet, client=(llm, "m", "groq"))

    assert report.generated_by == "llm"
    assert [p.text for p in report.key_evidence] == ["Conflict K was 0.452."]
    assert len(report.grounding_issues) == 3


def test_fully_ungrounded_llm_output_falls_back_to_template(scam_pipeline_result):
    sheet = build_from_confidence_result(scam_pipeline_result)
    llm = FakeLLM({"summary": "Probably a scam, 99% sure.", "summary_cites": ["Q1"], "key_evidence": []})
    report = explain_evidence_sheet(sheet, client=(llm, "m", "groq"))

    assert report.generated_by == "template"
    assert any("99%" in issue for issue in report.grounding_issues)


# ──────────────────────────────────────────────────────────
# 6. LLM failure
# ──────────────────────────────────────────────────────────

def test_llm_error_falls_back_to_template(scam_pipeline_result):
    llm = FakeLLM(error=RuntimeError("rate limited"))
    report = explain_confidence_result(scam_pipeline_result, client=(llm, "m", "groq"))
    assert report.generated_by == "template"
    assert any("rate limited" in issue for issue in report.grounding_issues)


def test_use_llm_false_skips_client(scam_pipeline_result):
    llm = FakeLLM({"summary": "x"})
    report = explain_confidence_result(scam_pipeline_result, client=(llm, "m", "groq"), use_llm=False)
    assert report.generated_by == "template"
    assert llm.calls == []
