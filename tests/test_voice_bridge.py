"""
tests/test_voice_bridge.py — Obj2 → Obj3 → DST wiring in voice_verification_bridge.py
and the UI entry point in pipeline_bridge.py. The voice call itself is mocked.

Run with:
    python -m pytest tests/test_voice_bridge.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import objective3_voice_agent
import voice_verification_bridge as bridge
from schemas.credibility_schema import (
    CredibilityEvaluation,
    CredibilityFlag,
    EntityConsensusScore,
    EvaluatedDocument,
    MRAVTriple,
    Objective2OutputPayload,
)

ENTITIES = {
    "company_name": "Apex Tech Solutions",
    "job_role": "Software Engineer",
    "recruiter_name": "Sarah Johnson",
    "recruiter_email": "sarah@gmail.com",
    "domain_or_website": "apextech.com",
    "location": "San Francisco, CA",
}

EVIDENCE = [
    {"page_content": "legit", "metadata": {"source_type": "legitimate_job_record", "fraudulent": 0}, "distance": 0.6},
    {"page_content": "scam", "metadata": {"source_type": "scam_report", "fraudulent": 1}, "distance": 1.4},
]

CONFIRMING_SIGNALS = {
    "company_reached": True,
    "role_actively_hiring": True,
    "recruiter_recognized": True,
    "company_denied_role": False,
    "suspicious_response": False,
    "source_confidence": 1.0,
    "extraction_confidence": 0.9,
    "number_source": "webrtc_simulation",
    "response_summary": "HR confirmed the role is open.",
}


@pytest.fixture
def fake_call(monkeypatch):
    """Replace the real voice pipeline; records the arguments it was called with."""
    calls = []

    def _install(voice_result):
        def _run(claim_id, entity, config, event_push=None):
            calls.append({"claim_id": claim_id, "entity": entity})
            return voice_result
        monkeypatch.setattr(objective3_voice_agent, "run_voice_verification", _run)
        return calls

    return _install


def test_distance_to_similarity():
    assert bridge.distance_to_similarity(0.0) == 1.0
    assert bridge.distance_to_similarity(1.0) == pytest.approx(0.5)
    assert bridge.distance_to_similarity(4.0) == 0.0


def test_check_voice_services_reports_unreachable_bridge():
    message = bridge.check_voice_services({"webrtc_bridge_url": "http://127.0.0.1:9"}, timeout=0.5)
    assert message is not None
    assert "call_bridge.py" in message


def test_verify_offer_by_phone_combines_confirming_call(fake_call):
    calls = fake_call({"pipeline_completed": True, "voice_signals": CONFIRMING_SIGNALS})

    result = bridge.verify_offer_by_phone(
        "UI_t", ENTITIES, EVIDENCE,
        suspicion_reasons=["WARNING: free email"],
        voice_config={},
    )

    assert len(calls) == 1
    entity = calls[0]["entity"]
    assert entity["recruiter_name"] == "Sarah Johnson"
    assert entity["role"] == "Software Engineer"
    assert entity["suspicion_reasons"] == ["WARNING: free email"]

    assert result["voice_mass"]["REAL"] > result["voice_mass"]["FAKE"]
    assert result["final_belief_real"] > result["final_belief_fake"]
    total = result["final_belief_real"] + result["final_belief_fake"] + result["final_belief_uncertain"]
    assert total == pytest.approx(1.0, abs=1e-3)


def test_verify_offer_by_phone_without_signals_uses_documents_only(fake_call):
    fake_call({"pipeline_completed": False, "skip_reason": "no_official_number_found"})
    result = bridge.verify_offer_by_phone("UI_t", ENTITIES, EVIDENCE, voice_config={})
    assert result["voice_mass"] is None
    assert result["voice_verification"]["skip_reason"] == "no_official_number_found"
    assert result["final_verdict"] is not None


def test_verify_offer_by_phone_reports_pipeline_exception(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("bridge exploded")
    monkeypatch.setattr(objective3_voice_agent, "run_voice_verification", _boom)
    result = bridge.verify_offer_by_phone("UI_t", ENTITIES, EVIDENCE, voice_config={})
    assert "bridge exploded" in result["voice_error"]
    assert result["final_verdict"] is not None


def _obj2_output(routing):
    doc = EvaluatedDocument(
        source="REG_001",
        source_type="registry",
        authority=1.0,
        freshness=0.9,
        consensus=0.8,
        credibility_index=0.86,
        credibility_tier="trusted",
        consensus_breakdown=EntityConsensusScore(
            company_name_match_score=1.0, domain_match=True, email_domain_trust=0.0, location_match=True
        ),
        mrav_triple=MRAVTriple(hypothesis="Confirms", explanation="ok", consistency_score=0.9),
    )
    return Objective2OutputPayload(
        offer_id="OFFER_T",
        credibility_evaluation=CredibilityEvaluation(
            overall_trust_score=0.5,
            flags_triggered=[CredibilityFlag(severity="CRITICAL", message="Domain mismatch")],
        ),
        evaluated_evidence=[doc],
        routing_decision=routing,
    )


def test_run_full_verification_runs_voice_when_routed(fake_call):
    calls = fake_call({"pipeline_completed": True, "voice_signals": CONFIRMING_SIGNALS})
    result = bridge.run_full_verification(
        _obj2_output("PROCEED_TO_VOICE_VERIFICATION"), ENTITIES, voice_config={}
    )
    assert len(calls) == 1
    assert calls[0]["entity"]["suspicion_reasons"] == ["Domain mismatch"]
    assert result["voice_mass"] is not None
    assert "Real" in result["final_verdict"]


def test_run_full_verification_skips_voice_otherwise(fake_call):
    calls = fake_call({"pipeline_completed": True, "voice_signals": CONFIRMING_SIGNALS})
    result = bridge.run_full_verification(
        _obj2_output("PROCEED_TO_CONFIDENCE_FRAMEWORK"), ENTITIES, voice_config={}
    )
    assert calls == []
    assert result["voice_verification"] is None
    assert result["final_verdict"] is not None


def test_handle_phone_verification_requires_services(monkeypatch):
    import pipeline_bridge
    monkeypatch.setattr(pipeline_bridge, "check_voice_services", lambda: "call bridge down")
    with pytest.raises(RuntimeError, match="call bridge down"):
        pipeline_bridge.handle_phone_verification(
            {"offer_id": "UI_t", "extracted_entities": ENTITIES, "retrieved_evidence": EVIDENCE, "flags": []}
        )


def test_handle_phone_verification_passes_analysis_through(monkeypatch, fake_call):
    import pipeline_bridge
    calls = fake_call({"pipeline_completed": True, "voice_signals": CONFIRMING_SIGNALS})
    monkeypatch.setattr(pipeline_bridge, "check_voice_services", lambda: None)
    result = pipeline_bridge.handle_phone_verification(
        {"offer_id": "UI_t", "extracted_entities": ENTITIES, "retrieved_evidence": EVIDENCE, "flags": ["F1"]},
    )
    assert calls[0]["entity"]["suspicion_reasons"] == ["F1"]
    assert result["offer_id"] == "UI_t"
