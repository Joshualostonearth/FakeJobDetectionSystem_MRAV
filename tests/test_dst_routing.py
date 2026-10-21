"""
Tests for the Objective 2 DST sufficiency score and voice-agent routing.
"""

from obj2dst import (
    score_dst_sufficiency,
    SUFFICIENT_BELIEF_THRESHOLD,
    HIGH_CONFLICT_THRESHOLD,
    ROUTE_TO_VOICE_AGENT,
    PROCEED_TO_CONFIDENCE_FRAMEWORK,
)
from full_pipeline_orchestrator import run_full_pipeline
from main import get_demo_payload


def _mass(real, fake, conflict=0.0):
    return {"REAL": real, "FAKE": fake, "UNCERTAIN": 1.0 - real - fake, "conflict": conflict}


class TestScoreDstSufficiency:
    def test_strong_real_goes_to_objective_4(self):
        result = score_dst_sufficiency(_mass(0.87, 0.0))
        assert result["dst_score"] == 0.87
        assert result["dst_status"] == "SUFFICIENT_REAL"
        assert result["routing_decision"] == PROCEED_TO_CONFIDENCE_FRAMEWORK

    def test_strong_fake_goes_to_objective_4(self):
        result = score_dst_sufficiency(_mass(0.05, 0.80))
        assert result["dst_status"] == "SUFFICIENT_FAKE"
        assert result["routing_decision"] == PROCEED_TO_CONFIDENCE_FRAMEWORK

    def test_weak_belief_goes_to_voice_agent(self):
        result = score_dst_sufficiency(_mass(0.459, 0.0))
        assert result["dst_status"] == "NOT_SUFFICIENT_INFORMATION"
        assert result["routing_decision"] == ROUTE_TO_VOICE_AGENT

    def test_threshold_is_inclusive(self):
        result = score_dst_sufficiency(_mass(SUFFICIENT_BELIEF_THRESHOLD, 0.0))
        assert result["routing_decision"] == PROCEED_TO_CONFIDENCE_FRAMEWORK

    def test_high_conflict_goes_to_voice_agent(self):
        result = score_dst_sufficiency(_mass(0.70, 0.10, conflict=HIGH_CONFLICT_THRESHOLD + 0.01))
        assert result["dst_status"] == "NOT_SUFFICIENT_INFORMATION"
        assert result["routing_decision"] == ROUTE_TO_VOICE_AGENT

    def test_no_evidence_goes_to_voice_agent(self):
        result = score_dst_sufficiency({"REAL": 0.0, "FAKE": 0.0, "UNCERTAIN": 1.0, "conflict": 0.0})
        assert result["dst_score"] == 0.0
        assert result["routing_decision"] == ROUTE_TO_VOICE_AGENT


def _sparse_payload():
    payload = get_demo_payload()
    payload["retrieved_evidence"] = {"registry_records": [], "linkedin_data": [], "scam_reports": []}
    return payload


_VOICE = {
    "company_reached": True,
    "company_denied_role": True,
    "suspicious_response": False,
    "source_confidence": 0.90,
    "extraction_confidence": 0.85,
}


class TestPipelineRouting:
    def test_sufficient_evidence_skips_voice_agent(self):
        calls = []
        out = run_full_pipeline(get_demo_payload(), voice_agent_fn=lambda p: calls.append(p) or _VOICE)
        assert out["obj2_dst"]["routing_decision"] == PROCEED_TO_CONFIDENCE_FRAMEWORK
        assert out["voice_verification"]["status"] == "SKIPPED"
        assert calls == []
        assert out["final_verification"]["provisional"] is False

    def test_insufficient_evidence_calls_voice_agent(self):
        calls = []
        out = run_full_pipeline(_sparse_payload(), voice_agent_fn=lambda p: calls.append(p) or _VOICE)
        assert out["obj2_dst"]["dst_status"] == "NOT_SUFFICIENT_INFORMATION"
        assert out["obj2_credibility"]["routing_decision"] == ROUTE_TO_VOICE_AGENT
        assert len(calls) == 1
        assert out["voice_verification"]["status"] == "COMPLETED"
        assert out["final_verification"]["final_verdict"] == "HIGH_RISK_SCAM"

    def test_insufficient_evidence_without_voice_is_provisional(self):
        out = run_full_pipeline(_sparse_payload())
        assert out["voice_verification"]["status"] == "REQUIRED_NOT_RUN"
        assert out["final_verification"]["provisional"] is True


class TestDomainMismatch:
    def test_mismatched_record_counts_as_fake(self):
        from obj2dst import DSTEngine
        engine = DSTEngine()
        matching = engine.assign_mass_function(0.9, "registry", 0, 0.9)
        mismatched = engine.assign_mass_function(0.9, "registry", 0, 0.9, domain_mismatch=True)
        assert matching["REAL"] > 0 and matching["FAKE"] == 0
        assert mismatched["FAKE"] == matching["REAL"] and mismatched["REAL"] == 0

    def test_demo_scam_with_fake_domain_is_sufficient_fake(self):
        out = run_full_pipeline(get_demo_payload())
        assert out["obj2_dst"]["dst_status"] == "SUFFICIENT_FAKE"
        assert out["final_verification"]["final_verdict"] == "HIGH_RISK_SCAM"

    def test_matching_domain_stays_real(self):
        payload = get_demo_payload()
        payload["extracted_entities"]["domain_or_website"] = "apextech.com"
        payload["extracted_entities"]["email_or_phone"] = "sarah.johnson@apextech.com"
        payload["retrieved_evidence"]["scam_reports"] = []
        out = run_full_pipeline(payload)
        assert out["obj2_dst"]["dst_status"] == "SUFFICIENT_REAL"
