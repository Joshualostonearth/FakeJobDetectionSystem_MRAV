"""
Tests for the Objective 1 -> Objective 2 connection (objective1_intake.py).
Entity extraction and retrieval are stubbed so no LLM or FAISS index is needed.
"""

import models
import objective1_intake
from models import JobOfferEntities
from schemas.input_schema import Objective2InputPayload


_ENTITIES = JobOfferEntities(
    company_name="Apex Tech Solutions",
    job_role="Software Engineer",
    recruiter_email="hr@apextech-careers.org",
    domain_or_website="apextech-careers.org",
    location="San Francisco, CA",
)


def _doc(doc_id, source_type, fraudulent, domain):
    return {
        "doc_id": doc_id,
        "content": "Apex Tech Solutions record",
        "similarity_score": 0.4,
        "metadata": {
            "source_type": source_type,
            "fraudulent": fraudulent,
            "company_name": "Apex Tech Solutions",
            "domain": domain,
            "credibility_tier": "official",
            "last_updated": "2026-06-01",
        },
    }


class _Retriever:
    def __init__(self, bundle, loaded=True):
        self.vector_db = object() if loaded else None
        self._bundle = bundle
        self.calls = 0

    def retrieve_evidence_bundle(self, entities, top_k=5):
        self.calls += 1
        return self._bundle


def _success_bundle():
    return {
        "status": "SUCCESS",
        "retrieved_evidence": {
            "legitimate_job_records": [],
            "registry_records": [_doc("REG_1", "registry", 0, "apextech.com")],
            "linkedin_data": [],
            "scam_reports": [_doc("SCAM_1", "scam_report", 1, "apextech-careers.org")],
        },
    }


def _stub_extraction(monkeypatch):
    monkeypatch.setattr(models, "extract_entities_from_text", lambda text, api_key="": _ENTITIES)


class TestBuildObjective2Payload:
    def test_payload_matches_objective2_contract(self, monkeypatch):
        _stub_extraction(monkeypatch)
        payload = objective1_intake.build_objective2_payload(
            text="offer text", offer_id="OFFER_T1", retriever=_Retriever(_success_bundle())
        )
        payload.pop("objective1")
        validated = Objective2InputPayload(**payload)
        assert validated.extracted_entities.email_or_phone == "hr@apextech-careers.org"
        assert len(validated.retrieved_evidence.registry_records) == 1
        assert len(validated.retrieved_evidence.scam_reports) == 1

    def test_missing_index_yields_no_evidence(self, monkeypatch):
        _stub_extraction(monkeypatch)
        retriever = _Retriever(_success_bundle(), loaded=False)
        payload = objective1_intake.build_objective2_payload(text="offer text", retriever=retriever)
        assert retriever.calls == 0
        assert payload["objective1"]["retrieval_status"] == "NO_INDEX"
        assert payload["objective1"]["documents_retrieved"] == 0

    def test_failed_retrieval_yields_no_evidence(self, monkeypatch):
        _stub_extraction(monkeypatch)
        retriever = _Retriever({"status": "FAILED", "retrieved_evidence": []})
        payload = objective1_intake.build_objective2_payload(text="offer text", retriever=retriever)
        assert payload["objective1"]["documents_retrieved"] == 0


class TestVerifyOfferFromInput:
    def test_runs_objectives_1_to_4(self, monkeypatch):
        _stub_extraction(monkeypatch)
        out = objective1_intake.verify_offer_from_input(
            text="offer text", retriever=_Retriever(_success_bundle())
        )
        assert out["objective1"]["documents_retrieved"] == 2
        assert out["obj2_dst"]["dst_status"] == "SUFFICIENT_FAKE"
        assert out["final_verification"]["final_verdict"] == "HIGH_RISK_SCAM"

    def test_no_evidence_routes_to_voice_agent(self, monkeypatch):
        _stub_extraction(monkeypatch)
        calls = []
        out = objective1_intake.verify_offer_from_input(
            text="offer text",
            retriever=_Retriever(_success_bundle(), loaded=False),
            voice_agent_fn=lambda payload: calls.append(payload) or None,
        )
        assert out["obj2_dst"]["dst_status"] == "NOT_SUFFICIENT_INFORMATION"
        assert len(calls) == 1
        assert out["voice_verification"]["status"] == "REQUIRED_NOT_RUN"
