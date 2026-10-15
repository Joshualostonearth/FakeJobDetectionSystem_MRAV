"""The app's analysis (pipeline_bridge.analyze_extracted_offer) runs run_full_pipeline."""

from unittest.mock import patch

import pipeline_bridge
from models import JobOfferEntities
from schemas.credibility_schema import MRAVTriple


class _StubRetriever:
    def retrieve_evidence_bundle(self, entities, top_k=3, query_text=None):
        meta = {"row_id": "1", "fraudulent": 0, "source_type": "legitimate_job_record", "authority_tier": 3}
        return {"status": "SUCCESS", "retrieved_evidence": {
            "legitimate_job_records": [{"content": "Title: Driver. Location: US, TX, Houston.",
                                        "similarity_score": 0.6, "metadata": meta}],
            "scam_reports": [], "registry_records": [], "linkedin_data": []}}


NEUTRAL = MRAVTriple(hypothesis="Neutral", explanation="test", consistency_score=0.5)


def _analyze(prob):
    with patch("credibility_engine.generate_alignment_triple", return_value=NEUTRAL), \
         patch.object(pipeline_bridge, "predict_scam_probability", return_value=prob):
        return pipeline_bridge.analyze_extracted_offer(
            "Driver job in Houston", JobOfferEntities(company_name="Acme", job_role="Driver"), _StubRetriever())


def test_scam_like_text_is_high_risk_and_auto_calls():
    res = _analyze(0.99)
    assert res["final_verdict"] in ("HIGH_RISK_SCAM", "SUSPICIOUS_FAKE")
    assert res["risk_level"] == "HIGH RISK"
    assert res["auto_call"] is True
    assert any("Text classifier" in f for f in res["flags"])


def test_clean_text_is_low_risk():
    res = _analyze(0.01)
    assert res["final_verdict"] == "VERIFIED_REAL"
    assert res["risk_level"] == "LOW RISK"
    assert res["voice_recommended"] is False
    assert 0 <= res["trust_score"] <= 100
