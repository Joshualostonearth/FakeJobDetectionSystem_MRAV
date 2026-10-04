"""
Integration tests for the Credibility Engine orchestrator.

These tests mock the LLM alignment calls to avoid requiring a running
Ollama instance. All other sub-engines run with real logic.
"""

import pytest
from unittest.mock import patch, MagicMock

from schemas.input_schema import (
    JobOfferEntities,
    RetrivedDocMetadata,
    VectorDocument,
    RetrievedEvidenceBundle,
    Objective2InputPayload,
)
from schemas.credibility_schema import MRAVTriple
from credibility_engine import evaluate_credibility, _classify_tier


def _mock_alignment_triple(*args, **kwargs) -> MRAVTriple:
    """Returns a neutral M-RAV triple for testing."""
    return MRAVTriple(
        hypothesis="Neutral",
        explanation="Mocked alignment for testing.",
        consistency_score=0.5,
    )


def _build_test_payload() -> Objective2InputPayload:
    """Build a realistic test payload with mixed evidence."""
    return Objective2InputPayload(
        offer_id="TEST_001",
        extracted_entities=JobOfferEntities(
            company_name="Apex Technologies Inc.",
            job_role="Senior Software Engineer",
            domain_or_website="apextech-jobs.org",
            email_or_phone="recruiter@gmail.com",
            location="New York, NY",
        ),
        retrieved_evidence=RetrievedEvidenceBundle(
            registry_records=[
                VectorDocument(
                    doc_id="REG_001",
                    content="Apex Technologies Inc. registered in Delaware. Official domain: apextech.com. New York, NY.",
                    similarity_score=0.92,
                    metadata=RetrivedDocMetadata(
                        source_type="registry",
                        company_name="Apex Technologies Inc.",
                        domain="apextech.com",
                        credibility_tier="official",
                        last_updated="2026-06-15",
                    ),
                ),
            ],
            linkedin_data=[
                VectorDocument(
                    doc_id="LI_001",
                    content="Apex Technologies - Enterprise Software. New York, NY. Website: apextech.com",
                    similarity_score=0.85,
                    metadata=RetrivedDocMetadata(
                        source_type="linkedin",
                        company_name="Apex Technologies",
                        domain="apextech.com",
                        credibility_tier="high",
                        last_updated="2026-07-01",
                    ),
                ),
            ],
            scam_reports=[
                VectorDocument(
                    doc_id="SCAM_001",
                    content="Scam alert: apextech-jobs.org is a fake domain used to impersonate Apex Technologies.",
                    similarity_score=0.78,
                    metadata=RetrivedDocMetadata(
                        source_type="scam_report",
                        company_name="Apex Technologies",
                        domain="apextech-jobs.org",
                        credibility_tier="community",
                        last_updated="2026-07-20",
                    ),
                ),
            ],
        ),
    )


class TestClassifyTier:
    """Verify credibility tier classification."""

    def test_discard(self):
        assert _classify_tier(0.1) == "discard"
        assert _classify_tier(0.29) == "discard"

    def test_low_confidence(self):
        assert _classify_tier(0.3) == "low_confidence"
        assert _classify_tier(0.49) == "low_confidence"

    def test_trusted(self):
        assert _classify_tier(0.5) == "trusted"
        assert _classify_tier(0.9) == "trusted"

    def test_boundary_discard_low(self):
        assert _classify_tier(0.3) == "low_confidence"  # >= 0.3 is low_confidence

    def test_boundary_low_trusted(self):
        assert _classify_tier(0.5) == "trusted"  # >= 0.5 is trusted


class TestCredibilityEngine:
    """Integration tests for the full credibility evaluation pipeline."""

    @patch("credibility_engine.generate_alignment_triple", side_effect=_mock_alignment_triple)
    def test_output_structure(self, mock_alignment):
        """Verify the output payload has the correct structure."""
        payload = _build_test_payload()
        result = evaluate_credibility(payload)

        assert result.offer_id == "TEST_001"
        assert result.routing_decision == "PROCEED_TO_CONFIDENCE_FRAMEWORK"
        assert 0.0 <= result.credibility_evaluation.overall_trust_score <= 1.0
        assert isinstance(result.credibility_evaluation.flags_triggered, list)
        assert isinstance(result.evaluated_evidence, list)
        assert isinstance(result.discarded_evidence, list)

    @patch("credibility_engine.generate_alignment_triple", side_effect=_mock_alignment_triple)
    def test_all_docs_evaluated(self, mock_alignment):
        """All 3 documents should be processed (retained + discarded = total)."""
        payload = _build_test_payload()
        result = evaluate_credibility(payload)

        total = len(result.evaluated_evidence) + len(result.discarded_evidence)
        assert total == 3

    @patch("credibility_engine.generate_alignment_triple", side_effect=_mock_alignment_triple)
    def test_registry_gets_high_authority(self, mock_alignment):
        """Registry document should have authority = 1.0."""
        payload = _build_test_payload()
        result = evaluate_credibility(payload)

        all_docs = result.evaluated_evidence + result.discarded_evidence
        reg_doc = next(d for d in all_docs if d.source_type == "registry")
        assert reg_doc.authority == 1.0

    @patch("credibility_engine.generate_alignment_triple", side_effect=_mock_alignment_triple)
    def test_domain_mismatch_flagged(self, mock_alignment):
        """Domain mismatch between offer and registry should generate a CRITICAL flag."""
        payload = _build_test_payload()
        result = evaluate_credibility(payload)

        critical_flags = [
            f for f in result.credibility_evaluation.flags_triggered
            if f.severity == "CRITICAL" and "Domain mismatch" in f.message
        ]
        assert len(critical_flags) >= 1

    @patch("credibility_engine.generate_alignment_triple", side_effect=_mock_alignment_triple)
    def test_free_email_flagged(self, mock_alignment):
        """Gmail recruiter email should generate a WARNING flag."""
        payload = _build_test_payload()
        result = evaluate_credibility(payload)

        email_flags = [
            f for f in result.credibility_evaluation.flags_triggered
            if f.severity == "WARNING" and "public" in f.message.lower()
        ]
        assert len(email_flags) >= 1

    @patch("credibility_engine.generate_alignment_triple", side_effect=_mock_alignment_triple)
    def test_scam_report_flagged(self, mock_alignment):
        """Presence of scam reports should generate a CRITICAL flag."""
        payload = _build_test_payload()
        result = evaluate_credibility(payload)

        scam_flags = [
            f for f in result.credibility_evaluation.flags_triggered
            if f.severity == "CRITICAL" and "SCAM" in f.message.upper()
        ]
        assert len(scam_flags) >= 1

    @patch("credibility_engine.generate_alignment_triple", side_effect=_mock_alignment_triple)
    def test_each_doc_has_mrav_triple(self, mock_alignment):
        """Every evaluated document should have an M-RAV triple."""
        payload = _build_test_payload()
        result = evaluate_credibility(payload)

        all_docs = result.evaluated_evidence + result.discarded_evidence
        for doc in all_docs:
            assert doc.mrav_triple is not None
            assert doc.mrav_triple.hypothesis in ("Confirms", "Contradicts", "Neutral")
            assert 0.0 <= doc.mrav_triple.consistency_score <= 1.0

    @patch("credibility_engine.generate_alignment_triple", side_effect=_mock_alignment_triple)
    def test_credibility_tiers_assigned(self, mock_alignment):
        """Each document should be classified into a valid credibility tier."""
        payload = _build_test_payload()
        result = evaluate_credibility(payload)

        all_docs = result.evaluated_evidence + result.discarded_evidence
        valid_tiers = {"discard", "low_confidence", "trusted"}
        for doc in all_docs:
            assert doc.credibility_tier in valid_tiers

    @patch("credibility_engine.generate_alignment_triple", side_effect=_mock_alignment_triple)
    def test_empty_evidence_bundle(self, mock_alignment):
        """An empty evidence bundle should produce a CRITICAL 'no evidence' flag."""
        payload = Objective2InputPayload(
            offer_id="EMPTY_001",
            extracted_entities=JobOfferEntities(
                company_name="Nonexistent Corp",
                job_role="Developer",
            ),
            retrieved_evidence=RetrievedEvidenceBundle(),
        )
        result = evaluate_credibility(payload)

        assert result.credibility_evaluation.overall_trust_score == 0.0
        assert any(
            "All evidence documents were discarded" in f.message
            for f in result.credibility_evaluation.flags_triggered
        )
        assert len(result.evaluated_evidence) == 0

    @patch("credibility_engine.generate_alignment_triple", side_effect=_mock_alignment_triple)
    def test_routing_always_confidence_framework(self, mock_alignment):
        """Routing should always be PROCEED_TO_CONFIDENCE_FRAMEWORK."""
        payload = _build_test_payload()
        result = evaluate_credibility(payload)
        assert result.routing_decision == "PROCEED_TO_CONFIDENCE_FRAMEWORK"
