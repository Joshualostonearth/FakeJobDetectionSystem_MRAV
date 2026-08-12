"""Unit tests for the Cross-Source Consensus Evaluator."""

import pytest

from schemas.input_schema import JobOfferEntities, VectorDocument, RetrivedDocMetadata
from engines.consensus_evaluator import (
    calculate_consensus,
    _extract_root_domain,
    _extract_email_domain,
    _compute_email_trust,
)


def _make_entities(**overrides) -> JobOfferEntities:
    """Helper to create a JobOfferEntities with sensible defaults."""
    defaults = {
        "company_name": "Apex Technologies Inc.",
        "job_role": "Software Engineer",
        "domain_or_website": "apextech.com",
        "email_or_phone": "recruiter@apextech.com",
        "location": "New York, NY",
    }
    defaults.update(overrides)
    return JobOfferEntities(**defaults)


def _make_doc(
    company_name="Apex Technologies Inc.",
    domain="apextech.com",
    content="Apex Technologies Inc. located in New York, NY.",
    source_type="registry",
    **meta_overrides,
) -> VectorDocument:
    """Helper to create a VectorDocument with sensible defaults."""
    meta = RetrivedDocMetadata(
        source_type=source_type,
        company_name=company_name,
        domain=domain,
        **meta_overrides,
    )
    return VectorDocument(
        doc_id="test_doc",
        content=content,
        similarity_score=0.9,
        metadata=meta,
    )


class TestDomainExtraction:
    """Verify tldextract-based domain parsing."""

    def test_simple_domain(self):
        assert _extract_root_domain("apextech.com") == "apextech.com"

    def test_subdomain(self):
        assert _extract_root_domain("www.apextech.com") == "apextech.com"

    def test_url_with_path(self):
        assert _extract_root_domain("https://careers.apextech.com/jobs") == "apextech.com"

    def test_none(self):
        assert _extract_root_domain(None) is None

    def test_empty(self):
        assert _extract_root_domain("") is None


class TestEmailDomainExtraction:
    """Verify email domain parsing."""

    def test_normal_email(self):
        assert _extract_email_domain("user@apextech.com") == "apextech.com"

    def test_no_at_sign(self):
        assert _extract_email_domain("+1-555-0123") is None

    def test_none(self):
        assert _extract_email_domain(None) is None


class TestEmailTrust:
    """Verify email trust scoring."""

    def test_corporate_match(self):
        """Email matching claimed domain should score 1.0."""
        assert _compute_email_trust("hr@apextech.com", "apextech.com") == 1.0

    def test_free_provider(self):
        """Gmail address should score 0.0."""
        assert _compute_email_trust("user@gmail.com", "apextech.com") == 0.0

    def test_yahoo(self):
        assert _compute_email_trust("user@yahoo.com", "apextech.com") == 0.0

    def test_non_free_non_matching(self):
        """A real corporate email that doesn't match claimed domain → 0.5."""
        assert _compute_email_trust("user@subsidiary.com", "apextech.com") == 0.5

    def test_no_email(self):
        """No email provided → neutral 0.5."""
        assert _compute_email_trust(None, "apextech.com") == 0.5

    def test_phone_number(self):
        """Phone number (no @) → neutral 0.5."""
        assert _compute_email_trust("+1-555-0123", "apextech.com") == 0.5


class TestConsensusScoring:
    """Verify end-to-end consensus scoring."""

    def test_perfect_match(self):
        """All fields matching should produce a high consensus score."""
        entities = _make_entities()
        doc = _make_doc()
        score, breakdown = calculate_consensus(entities, doc)
        assert score >= 0.85
        assert breakdown.domain_match is True
        assert breakdown.email_domain_trust == 1.0
        assert breakdown.company_name_match_score >= 0.95

    def test_domain_mismatch(self):
        """Different domains should reduce score and flag domain_match=False."""
        entities = _make_entities(domain_or_website="apextech-jobs.org")
        doc = _make_doc(domain="apextech.com")
        score, breakdown = calculate_consensus(entities, doc)
        assert breakdown.domain_match is False
        # Score should be lower than perfect match
        assert score < 0.85

    def test_free_email_penalty(self):
        """Free email provider should drag down the score."""
        entities = _make_entities(email_or_phone="recruiter@gmail.com")
        doc = _make_doc()
        score, breakdown = calculate_consensus(entities, doc)
        assert breakdown.email_domain_trust == 0.0
        # Should be lower than corporate email version
        perfect_entities = _make_entities()
        perfect_score, _ = calculate_consensus(perfect_entities, doc)
        assert score < perfect_score

    def test_company_name_fuzzy_match(self):
        """Slightly different company names should still get a reasonable score."""
        entities = _make_entities(company_name="Apex Tech Inc")
        doc = _make_doc(company_name="Apex Technologies Incorporated")
        score, breakdown = calculate_consensus(entities, doc)
        assert breakdown.company_name_match_score >= 0.60

    def test_completely_different_company(self):
        """Totally different company names should score very low on name match."""
        entities = _make_entities(company_name="Zenith Global Solutions")
        doc = _make_doc(company_name="Apex Technologies Inc.")
        score, breakdown = calculate_consensus(entities, doc)
        assert breakdown.company_name_match_score < 0.50

    def test_location_match_in_content(self):
        """Location mentioned in document content should be detected."""
        entities = _make_entities(location="New York")
        doc = _make_doc(content="The company is headquartered in New York, NY.")
        _, breakdown = calculate_consensus(entities, doc)
        assert breakdown.location_match is True

    def test_location_no_match(self):
        """Location NOT in document content should return False."""
        entities = _make_entities(location="San Francisco")
        doc = _make_doc(content="The company is headquartered in New York, NY.")
        _, breakdown = calculate_consensus(entities, doc)
        assert breakdown.location_match is False

    def test_no_doc_metadata_fields(self):
        """Missing metadata fields should degrade gracefully, not crash."""
        entities = _make_entities()
        doc = _make_doc(company_name=None, domain=None, content="Some generic text.")
        score, breakdown = calculate_consensus(entities, doc)
        assert 0.0 <= score <= 1.0
        assert breakdown.company_name_match_score == 0.0
        assert breakdown.domain_match is False
