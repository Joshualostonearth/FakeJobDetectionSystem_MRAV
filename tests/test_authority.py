"""Unit tests for the Authority Evaluator sub-engine."""

import pytest

from schemas.input_schema import RetrivedDocMetadata
from engines.authority_evaluator import calculate_authority


class TestAuthorityTiers:
    """Verify that all documented source types map to the correct tier weights."""

    def test_tier1_registry(self):
        meta = RetrivedDocMetadata(source_type="registry", credibility_tier="official")
        assert calculate_authority(meta) == 1.0

    def test_tier1_government_registry(self):
        meta = RetrivedDocMetadata(source_type="government_registry")
        assert calculate_authority(meta) == 1.0

    def test_tier2_linkedin(self):
        meta = RetrivedDocMetadata(source_type="linkedin", credibility_tier="high")
        assert calculate_authority(meta) == 0.75

    def test_tier2_linkedin_verified(self):
        meta = RetrivedDocMetadata(source_type="linkedin_verified")
        assert calculate_authority(meta) == 0.75

    def test_tier3_aggregator(self):
        meta = RetrivedDocMetadata(source_type="third_party_aggregator")
        assert calculate_authority(meta) == 0.50

    def test_tier3_glassdoor(self):
        meta = RetrivedDocMetadata(source_type="glassdoor")
        assert calculate_authority(meta) == 0.50

    def test_tier3_crunchbase(self):
        meta = RetrivedDocMetadata(source_type="crunchbase")
        assert calculate_authority(meta) == 0.50

    def test_tier4_scam_report(self):
        meta = RetrivedDocMetadata(source_type="scam_report")
        assert calculate_authority(meta) == 0.30

    def test_tier4_reddit(self):
        meta = RetrivedDocMetadata(source_type="reddit")
        assert calculate_authority(meta) == 0.30

    def test_unknown_source_type_with_known_tier(self):
        """When source_type is unknown, fall back to credibility_tier."""
        meta = RetrivedDocMetadata(source_type="unknown_source", credibility_tier="official")
        assert calculate_authority(meta) == 1.0

    def test_unknown_source_type_with_unknown_tier(self):
        """When both are unknown, return the default weight."""
        meta = RetrivedDocMetadata(source_type="random_blog", credibility_tier="unknown")
        assert calculate_authority(meta) == 0.20

    def test_case_insensitive(self):
        """Source type matching should be case-insensitive."""
        meta = RetrivedDocMetadata(source_type="REGISTRY")
        assert calculate_authority(meta) == 1.0

    def test_whitespace_handling(self):
        """Source type with leading/trailing whitespace should still match."""
        meta = RetrivedDocMetadata(source_type="  linkedin  ")
        assert calculate_authority(meta) == 0.75
