"""Unit tests for the Freshness Evaluator sub-engine (Plateau Scale Sigmoid)."""

import pytest
from datetime import date

from schemas.input_schema import RetrivedDocMetadata
from engines.freshness_evaluator import calculate_freshness, _sigmoid_freshness


class TestSigmoidFunction:
    """Verify the plateau-scale sigmoid produces expected behavior."""

    def test_zero_age_near_one(self):
        """A brand-new document (0 months) should score very close to 1.0."""
        score = _sigmoid_freshness(0.0)
        assert score >= 0.95

    def test_midpoint_is_half(self):
        """At t_mid (default 12 months), score should be exactly 0.5."""
        score = _sigmoid_freshness(12.0)
        assert abs(score - 0.5) < 0.01

    def test_old_document_near_floor(self):
        """A very old document (36 months) should be near the floor."""
        score = _sigmoid_freshness(36.0)
        assert score <= 0.10

    def test_monotonically_decreasing(self):
        """Score should decrease as age increases."""
        scores = [_sigmoid_freshness(t) for t in range(0, 30)]
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1]

    def test_floor_enforced(self):
        """Score should never go below the floor value (0.05)."""
        score = _sigmoid_freshness(100.0)
        assert score >= 0.05


class TestCalculateFreshness:
    """Verify the full freshness calculation pipeline."""

    def test_recent_document(self):
        """A 1-month-old document should score near 1.0."""
        meta = RetrivedDocMetadata(
            source_type="registry",
            last_updated="2026-07-11",
        )
        score, flags = calculate_freshness(meta, reference_date=date(2026, 8, 11))
        assert score >= 0.9
        assert len(flags) == 0

    def test_one_year_old_document(self):
        """A 12-month-old document should score ~0.5."""
        meta = RetrivedDocMetadata(
            source_type="linkedin",
            last_updated="2025-08-11",
        )
        score, flags = calculate_freshness(meta, reference_date=date(2026, 8, 11))
        assert 0.4 <= score <= 0.6

    def test_two_year_old_document(self):
        """A 24-month-old document should score near the floor."""
        meta = RetrivedDocMetadata(
            source_type="linkedin",
            last_updated="2024-08-11",
        )
        score, flags = calculate_freshness(meta, reference_date=date(2026, 8, 11))
        assert score <= 0.15

    def test_missing_last_updated(self):
        """Missing date should return penalty score and an INFO flag."""
        meta = RetrivedDocMetadata(source_type="registry", last_updated=None)
        score, flags = calculate_freshness(meta)
        assert score == 0.3
        assert len(flags) == 1
        assert "No last_updated" in flags[0]

    def test_unparseable_date(self):
        """Garbage date string should return penalty score and a WARNING flag."""
        meta = RetrivedDocMetadata(source_type="registry", last_updated="not-a-date")
        score, flags = calculate_freshness(meta)
        assert score == 0.3
        assert any("Could not parse" in f for f in flags)

    def test_new_registration_anomaly(self):
        """Registry document < 30 days old should trigger a freshness anomaly flag."""
        meta = RetrivedDocMetadata(
            source_type="registry",
            last_updated="2026-07-25",
        )
        score, flags = calculate_freshness(meta, reference_date=date(2026, 8, 11))
        assert score >= 0.95  # Still fresh, high score
        assert any("FRESHNESS_ANOMALY" in f for f in flags)

    def test_new_registration_non_registry_no_flag(self):
        """Non-registry document < 30 days old should NOT trigger the anomaly flag."""
        meta = RetrivedDocMetadata(
            source_type="linkedin",
            last_updated="2026-07-25",
        )
        score, flags = calculate_freshness(meta, reference_date=date(2026, 8, 11))
        assert score >= 0.95
        assert not any("FRESHNESS_ANOMALY" in f for f in flags)

    def test_future_date(self):
        """A future-dated document should score 1.0 and get a WARNING flag."""
        meta = RetrivedDocMetadata(
            source_type="linkedin",
            last_updated="2027-01-01",
        )
        score, flags = calculate_freshness(meta, reference_date=date(2026, 8, 11))
        assert score == 1.0
        assert any("future" in f.lower() for f in flags)
