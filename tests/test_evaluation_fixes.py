"""Tests for the fixes made while evaluating on the EMSCAD splits."""

from confidence_calibrator import apply_verification_override
from engines.authority_evaluator import calculate_authority
from engines.consensus_evaluator import calculate_consensus
from models import JobOfferEntities as ExtractedEntities, clean_entities
from obj2dst import DSTEngine
from schemas.credibility_schema import CredibilityFlag
from schemas.input_schema import JobOfferEntities, RetrivedDocMetadata, VectorDocument

DST = {"REAL": 0.8, "FAKE": 0.1, "UNCERTAIN": 0.1, "conflict": 0.0}


def test_warning_flag_mentioning_scam_does_not_override():
    flags = [CredibilityFlag(severity="WARNING", message="LLM says this looks like a scam template")]
    result = apply_verification_override(0.9, DST, flags)
    assert result["override_applied"] is False
    assert result["final_verdict"] == "VERIFIED_REAL"


def test_critical_flag_still_overrides():
    flags = [CredibilityFlag(severity="CRITICAL", message="Domain mismatch")]
    result = apply_verification_override(0.9, DST, flags)
    assert result["override_applied"] is True
    assert result["is_scam"] is True


def test_critical_string_flag_overrides():
    result = apply_verification_override(0.9, DST, ["CRITICAL: Top similarity match is fraudulent"])
    assert result["override_applied"] is True


def test_emscad_records_use_authority_tier():
    assert calculate_authority(RetrivedDocMetadata(source_type="legitimate_job_record", authority_tier=3)) == 0.50
    assert calculate_authority(RetrivedDocMetadata(source_type="unknown_kind", authority_tier=4)) == 0.30


def test_missing_fields_are_neutral_in_consensus():
    entities = JobOfferEntities(company_name="", job_role="Driver", location=None)
    doc = VectorDocument(doc_id="ROW1", content="Title: Driver.", similarity_score=0.8,
                         metadata=RetrivedDocMetadata(source_type="legitimate_job_record"))
    score, _ = calculate_consensus(entities, doc)
    # name, domain and location unknown -> 0.5 each; no email -> 0.5
    assert abs(score - 0.5) < 1e-9


def test_classifier_mass_sums_to_one():
    m = DSTEngine().classifier_mass(0.8, reliability=0.9)
    assert abs(m["REAL"] + m["FAKE"] + m["UNCERTAIN"] - 1.0) < 1e-9
    assert m["FAKE"] > m["REAL"]


def test_clean_entities_drops_placeholders():
    e = clean_entities(ExtractedEntities(company_name="None", job_role="Engineer", domain_or_website="N/A"))
    assert e.company_name == ""
    assert e.domain_or_website is None
    assert e.job_role == "Engineer"
