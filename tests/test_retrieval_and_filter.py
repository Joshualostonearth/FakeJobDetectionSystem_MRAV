"""
tests/test_retrieval_and_filter.py — FAISS query construction, the UI's
top-match scam check, and the M-RAV alignment fallback. No Ollama or
FAISS index required.

Run with:
    python -m pytest tests/test_retrieval_and_filter.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import JobOfferEntities
from search_engine import EvidenceRetriever
from engines import credibility_engine as simple_engine
from engines import alignment_chain
from schemas.input_schema import JobOfferEntities as Obj2Entities, RetrivedDocMetadata, VectorDocument


def _retriever_with_mock_db():
    retriever = EvidenceRetriever.__new__(EvidenceRetriever)
    retriever.vector_db = MagicMock()
    retriever.vector_db.similarity_search_with_score.return_value = []
    return retriever


def test_legit_and_scam_searches_use_the_same_query():
    retriever = _retriever_with_mock_db()
    entities = JobOfferEntities(
        company_name="Apex Tech", job_role="Data Analyst", location="Austin, TX",
        recruiter_email="hr@apex-careers.org",
    )

    retriever.retrieve_evidence_bundle(entities, top_k=3)

    calls = retriever.vector_db.similarity_search_with_score.call_args_list
    assert [c.kwargs["filter"]["source_type"] for c in calls] == ["legitimate_job_record", "scam_report"]
    queries = {c.kwargs["query"] for c in calls}
    assert queries == {"Apex Tech Data Analyst Austin, TX hr@apex-careers.org"}


def test_empty_entities_skip_search():
    retriever = _retriever_with_mock_db()
    bundle = retriever.retrieve_evidence_bundle(JobOfferEntities(), top_k=3)
    assert bundle["status"] == "SUCCESS"
    retriever.vector_db.similarity_search_with_score.assert_not_called()


def _doc(fraudulent, distance):
    return {"page_content": "x", "metadata": {"fraudulent": fraudulent}, "distance": distance}


ENTITIES = {"company_name": "Apex Tech", "recruiter_email": "hr@apextech.com"}


@patch.object(simple_engine, "evaluate_mrav_consistency", return_value={"mrav_analysis": "VERDICT: SUSPICIOUS"})
def test_top_match_scam_detected_even_when_listed_after_legit(_mock):
    # Legit records come first in the list, but the scam report is closer.
    evidence = [_doc(0, 1.2), _doc(0, 1.3), _doc(1, 0.4)]
    result = simple_engine.Run_Objective_2_Filter(ENTITIES, evidence)
    assert any("known fraudulent" in f for f in result["flags_raised"])
    assert result["final_trust_score"] == 50.0


@patch.object(simple_engine, "evaluate_mrav_consistency", return_value={"mrav_analysis": "VERDICT: LEGITIMATE"})
def test_top_match_legit_raises_no_scam_flag(_mock):
    evidence = [_doc(1, 1.1), _doc(0, 0.3)]
    result = simple_engine.Run_Objective_2_Filter(ENTITIES, evidence)
    assert not any("known fraudulent" in f for f in result["flags_raised"])
    assert result["final_trust_score"] == 100.0


@patch.object(simple_engine, "evaluate_mrav_consistency", return_value={"mrav_analysis": ""})
def test_no_evidence_is_handled(_mock):
    result = simple_engine.Run_Objective_2_Filter(ENTITIES, [])
    assert result["final_trust_score"] == 100.0


def test_alignment_failure_is_neutral_not_contradiction():
    entities = Obj2Entities(company_name="Apex Tech", job_role="Data Analyst")
    doc = VectorDocument(
        doc_id="D1", content="text", similarity_score=0.5,
        metadata=RetrivedDocMetadata(source_type="legitimate_job_record", fraudulent=0),
    )
    with patch.object(alignment_chain.key_pool, "execute", side_effect=RuntimeError("model 'ministral-3' not found")):
        triple = alignment_chain.generate_alignment_triple(entities, doc)

    assert triple.hypothesis == "Neutral"
    assert triple.consistency_score == 0.5
    assert "not found" in triple.explanation
