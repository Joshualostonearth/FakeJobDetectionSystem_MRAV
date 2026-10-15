"""
evaluation_baseline_original.py — Score the ORIGINAL project code (a git checkout of the
commit before the evaluation fixes) on the same evaluation rows, for a before/after comparison.

Run it with the original checkout first on sys.path:

    git worktree add --detach <wt> 4ac50c1
    cd <wt> && python <project>/evaluation_baseline_original.py <csv> <entities.jsonl> <out_dir>

Steps reproduced exactly as the original code does them:
    app path   handle_ui_submission steps 7-10: entity-string FAISS query (top_k=3),
               Run_Objective_2_Filter (no text classifier)
    DST        verify_offer_by_phone mass construction, voice step omitted
    final      full_pipeline_orchestrator.run_full_pipeline (original Obj 2, synthetic Platt model)

LLM outputs that never change a score are not re-generated: the M-RAV verdict text is
display-only, and alignment triples are set to Neutral ("NOT_RUN"). In the original
override, alignment text containing "scam" could also force a scam verdict; rows where that
could matter are counted in the output as 'verdict_may_depend_on_llm_text'.
"""

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())  # the original checkout

import pandas as pd

TEXT_FIELDS = [
    ("title", "Job Title"), ("location", "Location"), ("department", "Department"),
    ("salary_range", "Salary Range"), ("employment_type", "Employment Type"),
    ("required_experience", "Required Experience"), ("required_education", "Required Education"),
    ("industry", "Industry"), ("function", "Function"), ("company_profile", "Company Profile"),
    ("description", "Description"), ("requirements", "Requirements"), ("benefits", "Benefits"),
]


def main(csv_path, entities_path, out_dir):
    import credibility_engine as root_credibility
    import engines.credibility_engine as app_credibility
    from engines.credibility_engine import Run_Objective_2_Filter
    from full_pipeline_orchestrator import run_full_pipeline
    from models import JobOfferEntities
    from obj2dst import DSTEngine
    from schemas.credibility_schema import MRAVTriple
    from search_engine import EvidenceRetriever
    from voice_verification_bridge import distance_to_similarity, _combine_masses

    app_credibility.evaluate_mrav_consistency = lambda e, ev: {"mrav_analysis": "NOT_RUN"}
    root_credibility.generate_alignment_triple = lambda e, d: MRAVTriple(
        hypothesis="Neutral", explanation="NOT_RUN", consistency_score=0.5)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"
    if pred_file.exists():
        pred_file.unlink()

    df = pd.read_csv(csv_path)
    ents = {}
    with open(entities_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ents[r["job_id"]] = r["entities"]

    retriever = EvidenceRetriever(index_folder="faiss_index")
    dst = DSTEngine()
    t0 = time.time()
    for n, r in enumerate(df.to_dict("records"), 1):
        job_id = int(r["job_id"])
        rec = {"job_id": job_id, "true_label": int(r["fraudulent"]), "errors": {}, "llm_cached": False,
               "scam_probability": None}
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            entities = JobOfferEntities(**ents[job_id])
            entities_dict = entities.model_dump()
            bundle = retriever.retrieve_evidence_bundle(entities, top_k=3)
            evidence = []
            for cat in ["legitimate_job_records", "scam_reports", "registry_records", "linkedin_data"]:
                for item in bundle.get("retrieved_evidence", {}).get(cat, []):
                    evidence.append({"page_content": item.get("content", ""), "metadata": item.get("metadata", {}),
                                     "distance": float(item.get("similarity_score", 0.0))})
            evidence.sort(key=lambda d: d["distance"])
            obj2 = Run_Objective_2_Filter(entities_dict, evidence)

        rec["app"] = {
            "entities": entities_dict, "trust_score": float(obj2["final_trust_score"]),
            "risk_level": obj2["risk_level"], "flags": obj2["flags_raised"], "mrav_explanation": "NOT_RUN",
            "evidence": [{"row_id": e["metadata"].get("row_id"), "fraudulent": e["metadata"].get("fraudulent"),
                          "distance": e["distance"]} for e in evidence],
        }

        masses = []
        for ev in evidence:
            meta = ev["metadata"]
            st = meta.get("source_type", "unknown")
            masses.append(dst.assign_mass_function(
                similarity_score=distance_to_similarity(ev["distance"]), source_type=st,
                is_fraudulent=1 if meta.get("fraudulent") == 1 or st == "scam_report" else 0))
        bridge = {}
        with contextlib.redirect_stdout(log):
            _combine_masses(dst, masses, bridge)
        comb = bridge.get("dst_combined") or {}
        rec["evidence_dst"] = {"verdict": bridge["final_verdict"], "real": comb.get("REAL", 0.0),
                               "fake": comb.get("FAKE", 0.0), "uncertain": comb.get("UNCERTAIN", 1.0),
                               "conflict": comb.get("conflict", 0.0)}

        bundle_docs = {"legitimate_job_records": [], "registry_records": [], "linkedin_data": [], "scam_reports": []}
        for ev in evidence:
            meta = ev["metadata"]
            key = {"legitimate_job_record": "legitimate_job_records", "scam_report": "scam_reports"}.get(
                meta.get("source_type", ""), "registry_records")
            bundle_docs[key].append({"doc_id": f"ROW{meta.get('row_id', '')}", "content": ev["page_content"],
                                     "similarity_score": distance_to_similarity(ev["distance"]), "metadata": meta})
        payload = {"offer_id": f"TEST_{job_id}", "retrieved_evidence": bundle_docs, "extracted_entities": {
            "company_name": entities_dict.get("company_name") or "", "job_role": entities_dict.get("job_role") or "",
            "recruiter_name": entities_dict.get("recruiter_name"), "domain_or_website": entities_dict.get("domain_or_website"),
            "email_or_phone": entities_dict.get("recruiter_email"), "salary_or_stipend": None,
            "location": entities_dict.get("location")}}
        try:
            with contextlib.redirect_stdout(log):
                full = run_full_pipeline(payload, None)
            cred = full["obj2_credibility"]
            fv = full["final_verification"]
            flags = cred["credibility_evaluation"]["flags_triggered"]
            rec["orchestrator"] = {
                "overall_trust_score": cred["credibility_evaluation"]["overall_trust_score"],
                "routing_decision": cred["routing_decision"],
                "flags": [f"{f['severity']}: {f['message']}" for f in flags],
                "retained_docs": len(cred["evaluated_evidence"]), "discarded_docs": len(cred["discarded_evidence"]),
                "mrav_hypotheses": [],
                "dst": {k: full["dst_combined_masses"].get(k) for k in ("REAL", "FAKE", "UNCERTAIN", "conflict")},
                "final_verdict": fv["final_verdict"], "calibrated_confidence": fv["calibrated_confidence"],
                "is_scam": fv["is_scam"], "override_applied": fv["override_applied"],
                "override_reason": fv["override_reason"],
                # Without a CRITICAL flag, an LLM alignment text containing "scam" could have triggered the override
                "verdict_may_depend_on_llm_text": not any(f["severity"] == "CRITICAL" for f in flags),
            }
        except Exception as e:
            rec["errors"]["orchestrator"] = f"{type(e).__name__}: {e}"

        with open(pred_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        if n % 100 == 0 or n == len(df):
            print(f"[baseline] {n}/{len(df)} {(time.time() - t0) / n:.2f}s/row", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
