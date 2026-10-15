"""
evaluate_test_split.py — Evaluate the text pipeline (Objectives 1, 2 and 4) on EMSCAD splits.

OCR (image input) and Objective 3 (voice agent) are not exercised: the data is
text-only and the voice step needs a live phone call.

The Ollama-backed steps are run once and cached, so the deterministic steps can be
re-scored quickly:

    extract   Obj 1a  models.extract_entities_from_text                -> entities.jsonl
    llm       Obj 2   M-RAV verdict (engines.credibility_engine) and
                      per-document alignment triples (engines.alignment_chain) -> llm.jsonl
    score     Obj 1b  FAISS retrieval + scam_classifier
              Obj 2   app filter (pipeline_bridge.analyze_extracted_offer)
              Obj 4   DST over FAISS evidence + classifier (phone-verify path, no call)
              Obj 2+4 full_pipeline_orchestrator.run_full_pipeline (no voice)
                      -> <out_dir>/predictions.jsonl
    fit-platt         refit platt_model.pkl on the DST margins of a *validation* score run
    metrics           -> <out_dir>/performance_metrics.json, predictions_flat.csv

Usage:
    python evaluate_test_split.py extract   <csv> <entities.jsonl>
    python evaluate_test_split.py llm       <csv> <entities.jsonl> <llm.jsonl>
    python evaluate_test_split.py score     <csv> <entities.jsonl> <out_dir> [<llm.jsonl>]
    python evaluate_test_split.py fit-platt <val_out_dir>
    python evaluate_test_split.py metrics   <out_dir>
"""

import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scam_classifier import posting_to_text  # noqa: F401  (re-exported for eval_llm_stage.py)

ROOT = Path(__file__).parent


def _read_jsonl(path) -> dict:
    """job_id -> last record for that job_id."""
    out = {}
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["job_id"]] = r
    return out


def _append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _entities(rec: dict):
    from models import JobOfferEntities, clean_entities
    return clean_entities(JobOfferEntities(**rec["entities"]))


def _evidence_to_payload(offer_id: str, entities: dict, retrieved_evidence: list) -> dict:
    """Map the app's entities/evidence onto Objective2InputPayload for the orchestrator."""
    from voice_verification_bridge import distance_to_similarity

    bundle = {"legitimate_job_records": [], "registry_records": [], "linkedin_data": [], "scam_reports": []}
    for ev in retrieved_evidence:
        meta = ev.get("metadata", {}) or {}
        source_type = meta.get("source_type", "")
        doc = {
            "doc_id": f"ROW{meta.get('row_id', '')}",
            "content": ev.get("page_content", ""),
            # VectorDocument.similarity_score is "higher = closer"; FAISS returns squared L2 distance
            "similarity_score": distance_to_similarity(float(ev.get("distance", 2.0))),
            "metadata": meta,
        }
        key = {"legitimate_job_record": "legitimate_job_records", "scam_report": "scam_reports",
               "linkedin_verified": "linkedin_data"}.get(source_type, "registry_records")
        bundle[key].append(doc)

    return {
        "offer_id": offer_id,
        "extracted_entities": {
            "company_name": entities.get("company_name") or "",
            "job_role": entities.get("job_role") or "",
            "recruiter_name": entities.get("recruiter_name"),
            "domain_or_website": entities.get("domain_or_website"),
            "email_or_phone": entities.get("recruiter_email"),
            "salary_or_stipend": None,
            "location": entities.get("location"),
        },
        "retrieved_evidence": bundle,
    }


# ──────────────────────────────────────────────────────────
# extract / llm (Ollama)
# ──────────────────────────────────────────────────────────

def extract(csv_path, out_path):
    from models import extract_entities_from_text, OLLAMA_MODEL

    df = pd.read_csv(csv_path)
    done = _read_jsonl(out_path)
    todo = df[~df["job_id"].isin(done)]
    print(f"[extract] model={OLLAMA_MODEL} rows={len(df)} done={len(done)} todo={len(todo)}", flush=True)
    t_start = time.time()
    for n, (_, r) in enumerate(todo.iterrows(), 1):
        t0 = time.time()
        ent = extract_entities_from_text(posting_to_text(r.to_dict()))
        _append_jsonl(out_path, {"job_id": int(r["job_id"]), "model": OLLAMA_MODEL,
                                 "entities": ent.model_dump(), "seconds": round(time.time() - t0, 2)})
        if n % 10 == 0 or n == len(todo):
            rate = (time.time() - t_start) / n
            print(f"[extract] {len(done) + n}/{len(df)} {rate:.1f}s/row ETA {rate * (len(todo) - n) / 60:.0f} min", flush=True)


def llm(csv_path, entities_path, out_path):
    """Run the M-RAV LLM steps on exactly the evidence the score stage uses."""
    from engines.alignment_chain import generate_alignment_triple
    from engines.credibility_engine import OLLAMA_MODEL
    from pipeline_bridge import analyze_extracted_offer
    from schemas.input_schema import Objective2InputPayload
    from search_engine import EvidenceRetriever

    df = pd.read_csv(csv_path)
    ents = _read_jsonl(entities_path)
    done = _read_jsonl(out_path)
    todo = df[~df["job_id"].isin(done)]
    print(f"[llm] model={OLLAMA_MODEL} rows={len(df)} done={len(done)} todo={len(todo)}", flush=True)
    retriever = EvidenceRetriever(index_folder=str(ROOT / "faiss_index"))
    t_start = time.time()
    for n, (_, r) in enumerate(todo.iterrows(), 1):
        job_id = int(r["job_id"])
        t0 = time.time()
        rec = {"job_id": job_id, "model": OLLAMA_MODEL, "errors": []}
        try:
            entities = _entities(ents[job_id])
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                ui = analyze_extracted_offer(posting_to_text(r.to_dict()), entities, retriever)
                rec["mrav"] = ui["explanation"]
                payload = Objective2InputPayload(**_evidence_to_payload(f"TEST_{job_id}", ui["extracted_entities"],
                                                                        ui["retrieved_evidence"]))
                ev = payload.retrieved_evidence
                rec["align"] = {
                    d.doc_id: generate_alignment_triple(payload.extracted_entities, d).model_dump()
                    for d in ev.legitimate_job_records + ev.registry_records + ev.linkedin_data + ev.scam_reports
                }
            rec["errors"] = [ln for ln in log.getvalue().splitlines()
                             if re.search(r"\[Error\]|\[Warning\] LLM|M-RAV evaluation error|\[KeyPool\] Error", ln)]
        except Exception as e:
            rec["errors"].append(f"{type(e).__name__}: {e}")
        rec["seconds"] = round(time.time() - t0, 2)
        _append_jsonl(out_path, rec)
        if n % 10 == 0 or n == len(todo):
            rate = (time.time() - t_start) / n
            print(f"[llm] {len(done) + n}/{len(df)} {rate:.1f}s/row ETA {rate * (len(todo) - n) / 60:.0f} min", flush=True)


# ──────────────────────────────────────────────────────────
# score (deterministic; LLM outputs come from the caches)
# ──────────────────────────────────────────────────────────

def score(csv_path, entities_path, out_dir, llm_path=None):
    import credibility_engine as root_credibility
    import engines.credibility_engine as app_credibility
    from full_pipeline_orchestrator import run_full_pipeline, CLASSIFIER_RELIABILITY
    from obj2dst import DSTEngine
    from pipeline_bridge import analyze_extracted_offer
    from schemas.credibility_schema import MRAVTriple
    from search_engine import EvidenceRetriever
    from voice_verification_bridge import distance_to_similarity, _combine_masses

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.jsonl"
    if pred_file.exists():
        pred_file.unlink()

    df = pd.read_csv(csv_path)
    ents = _read_jsonl(entities_path)
    cache = _read_jsonl(llm_path)
    current = {"job_id": None}

    # Serve the LLM steps from the cache. Neither output changes a score or verdict
    # (M-RAV text is display-only; alignment hypotheses only add WARNING flags).
    def cached_mrav(entities, retrieved_evidence):
        c = cache.get(current["job_id"])
        return {"mrav_analysis": c["mrav"] if c and "mrav" in c else "NOT_RUN"}

    def cached_alignment(entities, doc):
        c = cache.get(current["job_id"])
        if c and doc.doc_id in c.get("align", {}):
            return MRAVTriple(**c["align"][doc.doc_id])
        return MRAVTriple(hypothesis="Neutral", explanation="NOT_RUN", consistency_score=0.5)

    app_credibility.evaluate_mrav_consistency = cached_mrav
    root_credibility.generate_alignment_triple = cached_alignment

    retriever = EvidenceRetriever(index_folder=str(ROOT / "faiss_index"))
    dst = DSTEngine()
    t_start = time.time()
    for n, r in enumerate(df.to_dict("records"), 1):
        job_id = int(r["job_id"])
        current["job_id"] = job_id
        rec = {"job_id": job_id, "true_label": int(r["fraudulent"]), "errors": {},
               "llm_cached": job_id in cache}
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                ui = analyze_extracted_offer(posting_to_text(r), _entities(ents[job_id]), retriever)
        except Exception as e:
            rec["errors"]["app"] = f"{type(e).__name__}: {e}"
            _append_jsonl(pred_file, rec)
            continue

        rec["scam_probability"] = ui["scam_probability"]
        rec["app"] = {
            "entities": ui["extracted_entities"],
            "trust_score": ui["trust_score"],
            "risk_level": ui["risk_level"],
            "flags": ui["flags"],
            "mrav_explanation": ui["explanation"],
            "evidence": [{"row_id": ev["metadata"].get("row_id"), "fraudulent": ev["metadata"].get("fraudulent"),
                          "distance": ev["distance"]} for ev in ui["retrieved_evidence"]],
        }

        # FAISS evidence + classifier -> DST, as verify_offer_by_phone builds it (voice step omitted)
        try:
            masses = []
            for ev in ui["retrieved_evidence"]:
                meta = ev.get("metadata", {}) or {}
                st = meta.get("source_type", "unknown")
                masses.append(dst.assign_mass_function(
                    similarity_score=distance_to_similarity(float(ev.get("distance", 2.0))),
                    source_type=st,
                    is_fraudulent=1 if meta.get("fraudulent") == 1 or st == "scam_report" else 0,
                ))
            if ui["scam_probability"] is not None:
                masses.append(dst.classifier_mass(ui["scam_probability"], CLASSIFIER_RELIABILITY))
            bridge = {}
            with contextlib.redirect_stdout(log):
                _combine_masses(dst, masses, bridge)
            comb = bridge.get("dst_combined") or {}
            rec["evidence_dst"] = {"verdict": bridge["final_verdict"], "real": comb.get("REAL", 0.0),
                                   "fake": comb.get("FAKE", 0.0), "uncertain": comb.get("UNCERTAIN", 1.0),
                                   "conflict": comb.get("conflict", 0.0)}
        except Exception as e:
            rec["errors"]["evidence_dst"] = f"{type(e).__name__}: {e}"

        # Full orchestrator: Obj 2 evaluate_credibility -> Obj 4 DST + Platt + override (no voice)
        try:
            payload = _evidence_to_payload(f"TEST_{job_id}", ui["extracted_entities"], ui["retrieved_evidence"])
            with contextlib.redirect_stdout(log):
                full = run_full_pipeline(payload, None, scam_probability=ui["scam_probability"])
            cred = full["obj2_credibility"]
            fv = full["final_verification"]
            rec["orchestrator"] = {
                "overall_trust_score": cred["credibility_evaluation"]["overall_trust_score"],
                "routing_decision": cred["routing_decision"],
                "flags": [f"{f['severity']}: {f['message']}" for f in cred["credibility_evaluation"]["flags_triggered"]],
                "retained_docs": len(cred["evaluated_evidence"]),
                "discarded_docs": len(cred["discarded_evidence"]),
                "mrav_hypotheses": [d["mrav_triple"]["hypothesis"] for d in cred["evaluated_evidence"] + cred["discarded_evidence"]
                                    if d["mrav_triple"]["explanation"] != "NOT_RUN"],
                "dst": {k: full["dst_combined_masses"].get(k) for k in ("REAL", "FAKE", "UNCERTAIN", "conflict")},
                "final_verdict": fv["final_verdict"],
                "calibrated_confidence": fv["calibrated_confidence"],
                "is_scam": fv["is_scam"],
                "override_applied": fv["override_applied"],
                "override_reason": fv["override_reason"],
            }
        except Exception as e:
            rec["errors"]["orchestrator"] = f"{type(e).__name__}: {e}"

        _append_jsonl(pred_file, rec)
        if n % 100 == 0 or n == len(df):
            print(f"[score] {n}/{len(df)} {(time.time() - t_start) / n:.2f}s/row", flush=True)


def fit_platt(val_out_dir):
    """Refit Obj 4 Platt scaling on validation DST margins (labels: 1 = real, 0 = fake)."""
    from confidence_calibrator import fit_and_save_platt_model, MODEL_FILE_PATH

    recs = [r for r in _read_jsonl(Path(val_out_dir) / "predictions.jsonl").values() if "orchestrator" in r]
    s = np.array([r["orchestrator"]["dst"]["REAL"] - r["orchestrator"]["dst"]["FAKE"] for r in recs])
    y = np.array([1 - r["true_label"] for r in recs])
    scaler = fit_and_save_platt_model(MODEL_FILE_PATH, s, y)
    lr = scaler.model
    print(f"[fit-platt] fitted on {len(s)} validation offers ({int((y == 0).sum())} fake): "
          f"coef={lr.coef_[0][0]:.4f} intercept={lr.intercept_[0]:.4f} -> {MODEL_FILE_PATH}")


# ──────────────────────────────────────────────────────────
# metrics
# ──────────────────────────────────────────────────────────

def _binary_metrics(y_true, y_pred, y_score=None):
    """Fake (fraudulent=1) is the positive class."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                 matthews_corrcoef, precision_score, recall_score, roc_auc_score,
                                 average_precision_score)
    if not len(y_true):
        return None
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, zero_division=0),
        "precision_real": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_real": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_real": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "confusion_matrix": {"tn_real_as_real": int(tn), "fp_real_as_fake": int(fp),
                             "fn_fake_as_real": int(fn), "tp_fake_as_fake": int(tp)},
    }
    if y_score is not None and len(set(y_true)) == 2:
        m["roc_auc"] = roc_auc_score(y_true, y_score)
        m["pr_auc"] = average_precision_score(y_true, y_score)
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


def _calibration(y_true_real, p_real, bins=10):
    """Brier score and expected calibration error for P(Real)."""
    y = np.asarray(y_true_real, dtype=float)
    p = np.asarray(p_real, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.sum():
            ece += mask.sum() / len(p) * abs(p[mask].mean() - y[mask].mean())
            table.append({"bin": f"[{lo:.1f},{hi:.1f}{']' if hi == 1 else ')'}", "count": int(mask.sum()),
                          "mean_predicted_p_real": round(float(p[mask].mean()), 4),
                          "observed_fraction_real": round(float(y[mask].mean()), 4)})
    return {"brier_score_p_real": round(float(np.mean((p - y) ** 2)), 4), "ece_p_real": round(float(ece), 4),
            "reliability_bins": table}


def _mrav_verdict(text):
    m = re.search(r"VERDICT\s*:\s*[\[\*\s]*(LEGITIMATE|SUSPICIOUS|FRAUDULENT)", text or "", re.IGNORECASE)
    return m.group(1).upper() if m else None


def _dist(values, labels):
    from collections import Counter
    return {"real": dict(Counter(v for v, y in zip(values, labels) if y == 0)),
            "fake": dict(Counter(v for v, y in zip(values, labels) if y == 1))}


def metrics(out_dir):
    from collections import Counter

    out_dir = Path(out_dir)
    rows = list(_read_jsonl(out_dir / "predictions.jsonl").values())
    y_all = [r["true_label"] for r in rows]
    report = {
        "dataset": {"rows_evaluated": len(rows),
                    "label_counts": {"real_0": y_all.count(0), "fake_1": y_all.count(1)},
                    "positive_class": "fake (fraudulent=1)"},
        "errors": {k: sum(k in r["errors"] for r in rows) for k in ("app", "evidence_dst", "orchestrator")},
        "llm_outputs_cached_rows": sum(r.get("llm_cached", False) for r in rows),
    }

    # Obj 1a — entity extraction coverage
    app = [r for r in rows if "app" in r]
    fields = ["company_name", "job_role", "recruiter_name", "recruiter_email", "domain_or_website", "location"]
    filled = lambda r, f: bool((r["app"]["entities"].get(f) or "").strip())
    report["obj1a_entity_extraction"] = {
        "note": "EMSCAD has no gold entity annotations, so extraction coverage is reported, not accuracy.",
        "field_non_empty_rate": {f: round(sum(filled(r, f) for r in app) / max(len(app), 1), 4) for f in fields},
    }

    # Obj 1b — retrieval and classifier as standalone signals
    y = [r["true_label"] for r in app]
    top1 = [int(sorted(r["app"]["evidence"], key=lambda e: e["distance"])[0]["fraudulent"] == 1)
            if r["app"]["evidence"] else 0 for r in app]
    p = [r.get("scam_probability") for r in app]
    report["obj1b_retrieval_top1_neighbour"] = _binary_metrics(y, top1)
    if p and all(v is not None for v in p):
        report["obj1b_text_classifier_p_ge_0.5"] = _binary_metrics(y, [int(v >= 0.5) for v in p], p)

    # Obj 2 — app filter (what the Streamlit UI shows)
    risk = [r["app"]["risk_level"] for r in app]
    fake_score = [(100 - r["app"]["trust_score"]) / 100 for r in app]
    mrav = [(r["true_label"], _mrav_verdict(r["app"]["mrav_explanation"])) for r in app
            if r["app"]["mrav_explanation"] != "NOT_RUN"]
    mrav_ok = [(yy, v) for yy, v in mrav if v]
    report["obj2_app_filter"] = {
        "risk_level_distribution": _dist(risk, y),
        "fake_if_HIGH_RISK": _binary_metrics(y, [int(v == "HIGH RISK") for v in risk], fake_score),
        "fake_if_HIGH_or_MEDIUM_RISK": _binary_metrics(y, [int(v in ("HIGH RISK", "MEDIUM RISK")) for v in risk], fake_score),
        "flag_frequency": dict(Counter(re.sub(r"\(P\(scam\)=[\d.]+\)", "", f).strip() for r in app for f in r["app"]["flags"])),
        "mrav_llm_verdict": {
            "note": "VERDICT line of the M-RAV explanation (display-only, does not change the trust score).",
            "rows_with_llm_output": len(mrav), "unparseable": len(mrav) - len(mrav_ok),
            "distribution": _dist([v for _, v in mrav_ok], [yy for yy, _ in mrav_ok]),
            "fake_if_SUSPICIOUS_or_FRAUDULENT": _binary_metrics([yy for yy, _ in mrav_ok], [int(v != "LEGITIMATE") for _, v in mrav_ok]),
            "fake_if_FRAUDULENT": _binary_metrics([yy for yy, _ in mrav_ok], [int(v == "FRAUDULENT") for _, v in mrav_ok]),
        },
    }

    # Obj 4 — DST over FAISS evidence + classifier (phone-verify path without the call)
    ed = [r for r in rows if "evidence_dst" in r]
    y = [r["true_label"] for r in ed]
    v = [r["evidence_dst"]["verdict"] for r in ed]
    betp_fake = [r["evidence_dst"]["fake"] + r["evidence_dst"]["uncertain"] / 2 for r in ed]
    decided = [(yy, vv) for yy, vv in zip(y, v) if not vv.startswith("UNCERTAIN")]
    report["obj4_evidence_dst_without_voice"] = {
        "verdict_distribution": _dist(v, y),
        "abstention_rate": round(1 - len(decided) / max(len(ed), 1), 4),
        "fake_if_BetP_fake_ge_0.5_all_rows": _binary_metrics(y, [int(b >= 0.5) for b in betp_fake], betp_fake),
        "on_decided_only": _binary_metrics([yy for yy, _ in decided], [int(vv.endswith("Fake")) for _, vv in decided]),
    }

    # Obj 2 + Obj 4 — full orchestrator (final system verdict)
    orc = [r for r in rows if "orchestrator" in r]
    y = [r["true_label"] for r in orc]
    fv = [r["orchestrator"]["final_verdict"] for r in orc]
    p_real = [r["orchestrator"]["calibrated_confidence"] for r in orc]
    decided = [(yy, vv) for yy, vv in zip(y, fv) if vv != "UNCERTAIN_NEEDS_VERIFICATION"]
    report["obj2_obj4_final_verdict_without_voice"] = {
        "routing_distribution": _dist([r["orchestrator"]["routing_decision"] for r in orc], y),
        "final_verdict_distribution": _dist(fv, y),
        "override_applied_rate": round(sum(r["orchestrator"]["override_applied"] for r in orc) / max(len(orc), 1), 4),
        "is_scam_all_rows": _binary_metrics(y, [int(r["orchestrator"]["is_scam"]) for r in orc], [1 - q for q in p_real]),
        "abstention_rate_UNCERTAIN_NEEDS_VERIFICATION": round(1 - len(decided) / max(len(orc), 1), 4),
        "on_decided_only": _binary_metrics([yy for yy, _ in decided],
                                           [int(vv in ("HIGH_RISK_SCAM", "SUSPICIOUS_FAKE")) for _, vv in decided]),
        "platt_calibration": _calibration([1 - yy for yy in y], p_real) if orc else None,
        "mrav_alignment_hypotheses": dict(Counter(h for r in orc for h in r["orchestrator"]["mrav_hypotheses"])),
    }

    with open(out_dir / "performance_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    flat = [{
        "job_id": r["job_id"], "true_label": r["true_label"], "scam_probability": r.get("scam_probability"),
        "company_name": r.get("app", {}).get("entities", {}).get("company_name"),
        "job_role": r.get("app", {}).get("entities", {}).get("job_role"),
        "app_trust_score": r.get("app", {}).get("trust_score"), "app_risk_level": r.get("app", {}).get("risk_level"),
        "app_mrav_verdict": _mrav_verdict(r.get("app", {}).get("mrav_explanation")),
        "evidence_dst_verdict": r.get("evidence_dst", {}).get("verdict"),
        "orch_trust": r.get("orchestrator", {}).get("overall_trust_score"),
        "orch_routing": r.get("orchestrator", {}).get("routing_decision"),
        "orch_final_verdict": r.get("orchestrator", {}).get("final_verdict"),
        "orch_p_real": r.get("orchestrator", {}).get("calibrated_confidence"),
        "orch_override": r.get("orchestrator", {}).get("override_applied"),
        "errors": ";".join(r["errors"]),
    } for r in rows]
    pd.DataFrame(flat).to_csv(out_dir / "predictions_flat.csv", index=False)
    return report


if __name__ == "__main__":
    cmds = {"extract": extract, "llm": llm, "score": score, "fit-platt": fit_platt, "metrics": metrics}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    result = cmds[sys.argv[1]](*sys.argv[2:])
    if sys.argv[1] == "metrics":
        print(json.dumps(result, indent=2))
