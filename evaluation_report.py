"""
evaluation_report.py — Write evaluation_results/PERFORMANCE_REPORT.md from the metrics files.

    python evaluation_report.py <after_out_dir> <baseline_out_dir> <report.md> [<previous_app_out_dir>]

Every number in the report is read from the predictions/metrics files produced by
evaluate_test_split.py and evaluation_baseline_original.py; nothing is typed in by hand
except the reference values of Paper 1.
"""

import json
import sys
from pathlib import Path

import numpy as np

PAPER_1 = {"precision": 0.968, "recall": 0.851, "f1_macro": 0.951}


def _load_rows(out_dir):
    with open(Path(out_dir) / "predictions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _bootstrap(y, pred, n=2000, seed=0):
    """95% percentile bootstrap intervals for precision/recall (fake) and macro-F1."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    stats = {"precision_fake": [], "recall_fake": [], "f1_macro": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        stats["precision_fake"].append(precision_score(y[i], pred[i], zero_division=0))
        stats["recall_fake"].append(recall_score(y[i], pred[i], zero_division=0))
        stats["f1_macro"].append(f1_score(y[i], pred[i], average="macro", zero_division=0))
    return {k: (round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)) for k, v in stats.items()}


def _systems(rows):
    """Binary predictions (1 = fake) for each evaluated system, from the raw records."""
    y = [r["true_label"] for r in rows]
    out = {}
    app = [r for r in rows if "app" in r]
    if len(app) == len(rows):
        out["App risk level: HIGH RISK = fake"] = [int(r["app"]["risk_level"] == "HIGH RISK") for r in rows]
        out["App risk level: HIGH or MEDIUM RISK = fake"] = [int(r["app"]["risk_level"] in ("HIGH RISK", "MEDIUM RISK")) for r in rows]
        out["Obj 1b nearest FAISS neighbour label"] = [
            int(sorted(r["app"]["evidence"], key=lambda e: e["distance"])[0]["fraudulent"] == 1) if r["app"]["evidence"] else 0
            for r in rows]
    if all(r.get("scam_probability") is not None for r in rows):
        out["Obj 1b text classifier alone (P(scam) >= 0.5)"] = [int(r["scam_probability"] >= 0.5) for r in rows]
    if all("evidence_dst" in r for r in rows):
        out["Obj 4 DST over evidence (BetP(fake) >= 0.5)"] = [
            int(r["evidence_dst"]["fake"] + r["evidence_dst"]["uncertain"] / 2 >= 0.5) for r in rows]
    if all("orchestrator" in r for r in rows):
        out["Obj 2+4 final verdict (is_scam)"] = [int(r["orchestrator"]["is_scam"]) for r in rows]
    return y, out


def _row(name, y, pred, ci=False):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    p = precision_score(y, pred, zero_division=0)
    r = recall_score(y, pred, zero_division=0)
    f = f1_score(y, pred, average="macro", zero_division=0)
    a = accuracy_score(y, pred)
    tp = sum(1 for t, q in zip(y, pred) if t == 1 and q == 1)
    fp = sum(1 for t, q in zip(y, pred) if t == 0 and q == 1)
    fn = sum(1 for t, q in zip(y, pred) if t == 1 and q == 0)
    cells = [name, f"{p:.3f}", f"{r:.3f}", f"{f:.3f}", f"{a:.3f}", f"{tp} / {fp} / {fn}"]
    if ci:
        b = _bootstrap(y, pred)
        cells.append(f"P {b['precision_fake'][0]}–{b['precision_fake'][1]}, R {b['recall_fake'][0]}–{b['recall_fake'][1]}, "
                     f"F1 {b['f1_macro'][0]}–{b['f1_macro'][1]}")
    return "| " + " | ".join(cells) + " |"


def main(after_dir, baseline_dir, report_path, previous_dir=None):
    after_rows = _load_rows(after_dir)
    base_rows = _load_rows(baseline_dir)
    after = json.load(open(Path(after_dir) / "performance_metrics.json", encoding="utf-8"))
    base = json.load(open(Path(baseline_dir) / "performance_metrics.json", encoding="utf-8"))
    y, after_sys = _systems(after_rows)
    yb, base_sys = _systems(base_rows)
    final = after_sys["Obj 2+4 final verdict (is_scam)"]

    from sklearn.metrics import f1_score, precision_score, recall_score
    fp_, fr_, ff_ = (precision_score(y, final), recall_score(y, final), f1_score(y, final, average="macro"))

    hdr = "| System | Precision (fake) | Recall (fake) | Macro-F1 | Accuracy | TP / FP / FN |"
    sep = "|---|---|---|---|---|---|"
    lines = [
        "# Performance Report — Fake Job Detection System (text pipeline)",
        "",
        f"Test data: {len(y)} postings sampled (stratified, seed 42) from `data/splits/test.csv` — "
        f"{y.count(0)} real, {y.count(1)} fraudulent. Positive class = fraudulent. "
        "OCR and Objective 3 (voice agent) are not part of this evaluation.",
        "",
        "## Headline: final system verdict vs Paper 1",
        "",
        "| | Precision | Recall | Macro-F1 |",
        "|---|---|---|---|",
        f"| Paper 1 (reported) | {PAPER_1['precision']:.3f} | {PAPER_1['recall']:.3f} | {PAPER_1['f1_macro']:.3f} |",
        f"| This project — Obj 2+4 final verdict | {fp_:.3f} | {fr_:.3f} | {ff_:.3f} |",
        "",
        "95% bootstrap intervals for this project are in the table below. Paper 1 does not state its test set, "
        "so the two rows are not measured on the same postings.",
        "",
        "## After the fixes (current code)",
        "",
        "The app's analysis (`pipeline_bridge.analyze_extracted_offer`) runs `run_full_pipeline`; its risk level is "
        "the final verdict mapped as HIGH_RISK_SCAM/SUSPICIOUS_FAKE → HIGH, UNCERTAIN_NEEDS_VERIFICATION → MEDIUM "
        "(call recommended), VERIFIED_REAL → LOW. No phone call is part of these numbers.",
        "",
        hdr + " 95% bootstrap CI |",
        sep + "---|",
    ]
    lines += [_row(n, y, p, ci=True) for n, p in after_sys.items()]
    if previous_dir:
        yp, prev_sys = _systems(_load_rows(previous_dir))
        lines += ["", "## Previous app version (fixed code, app risk from Run_Objective_2_Filter), same postings", "",
                  hdr, sep]
        lines += [_row(n, yp, p) for n, p in prev_sys.items() if n.startswith("App risk level")]
    lines += ["", "## Original code (before any change), same postings", "", hdr, sep]
    lines += [_row(n, yb, p) for n, p in base_sys.items()]
    mrav = after["obj2_app_filter"]["mrav_llm_verdict"]
    lines += [
        "",
        "## Other measurements",
        "",
        f"- Obj 1a entity extraction coverage (non-empty rate): {after['obj1a_entity_extraction']['field_non_empty_rate']}",
        f"- Obj 2+4 abstention (UNCERTAIN_NEEDS_VERIFICATION): "
        f"{after['obj2_obj4_final_verdict_without_voice']['abstention_rate_UNCERTAIN_NEEDS_VERIFICATION']}",
        f"- Obj 2+4 final verdict distribution: {after['obj2_obj4_final_verdict_without_voice']['final_verdict_distribution']}",
        f"- Obj 4 Platt calibration (P(real)): Brier {after['obj2_obj4_final_verdict_without_voice']['platt_calibration']['brier_score_p_real']}, "
        f"ECE {after['obj2_obj4_final_verdict_without_voice']['platt_calibration']['ece_p_real']}",
        f"- M-RAV LLM verdict (display text) available for {mrav['rows_with_llm_output']} of {len(y)} postings; "
        f"distribution {mrav['distribution']}",
        f"- Pipeline errors: {after['errors']}",
        "",
        "Full numbers: `performance_metrics.json` in each results folder; per-posting outputs: `predictions_flat.csv`.",
    ]
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:5])
