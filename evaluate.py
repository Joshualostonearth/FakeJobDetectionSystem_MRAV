"""
evaluate.py — Full Evaluation Pipeline (Steps 1 through 8).

Runs end-to-end evaluation on test_dataset.csv, computes F1-micro, F1-macro,
Precision, Recall, Abstention Rate, generates confusion_matrix.png, and performs
Platt Calibration sanity-checking.
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from full_pipeline_orchestrator import run_full_pipeline

CSV_FILE_PATH = Path(__file__).parent / "test_dataset.csv"
CONFUSION_MATRIX_PATH = Path(__file__).parent / "confusion_matrix.png"


# ──────────────────────────────────────────────────────────
# Step 1 — verify_offer() wrapper function
# ──────────────────────────────────────────────────────────

def verify_offer(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 1: End-to-end pipeline call function for a single offer.

    Args:
        row: Dict from test_dataset.csv

    Returns:
        Dict with keys:
            verdict:               'real' | 'fake' | 'unable_to_verify'
            calibrated_confidence: float
            conflict_k:            float
            final_verdict_raw:     str
    """
    raw_payload = {
        "offer_id": row["offer_id"],
        "extracted_entities": {
            "company_name": row["company_name"],
            "job_role": row["job_role"],
            "recruiter_name": "Test Recruiter",
            "domain_or_website": row["domain"],
            "email_or_phone": row["recruiter_email"],
            "salary_or_stipend": "$120,000/year",
            "location": row["location"],
        },
        "retrieved_evidence": {
            "registry_records": [
                {
                    "doc_id": f"REC_{row['offer_id']}",
                    "content": row["doc_content"],
                    "similarity_score": float(row["similarity_score"]),
                    "metadata": {
                        "source_type": row["source_type"],
                        "company_name": row["company_name"],
                        "domain": row["domain"],
                        "credibility_tier": "official" if row["source_type"] == "registry" else "high",
                        "last_updated": "2026-06-01",
                        "fraudulent": int(row["is_fraudulent"]),
                        "source_url": f"https://{row['domain']}/careers",
                    },
                }
            ],
            "linkedin_data": [],
            "scam_reports": [],
        },
    }

    voice_signals = None
    if row.get("voice_denied") == "True" or row.get("voice_suspicious") == "True":
        voice_signals = {
            "company_reached": True,
            "company_denied_role": row.get("voice_denied") == "True",
            "suspicious_response": row.get("voice_suspicious") == "True",
            "source_confidence": 0.90,
            "extraction_confidence": 0.85,
        }

    pipeline_result = run_full_pipeline(raw_payload, voice_signals)
    verdict_raw = pipeline_result["final_verification"]["final_verdict"]
    cal_conf = pipeline_result["final_verification"]["calibrated_confidence"]
    conflict_k = pipeline_result["final_verification"]["conflict_factor_k"]

    # Map raw verdict string to evaluation taxonomy: 'real', 'fake', 'unable_to_verify'
    if verdict_raw == "VERIFIED_REAL":
        verdict = "real"
    elif verdict_raw in ("HIGH_RISK_SCAM", "SUSPICIOUS_FAKE"):
        verdict = "fake"
    else:
        verdict = "unable_to_verify"

    return {
        "verdict": verdict,
        "calibrated_confidence": cal_conf,
        "conflict_k": conflict_k,
        "final_verdict_raw": verdict_raw,
    }


# ──────────────────────────────────────────────────────────
# Main Evaluation Loop (Steps 2 through 8)
# ──────────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 70)
    print("RUNNING PIPELINE EVALUATION SUITE (STEPS 1 TO 8)")
    print("=" * 70)

    # ── Step 2: Load labeled test dataset ─────────────────────────────
    if not CSV_FILE_PATH.exists():
        from generate_test_dataset import generate_csv
        generate_csv()

    test_rows = []
    with open(CSV_FILE_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            test_rows.append(r)

    print(f"\n[Step 2] Loaded {len(test_rows)} labeled test offers from {CSV_FILE_PATH.name}")

    # ── Step 3: Loop through test dataset & call verify_offer() ───────
    y_true = []
    predictions = []
    probabilities = []
    conflicts = []

    print("\n[Step 3] Running predictions on test offers...")
    for i, r in enumerate(test_rows, 1):
        true_lbl = int(r["true_label"])
        y_true.append(true_lbl)

        res = verify_offer(r)
        pred_v = res["verdict"]
        predictions.append(pred_v)
        probabilities.append(res["calibrated_confidence"])
        conflicts.append(res["conflict_k"])

        print(
            f"  [{i:02d}/{len(test_rows)}] Offer: {r['offer_id']:<15} | "
            f"True: {'REAL' if true_lbl==1 else 'FAKE'} | "
            f"Predicted: {pred_v.upper():<18} | "
            f"Conf: {res['calibrated_confidence']:.2%} | "
            f"K: {res['conflict_k']:.4f}"
        )

    y_true = np.array(y_true)

    # ── Step 4: Handle 'Unable to Verify' & Abstention Rate ───────────
    confident_mask = [p != "unable_to_verify" for p in predictions]
    abstention_rate = 1.0 - (sum(confident_mask) / len(predictions))

    y_true_confident = y_true[confident_mask]
    preds_confident_str = [p for p, m in zip(predictions, confident_mask) if m]
    preds_confident = [1 if p == "real" else 0 for p in preds_confident_str]

    print("-" * 70)
    print(f"[Step 4] ABSTENTION ANALYSIS:")
    print(f"  Total Test Offers     : {len(predictions)}")
    print(f"  Confident Predictions : {sum(confident_mask)}")
    print(f"  Unable to Verify      : {len(predictions) - sum(confident_mask)}")
    print(f"  Abstention Rate       : {abstention_rate:.2%}")
    print("-" * 70)

    # ── Step 5: Compute F1-micro, F1-macro, Precision, Recall ─────────
    f1_micro = f1_score(y_true_confident, preds_confident, average="micro")
    f1_macro = f1_score(y_true_confident, preds_confident, average="macro")
    precision = precision_score(y_true_confident, preds_confident)
    recall = recall_score(y_true_confident, preds_confident)

    print("\n[Step 5] PERFORMANCE METRICS (on confident predictions):")
    print(f"  F1-Micro  : {f1_micro:.4f}")
    print(f"  F1-Macro  : {f1_macro:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")

    # ── Step 6: Generate Confusion Matrix Plot ────────────────────────
    cm = confusion_matrix(y_true_confident, preds_confident)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake (0)", "Real (1)"],
        yticklabels=["Fake (0)", "Real (1)"],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("Fake Job Detection Pipeline — Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    print(f"\n[Step 6] Confusion matrix heatmap saved to: {CONFUSION_MATRIX_PATH.name}")

    # ── Step 7: Report complete summary metrics ───────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY REPORT (M-RAV BENCHMARK FORMAT)")
    print("=" * 70)
    print(f"Total Test Set Size    : {len(predictions)}")
    print(f"Abstention Rate        : {abstention_rate:.2%}")
    print(f"Evaluated (Confident)  : {len(preds_confident)}")
    print(f"F1-Micro Score         : {f1_micro:.4f}")
    print(f"F1-Macro Score         : {f1_macro:.4f}")
    print(f"Precision (Real)       : {precision:.4f}")
    print(f"Recall (Real)          : {recall:.4f}")
    print("-" * 70)
    print("CONFUSION MATRIX BREAKDOWN:")
    print(f"  True Negatives (Fake -> Fake) : {cm[0][0]}")
    print(f"  False Positives (Fake -> Real): {cm[0][1]}")
    print(f"  False Negatives (Real -> Fake): {cm[1][0]}")
    print(f"  True Positives (Real -> Real) : {cm[1][1]}")
    print("=" * 70)

    # ── Step 8: Platt Calibration Sanity-Check ───────────────────────
    print("\n[Step 8] PLATT CALIBRATION SANITY CHECK:")
    high_conf_indices = [i for i, p in enumerate(probabilities) if p >= 0.80]
    if high_conf_indices:
        high_conf_true = y_true[high_conf_indices]
        high_conf_preds = [1 if predictions[i] == "real" else 0 for i in high_conf_indices]
        high_conf_acc = np.mean(high_conf_true == high_conf_preds)
        avg_predicted_prob = np.mean([probabilities[i] for i in high_conf_indices])
        print(f"  High Confidence Bins (prob >= 80%):")
        print(f"    Count                  : {len(high_conf_indices)}")
        print(f"    Avg Predicted Prob P   : {avg_predicted_prob:.2%}")
        print(f"    Empirical Accuracy     : {high_conf_acc:.2%}")
        print(f"    Calibration Error |P-A|: {abs(avg_predicted_prob - high_conf_acc):.4f}")
    else:
        print("  No offers in >=80% confidence bin.")

    low_conf_indices = [i for i, p in enumerate(probabilities) if p <= 0.25]
    if low_conf_indices:
        low_conf_true = y_true[low_conf_indices]
        low_conf_preds = [1 if predictions[i] == "real" else 0 for i in low_conf_indices]
        low_conf_acc = np.mean(low_conf_true == low_conf_preds)
        avg_pred_fake_prob = np.mean([1.0 - probabilities[i] for i in low_conf_indices])
        print(f"\n  Low Confidence / Scam Bins (prob <= 25%):")
        print(f"    Count                  : {len(low_conf_indices)}")
        print(f"    Avg Predicted Scam Prob: {avg_pred_fake_prob:.2%}")
        print(f"    Empirical Accuracy     : {low_conf_acc:.2%}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    run_evaluation()
