"""
Parity test: M4Detector on the EMSCAD test split vs the Kaggle run's predictions.

Gate (from the M4 integration brief, Step 2):
  - label agreement with test_predictions_mravplus.csv >= 99.5 %
  - P_cheap within ~1e-4 on non-REVIEW rows
  - fraud-F1 within ~0.01 of 0.887

Needs models/mravplus_bundle.joblib, models/stage2.pt and data/fake_job_postings.csv
(skipped otherwise). About 3-6 minutes on CPU (MiniLM over 3,151 postings + DistilBERT on
the ~1 % REVIEW rows).

    python -m pytest tests/test_m4_parity.py -v -s
    python tests/test_m4_parity.py          # also writes results/m4_parity_metrics.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mravplus_infer as m4  # noqa: E402

FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "test_predictions_mravplus.csv"
KAGGLE_F1 = 0.8873


def _skip_reason():
    if m4.find_model_file(m4.BUNDLE_FILENAME) is None:
        return "models/mravplus_bundle.joblib missing"
    if m4.find_model_file(m4.STAGE2_FILENAME) is None:
        return "models/stage2.pt missing"
    if not m4.DATA_CSV.is_file():
        return "data/fake_job_postings.csv missing"
    if not FIXTURE.is_file():
        return "tests/fixtures/test_predictions_mravplus.csv missing"
    return None


def run_parity():
    import pandas as pd
    from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)

    t0 = time.time()
    det = m4.M4Detector()
    tr, va, te = m4.load_emscad_splits()
    ref = pd.read_csv(FIXTURE)

    checks = {
        "split_sizes": [len(tr), len(va), len(te)],
        "train_matches_bundle": bool(len(tr) == len(det.y_tr) and (tr.fraudulent.values == det.y_tr).all()),
        "fixture_rows": len(ref),
        "fixture_y_matches_test_labels": bool(len(ref) == len(te) and (ref.y.values == te.fraudulent.values).all()),
    }
    if not (checks["train_matches_bundle"] and checks["fixture_y_matches_test_labels"]):
        return {"checks": checks}

    # MiniLM reproduces the bundle's training embeddings?
    e = det.embed(tr.text.iloc[:64].tolist())
    checks["E_tr_reembed_min_cos"] = float(np.min(np.sum(e * det.E_tr[:64], axis=1)))

    rows = []
    for rec in te.drop(columns=["text"]).to_dict("records"):
        row, _ = det.build_row("", rec)
        rows.append(row)
    # The rebuilt cleaned text must equal the split's 'text' column
    checks["cleaned_text_matches"] = bool(all(r["text"] == t for r, t in zip(rows, te.text)))

    out = det.score_rows(rows)
    t_score = time.time() - t0

    q = np.array([o["q"] for o in out])
    p_cheap = np.array([o["p_cheap"] for o in out])
    zone = np.array([o["zone"] for o in out])
    p_fake = np.array([o["p_fake"] for o in out])
    pred = np.array([o["pred"] for o in out])
    y = te.fraudulent.values

    nonrev = ref.zone.values != 1
    review = ref.zone.values == 1
    p2_mine = np.array([o["p2"] if o["p2"] is not None else np.nan for o in out])
    both_rev = review & (zone == 1)

    # Kaggle reported probability: P_final in REVIEW, P_cheap elsewhere (as in the notebook's route())
    ref_prob = np.where(ref.zone.values == 1, ref.P_final.values, ref.P_cheap.values)

    def metrics(yhat, prob):
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        return {
            "precision": float(precision_score(y, yhat)), "recall": float(recall_score(y, yhat)),
            "f1": float(f1_score(y, yhat)), "roc_auc": float(roc_auc_score(y, prob)),
            "pr_auc": float(average_precision_score(y, prob)), "brier": float(brier_score_loss(y, prob)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

    disagree = np.where(pred != ref.pred_M4.values)[0]
    return {
        "checks": checks,
        "seconds": round(t_score, 1),
        "device": det.device,
        "label_agreement": float(np.mean(pred == ref.pred_M4.values)),
        "n_label_disagreements": int(len(disagree)),
        "disagreements": [
            {"row": int(i), "y": int(y[i]), "kaggle_zone": int(ref.zone[i]), "my_zone": int(zone[i]),
             "kaggle_P_final": float(ref.P_final[i]), "my_p_fake": float(p_fake[i]),
             "kaggle_p2": float(ref.p2[i]), "my_p2": None if np.isnan(p2_mine[i]) else float(p2_mine[i])}
            for i in disagree
        ],
        "zone_agreement": float(np.mean(zone == ref.zone.values)),
        "zone_counts_mine": {m4.ZONE_NAMES[k]: int(np.sum(zone == k)) for k in (0, 1, 2)},
        "zone_counts_kaggle": {m4.ZONE_NAMES[k]: int(np.sum(ref.zone.values == k)) for k in (0, 1, 2)},
        "q_max_abs_diff": float(np.max(np.abs(q - ref.q.values))),
        "p_cheap_max_abs_diff_nonreview": float(np.max(np.abs(p_cheap[nonrev] - ref.P_cheap.values[nonrev]))),
        "p_cheap_max_abs_diff_all": float(np.max(np.abs(p_cheap - ref.P_cheap.values))),
        "p2_max_abs_diff_review": (float(np.nanmax(np.abs(p2_mine[both_rev] - ref.p2.values[both_rev])))
                                   if both_rev.any() else None),
        "p_final_max_abs_diff_review": (float(np.max(np.abs(p_fake[both_rev] - ref.P_final.values[both_rev])))
                                        if both_rev.any() else None),
        "escalation_rate": float(np.mean(zone == 1)),
        "metrics_mine": metrics(pred, p_fake),
        "metrics_kaggle_csv": metrics(ref.pred_M4.values, ref_prob),
    }


@pytest.fixture(scope="module")
def parity():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    res = run_parity()
    print("\n" + json.dumps({k: v for k, v in res.items() if k != "disagreements"}, indent=2))
    return res


def test_data_and_ordering(parity):
    c = parity["checks"]
    assert c["split_sizes"] == [11340, 1261, 3151]
    assert c["train_matches_bundle"]
    assert c["fixture_y_matches_test_labels"]
    assert c["cleaned_text_matches"]
    assert c["E_tr_reembed_min_cos"] > 0.999


def test_label_agreement(parity):
    assert parity["label_agreement"] >= 0.995


def test_p_cheap_matches_outside_review(parity):
    assert parity["p_cheap_max_abs_diff_nonreview"] < 1e-4


def test_fraud_f1_close_to_kaggle(parity):
    assert abs(parity["metrics_mine"]["f1"] - KAGGLE_F1) <= 0.01


if __name__ == "__main__":
    reason = _skip_reason()
    if reason:
        sys.exit(f"Cannot run parity: {reason}")
    res = run_parity()
    out = PROJECT_ROOT / "results" / "m4_parity_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"\nwritten {out}")
