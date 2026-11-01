"""
XAI (Objective 5) tests for M4.

  - linear attribution identity: sum(contributions) + intercept == mean SVM decision function (1e-6)
  - token aggregation conserves the text contribution
  - Integrated Gradients completeness on a REVIEW-zone posting
  - deletion (faithfulness): deleting the top fraud-leaning words of high-confidence frauds
    lowers p_fake on average over >= 30 test frauds

Needs the model files and data/fake_job_postings.csv (skipped otherwise).
    python -m pytest tests/test_xai.py -v -s
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mravplus_infer as m4  # noqa: E402

FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "test_predictions_mravplus.csv"
N_DELETION = 30
RESULTS = PROJECT_ROOT / "results" / "xai_test_results.json"

if (m4.find_model_file(m4.BUNDLE_FILENAME) is None or m4.find_model_file(m4.STAGE2_FILENAME) is None
        or not m4.DATA_CSV.is_file() or not FIXTURE.is_file()):
    pytest.skip("model files, data CSV or parity fixture missing", allow_module_level=True)


def _record(key, value):
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.is_file() else {}
    data[key] = value
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.fixture(scope="module")
def det():
    return m4.M4Detector()


@pytest.fixture(scope="module")
def test_split():
    _, _, te = m4.load_emscad_splits()
    ref = pd.read_csv(FIXTURE)
    assert (ref.y.values == te.fraudulent.values).all()
    return te.drop(columns=["text"]), ref


def _fields(te, i):
    return te.iloc[i].to_dict()


def test_linear_identity_and_token_sum(det, test_split):
    from xai import linear_explainer
    te, _ = test_split
    lin = linear_explainer(det)
    errs, tok_errs = [], []
    for i in range(0, len(te), 100):  # 32 postings across the split
        row, _ = det.build_row("", _fields(te, i))
        e = lin.explain(row, field_labels=tuple(det.FIELDS))
        errs.append(e["identity_error"])
        tok_errs.append(e["token_sum_error"])
    _record("linear_identity", {"n": len(errs), "max_identity_error": max(errs), "max_token_sum_error": max(tok_errs)})
    assert max(errs) < 1e-6
    assert max(tok_errs) < 1e-6


def test_linear_free_text(det):
    from xai import linear_explainer
    row, _ = det.build_row("Earn $500 a day from home, no experience needed. Pay a small registration fee.")
    e = linear_explainer(det).explain(row, field_labels=tuple(det.FIELDS))
    assert e["identity_error"] < 1e-6
    assert e["top_fraud_terms"] or e["top_legit_terms"]


def test_integrated_gradients_completeness(det, test_split):
    from xai.ig import integrated_gradients
    te, ref = test_split
    review_rows = np.where(ref.zone.values == 1)[0][:3]
    out = []
    for i in review_rows:
        row, _ = det.build_row("", _fields(te, int(i)))
        ig = integrated_gradients(det, row, n_steps=50)
        out.append({"row": int(i), "f_input": ig["f_input"], "f_baseline": ig["f_baseline"],
                    "attribution_sum": ig["attribution_sum"], "residual": ig["completeness_residual"],
                    "relative": ig["completeness_relative"], "method": ig["method"]})
    _record("integrated_gradients", out)
    for o in out:
        assert o["relative"] <= 0.05, o


def test_deletion_lowers_p_fake(det, test_split):
    from xai import deletion_test
    te, ref = test_split
    # high-confidence frauds: true frauds that the Kaggle run put in the HIGH zone
    idx = np.where((ref.y.values == 1) & (ref.zone.values == 2))[0][:N_DELETION]
    assert len(idx) >= N_DELETION
    rows = []
    for i in idx:
        f = _fields(te, int(i))
        res = det.predict("", f)
        d = deletion_test(det, "", f, res, ks=(5, 10), seed=int(i))
        rows.append({r["k"]: (r["delta"], r["delta_random"]) for r in d["results"]})
    summary = {}
    for k in (5, 10):
        top = np.array([r[k][0] for r in rows])
        rnd = np.array([r[k][1] for r in rows])
        summary[f"k={k}"] = {
            "n": len(rows),
            "mean_delta_top": float(top.mean()), "median_delta_top": float(np.median(top)),
            "share_lowered_top": float(np.mean(top < 0)),
            "mean_delta_random": float(rnd.mean()), "median_delta_random": float(np.median(rnd)),
        }
    _record("deletion", summary)
    print(json.dumps(summary, indent=2))
    assert summary["k=5"]["mean_delta_top"] < 0
    assert summary["k=10"]["mean_delta_top"] < 0
