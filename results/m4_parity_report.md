# M4 parity report (Step 2)

Local `mravplus_infer.M4Detector` vs the Kaggle run's `test_predictions_mravplus.csv`, on the 3,151-posting EMSCAD test split rebuilt with seed 42 (section 3.1 of the brief).

- Command: `.venv-m4/Scripts/python tests/test_m4_parity.py` (same code as `pytest tests/test_m4_parity.py`)
- Environment: Windows 11, CPU only, Python 3.12.6, scikit-learn 1.6.1 (= bundle's `sklearn_version`), torch 2.15.0+cpu, transformers 4.57.6, sentence-transformers 5.7.0
- Run time: 216 s (MiniLM over 3,151 postings + DistilBERT on the 33 REVIEW rows)
- Full output: `results/m4_parity_metrics.json`

## Gate

| Requirement | Result | Pass |
|---|---|---|
| Label agreement with Kaggle `pred_M4` >= 99.5 % | **100 %** (0 of 3,151 disagree) | yes |
| `P_cheap` within ~1e-4 on non-REVIEW rows | max abs diff **6.4e-8** | yes |
| Fraud-F1 within ~0.01 of 0.887 | **0.8873** | yes |

## Checks before comparing

| Check | Result |
|---|---|
| split sizes train / val / test | 11,340 / 1,261 / 3,151 |
| rebuilt train labels == bundle `y_tr` (element-wise) | True |
| CSV `y` == rebuilt `te.fraudulent` (row by row) | True |
| rebuilt cleaned text == split `text` column | True |
| MiniLM re-embedding of 64 train postings vs bundle `E_tr` | min cosine 0.99999994 |

## Component agreement

| Quantity | Max abs diff vs Kaggle |
|---|---|
| `q` (M1 classifier) | 1.1e-16 |
| `P_cheap`, all rows | 6.7e-8 |
| zone (LOW/REVIEW/HIGH) agreement | 100 % (2,998 / 33 / 120 both) |
| `p2` (DistilBERT) on the 33 REVIEW rows | 0.0027 (CPU here vs GPU on Kaggle) |
| `P_final` on the 33 REVIEW rows | 0.0008 |

Escalation rate (REVIEW zone): 1.05 % (33 / 3,151).

## Test metrics (fraud = positive class)

| | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|
| Local M4 (this run) | 0.9173 | 0.8592 | 0.8873 | 0.9904 | 0.9255 | 0.00901 | 2998 | 11 | 20 | 122 |
| Kaggle CSV | 0.9173 | 0.8592 | 0.8873 | 0.9904 | 0.9255 | 0.00901 | 2998 | 11 | 20 | 122 |

Probability used for ROC/PR-AUC and Brier: `P_final` in the REVIEW zone, `P_cheap` elsewhere (as the notebook's `route()` reports it).

Caveat (also in the README): one run, one seed, 142 test frauds; exact duplicates removed but near-duplicates may remain.
