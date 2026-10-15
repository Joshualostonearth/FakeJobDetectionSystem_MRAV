# Performance Report — Fake Job Detection System (text pipeline)

Test data: 1000 postings sampled (stratified, seed 42) from `data/splits/test.csv` — 951 real, 49 fraudulent. Positive class = fraudulent. OCR and Objective 3 (voice agent) are not part of this evaluation.

## Headline: final system verdict vs Paper 1

| | Precision | Recall | Macro-F1 |
|---|---|---|---|
| Paper 1 (reported) | 0.968 | 0.851 | 0.951 |
| This project — Obj 2+4 final verdict | 0.925 | 0.755 | 0.912 |

95% bootstrap intervals for this project are in the table below. Paper 1 does not state its test set, so the two rows are not measured on the same postings.

## After the fixes (current code)

| System | Precision (fake) | Recall (fake) | Macro-F1 | Accuracy | TP / FP / FN | 95% bootstrap CI |
|---|---|---|---|---|---|---|
| Obj 2 app filter (HIGH RISK = fake) | 0.932 | 0.837 | 0.938 | 0.989 | 41 / 3 / 8 | P 0.846–1.0, R 0.723–0.935, F1 0.897–0.971 |
| Obj 2 app filter (HIGH or MEDIUM RISK = fake) | 0.880 | 0.898 | 0.942 | 0.989 | 44 / 6 / 5 | P 0.78–0.964, R 0.804–0.976, F1 0.902–0.97 |
| Obj 1b nearest FAISS neighbour label | 0.844 | 0.551 | 0.826 | 0.973 | 27 / 5 / 22 | P 0.697–0.963, R 0.415–0.692, F1 0.763–0.884 |
| Obj 1b text classifier alone (P(scam) >= 0.5) | 0.927 | 0.776 | 0.919 | 0.986 | 38 / 3 / 11 | P 0.837–1.0, R 0.655–0.889, F1 0.872–0.957 |
| Obj 4 DST over evidence (BetP(fake) >= 0.5) | 0.902 | 0.755 | 0.907 | 0.984 | 37 / 4 / 12 | P 0.8–0.978, R 0.63–0.867, F1 0.857–0.946 |
| Obj 2+4 final verdict (is_scam) | 0.925 | 0.755 | 0.912 | 0.985 | 37 / 3 / 12 | P 0.83–1.0, R 0.63–0.87, F1 0.862–0.951 |

## Original code (before any change), same postings

| System | Precision (fake) | Recall (fake) | Macro-F1 | Accuracy | TP / FP / FN |
|---|---|---|---|---|---|
| Obj 2 app filter (HIGH RISK = fake) | 0.333 | 0.204 | 0.611 | 0.941 | 10 / 20 / 39 |
| Obj 2 app filter (HIGH or MEDIUM RISK = fake) | 0.333 | 0.204 | 0.611 | 0.941 | 10 / 20 / 39 |
| Obj 1b nearest FAISS neighbour label | 0.333 | 0.204 | 0.611 | 0.941 | 10 / 20 / 39 |
| Obj 4 DST over evidence (BetP(fake) >= 0.5) | 0.115 | 0.653 | 0.519 | 0.737 | 32 / 246 / 17 |
| Obj 2+4 final verdict (is_scam) | 0.063 | 0.980 | 0.260 | 0.287 | 48 / 712 / 1 |

## Other measurements

- Obj 1a entity extraction coverage (non-empty rate): {'company_name': 0.0, 'job_role': 0.795, 'recruiter_name': 0.016, 'recruiter_email': 0.031, 'domain_or_website': 0.152, 'location': 0.776}
- Obj 2+4 abstention (UNCERTAIN_NEEDS_VERIFICATION): 0.003
- Obj 2+4 final verdict distribution: {'real': {'VERIFIED_REAL': 948, 'SUSPICIOUS_FAKE': 2, 'HIGH_RISK_SCAM': 1}, 'fake': {'HIGH_RISK_SCAM': 35, 'SUSPICIOUS_FAKE': 2, 'VERIFIED_REAL': 9, 'UNCERTAIN_NEEDS_VERIFICATION': 3}}
- Obj 4 Platt calibration (P(real)): Brier 0.0121, ECE 0.0074
- M-RAV LLM verdict (display text) available for 0 of 1000 postings; distribution {'real': {}, 'fake': {}}
- Pipeline errors: {'app': 0, 'evidence_dst': 0, 'orchestrator': 0}

Full numbers: `performance_metrics.json` in each results folder; per-posting outputs: `predictions_flat.csv`.
