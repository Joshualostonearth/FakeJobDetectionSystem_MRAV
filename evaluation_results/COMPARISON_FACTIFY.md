# Comparison: truly new test postings vs FACTIFY (M-RAV)

**Our evaluation set:** test postings from `data/splits/test.csv` whose job description does not appear in
`data/splits/train.csv` — 845 postings (808 real, 37 scam). Final version of the app
(`pipeline_bridge.analyze_extracted_offer` → `run_full_pipeline`), llama3.2 for entity extraction, no phone call.
Scam = HIGH RISK.

| | F1-micro (≈ accuracy) | Macro F1 | Macro precision | Macro recall |
|---|---|---|---|---|
| FACTIFY, gold, llama-3.1-70B (reported) | 72.33 | 74.87 | 79.18 | 73.78 |
| This project — truly new postings, HIGH RISK = scam | 98.34 | 88.63 | 95.56 | 83.66 |
| Baseline — always predict 'real' (same postings) | 95.62 | 48.88 | 47.81 | 50.00 |

Confusion matrix (this project): real→real 806, real→scam 2, scam→real 12, scam→scam 25.

## Notes
- The tasks differ: FACTIFY is 5-class multimodal claim verification; this project is binary job-posting
  scam detection on text (EMSCAD). The numbers are not a like-for-like comparison.
- About 95% of these postings are real, so F1-micro is inflated (the always-'real' baseline already scores
  95.62). Macro F1 / macro recall weight both classes equally and are the fairer figures.
- With 37 scams in the set, one scam changes scam recall by 2.7 points.
