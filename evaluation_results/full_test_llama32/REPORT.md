# Full test evaluation — truly new postings, llama3.2, no phone call

**Data:** every posting in `data/splits/test.csv` whose description (leading/trailing spaces stripped) does not
appear in `data/splits/train.csv`, with repeats inside the test split removed: **2,770 postings (2,667 real, 103 scam)**
→ `test_truly_new.csv`.

**Pipeline:** `pipeline_bridge.analyze_extracted_offer` → `run_full_pipeline`
(Obj 1 entity extraction + FAISS retrieval + text classifier → Obj 2 credibility + M-RAV alignment →
Obj 4 DST combination → Platt calibration → override → final verdict). No voice call.
Both LLM steps (entity extraction, M-RAV alignment) use **llama3.2** on Ollama; M-RAV ran on all 2,770 postings.
Scam = HIGH RISK (HIGH_RISK_SCAM / SUSPICIOUS_FAKE).

## Results

| | F1-micro (≈ accuracy) | Macro F1 | Macro precision | Macro recall |
|---|---|---|---|---|
| FACTIFY, gold, llama-3.1-70B (reported) | 72.33 | 74.87 | 79.18 | 73.78 |
| This project — 2,770 truly new postings, HIGH RISK = scam | **98.66** | **89.76** | **94.12** | **86.24** |
| Baseline — always predict 'real' (same postings) | 96.28 | 49.05 | 48.14 | 50.00 |

- Scam class: precision 89.29, recall 72.82, F1 80.21.
- Confusion matrix: real→real 2658, real→scam 9, scam→real 28, scam→scam 75.
- Platt-calibrated P(real) as a scam score: ROC-AUC 0.919, PR-AUC 0.707.
- Final verdicts: VERIFIED_REAL 1255, UNCERTAIN_NEEDS_VERIFICATION 1431, HIGH_RISK_SCAM 74, SUSPICIOUS_FAKE 10.

## Notes
- The tasks differ: FACTIFY is 5-class multimodal claim verification; this is binary job-posting scam detection
  on text (EMSCAD). The comparison is not like-for-like.
- 1,427 postings (52%) get the UNVERIFIED override (VERIFIED_REAL → UNCERTAIN, i.e. MEDIUM RISK, call
  recommended). llama3.2 labels most retrieved records "Contradicts" (6,883 Contradicts, 2,079 Confirms,
  26 Neutral), so "no record confirms the offer" is common. This does not change the HIGH RISK = scam numbers,
  but counting MEDIUM as scam gives scam recall 99.0 at precision 6.7 (1,413 real postings flagged).
- With 103 scams, one scam changes scam recall by about 1 point.

## Re-run
```bash
export OLLAMA_MODEL=llama3.2
D=evaluation_results/full_test_llama32
python evaluate_test_split.py extract $D/test_truly_new.csv $D/entities.jsonl
python evaluate_test_split.py llm     $D/test_truly_new.csv $D/entities.jsonl $D/llm_mrav.jsonl
python evaluate_test_split.py score   $D/test_truly_new.csv $D/entities.jsonl $D/results $D/llm_mrav.jsonl
python evaluate_test_split.py metrics $D/results
```
Steps 1–2 skip rows already in their cache files, so step 3 can be re-run alone (~2 min) after code changes.
