# End-to-end image check (Step 6)

Command: `OLLAMA_HOST=http://127.0.0.1:9 .venv-m4/Scripts/python scripts/poster_images.py` (Ollama unreachable, voice services off). Run on the final code (commit `9772a6b`) in 358 s on CPU. Raw output: `results/e2e_image_results.json`; the rendered posters are in `results/poster_images/`.

**This is a small, synthetic check.** It uses 40 EMSCAD test postings (20 fraud, 20 legit, picked with seed 42), rendered as clean, computer-generated PNGs: title line, then short labelled sections, each section cut to 350 characters. Real phone photos of posters (blur, angles, lighting, complex layouts) will do worse.

## Three paths through the app pipeline

- **structured**: M4 on the original structured EMSCAD posting (what M4 was validated on)
- **text**: the poster's exact text pasted into the app (free text, no metadata)
- **image**: the PNG through EasyOCR, then the same app pipeline

| Path | Scored | Accuracy | Fraud recall | Fraud precision | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| structured | 40 | 0.850 | 0.75 | 0.94 | 15 | 1 | 5 | 19 |
| text | 40 | 0.950 | 0.95 | 0.95 | 19 | 1 | 1 | 19 |
| image (OCR) | 40 | 0.925 | 0.90 | 0.95 | 18 | 1 | 2 | 19 |

Verdict agreement: text vs image **39 / 40** (97.5 %), structured vs text 36 / 40, structured vs image 37 / 40. No poster was refused by the no-job-offer check, and there were no errors.

The one text/image disagreement (test row 2571, a fraud) is borderline. Text gave P(fraud) 0.603 (REVIEW zone, FAKE); OCR gave 0.046 (REVIEW zone, NOT FAKE). Small OCR differences flipped a case that went to DistilBERT.

## How to read the higher poster scores

The poster paths score *higher* than the structured path on these 40. This should **not** be read as posters being easier. Poster text carries no metadata, and in M4 missing metadata (`loc_Unknown`, `required_experience_Unknown`, `employment_type_Unknown`) pushes towards FAKE (see `results/xai_global_report.md`). Here that push turned 4 structured-path misses into catches (rows 266, 270, 1275, 2571 on the text path). It did not cause extra false positives among these 20 legit postings, but the same push applies to every legitimate poster. With 20 + 20 postings, the differences between paths are within noise.

## richmond_offer.png (no ground-truth label)

Verdict FAKE, P(fraud) 0.634, zone HIGH; 622 OCR characters. The explanation shows the letter's words are net slightly legit-leaning (word + char contributions −0.04). The score comes from missing metadata (+0.41) and from 3 of the 5 nearest training postings being fraudulent, at a low top similarity of 0.42.

## History

A first run (before commit `a8cdb6f`) refused 5 of the 40 posters (4 frauds, 1 legit) as "no job offer", because the regex role detection missed all-caps or unlisted titles. That was fixed (a short first line counts as the role when the text has job vocabulary), and the numbers above come from the fixed code.
