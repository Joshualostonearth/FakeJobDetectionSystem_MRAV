# M4 integration: status report

Branch `m4-integration` (local, not pushed), created from `origin/obj4-after-call`. It lives in the worktree `.claude/worktrees/m4-integration`, so the uncommitted work on `main` was not touched. Date: 2026-11-01.

**Bottom line:** image or text goes through OCR and M4, which gives FAKE / NOT FAKE plus an explanation, and it runs with Ollama and the voice services switched off. The local M4 reproduces the Kaggle run exactly on the 3,151-posting test set (fraud-F1 0.8873). The voice-call fusion is wired up and tested with a mocked call only. It is not validated.

## 1. What was done, step by step

| Step | Result | Commit(s) |
|---|---|---|
| 0. Branch audit | `obj4-after-call` chosen as base. Ported from `integrated2`: the no-job-offer check (rewritten) and the `.claude/worktrees/` ignore line. Nothing ported from `main`: its committed `server.js` has unresolved merge-conflict markers, and its voice files are older than the base's. Details: `results/branch_audit.md` | `8e82617`, `c4926c4` |
| Repo hygiene | Untracked (still on disk): `data/*.csv`, `scam_classifier.pkl`, `platt_model.pkl`, `webrtc-call.zip`, `webrtc-call/server/node_modules`. `*.joblib`, `*.pt`, `models/` and `.venv-m4/` are ignored | `e256ea9` |
| 1. M4 inference | `mravplus_infer.py`, `M4Detector`: M1 TF-IDF SVM, MiniLM 5-NN Dempster-Shafer masses, `platt_cheap`, LOW / REVIEW / HIGH routing, then the DistilBERT gate and `platt_final` in REVIEW. Loads `stage2.pt` with `strict=True`. Free text goes into `description`, with optional structured fields. Warns on short or unstructured input and on a scikit-learn version mismatch. Works offline from the Hugging Face cache | `cce9e99`, `5945b10` |
| 2. Parity test | 100 % agreement with the Kaggle predictions (section 2) | `5945b10` |
| 3. LLM-free bridge | `pipeline_bridge.py`: OCR, strip `[SOURCE]` tags, regex entities, M4, XAI. No langchain, Ollama, FAISS or legacy engines (checked by a test). Old bridge moved to `legacy/pipeline_bridge_llm.py` | `5befb87` |
| 4. UI + voice | `app.py`: verdict card, limitation note, optional structured fields, Objective 5 card, 5 nearest training postings instead of FAISS evidence, post-call card labelled NOT validated. `post_call_fusion.py`: M4 mass (r = 0.9) combined with the voice mass, BetP, and the repo's voice overrides and bands. `start_all.sh --no-voice` and a new `start_all.ps1 -NoVoice` for Windows | `c2bd499` |
| 5. Explainable AI | `xai/`: exact SVM term attributions, DST leave-one-out effects, neighbours, Integrated Gradients and gate weights for DistilBERT, on-demand deletion check, template summary (no LLM). Global report: `results/xai_global_report.md` | `67f186b`, `a8cdb6f` |
| 6. Image test | 40 synthetic posters plus `richmond_offer.png` (section 3) | `9772a6b` (extractor fixes it found) |
| 6b / 7. Packaging | `requirements.txt` (was UTF-16, now UTF-8; scikit-learn pinned to 1.6.1 as in the bundle, CPU torch, no LLM packages), `requirements-voice.txt`, `scripts/download_models.py`, README with model card and limitations | `0672329`; README not yet committed |

Environment: new venv `.venv-m4` (Python 3.12.6, scikit-learn 1.6.1, torch 2.15.0+cpu, transformers 4.57.6). Your old `venv/` was not changed. Model files are in `models/` (git-ignored). `stage2.zip` was the PyTorch file itself, copied as `models/stage2.pt`.

## 2. Parity with the Kaggle run (3,151 test postings)

| | Local M4 | Kaggle CSV |
|---|---|---|
| Label agreement | 100 % (0 disagreements) | - |
| Zone agreement (LOW / REVIEW / HIGH) | 100 % (2,998 / 33 / 120) | 2,998 / 33 / 120 |
| Max diff of `q` / `P_cheap` | 1.1e-16 / 6.4e-8 | - |
| Max diff of DistilBERT `p2` on REVIEW rows | 0.0027 (CPU vs GPU, no verdict changed) | - |
| Fraud precision / recall / F1 | 0.9173 / 0.8592 / **0.8873** | 0.9173 / 0.8592 / 0.8873 |
| ROC-AUC / PR-AUC / Brier | 0.9904 / 0.9255 / 0.0090 | same |
| FP / FN | 11 / 20 | 11 / 20 |
| Escalated to DistilBERT | 1.05 % (33) | 1.05 % |

All gate targets met. Details: `results/m4_parity_report.md`.

## 3. Image tests

**Synthetic posters (first run, before the extractor fixes).** 40 EMSCAD test postings (20 fraud, 20 legit, seed 42) rendered as clean PNGs, then image → EasyOCR → app pipeline:

| Path | Scored | Accuracy | Fraud recall | Fraud precision |
|---|---|---|---|---|
| Original structured posting | 40 | 0.85 | 0.75 | 0.94 |
| Poster text pasted | 35 | 0.94 | 0.94 | 0.94 |
| Poster image (OCR) | 35 | 0.94 | 0.94 | 0.94 |

- Text path vs image path: the same verdict on 35 of 35.
- The no-job-offer check refused 5 of the 40 posters (4 frauds, 1 legit). The cause was regex role detection missing all-caps or unlisted titles. Fixed in `a8cdb6f`. **A rerun with the fixed extractor is in progress, and this table will be updated from it.**
- This is a small check on clean, computer-rendered text. Real phone photos will do worse.

**`richmond_offer.png`** (no ground-truth label): FAKE, P(fraud) 0.634, zone HIGH. The explanation shows the letter's own words are net slightly legit-leaning (−0.04). The verdict comes from missing metadata (+0.41: no location, experience or employment type) and from 3 of 5 neighbours being fraud, at a low similarity of 0.42. This is the distribution shift in action.

**Acceptance test** `tests/test_e2e_image.py`, with Ollama blocked and no voice services: 3 / 3 passed (Richmond, 6 posters with image/text agreement 6/6, a blank image gives a clear message).

## 4. Explainable AI checks

| Check | Result |
|---|---|
| SVM attribution identity (contributions + intercept = decision value), 32 postings | max error 1.3e-15 |
| Integrated Gradients completeness, 3 REVIEW-zone postings | residual ≤ 0.33 % of the output change |
| Deletion test, 30 high-confidence test frauds: top-5 / top-10 fraud words removed | P(fraud) −0.179 / −0.315 on average (lowered for 97 % / 100 % of them); random words removed: +0.002 |

Global findings (`results/xai_global_report.md`):
- Missing metadata (`*_Unknown`) pushes towards FAKE.
- A company logo (`has_company_logo_1`) is the strongest legit signal.
- EMSCAD's anonymised `#email#` / `#phone#` placeholders are strong fraud-leaning tokens that real posters never contain.

## 5. Tests run

| Test | Result |
|---|---|
| `tests/test_m4_parity.py` (run as a script; same function the pytest asserts on) | all gate checks met |
| `tests/test_xai.py` | 4 / 4 passed |
| `tests/test_m4_bridge.py` | 12 / 12 passed on the full run; after the extractor fixes, 12 / 12 on the fast subset (the 2 slow tests were not re-run) |
| `tests/test_app_ui.py` (Streamlit AppTest, headless) | 3 / 3 passed |
| `tests/test_e2e_image.py` | 3 / 3 passed (before the last regex commit `9772a6b`) |
| `tests/test_voice_to_mass.py`, `test_objective3.py`, `test_voice_bridge.py` | 35 / 35 passed |
| `tests/test_dst.py`, `test_objective4.py` | 15 / 15 passed |
| Old-pipeline tests (`test_authority`, `test_consensus`, `test_credibility_engine`, `test_evaluation_fixes`, `test_freshness`, `test_retrieval_and_filter`, `legacy/tests/test_app_full_pipeline`) | **not run**: they need langchain-community, FAISS and rapidfuzz, which are deliberately not in the M4 env |
| Real app launch: `start_all.ps1 -NoVoice` | Streamlit served on :8501 (health ok, HTTP 200) and stopped cleanly |

Still to do before calling it finished: one final clean run of the whole list above on the last commit, including `pytest tests/test_m4_parity.py`.

## 6. How to run

```powershell
cd "C:\Users\Joshua Sunny\OneDrive\Desktop\FakeJobs\.claude\worktrees\m4-integration"
.\start_all.ps1 -NoVoice        # app only, no Ollama needed  ->  http://localhost:8501
.\start_all.ps1                 # with voice (needs requirements-voice.txt, whisper_Test\.venv, npm install, Ollama)
```

## 7. Untested, mocked or weak

- **Voice fusion is not validated**, and no real call was placed. The post-call path was tested only with mocked voice signals.
- **Distribution shift:** M4 was validated on structured EMSCAD postings. Posters lack metadata, which by itself pushes towards FAKE; filling in the optional structured fields in the UI reduces this.
- **Ollama "off"** was simulated (dead `OLLAMA_HOST` port plus guards that fail if any LLM function is called), because your Ollama was running and I did not stop it.
- **No browser was used.** The UI was checked with Streamlit AppTest, and AppTest cannot drive the file uploader, so the image upload in the UI was simulated with the bridge's result.
- **The regex entity extractor is heuristic.** It only feeds the call and the no-job-offer check, never the verdict, but the check can still refuse a real posting whose text has neither a recognisable role nor at least two job words.
- **`start_all.sh` on Linux/macOS** was syntax-checked only, not run.
- **Commit `5945b10`** also contains the two file moves into `legacy/` (staged by mistake), so at that commit `app.py` points at a missing bridge; fixed in the next commit. History was not rewritten.
- **Metrics come from one run and one seed** (142 test frauds), and near-duplicate postings may remain. M4 beats our own re-implementation of the Arrays paper, not its printed numbers.
