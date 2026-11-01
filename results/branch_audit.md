# Branch audit (Step 0, read-only)

Date: 2026-11-01. Remote: `origin` = `Joshualostonearth/FakeJobDetectionSystem_MRAV`, fetched with `git fetch --all --prune` just before this audit.
No branch was merged or rebased. "Ahead/behind" is relative to `origin/main`.

## Branches found

| Branch | Ahead / behind main | Last commit | Relationship |
|---|---|---|---|
| `main` | - | 2026-09-29 | 3 commits not in any integration branch (see below) |
| `Integrated` | 7 / 3 | 2026-09-30 | ancestor of `obj4-after-call` and `integrated2` |
| `Joshuas_work` | 7 / 3 | 2026-10-04 | ancestor of `obj4-after-call` and `integrated2` |
| `objective4-confidence` | 8 / 3 | 2026-10-04 | ancestor of `obj4-after-call` and `integrated2` |
| `allan-web-verification` | 0 / 4 | 2026-09-22 | fully contained in `main` |
| `person1-audio-pipeline` | 0 / 10 | 2026-09-21 | fully contained in `main` |
| `obj4-after-call` | 26 / 3 | 2026-10-20 | **chosen base**; shares history with `integrated2`, then 2 own commits |
| `integrated2` | 33 / 3 | 2026-10-29 | shares history with `obj4-after-call`, then 9 own commits |
| `explanable-ai` | 2 / 0 | 2026-10-21 | not in the brief's list; `main` + 2 commits (old Objective 5 LLM explainer) |

`Integrated`, `Joshuas_work`, `objective4-confidence`, `allan-web-verification` and `person1-audio-pipeline` have no commits that the base lacks, so there is nothing to port from them.

## Base choice: `obj4-after-call`

It has the most complete working UI + OCR + voice-call flow:

- `app.py` (Streamlit UI, palette, upload + text area, OCR preview, evidence expander, phone-verification section, post-call card showing `final_verdict`, beliefs, `calibrated_confidence`, `override_reason`)
- `ocr_engine.py` (EasyOCR, `sanitize_ocr_text`, `merge_text_and_ocr`)
- `pipeline_bridge.py` (`handle_ui_submission`, `handle_phone_verification`, `VERDICT_TO_RISK_LEVEL`)
- `voice_verification_bridge.py` + `objective3_voice_agent/` (incl. `voice_to_mass.py`) + `whisper_Test/call_bridge.py` + `webrtc-call/`
- `confidence_calibrator.apply_verification_override` with the two voice overrides the brief asks to keep
- `start_all.sh` (one-command launcher, Linux/macOS bash)
- voice tests: `tests/test_voice_to_mass.py`, `tests/test_objective3.py`, `tests/test_voice_bridge.py`

`integrated2` has the same UI/voice base, but not the post-call Objective 4 wiring (commit `68d6d13`) that the brief's post-call section is modelled on. Its extra work is mostly the old-pipeline XAI.

## What other branches have that the base lacks

### `integrated2` (9 own commits)

| Item | Matters for | Decision |
|---|---|---|
| `a3c42c3` refuse to score input with neither a company name nor a job role (in `pipeline_bridge.handle_ui_submission`) | UI / OCR (stops "conference backdrop rated 99% real") | **Port the idea**, reimplemented on the regex entity extractor in the new `pipeline_bridge.py` (overlaps a file I am rewriting, so it needs your OK) |
| `.gitignore`: `.claude/worktrees/` | repo hygiene | **Port** (one line, no conflict) |
| `explainability.py`, `tests/test_explainability.py`, app.py "Why This Verdict" card (`e8cf746`, `246b881`, `f200915`, `1732086`, `8863ef9`) | old classification only: SHAP over `scam_classifier.pkl` + FAISS/DST sources, summary written by llama3.2 | **Skip**. It is replaced by the new LLM-free `xai/` for M4 |
| `full_pipeline_orchestrator.py`, `confidence_calibrator.py` changes | old classification only | **Skip** (the calibrator change is the same one the base already has) |
| `whisper_Test/responses/*.json` (3 recorded call transcripts) | data output of past calls | **Skip** |
| `requirements.txt` change | old pipeline deps | **Skip** (rewritten in Step 7) |

### `main` (3 own commits: `24264dd`, `4c522a2`, `8fa3f18`)

| Item | Matters for | Decision |
|---|---|---|
| `ai/*.py`, `integration/voice_agent_api.py`, `schemas/voice_schemas.py`, `tests/test_person2_voice_agent.py` | an older, separate "Person 2" voice agent; `app.py` on the base does not use it | **Skip** |
| `webrtc-call/server/server.js` | voice call | **Skip**: the committed version on `main` contains unresolved merge-conflict markers (`<<<<<<< HEAD` / `>>>>>>> eacde25`) |
| `webrtc-call/public/index.html`, `package.json` | voice call UI | **Skip**: belongs to the broken `server.js` above; the base has a working `public/` + `server.js` |
| `whisper_Test/{assistant,caller,server,voice_tools}.py` | voice agent | **Skip**: older than the base's versions (the base adds `offer_text`-aware questions, configurable `OLLAMA_MODEL`) |
| `obj2dst.py`, `engines/*`, `models.py`, `search_engine.py`, `credibility_engine.py`, `testscraper.py` | old classification / web scraping | **Skip** |

Your **uncommitted** changes on `main` (`whisper_Test/caller.py`, `voice_tools.py`, `webrtc-call/server/server.js`, `whisper_Test/evaluation/`, ...) were not touched. Compared with the base, `caller.py` there is the older variant (hard-coded `OLLAMA_MODEL = "qwen3-vl:latest"`, no `offer_text`). They are not ported.

### `explanable-ai` (2 own commits)

| Item | Matters for | Decision |
|---|---|---|
| `objective5_explanation/` (LLM-grounded explainer over the old pipeline), `objective1_intake.py`, `number_sourcer.py` | old classification / LLM explanation | **Skip** (brief: XAI must be deterministic and LLM-free) |
| `engines/alignment_chain.py` timeout | old M-RAV LLM | **Skip** |

## Things on the base that conflict with the brief's rules

The base branch **tracks** files the brief says must never be committed:

- `data/fake_job_postings.csv` (50 MB) and `data/splits/*.csv`
- `scam_classifier.pkl` (47.5 MB), `platt_model.pkl`, `webrtc-call.zip`
- `webrtc-call/server/node_modules/` (795 files)

Proposal: `git rm --cached` them in one commit and add them to `.gitignore`. That removes them from the branch tip only (no history rewrite); the files stay on disk.

## Inputs located

| File | Status |
|---|---|
| `mravplus_bundle.joblib` | found in the main checkout; keys and values match the brief (`r_best={'q':0.95,'a':0.0,'k':0.5,'m':0.0}`, `rb4=0.9`, `lo4=0.035`, `hi4=0.57`, `meta_dim=632`, `K_NN=5`); **saved with scikit-learn 1.6.1** (the venv has 1.9.0) |
| `stage2.pt` | **not found under that name**. `stage2.zip` (266 MB) is a valid PyTorch archive (`stage2/data.pkl` inside) and `torch.load(..., weights_only=True)` returns a 114-key state dict with the `HybridGate` key names (`bert.*`, `meta.*`, `p_text`, `p_meta`, `gate.*`, `head.*`). The `stage2/` folder next to it is an unzipped copy |
| `test_predictions_mravplus.csv` | found; 3,151 rows; zones LOW 2,998 / REVIEW 33 / HIGH 120 |
| `mravplus_app_scaffold/` | not found; the inference module will be written from section 3 of the brief |

Data check (section 3.1, run on the base's `data/fake_job_postings.csv`): 17,880 rows / 866 fraud -> 15,752 after dedup -> train/val/test 11,340 / 1,261 / 3,151 (142 test frauds). `len(tr) == len(E_tr)` and `tr.fraudulent == y_tr` element-wise: both True. `test_predictions_mravplus.csv.y == te.fraudulent` row by row: True.
