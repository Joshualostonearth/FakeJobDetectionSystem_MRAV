# Fake Job Detection System (MRAV+ "M4")

Upload an image of a job poster or offer letter (or paste the text, or both) and get:

1. the OCR text of the image (EasyOCR),
2. a **FAKE / NOT FAKE** flag with risk level and calibrated confidence from the **M4** model,
3. an **explanation** of why (Objective 5, explainable AI, no LLM),
4. optionally a **phone verification call** (Objective 3 voice agent) whose result is fused into a post-call verdict (**not validated**, see below).

No LLM is used to classify. Ollama is only used by the optional voice agent (question generation / answer analysis) and as an optional entity-extraction fallback when a call is placed. Image -> verdict works with Ollama switched off.

## Architecture

```
image ──► EasyOCR ──┐
                    ├─► merge, strip [SOURCE] tags ──► regex entities (company, role, contact; for the call only)
text ───────────────┘                │
                                     ▼
                              M4Detector (mravplus_infer.py)
   ┌─────────────────────────────────┼──────────────────────────────────┐
   │ M1: word+char TF-IDF + one-hot  │ MiniLM embedding -> 5 nearest    │
   │ metadata -> calibrated LinearSVC│ EMSCAD training postings         │
   │ -> q = P(fraud)                 │ -> one DST mass each             │
   └───────────────┬─────────────────┴───────────────┬──────────────────┘
                   └──── Dempster-Shafer fusion ─────┘
                     s = m(FAKE) - m(REAL) -> platt_cheap -> P_cheap
           P_cheap < 0.035: LOW (NOT FAKE)   P_cheap >= 0.57: HIGH (FAKE)
           otherwise REVIEW (~1%): DistilBERT + metadata gate (stage2.pt) as one more
           DST source -> fuse again -> platt_final -> FAKE if >= 0.5
                                     │
                                     ▼
          xai/ (Objective 5): exact SVM term attributions, DST leave-one-out,
          neighbours, Integrated Gradients for DistilBERT, deletion check, template summary
                                     │
                                     ▼
                          Streamlit UI (app.py)
                                     │  optional "Verify by Phone"
                                     ▼
   voice agent (objective3_voice_agent, whisper_Test/call_bridge.py :5000, webrtc-call :3000, Ollama)
   -> voice mass -> post_call_fusion.py: M4 mass (r=0.9) ⊕ voice mass, BetP, Objective 4 overrides/bands
      (NOT validated)
```

Verdict mapping (`pipeline_bridge.py`): zone HIGH, or REVIEW with P(fraud) >= 0.5 -> **FAKE**, HIGH RISK (call starts automatically when the voice services are running); REVIEW with P(fraud) < 0.5 -> **NOT FAKE**, MEDIUM RISK (call recommended); LOW -> **NOT FAKE**, LOW RISK. The UI score is P(real) x 100 ("Calibrated confidence: Real").

## Setup

Python 3.12 (tested on Windows 11, CPU only).

```bash
py -3.12 -m venv .venv-m4                 # Linux/macOS: python3.12 -m venv .venv-m4
.venv-m4\Scripts\activate                 # Linux/macOS: source .venv-m4/bin/activate
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/download_models.py         # MiniLM ~90 MB, EasyOCR ~100 MB, DistilBERT tokenizer ~1 MB; asks first
```

`requirements.txt` pins **scikit-learn 1.6.1**, the version the bundle was pickled with. Other versions print a warning with the exact `pip install` command, because unpickling with another version can silently change results.

### Model files (not in git)

Put these in `models/` (or point `M4_MODEL_DIR` at their folder):

| File | What | Size |
|---|---|---|
| `models/mravplus_bundle.joblib` | fitted TF-IDF, encoder, calibrated SVM, MiniLM training embeddings, Platt models, DST weights, thresholds | ~31 MB |
| `models/stage2.pt` | DistilBERT hybrid-gate state dict (if your download is called `stage2.zip`, it is the same PyTorch file: rename it) | ~267 MB |

Both come from the Kaggle MRAV+ training notebook. Without the bundle the app shows a clear "Model files missing" message; without `stage2.pt` the REVIEW zone falls back to the cheap score with a warning.

`data/fake_job_postings.csv` (EMSCAD, Kaggle) is only needed for the tests and to build the neighbour snippets once (cached to `models/train_snippets.json`).

## Run

Without voice (only Streamlit; Ollama not needed):

```powershell
.\start_all.ps1 -NoVoice                  # Windows (or: powershell -ExecutionPolicy Bypass -File .\start_all.ps1 -NoVoice)
```
```bash
./start_all.sh --no-voice                 # Linux/macOS
# or simply:  .venv-m4/bin/python -m streamlit run app.py
```

With voice verification:

```bash
pip install -r requirements-voice.txt     # into .venv-m4
# voice services env, as before: whisper_Test/.venv with whisper_Test/requirements.txt
cd webrtc-call/server && npm install && cd ../..
ollama pull ministral-3                   # or set OLLAMA_MODEL
./start_all.sh                            # or .\start_all.ps1 ; add --no-tunnel / -NoTunnel to skip cloudflared
```

Open http://localhost:8501.

## Tests

```bash
python -m pytest tests/test_m4_parity.py -v -s   # M4 vs Kaggle predictions (~4 min)
python -m pytest tests/test_xai.py -v -s         # attribution identity, IG completeness, deletion test (~2 min)
python -m pytest tests/test_m4_bridge.py tests/test_app_ui.py tests/test_e2e_image.py -v
python -m pytest tests/test_voice_to_mass.py tests/test_objective3.py tests/test_voice_bridge.py -v   # needs requirements-voice.txt
python scripts/poster_images.py                  # 40 synthetic posters: structured vs text vs image
python scripts/xai_global_report.py              # results/xai_global_report.md
```

Old tests of the replaced LLM/FAISS pipeline live in `legacy/tests` and in `tests/` (Objective 2 engines); they need Ollama and the FAISS index.

## Model card: M4

- **What it is:** Dempster-Shafer fusion of (1) a calibrated LinearSVC over word (1-2-gram) + char_wb (2-5-gram) TF-IDF of the "field: value" posting text plus one-hot metadata, and (2) the 5 nearest EMSCAD training postings by MiniLM similarity; Platt-calibrated; about 1% of postings (the uncertain band) also get a DistilBERT + metadata attention-gate model as a third source.
- **Training data:** EMSCAD (Kaggle `fake_job_postings.csv`): 17,880 postings, 866 fraudulent; 15,752 after removing exact duplicate texts; stratified split train / val / test = 11,340 / 1,261 / 3,151 (seed 42).
- **Test results** (local re-run, `results/m4_parity_report.md`; identical to the Kaggle run):

  | Fraud precision | Fraud recall | Fraud F1 | ROC-AUC | PR-AUC | Brier | FP | FN | Escalated to DistilBERT |
  |---|---|---|---|---|---|---|---|---|
  | 0.9173 | 0.8592 | 0.8873 | 0.9904 | 0.9255 | 0.0090 | 11 | 20 | 1.05 % (33 / 3,151) |

- **Images:** on 40 synthetic, computer-rendered posters (20 fraud, 20 legit EMSCAD test postings), the image path (EasyOCR) scored accuracy 0.925, fraud recall 0.90, fraud precision 0.95, and agreed with the pasted-text path on 39 of 40 (`results/e2e_image_report.md`). That is a small, clean, synthetic check: real phone photos will do worse, and part of the poster recall comes from the missing-metadata push described below.

### Limitations (read before trusting a verdict)

- M4 was trained and tested on **structured EMSCAD postings** (title, company profile, description, requirements, benefits + metadata). Poster / OCR text is unstructured, so there is a **distribution shift**: the validated metrics apply to EMSCAD-style input, not to arbitrary images. In particular, missing metadata (`location`, `employment_type`, `required_experience` = Unknown) pushes the classifier towards FAKE, because fraudulent EMSCAD postings often left those fields empty; a bare poster always has them missing unless you fill in the optional structured fields.
- The test metrics come from **one run, one seed**, with a 57-fraud validation set (used to tune the weights and thresholds) and a 142-fraud test set. Exact duplicates were removed but near-duplicates may remain, which can favour the retrieval and TF-IDF parts.
- M4 does **not** beat the printed numbers of the Arrays paper (two-stage SVM + DistilBERT); it beats our own re-implementation of it.
- The **post-call fusion is not validated**: there is no labelled phone-call dataset, so the voice mass, its weight and the verdict bands have never been evaluated. Only M4's text score is validated.
- **OCR errors** change the text the model sees.
- EMSCAD is a historical English dataset; the model may not generalise to other regions, languages or newer scam styles.
- Explanations describe what drove this model's score. They are not proof that an offer is or is not a scam.

## Privacy

Everything runs locally; posting text is not sent to any external service (the optional Ollama calls go to your local Ollama). The M4 path (OCR -> verdict -> explanation) does not write posting text or phone numbers to files. The voice services print parts of the call (company, role, transcript snippets) to their console, which `start_all` redirects to `logs/` (git-ignored); delete those logs if that matters to you. The optional Cloudflare tunnel exposes the call page publicly while it runs.

## Repository map

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `pipeline_bridge.py` | UI entry points: OCR -> M4 -> XAI; phone verification |
| `mravplus_infer.py` | M4 inference (`M4Detector`) |
| `xai/` | Objective 5 explanations |
| `entity_extractor.py` | regex entities for the call (Ollama fallback only when calling) |
| `post_call_fusion.py` | M4 + voice DST fusion (not validated) |
| `ocr_engine.py` | EasyOCR wrapper |
| `objective3_voice_agent/`, `whisper_Test/`, `webrtc-call/`, `voice_verification_bridge.py` | voice agent stack |
| `legacy/`, `engines/`, `credibility_engine.py`, `search_engine.py`, `scam_classifier.py`, `full_pipeline_orchestrator.py`, `evaluate_test_split.py` | the replaced LLM/FAISS/DST classification pipeline, kept for reference; not used by the app |
| `results/` | branch audit, parity report, XAI and image-test reports |
