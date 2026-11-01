"""
mravplus_infer.py - inference for the MRAV+ "M4" fake-job model (no LLM).

M4 (trained on Kaggle EMSCAD; the bundle *is* the model, nothing here is tuned):

  1. M1 classifier: word + char TF-IDF of the "field: value" posting text plus one-hot
     metadata -> CalibratedClassifierCV(LinearSVC) -> q = P(fraud).
  2. Retrieval: MiniLM embedding of the cleaned text; the 5 nearest *training* postings
     each give a Dempster-Shafer mass.
  3. Cheap DST fusion -> margin s = m(FAKE) - m(REAL) -> platt_cheap -> P_cheap.
  4. Routing: P_cheap < lo -> LOW (not fake), P_cheap >= hi -> HIGH (fake), else REVIEW.
  5. REVIEW only: the DistilBERT + metadata gate (stage2.pt) is one more DST source
     (reliability rb4); fused again -> platt_final -> P_final; fraud if P_final >= 0.5.

Masses are [m(REAL), m(FAKE), m(UNCERTAIN)] throughout.

Usage:
    det = M4Detector()                      # finds models/mravplus_bundle.joblib, models/stage2.pt
    out = det.predict("We are hiring ...")  # free text goes into 'description'
    out = det.predict("", fields=row_dict)  # structured EMSCAD-style posting
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

BUNDLE_FILENAME = "mravplus_bundle.joblib"
STAGE2_FILENAME = "stage2.pt"
SNIPPETS_FILENAME = "train_snippets.json"
DATA_CSV = PROJECT_ROOT / "data" / "fake_job_postings.csv"

MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_MAX_SEQ_LEN = 256
DISTILBERT_NAME = "distilbert-base-uncased"
DISTILBERT_MAX_LEN = 512

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
META = ["telecommuting", "has_company_logo", "has_questions", "employment_type",
        "required_experience", "required_education", "industry", "function", "loc"]
# Metadata the UI may supply for free-text input (everything else in FIELDS is text)
BINARY_META = ("telecommuting", "has_company_logo", "has_questions")

ZONE_NAMES = {0: "LOW", 1: "REVIEW", 2: "HIGH"}
SHORT_INPUT_WORDS = 15
SNIPPET_CHARS = 300
SPLIT_SEED = 42


class ModelFilesMissing(FileNotFoundError):
    """The bundle or the DistilBERT weights are not where the app looks for them."""


# ──────────────────────────────────────────────────────────
# Text preparation (section 3.1 / 3.2 of the M4 spec)
# ──────────────────────────────────────────────────────────

def clean(s: str) -> str:
    s = s.lower().encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _as_text(v: Any) -> str:
    """Field value as the notebook saw it after fillna(''): NaN/None -> ''."""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v)


def _as_meta(v: Any) -> str:
    """META value after fillna('Unknown').astype(str); booleans become '0'/'1' like the CSV ints."""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "Unknown"
        return str(int(v)) if float(v).is_integer() else str(v)
    s = _as_text(v)
    return s if s != "" else "Unknown"


def loc_of(location: Any) -> str:
    loc = ",".join(_as_text(location).split(",")[:2])
    return loc if loc != "" else "Unknown"


def load_emscad_splits(csv_path: Path = DATA_CSV):
    """Rebuild the exact train/val/test split M4 was trained on (spec 3.1)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    df["text"] = df[TEXT_COLS].fillna("").agg(" ".join, axis=1).map(clean)
    df = df.drop_duplicates("text").reset_index(drop=True)
    full_tr, te = train_test_split(df, test_size=0.2, stratify=df.fraudulent, random_state=SPLIT_SEED)
    tr, va = train_test_split(full_tr, test_size=0.1, stratify=full_tr.fraudulent, random_state=SPLIT_SEED)
    return [d.reset_index(drop=True) for d in (tr, va, te)]


# ──────────────────────────────────────────────────────────
# Dempster-Shafer (spec 3.3)
# ──────────────────────────────────────────────────────────

def prob_mass(p: float, r: float) -> np.ndarray:
    return np.array([r * (1 - p), r * p, 1 - r])


def discount(m: Sequence[float], r: float) -> np.ndarray:
    return np.array([r * m[0], r * m[1], 1 - r * (m[0] + m[1])])


def combine(m1: Sequence[float], m2: Sequence[float]) -> Tuple[np.ndarray, float]:
    K = m1[0] * m2[1] + m1[1] * m2[0]
    R = m1[0] * m2[0] + m1[0] * m2[2] + m1[2] * m2[0]
    F = m1[1] * m2[1] + m1[1] * m2[2] + m1[2] * m2[1]
    U = m1[2] * m2[2]
    d = max(1 - K, 1e-12)
    return np.array([R / d, F / d, U / d]), K


def combine_all(masses: Sequence[Sequence[float]]) -> Tuple[np.ndarray, float]:
    """Fold with combine; total conflict = 1 - prod(1 - K_i)."""
    m = np.asarray(masses[0], dtype=float)
    keep = 1.0
    for nxt in masses[1:]:
        m, K = combine(m, nxt)
        keep *= 1 - K
    return m, 1 - keep


def knn_mass(sim: Sequence[float], lab: Sequence[int]) -> np.ndarray:
    masses = []
    for s_j, l_j in zip(sim, lab):
        l_j = int(l_j)
        s = float(np.clip(s_j, 0, 1)) * (0.8 if l_j == 1 else 0.7) * 0.7
        masses.append([s * (1 - l_j), s * l_j, 1 - s])
    return combine_all(masses)[0]


def source_masses(q: float, kn: np.ndarray, r: Dict[str, float],
                  p2: Optional[float] = None, rb: float = 0.0) -> Dict[str, np.ndarray]:
    """The (already discounted) mass of each source that `fuse` combines, in fusion order."""
    out = {"classifier": prob_mass(q, r["q"])}
    if r["k"] > 0:
        out["similar_postings"] = discount(kn, r["k"])
    if p2 is not None and rb > 0:
        out["distilbert"] = prob_mass(p2, rb)
    return out


def fuse(q: float, kn: np.ndarray, r: Dict[str, float],
         p2: Optional[float] = None, rb: float = 0.0) -> Tuple[float, float, np.ndarray]:
    """Returns (s = m(FAKE) - m(REAL), total conflict K, combined mass)."""
    m, K = combine_all(list(source_masses(q, kn, r, p2, rb).values()))
    return float(m[1] - m[0]), float(K), m


def platt(lr, s: float) -> float:
    return float(lr.predict_proba(np.array([[s]]))[0, 1])


def mass_dict(m: Sequence[float]) -> Dict[str, float]:
    return {"REAL": float(m[0]), "FAKE": float(m[1]), "UNCERTAIN": float(m[2])}



def hf_load(loader, name: str, **kw):
    """Load a Hugging Face asset from its cached snapshot folder when it is cached (a local path,
    so no network request at all), otherwise by name (downloads it). Run
    scripts/download_models.py once to fill the cache."""
    try:
        from huggingface_hub import snapshot_download
        local = snapshot_download(name, local_files_only=True)
    except Exception:
        local = None
    if local:
        try:
            return loader(local, **kw)
        except Exception:
            pass  # incomplete snapshot: fall back to the hub
    return loader(name, **kw)


# ──────────────────────────────────────────────────────────
# DistilBERT hybrid gate (spec 3.4) - must match stage2.pt key-for-key
# ──────────────────────────────────────────────────────────

def build_hybrid_gate(meta_dim: int):
    import torch.nn as nn
    import torch
    from transformers import AutoConfig, AutoModel

    class HybridGate(nn.Module):
        def __init__(self, meta_dim):
            super().__init__()
            # Config only (a few KB); the fine-tuned weights come from stage2.pt
            self.bert = AutoModel.from_config(hf_load(AutoConfig.from_pretrained, DISTILBERT_NAME))
            self.meta = nn.Sequential(nn.Linear(meta_dim, 128), nn.ReLU())
            self.p_text = nn.Linear(768, 128)
            self.p_meta = nn.Linear(128, 128)
            self.gate = nn.Sequential(nn.Linear(768 + 128, 64), nn.Tanh(), nn.Linear(64, 2))
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

        def forward(self, ids, mask, meta, inputs_embeds=None):
            if inputs_embeds is not None:  # used by the XAI integrated gradients
                h_t = self.bert(inputs_embeds=inputs_embeds, attention_mask=mask).last_hidden_state[:, 0]
            else:
                h_t = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
            h_m = self.meta(meta)
            w = torch.softmax(self.gate(torch.cat([h_t, h_m], 1)), 1)  # w[:,0]=text, w[:,1]=metadata
            fused = w[:, :1] * self.p_text(h_t) + w[:, 1:] * self.p_meta(h_m)
            return self.head(fused), w

    return HybridGate(meta_dim)


# ──────────────────────────────────────────────────────────
# File lookup
# ──────────────────────────────────────────────────────────

def _model_dirs() -> List[Path]:
    dirs = []
    if os.getenv("M4_MODEL_DIR"):
        dirs.append(Path(os.environ["M4_MODEL_DIR"]))
    dirs += [PROJECT_ROOT / "models", PROJECT_ROOT]
    return dirs


def find_model_file(name: str) -> Optional[Path]:
    for d in _model_dirs():
        if (d / name).is_file():
            return d / name
    return None


def _missing_message(name: str, what: str) -> str:
    where = ", ".join(str(d) for d in _model_dirs())
    return (
        f"{name} ({what}) was not found. Put it in {PROJECT_ROOT / 'models'} "
        f"(or set M4_MODEL_DIR). Searched: {where}. It is produced by the Kaggle MRAV+ training "
        "notebook; it is not in the git repository."
    )


# ──────────────────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────────────────

class M4Detector:
    """Loads the M4 bundle once; MiniLM and DistilBERT are loaded lazily on first use."""

    def __init__(self, bundle_path: Optional[os.PathLike] = None,
                 stage2_path: Optional[os.PathLike] = None, device: Optional[str] = None):
        bundle_path = Path(bundle_path) if bundle_path else find_model_file(BUNDLE_FILENAME)
        if bundle_path is None or not Path(bundle_path).is_file():
            raise ModelFilesMissing(_missing_message(BUNDLE_FILENAME, "the fitted M4 bundle, about 31 MB"))
        self.stage2_path = Path(stage2_path) if stage2_path else find_model_file(STAGE2_FILENAME)
        self.load_warnings: List[str] = []

        import joblib
        import sklearn

        with warnings.catch_warnings():
            # Reported below with the exact pip command instead
            warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
            b = joblib.load(bundle_path)
        self.bundle_path = Path(bundle_path)
        self.sklearn_version = b.get("sklearn_version")
        if self.sklearn_version and sklearn.__version__ != self.sklearn_version:
            msg = (f"scikit-learn {sklearn.__version__} is installed but the M4 bundle was saved with "
                   f"{self.sklearn_version}; results may be wrong. Fix: pip install "
                   f"scikit-learn=={self.sklearn_version}")
            warnings.warn(msg)
            self.load_warnings.append(msg)

        self.feats = b["feats"]
        self.ohe = b["ohe"]
        self.clf_q = b["clf_q"]
        self.E_tr = np.asarray(b["E_tr"], dtype=np.float32)
        self.y_tr = np.asarray(b["y_tr"]).astype(int)
        self.K_NN = int(b["K_NN"])
        self.platt_cheap = b["platt_cheap"]
        self.platt_final = b["platt_final"]
        self.r = dict(b["r_best"])
        self.rb4 = float(b["rb4"])
        self.lo = float(b["lo4"])
        self.hi = float(b["hi4"])
        self.meta_dim = int(b["meta_dim"])
        self.FIELDS = list(b["FIELDS"])

        n_ohe = sum(len(c) for c in self.ohe.categories_)
        if n_ohe != self.meta_dim:
            raise ValueError(f"Bundle is inconsistent: meta_dim={self.meta_dim} but the encoder has {n_ohe} columns")
        if len(self.E_tr) != len(self.y_tr):
            raise ValueError("Bundle is inconsistent: E_tr and y_tr have different lengths")

        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self._minilm = None
        self._bert = None
        self._tok = None
        self._snippets: Optional[List[Dict[str, str]]] = None
        self._snippets_tried = False

    # ── lazy models ──────────────────────────────────────

    @property
    def minilm(self):
        if self._minilm is None:
            from sentence_transformers import SentenceTransformer
            self._minilm = hf_load(SentenceTransformer, MINILM_NAME, device=self.device)
            self._minilm.max_seq_length = MINILM_MAX_SEQ_LEN
        return self._minilm

    def _load_bert(self):
        if self._bert is not None:
            return
        if self.stage2_path is None or not Path(self.stage2_path).is_file():
            raise ModelFilesMissing(_missing_message(STAGE2_FILENAME, "the DistilBERT hybrid-gate weights, about 270 MB"))
        import torch
        from transformers import AutoTokenizer

        model = build_hybrid_gate(self.meta_dim)
        state = torch.load(self.stage2_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)  # raises on any missing/unexpected key
        model.to(self.device).eval()
        self._bert = model
        self._tok = hf_load(AutoTokenizer.from_pretrained, DISTILBERT_NAME)

    @property
    def bert(self):
        self._load_bert()
        return self._bert

    @property
    def tokenizer(self):
        self._load_bert()
        return self._tok

    # ── input handling ───────────────────────────────────

    def build_row(self, text: str = "", fields: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Turn free text and/or structured fields into the posting the model expects.

        Free text (OCR / pasted) goes into 'description' (appended after fields['description']
        if both are given). Missing fields stay empty / 'Unknown'.
        """
        fields = dict(fields or {})
        input_warnings: List[str] = []
        text = (text or "").strip()

        raw = {c: _as_text(fields.get(c)) for c in set(self.FIELDS) | set(TEXT_COLS) | {"location"}}
        if text:
            raw["description"] = f"{raw['description']}\n{text}".strip() if raw["description"] else text

        meta = {c: _as_meta(fields.get(c)) for c in META if c != "loc"}
        meta["loc"] = loc_of(fields.get("location"))

        # 'full' text: META-overlapping fields carry 'Unknown' when missing and are skipped
        values = {c: (meta[c] if c in meta else raw[c]) for c in self.FIELDS}
        full = "\n".join(f"{c}: {v}" for c, v in values.items() if v.strip() and v != "Unknown")
        cleaned = clean(" ".join(raw[c] for c in TEXT_COLS))

        structured = [c for c in self.FIELDS if c != "description" and values[c].strip() and values[c] != "Unknown"]
        structured += [c for c in BINARY_META if meta[c] != "Unknown"]
        if not structured:
            input_warnings.append(
                "No structured fields (title, company profile, location, employment type, logo, ...) were "
                "given: the text was scored as a bare description. M4 was validated on structured EMSCAD "
                "postings, so this input is out of distribution."
            )
        n_words = len(cleaned.split())
        if n_words < SHORT_INPUT_WORDS:
            input_warnings.append(
                f"Very short input ({n_words} words): the verdict rests on very little text."
            )
        return {"full": full, "text": cleaned, "meta": meta, "raw": raw}, input_warnings

    def meta_frame(self, rows: Iterable[Dict[str, Any]]):
        import pandas as pd
        return pd.DataFrame([[r["meta"][c] for c in META] for r in rows], columns=META)

    def m1_matrix(self, rows: Sequence[Dict[str, Any]]):
        """Sparse M1 input: [TF-IDF(full) | one-hot(META)]."""
        from scipy.sparse import hstack
        return hstack([self.feats.transform([r["full"] for r in rows]),
                       self.ohe.transform(self.meta_frame(rows))]).tocsr()

    def embed(self, cleaned_texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self.minilm.encode(list(cleaned_texts), batch_size=batch_size, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def neighbours(self, emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Top-K training postings by cosine similarity: (indices [n,K], similarities [n,K])."""
        sims = emb @ self.E_tr.T
        idx = np.argsort(-sims, axis=1)[:, : self.K_NN]
        return idx, np.take_along_axis(sims, idx, axis=1)

    def distilbert_p2(self, rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """(p2 = P(fraud), gate weights [n,2] = [text, metadata])."""
        import torch
        enc = self.tokenizer([r["text"] for r in rows], truncation=True, max_length=DISTILBERT_MAX_LEN,
                             padding="max_length", return_tensors="pt")
        meta = torch.tensor(self.ohe.transform(self.meta_frame(rows)).toarray(), dtype=torch.float32)
        with torch.no_grad():
            logits, w = self.bert(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device),
                                  meta.to(self.device))
            p2 = torch.softmax(logits, 1)[:, 1]
        return p2.cpu().numpy().astype(float), w.cpu().numpy().astype(float)

    # ── snippets for the neighbour display ───────────────

    def snippets(self) -> Optional[List[Dict[str, str]]]:
        """Title + cleaned-text snippet per training posting (same order as E_tr), or None."""
        if self._snippets_tried:
            return self._snippets
        self._snippets_tried = True
        import json
        path = find_model_file(SNIPPETS_FILENAME)
        try:
            if path is not None:
                data = json.loads(path.read_text(encoding="utf-8"))
            elif DATA_CSV.is_file():
                data = build_train_snippets(self.y_tr)
                try:  # cache next to the bundle (git-ignored) so later starts skip the CSV rebuild
                    (self.bundle_path.parent / SNIPPETS_FILENAME).write_text(json.dumps(data), encoding="utf-8")
                except OSError:
                    pass
            else:
                return None
            if len(data) != len(self.y_tr):
                return None
            self._snippets = data
        except Exception:
            self._snippets = None
        return self._snippets

    # ── scoring ──────────────────────────────────────────

    def score_rows(self, rows: Sequence[Dict[str, Any]], with_distilbert: bool = True) -> List[Dict[str, Any]]:
        """Score prepared rows (see build_row). Batch path used by predict() and the parity test."""
        q = self.clf_q.predict_proba(self.m1_matrix(rows))[:, 1]
        idx, sims = self.neighbours(self.embed([r["text"] for r in rows]))
        out = []
        for i in range(len(rows)):
            lab = self.y_tr[idx[i]]
            kn = knn_mass(sims[i], lab)
            s_cheap, K_cheap, m_cheap = fuse(q[i], kn, self.r)
            p_cheap = platt(self.platt_cheap, s_cheap)
            zone = 0 if p_cheap < self.lo else (2 if p_cheap >= self.hi else 1)
            out.append({"q": float(q[i]), "kn": kn, "idx": idx[i], "sims": sims[i], "lab": lab,
                        "s_cheap": s_cheap, "K_cheap": K_cheap, "m_cheap": m_cheap,
                        "p_cheap": p_cheap, "zone": zone, "p2": None, "gate": None})
        review = [i for i, o in enumerate(out) if o["zone"] == 1]
        if review and with_distilbert:
            p2, w = self.distilbert_p2([rows[i] for i in review])
            for j, i in enumerate(review):
                o = out[i]
                s, K, m = fuse(o["q"], o["kn"], self.r, p2[j], self.rb4)
                o.update(p2=float(p2[j]), gate=w[j], s_final=s, K_final=K, m_final=m,
                         p_final=platt(self.platt_final, s))
        for o in out:
            if o["zone"] == 1 and o["p2"] is not None:
                o["p_fake"] = o["p_final"]
                o["pred"] = int(o["p_final"] >= 0.5)
            else:
                o["p_fake"] = o["p_cheap"]
                o["pred"] = int(o["zone"] == 2)
        return out

    def predict(self, text: str = "", fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        row, input_warnings = self.build_row(text, fields)
        warn = list(self.load_warnings) + input_warnings
        if not row["full"].strip() and not row["text"]:
            raise ValueError("Nothing to score: the input text and fields are empty.")
        try:
            o = self.score_rows([row])[0]
        except ModelFilesMissing as e:
            # REVIEW zone needs DistilBERT; fall back to the cheap score and say so
            o = self.score_rows([row], with_distilbert=False)[0]
            warn.append(f"REVIEW-zone posting scored without DistilBERT: {e}")
        used_bert = o["p2"] is not None
        masses = source_masses(o["q"], o["kn"], self.r, o["p2"], self.rb4 if used_bert else 0.0)
        combined = o["m_final"] if used_bert else o["m_cheap"]
        conflict = o["K_final"] if used_bert else o["K_cheap"]

        snip = self.snippets()
        nbrs = []
        for j, (ti, sim, lab) in enumerate(zip(o["idx"], o["sims"], o["lab"])):
            nbrs.append({
                "rank": j + 1,
                "index": int(ti),
                "similarity": float(sim),
                "label": int(lab),
                "label_name": "FRAUD" if int(lab) == 1 else "LEGIT",
                "title": snip[ti]["title"] if snip else "",
                "snippet": snip[ti]["snippet"] if snip else "",
            })
        if snip is None:
            warn.append("Training-posting snippets are unavailable (no models/train_snippets.json and no "
                        "data/fake_job_postings.csv); neighbours are shown without text.")

        return {
            "p_fake": float(o["p_fake"]),
            "p_real": float(1 - o["p_fake"]),
            "predicted_fraud": bool(o["pred"]),
            "zone": ZONE_NAMES[o["zone"]],
            "used_distilbert": used_bert,
            "p_cheap": float(o["p_cheap"]),
            "p_final": float(o["p_final"]) if used_bert else None,
            "belief_margin": float(o["s_final"] if used_bert else o["s_cheap"]),
            "conflict": float(conflict),
            "source_masses": {k: mass_dict(v) for k, v in masses.items()},
            "combined_masses": mass_dict(combined),
            "q_classifier": float(o["q"]),
            "p_distilbert": o["p2"],
            "gate_weights": ({"text": float(o["gate"][0]), "metadata": float(o["gate"][1])}
                             if o["gate"] is not None else None),
            "neighbours": nbrs,
            "neighbour_fraud_share": float(np.mean(o["lab"])),
            "thresholds": {"lo": self.lo, "hi": self.hi, "final": 0.5},
            "reliabilities": {**self.r, "distilbert": self.rb4},
            "warnings": warn,
            "_row": row,  # prepared input, reused by the XAI module
        }


def build_train_snippets(y_tr: Optional[np.ndarray] = None, csv_path: Path = DATA_CSV) -> List[Dict[str, str]]:
    """Rebuild the training split (spec 3.1) and return title + snippet per posting, in E_tr order."""
    tr, _, _ = load_emscad_splits(csv_path)
    if y_tr is not None and (len(tr) != len(y_tr) or not (tr.fraudulent.values == np.asarray(y_tr)).all()):
        raise ValueError("Rebuilt training split does not match the bundle's y_tr; snippets would be misaligned")
    return [{"title": _as_text(t), "snippet": x[:SNIPPET_CHARS]} for t, x in zip(tr.title, tr.text)]
