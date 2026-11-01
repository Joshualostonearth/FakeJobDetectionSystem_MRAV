"""
Synthetic poster check (Step 6): render EMSCAD test postings to plain PNG "posters" and compare
three paths through the app pipeline (Ollama and voice not used):

  structured : M4 on the original structured posting (what M4 was validated on)
  text       : the poster's exact text pasted into the app (free text -> 'description')
  image      : the PNG through EasyOCR -> the same app pipeline

This is a small synthetic check: clean computer-rendered text, 40 postings. Real phone photos
of posters will do worse (blur, layout, lighting).

    python scripts/poster_images.py              # 20 fraud + 20 legit, writes results/e2e_image_results.json
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "results" / "poster_images"
RESULTS = PROJECT_ROOT / "results" / "e2e_image_results.json"
SEED = 42
SECTION_CHARS = 350
WIDTH = 1100
MARGIN = 40

FONT_CANDIDATES = ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]


def _font(size: int):
    from PIL import ImageFont
    for p in FONT_CANDIDATES:
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def _s(v: Any) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()


def poster_text(rec: Dict[str, Any]) -> List[str]:
    """Plain poster layout: title line, then short labelled sections. Returns lines (first = title)."""
    lines = [_s(rec.get("title")) or "Job Opening"]
    for label, key in (("Location", "location"), ("Employment type", "employment_type"),
                       ("Salary", "salary_range")):
        if _s(rec.get(key)):
            lines.append(f"{label}: {_s(rec.get(key))}")
    for label, key in (("About us", "company_profile"), ("Job description", "description"),
                       ("Requirements", "requirements"), ("Benefits", "benefits")):
        v = " ".join(_s(rec.get(key)).split())
        if v:
            lines.append(f"{label}: {v[:SECTION_CHARS]}")
    return lines


def render(lines: List[str], path: Path) -> None:
    from PIL import Image, ImageDraw
    title_f, body_f = _font(36), _font(22)
    wrapped = [(textwrap.wrap(lines[0], 45), title_f)] + [(textwrap.wrap(l, 88), body_f) for l in lines[1:]]
    height = MARGIN * 2 + sum(len(w) * (f.size + 10) + 14 for w, f in wrapped)
    img = Image.new("RGB", (WIDTH, height), "white")
    d = ImageDraw.Draw(img)
    y = MARGIN
    for w, f in wrapped:
        for line in w:
            d.text((MARGIN, y), line, fill="black", font=f)
            y += f.size + 10
        y += 14
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def pick(te, n_each: int = 20, seed: int = SEED):
    rng = np.random.default_rng(seed)
    fraud = rng.choice(np.where(te.fraudulent.values == 1)[0], n_each, replace=False)
    legit = rng.choice(np.where(te.fraudulent.values == 0)[0], n_each, replace=False)
    return sorted(fraud.tolist() + legit.tolist())


def _run(fn):
    try:
        r = fn()
        return {"verdict": r["final_verdict"], "p_fake": r["p_fake"], "zone": r["zone"], "error": None,
                "ocr_chars": len(r.get("ocr_text") or "")}
    except Exception as e:
        return {"verdict": None, "p_fake": None, "zone": None, "error": f"{type(e).__name__}: {e}"}


def evaluate(n_each: int = 20, include_richmond: bool = True) -> Dict[str, Any]:
    import mravplus_infer as m4
    import pipeline_bridge as pb
    from ocr_engine import init_ocr_reader

    det = pb.get_detector()
    reader = init_ocr_reader(["en"], gpu=False)
    _, _, te = m4.load_emscad_splits()
    idx = pick(te, n_each)

    rows = []
    t0 = time.time()
    for i in idx:
        rec = te.drop(columns=["text"]).iloc[i].to_dict()
        lines = poster_text(rec)
        text = "\n".join(lines)
        png = OUT_DIR / f"test_{i:04d}_{'fraud' if rec['fraudulent'] else 'legit'}.png"
        render(lines, png)
        s = det.predict("", rec)
        rows.append({
            "test_row": int(i), "y": int(rec["fraudulent"]), "image": str(png.relative_to(PROJECT_ROOT)),
            "structured": {"verdict": "FAKE" if s["predicted_fraud"] else "NOT FAKE", "p_fake": s["p_fake"],
                           "zone": s["zone"], "error": None},
            "text": _run(lambda: pb.handle_ui_submission(None, text, detector=det, with_xai=False)),
            "image_path": _run(lambda: pb.handle_ui_submission(str(png), "", reader=reader, detector=det, with_xai=False)),
        })
    seconds = time.time() - t0

    def metrics(key):
        ok = [r for r in rows if r[key]["verdict"] is not None]
        y = np.array([r["y"] for r in ok])
        p = np.array([r[key]["verdict"] == "FAKE" for r in ok]).astype(int)
        tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
        fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
        return {"n_scored": len(ok), "n_errors": len(rows) - len(ok), "accuracy": float((p == y).mean()) if ok else None,
                "fraud_recall": tp / (tp + fn) if tp + fn else None,
                "fraud_precision": tp / (tp + fp) if tp + fp else None,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def agree(a, b):
        both = [r for r in rows if r[a]["verdict"] is not None and r[b]["verdict"] is not None]
        return {"n": len(both), "agreement": float(np.mean([r[a]["verdict"] == r[b]["verdict"] for r in both])) if both else None}

    out = {
        "n_postings": len(rows), "n_fraud": int(sum(r["y"] for r in rows)), "seed": SEED, "seconds": round(seconds, 1),
        "metrics": {k: metrics(k) for k in ("structured", "text", "image_path")},
        "agreement": {"text_vs_image": agree("text", "image_path"), "structured_vs_text": agree("structured", "text"),
                      "structured_vs_image": agree("structured", "image_path")},
        "errors": [{"test_row": r["test_row"], "path": k, "error": r[k]["error"]}
                   for r in rows for k in ("text", "image_path") if r[k]["error"]],
        "rows": rows,
    }
    if include_richmond and (PROJECT_ROOT / "richmond_offer.png").is_file():
        out["richmond_offer"] = _run(lambda: pb.handle_ui_submission(str(PROJECT_ROOT / "richmond_offer.png"), "",
                                                                     reader=reader, detector=det, with_xai=False))
    return out


if __name__ == "__main__":
    res = evaluate()
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
    print(f"written {RESULTS}")
