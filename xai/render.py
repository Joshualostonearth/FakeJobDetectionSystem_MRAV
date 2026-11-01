"""HTML helpers for the Objective 5 card (pure functions, no Streamlit import)."""

from __future__ import annotations

import html
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional

_PLACEHOLDER = re.compile(r"#(email|phone|url)_[0-9a-f]{8,}#", re.IGNORECASE)


def display_term(term: str) -> str:
    """EMSCAD anonymised e-mails/phones/URLs as '#email_<sha256>#'; show them as '#email#'."""
    return _PLACEHOLDER.sub(lambda m: f"#{m.group(1).lower()}#", str(term))


FRAUD_RGB = "163, 58, 43"   # red: pushes towards FAKE
LEGIT_RGB = "46, 125, 50"   # green: pushes towards REAL
DEEP_BLUE = "#326080"


def word_scores_from_ig(words: List[Dict]) -> Dict[str, float]:
    """IG word list -> mean attribution per (cleaned) word, for highlighting every occurrence."""
    tot: Dict[str, float] = defaultdict(float)
    n: Dict[str, int] = defaultdict(int)
    for w in words:
        tot[w["word"]] += w["attribution"]
        n[w["word"]] += 1
    return {k: tot[k] / n[k] for k in tot}


def highlight_html(text: str, scores: Dict[str, float], key: Optional[Callable[[str], str]] = None,
                   top_n: int = 15, max_chars: int = 6000) -> str:
    """
    Escape `text` and highlight the tokens whose score is among the top_n strongest in each
    direction. `key` maps a whitespace token to the score dict's key (default: lower-case).
    """
    key = key or (lambda t: t.lower())
    text = (text or "")[:max_chars]
    pos = sorted((v for v in scores.values() if v > 0), reverse=True)[:top_n]
    neg = sorted((v for v in scores.values() if v < 0))[:top_n]
    pos_cut = pos[-1] if pos else float("inf")
    neg_cut = neg[-1] if neg else float("-inf")
    vmax = max([abs(v) for v in pos + neg] or [1.0])

    out = []
    for part in re.split(r"(\s+)", text):
        if not part or part.isspace():
            out.append(html.escape(part).replace("\n", "<br>"))
            continue
        v = scores.get(key(part))
        esc = html.escape(part)
        if v is not None and (v >= pos_cut or v <= neg_cut) and v != 0:
            rgb = FRAUD_RGB if v > 0 else LEGIT_RGB
            alpha = 0.18 + 0.62 * min(abs(v) / vmax, 1.0)
            tip = f"{'towards FAKE' if v > 0 else 'towards REAL'}: {v:+.3f}"
            out.append(f"<span title='{tip}' style='background: rgba({rgb}, {alpha:.2f}); "
                       f"border-radius: 3px; padding: 0 2px;'>{esc}</span>")
        else:
            out.append(esc)
    return "".join(out)


def bars_html(items: List[Dict], label_key: str = "term", value_key: str = "contribution") -> str:
    """Horizontal bar chart (HTML) of signed values; red = towards FAKE, green = towards REAL."""
    if not items:
        return "<p style='font-size: 13px;'>None.</p>"
    vmax = max(abs(d[value_key]) for d in items) or 1.0
    rows = []
    for d in items:
        v = d[value_key]
        rgb = FRAUD_RGB if v > 0 else LEGIT_RGB
        width = 100 * abs(v) / vmax
        rows.append(
            "<div style='display: flex; align-items: center; gap: 8px; margin: 3px 0; font-size: 13px;'>"
            f"<div style='width: 38%; text-align: right; color: {DEEP_BLUE}; overflow: hidden; "
            f"text-overflow: ellipsis; white-space: nowrap;'>{html.escape(display_term(d[label_key]))}</div>"
            "<div style='flex: 1; background: #FFF1E7; border: 1px solid #B5D2E6; height: 14px;'>"
            f"<div style='width: {width:.1f}%; height: 100%; background: rgb({rgb});'></div></div>"
            f"<div style='width: 64px; color: {DEEP_BLUE};'>{v:+.3f}</div></div>"
        )
    return "".join(rows)
