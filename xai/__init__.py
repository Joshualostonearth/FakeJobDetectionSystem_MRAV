"""
Objective 5 - explainable AI for the M4 detector. Deterministic and LLM-free.

    from xai import explain
    result["xai"] = explain(detector, text, fields, m4_result)

Components:
  m1             exact linear term attributions of the M1 SVM (xai/linear.py)
  dst            per-source masses + leave-one-source-out effect on the probability
  retrieval      the 5 nearest training postings and their fraud share
  distilbert     Integrated Gradients + gate weights (REVIEW zone only; xai/ig.py)
  summary        plain-language summary from templates
  deletion       faithfulness check, run on demand (deletion_test)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence


from mravplus_infer import (TEXT_COLS, clean, combine_all, knn_mass, platt, source_masses)

from .linear import LinearExplainer
from .render import display_term

SOURCE_LABELS = {
    "classifier": "Text + metadata classifier (TF-IDF SVM)",
    "similar_postings": "Similar known postings (MiniLM nearest neighbours)",
    "distilbert": "DistilBERT + metadata gate",
}
# Fields whose words may be deleted in the faithfulness check (metadata is left alone)
DELETABLE_FIELDS = tuple(TEXT_COLS)

CAVEATS = [
    "M4 was trained and tested on structured EMSCAD job postings; poster/OCR text is unstructured, "
    "so its validated metrics do not directly apply to arbitrary images.",
    "Explanations describe what drove this model's score, not proof that the offer is or is not a scam.",
    "OCR errors can change the text the model sees.",
]

_linear_cache: Dict[int, LinearExplainer] = {}


def linear_explainer(detector) -> LinearExplainer:
    key = id(detector)
    if key not in _linear_cache:
        _linear_cache[key] = LinearExplainer(detector)
    return _linear_cache[key]


# ──────────────────────────────────────────────────────────
# DST decomposition
# ──────────────────────────────────────────────────────────

def dst_decomposition(detector, result: Dict[str, Any]) -> Dict[str, Any]:
    """Each source's mass and how much P(fake) moves when that source is left out."""
    sims = [n["similarity"] for n in result["neighbours"]]
    labs = [n["label"] for n in result["neighbours"]]
    kn = knn_mass(sims, labs)
    used_bert = result["used_distilbert"]
    p2 = result["p_distilbert"] if used_bert else None
    masses = source_masses(result["q_classifier"], kn, detector.r, p2, detector.rb4 if used_bert else 0.0)
    lr = detector.platt_final if used_bert else detector.platt_cheap
    platt_name = "platt_final" if used_bert else "platt_cheap"

    m_all, K_all = combine_all(list(masses.values()))
    p_all = platt(lr, float(m_all[1] - m_all[0]))

    rows = []
    for name, m in masses.items():
        rest = [v for k, v in masses.items() if k != name]
        if rest:
            m_wo, K_wo = combine_all(rest)
            p_wo = platt(lr, float(m_wo[1] - m_wo[0]))
        else:
            m_wo, K_wo, p_wo = None, 0.0, None
        says = {"classifier": result["q_classifier"],
                "similar_postings": result["neighbour_fraud_share"],
                "distilbert": p2}[name]
        rows.append({
            "source": name,
            "label": SOURCE_LABELS[name],
            "mass": {"REAL": float(m[0]), "FAKE": float(m[1]), "UNCERTAIN": float(m[2])},
            "says_p_fake": None if says is None else float(says),
            "p_fake_without": p_wo,
            "effect_on_p_fake": None if p_wo is None else float(p_all - p_wo),
            "conflict_without": float(K_wo),
        })
    return {
        "sources": rows,
        "combined": {"REAL": float(m_all[0]), "FAKE": float(m_all[1]), "UNCERTAIN": float(m_all[2])},
        "conflict": float(K_all),
        "residual_ignorance": float(m_all[2]),
        "p_fake": float(p_all),
        "calibrator": platt_name,
        "note": ("Leave-one-out effects reuse the same Platt calibrator, which was fitted on the full "
                 "fusion, so they are indicative rather than calibrated probabilities."),
    }


# ──────────────────────────────────────────────────────────
# Faithfulness (deletion) check
# ──────────────────────────────────────────────────────────

def _drop_tokens(s: str, drop: set, match_clean: bool) -> str:
    if not s:
        return s
    toks = s.split()
    key = (lambda t: clean(t)) if match_clean else (lambda t: t.lower())
    kept = [t for t in toks if key(t) not in drop]
    return " ".join(kept) if len(kept) != len(toks) else s


def delete_words(text: str, fields: Optional[Dict[str, Any]], words: Sequence[str],
                 match_clean: bool = False):
    drop = set(words)
    new_fields = dict(fields or {})
    for c in DELETABLE_FIELDS:
        v = new_fields.get(c)
        if isinstance(v, str):
            new_fields[c] = _drop_tokens(v, drop, match_clean)
    return _drop_tokens(text or "", drop, match_clean), new_fields


def deletable_tokens(text: str, fields: Optional[Dict[str, Any]]) -> set:
    toks = set((text or "").lower().split())
    for c in DELETABLE_FIELDS:
        v = (fields or {}).get(c)
        if isinstance(v, str):
            toks |= set(v.lower().split())
    return toks


def deletion_test(detector, text: str, fields: Optional[Dict[str, Any]], result: Dict[str, Any],
                  ks: Sequence[int] = (5, 10), seed: int = 0) -> Dict[str, Any]:
    """
    Delete the top-k fraud-leaning words (M1 attributions) and re-score through M4.
    A random-deletion baseline with the same k is reported next to it.
    """
    lin = result.get("xai", {}).get("m1") or linear_explainer(detector).explain(
        result["_row"], field_labels=tuple(detector.FIELDS))
    allowed = deletable_tokens(text, fields)
    ranked = [t for t, v in sorted(lin["token_scores_total"].items(), key=lambda kv: -kv[1])
              if v > 0 and t in allowed]
    pool = sorted(allowed)
    rng = random.Random(seed)
    base = result["p_fake"]
    out = []
    for k in ks:
        top = ranked[:k]
        t2, f2 = delete_words(text, fields, top)
        p_top = detector.predict(t2, f2)["p_fake"] if top else base
        rnd = rng.sample(pool, min(k, len(pool)))
        t3, f3 = delete_words(text, fields, rnd)
        p_rnd = detector.predict(t3, f3)["p_fake"] if rnd else base
        out.append({"k": k, "deleted": top, "p_fake_before": base, "p_fake_after": p_top,
                    "delta": p_top - base, "random_deleted": rnd, "p_fake_after_random": p_rnd,
                    "delta_random": p_rnd - base})
    return {"method": "delete top-k fraud-leaning words (M1 attributions) and re-score with M4",
            "results": out}


# ──────────────────────────────────────────────────────────
# Plain-language summary (templates, no LLM)
# ──────────────────────────────────────────────────────────

def _phrases(items: List[Dict[str, Any]], n: int = 3) -> str:
    terms = [f"\"{display_term(d['term'])}\"" for d in items[:n]]
    return ", ".join(terms) if terms else "none"


# Sum of 'Unknown'-metadata contributions (towards FAKE) above which the summary mentions it
MISSING_META_NOTE = 0.1


def plain_summary(result: Dict[str, Any], m1: Dict[str, Any], dst: Dict[str, Any],
                  bert: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    verdict = "FAKE" if result["predicted_fraud"] else "NOT FAKE"
    lines = [
        f"Verdict: {verdict} (calibrated probability of fraud {result['p_fake']:.0%}, zone {result['zone']}).",
        f"Words pushing towards fraud: {_phrases(m1['top_fraud_terms'])}.",
        f"Words pushing towards a legitimate posting: {_phrases(m1['top_legit_terms'])}.",
    ]
    missing = [d for d in m1["metadata"] if d["feature"].endswith("_Unknown") and d["contribution"] > 0]
    if sum(d["contribution"] for d in missing) > MISSING_META_NOTE:
        names = ", ".join(d["feature"].rsplit("_", 1)[0].replace("_", " ") for d in missing[:3])
        lines.append(
            f"Missing details ({names}) themselves push towards fraud in this model, because fraudulent "
            "training postings often left them empty; filling in the structured fields removes this effect."
        )
    nb = result["neighbours"]
    n_fraud = sum(n["label"] for n in nb)
    lines.append(
        f"{n_fraud} of the {len(nb)} most similar known postings were fraudulent "
        f"(highest similarity {max(n['similarity'] for n in nb):.2f})." if nb else
        "No similar known postings were retrieved."
    )
    if result["used_distilbert"]:
        g = result.get("gate_weights") or {}
        lines.append(
            f"The case was borderline, so the DistilBERT model was consulted: it gave P(fraud) = "
            f"{result['p_distilbert']:.2f}, relying {g.get('text', 0):.0%} on the text and "
            f"{g.get('metadata', 0):.0%} on the metadata."
        )
        if bert and bert.get("top_fraud_words"):
            words = ", ".join(f"\"{w['word']}\"" for w in bert["top_fraud_words"][:3])
            lines.append(f"DistilBERT's most fraud-leaning words: {words}.")
    else:
        lines.append("The cheap evidence was decisive, so DistilBERT was not needed.")
    K = dst["conflict"]
    level = "low" if K < 0.1 else "moderate" if K < 0.3 else "high"
    lines.append(f"Conflict between the evidence sources is {level} (K = {K:.2f}).")
    return {"text": " ".join(lines), "lines": lines, "caveats": list(CAVEATS),
            "source": "template (no LLM)"}


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────

def explain(detector, text: str, fields: Optional[Dict[str, Any]], result: Dict[str, Any],
            with_ig: bool = True, ig_steps: int = 40) -> Dict[str, Any]:
    """`text`/`fields` are the scored input; the prepared row in result['_row'] is what is explained."""
    row = result["_row"]
    m1 = linear_explainer(detector).explain(row, field_labels=tuple(detector.FIELDS))
    dst = dst_decomposition(detector, result)
    retrieval = {
        "neighbours": result["neighbours"],
        "fraud_share": result["neighbour_fraud_share"],
    }
    bert = None
    bert_error = None
    if result["used_distilbert"] and with_ig:
        try:
            from .ig import integrated_gradients
            bert = integrated_gradients(detector, row, n_steps=ig_steps)
        except Exception as e:  # never let the explanation break the verdict
            bert_error = f"Integrated Gradients failed: {e}"
    summary = plain_summary(result, m1, dst, bert)
    warnings: List[str] = []
    if m1["identity_error"] > 1e-6:
        warnings.append(f"Linear attribution identity off by {m1['identity_error']:.2e}.")
    if bert is not None and not bert["completeness_ok"]:
        warnings.append(
            f"Integrated Gradients completeness residual is {bert['completeness_relative']:.0%} of the "
            "output change; word attributions for DistilBERT are approximate.")
    if bert_error:
        warnings.append(bert_error)
    return {"m1": m1, "dst": dst, "retrieval": retrieval, "distilbert": bert, "summary": summary,
            "warnings": warnings}
