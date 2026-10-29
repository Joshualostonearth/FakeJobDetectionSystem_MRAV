"""
explainability.py — Objective 5: explains how the final verdict was reached.

The explanation is built from the pipeline's own numbers, so every statement in it is
grounded in what the system actually computed:

    1. Evidence attribution — each DST evidence source (retrieved records, text classifier,
       phone call) is scored two ways:
         - leave-one-out: the calibrated P(Real) recomputed with that source removed; the
           difference is how far that source moved the confidence;
         - SHAP (exact Shapley values, shap.ExactExplainer): the source's fair share of the
           move from "no evidence" to the actual confidence, averaged over every order of
           adding sources, so sources that repeat each other share the credit and all
           shares add up to the total.
       The leave-one-out effect ranks the sources in the summaries.
    1b. Text classifier words — SHAP (shap.LinearExplainer) splits the SVM's log-odds of
       scam over its TF-IDF features, which are mapped back onto the words of the offer.
    2. Combination — the combined DST masses, the conflict factor K, the Platt calibration
       and the verdict band the confidence falls in.
    3. Overrides — any rule that replaced the calibrated confidence (critical flags,
       phone call red flags, no record confirming the offer).

An optional local LLM summary (summarize_with_llm) rewrites these facts in plain language.
Its output is rejected if it quotes a number that is not in the facts, and the caller then
falls back to the deterministic summary.

Public API:
    explain_verdict(full_result, scam_probability=None, retrieved_evidence=None,
                    classifier_words=None) -> dict
    explain_classifier_words(text) -> Optional[dict]
    highlight_html(text, classifier_words) -> str
    explain_phone_verification(voice_result) -> dict
    summarize_with_llm(explanation, model=None) -> Optional[str]
"""

import html
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import shap

from obj2dst import combine_all
from confidence_calibrator import (
    calculate_confidence_score,
    VERIFIED_REAL_MIN_CONFIDENCE,
    SCAM_MAX_CONFIDENCE,
    HIGH_RISK_MAX_CONFIDENCE,
)

# Same threshold calculate_confidence_score uses to temper confidence towards 0.5
HIGH_CONFLICT_THRESHOLD: float = 0.4

# Leave-one-out change in P(Real) below which a source is reported as having no real effect
NEGLIGIBLE_EFFECT: float = 0.01

# Leave-one-out change in P(Real) from which an effect is called strong / moderate in words
STRONG_EFFECT: float = 0.20
MODERATE_EFFECT: float = 0.05

# Longest flag text passed to the LLM
MAX_LLM_FLAG_CHARS: int = 160

# Words listed on each side (towards scam / towards legitimate) for the text classifier
TOP_K_WORDS: int = 8
# Of those, how many go into the facts and the LLM summary
WORDS_IN_FACTS: int = 5

VERDICT_LABELS = {
    "VERIFIED_REAL": "Verified Real",
    "UNCERTAIN_NEEDS_VERIFICATION": "Uncertain, needs verification",
    "SUSPICIOUS_FAKE": "Suspicious, likely fake",
    "HIGH_RISK_SCAM": "High-risk scam",
}

SOURCE_LABELS = {
    "legitimate_job_record": "Legitimate job record",
    "scam_report": "Scam report",
    "scam_report_forum": "Scam forum report",
    "government_registry": "Government registry record",
    "linkedin_verified": "LinkedIn record",
    "text_classifier": "Text classifier",
    "voice_call": "Phone verification call",
}

VOICE_SIGNAL_LABELS = {
    "company_reached": "Company was reached by phone",
    "role_actively_hiring": "Company confirmed the role is open",
    "recruiter_recognized": "Company recognised the recruiter",
    "company_denied_role": "Company denied the role exists",
    "suspicious_response": "Response on the call was suspicious",
}


def _verdict_band(confidence: float) -> str:
    """Which verdict band a calibrated P(Real) falls in, with the band's limits."""
    if confidence >= VERIFIED_REAL_MIN_CONFIDENCE:
        return f"at or above {VERIFIED_REAL_MIN_CONFIDENCE:.2f}, the Verified Real band"
    if confidence <= HIGH_RISK_MAX_CONFIDENCE:
        return f"at or below {HIGH_RISK_MAX_CONFIDENCE:.2f}, the High-risk scam band"
    if confidence <= SCAM_MAX_CONFIDENCE:
        return f"between {HIGH_RISK_MAX_CONFIDENCE:.2f} and {SCAM_MAX_CONFIDENCE:.2f}, the Suspicious band"
    return f"between {SCAM_MAX_CONFIDENCE:.2f} and {VERIFIED_REAL_MIN_CONFIDENCE:.2f}, the Uncertain band"


def _record_titles(retrieved_evidence: Optional[List[Dict[str, Any]]]) -> Dict[str, str]:
    """doc_id ("ROW<row_id>", as built by pipeline_bridge) -> record title."""
    titles = {}
    for ev in retrieved_evidence or []:
        meta = ev.get("metadata", {}) or {}
        title = meta.get("title") or meta.get("company_name")
        if title:
            titles[f"ROW{meta.get('row_id', '')}"] = str(title)
    return titles


def _describe_source(mass: Dict[str, Any], docs: List[Dict[str, Any]], titles: Dict[str, str],
                     scam_probability: Optional[float]) -> Dict[str, Any]:
    """Label and plain-language detail for one evidence mass."""
    source = mass.get("source", "unknown")
    label = SOURCE_LABELS.get(source, source.replace("_", " ").capitalize())
    detail = ""

    doc_index = mass.get("doc_index")
    if doc_index is not None and doc_index < len(docs):
        doc = docs[doc_index]
        triple = doc.get("mrav_triple") or {}
        title = titles.get(doc.get("source", ""))
        if title:
            label = f"{label}: {title}"
        detail = f"Credibility {doc.get('credibility_index', 0.0):.2f} ({doc.get('credibility_tier', 'unknown')})"
        if triple.get("explanation") and triple["explanation"] != "NOT_RUN":
            detail += f"; M-RAV alignment: {triple.get('hypothesis', 'Neutral')}"
            if triple.get("hypothesis") == "Contradicts" and mass.get("REAL", 0.0) == 0.0 and mass.get("FAKE", 0.0) == 0.0:
                detail += " (the offer does not match this record, so it is not counted as support)"
    elif source == "text_classifier" and scam_probability is not None:
        detail = f"P(scam) = {scam_probability:.2f} from the offer's wording"
    elif source == "voice_call":
        detail = "Signals extracted from the verification call"

    return {"source": source, "label": label, "detail": detail}


def _effect_strength(effect: float) -> str:
    """An effect on P(Real) in words, so the LLM is not handed numbers it can misrelate."""
    size = abs(effect)
    if size >= STRONG_EFFECT:
        return "strongly"
    if size >= MODERATE_EFFECT:
        return "moderately"
    return "slightly"


def _short_flag(flag: str) -> str:
    """
    Flag text for the LLM: without doc ids and the M-RAV reasoning some flags carry
    ("Offer contradicts EMSCAD legitimate baseline: <full explanation>"), cut to one line.
    """
    text = re.sub(r"\s*\(doc: [^)]*\)", "", flag)
    text = re.sub(r"(Offer contradicts EMSCAD legitimate baseline):.*", r"\1", text, flags=re.S)
    text = " ".join(text.split())
    if len(text) > MAX_LLM_FLAG_CHARS:
        text = text[:MAX_LLM_FLAG_CHARS].rsplit(" ", 1)[0] + "…"
    return text


def _attribute_sources(evidence_masses: List[Dict[str, Any]], base_confidence: float) -> List[float]:
    """Leave-one-out effect of each mass on the calibrated P(Real), before overrides."""
    effects = []
    for i in range(len(evidence_masses)):
        rest = evidence_masses[:i] + evidence_masses[i + 1:]
        # combine_all([]) is total ignorance, so removing the only source is handled too
        without = calculate_confidence_score(combine_all(rest))
        effects.append(round(base_confidence - without, 4))
    return effects


def _shapley_sources(evidence_masses: List[Dict[str, Any]]) -> List[float]:
    """
    Exact Shapley value of each mass for the calibrated P(Real), before overrides.

    The "features" are the sources themselves (1 = included, 0 = left out); the value of a
    set of sources is calculate_confidence_score(combine_all(set)). With at most a handful
    of sources, shap.ExactExplainer evaluates every subset.
    """
    if not evidence_masses:
        return []

    def confidence(included: np.ndarray) -> np.ndarray:
        return np.array([
            calculate_confidence_score(combine_all([m for m, keep in zip(evidence_masses, row) if keep]))
            for row in included
        ])

    n = len(evidence_masses)
    explainer = shap.ExactExplainer(confidence, shap.maskers.Independent(np.zeros((1, n))))
    values = explainer(np.ones((1, n)), silent=True).values[0]
    return [round(float(v), 4) for v in values]


def explain_verdict(
    full_result: Dict[str, Any],
    scam_probability: Optional[float] = None,
    retrieved_evidence: Optional[List[Dict[str, Any]]] = None,
    classifier_words: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Explains a run_full_pipeline() result. classifier_words, from explain_classifier_words(),
    adds the words that drove the text classifier to the facts.

    Returns a dict with:
        verdict, verdict_label, confidence
        factors:     one per evidence source, strongest effect first, each with
                     label, detail, mass_real, mass_fake, effect (leave-one-out change in
                     P(Real)) and shap (Shapley share of P(Real) minus shap_base)
        shap_base:   P(Real) with no evidence at all, where the SHAP shares start from
        flags:       CRITICAL / WARNING flags from Objective 2
        combination: DST masses, conflict, calibrated confidence before overrides, band
        override:    reason, if an override replaced the calibrated confidence
        summary:     one paragraph built from the above
        facts:       the detailed statements behind the summary
        llm_facts:   the same facts for the LLM summary: effects in words, short flags,
                     the final confidence as the only number from the scoring
    """
    fv = full_result["final_verification"]
    dst = full_result["dst_combined_masses"]
    cred = full_result["obj2_credibility"]
    evidence_masses = full_result.get("evidence_masses", [])
    docs = cred.get("evaluated_evidence", [])
    titles = _record_titles(retrieved_evidence)

    verdict = fv["final_verdict"]
    confidence = float(fv["calibrated_confidence"])
    calibrated = calculate_confidence_score(dst) if evidence_masses else confidence
    conflict = float(dst.get("conflict", 0.0))

    factors = []
    effects = _attribute_sources(evidence_masses, calibrated)
    for mass, effect, shap_value in zip(evidence_masses, effects, _shapley_sources(evidence_masses)):
        factor = _describe_source(mass, docs, titles, scam_probability)
        factor.update({
            "mass_real": round(float(mass.get("REAL", 0.0)), 4),
            "mass_fake": round(float(mass.get("FAKE", 0.0)), 4),
            "effect": effect,
            "shap": shap_value,
        })
        factors.append(factor)
    factors.sort(key=lambda f: abs(f["effect"]), reverse=True)

    flags = list(dict.fromkeys(
        f"{f['severity']}: {f['message']}"
        for f in cred["credibility_evaluation"]["flags_triggered"]
        if f["severity"] in ("CRITICAL", "WARNING")
    ))

    override = fv.get("override_reason") if fv.get("override_applied") else None

    # ── Facts: the only statements the summaries may use ─────
    facts = [f"Final verdict: {VERDICT_LABELS.get(verdict, verdict)}, with calibrated confidence "
             f"that the offer is real of {confidence:.2f}."]
    for f in factors:
        if abs(f["effect"]) < NEGLIGIBLE_EFFECT:
            continue
        direction = "towards real" if f["effect"] > 0 else "towards scam"
        facts.append(f"{f['label']} moved the confidence {direction} by {abs(f['effect']):.2f}"
                     + (f" ({f['detail']})." if f["detail"] else "."))
    if not any(abs(f["effect"]) >= NEGLIGIBLE_EFFECT for f in factors):
        facts.append("No single evidence source changed the confidence noticeably.")
    word_facts = _word_facts(classifier_words)
    facts.extend(word_facts)
    facts.append(f"Combined evidence: belief real {dst.get('REAL', 0.0):.2f}, belief fake "
                 f"{dst.get('FAKE', 0.0):.2f}, uncertain {dst.get('UNCERTAIN', 1.0):.2f}, "
                 f"conflict between sources {conflict:.2f}.")
    if conflict > HIGH_CONFLICT_THRESHOLD:
        facts.append("The sources disagreed strongly, so the confidence was pulled towards 0.50.")
    facts.append(f"Calibrated confidence before overrides was {calibrated:.2f}.")
    for flag in flags:
        facts.append(f"Flag raised: {flag}")
    if override:
        facts.append(f"An override rule set the confidence to {confidence:.2f}: {override}")
    facts.append(f"A confidence of {confidence:.2f} is {_verdict_band(confidence)}.")
    if verdict == "UNCERTAIN_NEEDS_VERIFICATION":
        facts.append("The evidence is not enough to decide, so a phone verification call is needed.")

    # ── LLM facts: the same story ranked, with effects in words and short flags ─────
    # The only number is the final confidence, so the LLM cannot relate two numbers wrongly,
    # and the main reason is named so it does not pick its own.
    llm_facts = [f"Verdict: {VERDICT_LABELS.get(verdict, verdict)}. "
                 f"Confidence that the offer is real: {confidence:.0%}."]
    evidence = []
    for f in factors:
        if abs(f["effect"]) >= NEGLIGIBLE_EFFECT:
            direction = "a real offer" if f["effect"] > 0 else "a scam"
            evidence.append(f"{f['label']} {_effect_strength(f['effect'])} points to {direction}.")
    if override:
        # The override set the confidence, so it decided the result over the evidence
        llm_facts.append(f"Main reason: a safety rule changed the result: {_short_flag(override)}")
    elif evidence:
        llm_facts.append(f"Main reason: {evidence.pop(0)}")
    else:
        llm_facts.append("Main reason: no single piece of evidence changed the result noticeably.")
    llm_facts.extend(f"Less important evidence: {e}" for e in evidence)
    llm_facts.extend(word_facts)
    if conflict > HIGH_CONFLICT_THRESHOLD:
        llm_facts.append("The evidence sources disagree strongly with each other.")
    for short in dict.fromkeys(_short_flag(f) for f in flags):
        # Only CRITICAL flags affect the result, through the override above
        if short.startswith("WARNING"):
            llm_facts.append(f"Warning noted, it did not change the score: {short.removeprefix('WARNING: ')}")
        else:
            llm_facts.append(f"Critical warning: {short.removeprefix('CRITICAL: ')}")
    if verdict == "UNCERTAIN_NEEDS_VERIFICATION":
        llm_facts.append("The evidence is not enough to decide, so a phone verification call is needed.")

    return {
        "verdict": verdict,
        "verdict_label": VERDICT_LABELS.get(verdict, verdict),
        "confidence": confidence,
        "factors": factors,
        "flags": flags,
        "combination": {
            "real": round(float(dst.get("REAL", 0.0)), 4),
            "fake": round(float(dst.get("FAKE", 0.0)), 4),
            "uncertain": round(float(dst.get("UNCERTAIN", 1.0)), 4),
            "conflict": round(conflict, 4),
            "high_conflict": conflict > HIGH_CONFLICT_THRESHOLD,
            "calibrated_confidence": calibrated,
            "band": _verdict_band(confidence),
        },
        "override": override,
        "shap_base": calculate_confidence_score(combine_all([])),
        "summary": _summary(verdict, confidence, factors, override),
        "facts": facts,
        "llm_facts": llm_facts,
    }


def _word_facts(classifier_words: Optional[Dict[str, Any]]) -> List[str]:
    """Facts naming the words that pushed the text classifier each way."""
    if not classifier_words:
        return []
    facts = []
    for key, direction in (("top_scam_words", "a scam"), ("top_legit_words", "a real offer")):
        words = [w["word"] for w in classifier_words[key][:WORDS_IN_FACTS]]
        if words:
            facts.append(f"Words in the offer that made the text classifier lean towards {direction}: "
                         f"{', '.join(words)}.")
    return facts


# ── Text classifier words (SHAP on the linear SVM) ─────────

_linear_svm = None


def _svm_as_linear_model(model):
    """
    The classifier as one linear model of the log-odds of scam: (weights, intercept).

    The SVM is calibrated with a sigmoid per cross-validation fold, and each fold's log-odds
    -a_k * (coef_k . x + intercept_k) - b_k is linear in the TF-IDF features x; the folds are
    averaged. The model's P(scam) is the mean of the fold sigmoids, so it differs a little
    from sigmoid of this averaged log-odds.
    """
    global _linear_svm
    if _linear_svm is None or _linear_svm[0] is not model:
        folds = model.steps[-1][1].calibrated_classifiers_
        coef = np.mean([-f.calibrators[0].a_ * f.estimator.coef_.ravel() for f in folds], axis=0)
        intercept = float(np.mean([-f.calibrators[0].a_ * f.estimator.intercept_[0] - f.calibrators[0].b_
                                   for f in folds]))
        _linear_svm = (model, coef, intercept)
    return _linear_svm[1], _linear_svm[2]


def _feature_occurrences(vectorizer, chunks: List[str]) -> Dict[int, List[tuple]]:
    """
    Where each of the vectorizer's features occurs in the text: feature index -> list of
    occurrences, each a tuple of (chunk index, share of that occurrence).

    char_wb n-grams are built inside each whitespace-separated chunk, so each occurrence
    lies in one chunk; a word n-gram is split evenly between the chunks of its n words.
    """
    vocab = vectorizer.vocabulary_
    occurrences = defaultdict(list)
    if vectorizer.analyzer == "char_wb":
        analyze = vectorizer.build_analyzer()
        for i, chunk in enumerate(chunks):
            for gram in analyze(chunk):
                if gram in vocab:
                    occurrences[vocab[gram]].append(((i, 1.0),))
    else:
        preprocess, tokenize = vectorizer.build_preprocessor(), vectorizer.build_tokenizer()
        tokens = [(tok, i) for i, chunk in enumerate(chunks) for tok in tokenize(preprocess(chunk))]
        lo, hi = vectorizer.ngram_range
        for n in range(lo, hi + 1):
            for t in range(len(tokens) - n + 1):
                gram = " ".join(tok for tok, _ in tokens[t:t + n])
                if gram in vocab:
                    occurrences[vocab[gram]].append(tuple((i, 1.0 / n) for _, i in tokens[t:t + n]))
    return occurrences


def explain_classifier_words(text: str, top_k: int = TOP_K_WORDS) -> Optional[Dict[str, Any]]:
    """
    Which words of the offer pushed the text classifier towards scam or legitimate.

    SHAP (shap.LinearExplainer, with an empty posting as the background) gives each TF-IDF
    feature's contribution to the log-odds of scam; each feature's contribution is then
    shared among the words it occurs in. The word contributions add up exactly to
    log_odds - base_log_odds.

    Returns None for empty text, or a dict with:
        scam_probability: the model's P(scam)
        base_log_odds:    log-odds of scam for an empty posting (SHAP's expected value)
        log_odds:         base_log_odds plus all word contributions
        words:            [{text, start, end, contribution}] for each whitespace-separated
                          word of the text in order; contribution > 0 pushes towards scam
        top_scam_words / top_legit_words: [{word, contribution, count}], the same word's
                          occurrences summed (lowercase, punctuation stripped), strongest first
    """
    if not text or not text.strip():
        return None
    from scam_classifier import load_model

    model = load_model()
    coef, intercept = _svm_as_linear_model(model)
    union = model.steps[0][1]
    x = union.transform([text]).tocsr()

    explainer = shap.LinearExplainer((coef, intercept), shap.maskers.Independent(sp.csr_matrix((1, x.shape[1]))))
    shap_values = explainer(x)
    feature_values = np.asarray(shap_values.values).ravel()
    base = float(np.ravel(shap_values.base_values)[0])

    # Whitespace-separated words with their character offsets in the original text
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    chunks = [text[s:e] for s, e in spans]
    word_values = np.zeros(len(chunks))

    offset = 0
    for _, vectorizer in union.transformer_list:
        occurrences = _feature_occurrences(vectorizer, chunks)
        for j in x.indices[(x.indices >= offset) & (x.indices < offset + len(vectorizer.vocabulary_))]:
            occ = occurrences.get(j - offset)
            if not occ:
                continue
            for parts in occ:
                for i, share in parts:
                    word_values[i] += feature_values[j] * share / len(occ)
        offset += len(vectorizer.vocabulary_)

    words = [{"text": chunk, "start": s, "end": e, "contribution": float(c)}
             for chunk, (s, e), c in zip(chunks, spans, word_values)]

    totals = defaultdict(lambda: [0.0, 0])
    for w in words:
        word = re.sub(r"^\W+|\W+$", "", w["text"].lower())
        if word and w["contribution"] != 0.0:
            totals[word][0] += w["contribution"]
            totals[word][1] += 1
    ranked = sorted(({"word": word, "contribution": round(c, 4), "count": n} for word, (c, n) in totals.items()),
                    key=lambda d: d["contribution"])

    return {
        "scam_probability": float(model.predict_proba([text])[0][1]),
        "base_log_odds": base,
        "log_odds": base + float(word_values.sum()),
        "words": words,
        "top_scam_words": [d for d in reversed(ranked) if d["contribution"] > 0][:top_k],
        "top_legit_words": [d for d in ranked if d["contribution"] < 0][:top_k],
    }


def highlight_html(text: str, classifier_words: Dict[str, Any]) -> str:
    """
    The offer text as HTML, each word shaded by its SHAP contribution: red towards scam,
    blue towards legitimate, stronger colour for a bigger contribution.
    """
    words = classifier_words["words"]
    scale = max((abs(w["contribution"]) for w in words), default=0.0) or 1.0
    out: List[str] = []
    pos = 0
    for w in words:
        out.append(html.escape(text[pos:w["start"]]))
        c = w["contribution"]
        alpha = min(abs(c) / scale, 1.0) * 0.75
        rgb = "214, 69, 65" if c > 0 else "50, 96, 128"
        out.append(f'<span title="{c:+.3f}" style="background-color: rgba({rgb}, {alpha:.2f}); '
                   f'border-radius: 2px;">{html.escape(w["text"])}</span>')
        pos = w["end"]
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def _summary(verdict: str, confidence: float, factors: List[Dict[str, Any]], override: Optional[str]) -> str:
    """One deterministic paragraph naming the verdict and the sources that drove it."""
    parts = [f"Verdict: {VERDICT_LABELS.get(verdict, verdict)} "
             f"(confidence the offer is real: {confidence:.0%})."]
    towards_scam = [f["label"] for f in factors if f["effect"] <= -NEGLIGIBLE_EFFECT][:2]
    towards_real = [f["label"] for f in factors if f["effect"] >= NEGLIGIBLE_EFFECT][:2]
    if towards_scam:
        parts.append(f"Pointing to a scam: {', '.join(towards_scam)}.")
    if towards_real:
        parts.append(f"Pointing to a real offer: {', '.join(towards_real)}.")
    if not towards_scam and not towards_real:
        parts.append("None of the evidence moved the confidence noticeably.")
    if override:
        parts.append(f"Override applied: {override}")
    if verdict == "UNCERTAIN_NEEDS_VERIFICATION":
        parts.append("The evidence is not enough to decide; verify by phone.")
    return " ".join(parts)


def explain_phone_verification(voice_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explains a verify_offer_by_phone() result: what each call signal contributed to the
    voice evidence, and the combined belief after adding the call.
    """
    voice_mass = voice_result.get("voice_mass") or {}
    signals = []
    for key, contribution in (voice_mass.get("details") or {}).items():
        if contribution:
            signals.append({
                "label": VOICE_SIGNAL_LABELS.get(key, key.replace("_", " ").capitalize()),
                "contribution": contribution,
            })
    signals.sort(key=lambda s: abs(s["contribution"]), reverse=True)

    facts = []
    for s in signals:
        direction = "towards real" if s["contribution"] > 0 else "towards scam"
        facts.append(f"{s['label']}: {direction} ({s['contribution']:+.2f}).")
    if voice_mass:
        facts.append(f"Call evidence: belief real {voice_mass.get('REAL', 0.0):.2f}, belief fake "
                     f"{voice_mass.get('FAKE', 0.0):.2f}, uncertain {voice_mass.get('UNCERTAIN', 1.0):.2f}.")
        if voice_mass.get("combined_confidence") is not None:
            facts.append(f"Confidence in the call's phone number and transcript: "
                         f"{voice_mass['combined_confidence']:.2f}.")
    else:
        facts.append("The call produced no usable signals, so the verdict uses database evidence only.")
    if voice_result.get("final_belief_real") is not None:
        facts.append(f"After combining the call with the other evidence: belief real "
                     f"{voice_result['final_belief_real']:.2f}, belief fake {voice_result['final_belief_fake']:.2f}, "
                     f"uncertain {voice_result['final_belief_uncertain']:.2f}.")
    return {"signals": signals, "facts": facts}


# ── Optional LLM summary ──────────────────────────────────

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


# Rounding used to compare numbers; finer than 2 places so 99.7% does not pass as 1.00
_PRECISION: int = 4


def _numbers(text: str) -> set:
    """Numbers in text, each also as a fraction, so 0.82 and 82% compare equal."""
    found = set()
    for n in _NUMBER.findall(text):
        value = float(n)
        found.add(round(value, _PRECISION))
        found.add(round(value / 100.0, _PRECISION))  # percentages
    return found


def is_grounded(summary: str, facts: List[str]) -> bool:
    """True if every number the summary quotes appears in the facts."""
    allowed = _numbers(" ".join(facts))
    return all(round(float(n), _PRECISION) in allowed or round(float(n) / 100.0, _PRECISION) in allowed
               for n in _NUMBER.findall(summary))


# Words that claim strength, rank or certainty; a summary may only use those the facts use
CLAIM_WORDS = re.compile(
    r"\b(strongest|strongly|strong|most|main|primary|primarily|key|decisive|decisively|biggest|"
    r"major|identical|exact|exactly|definite|definitive|definitely|certain|certainly|clearly|"
    r"proves?|proof|confirmed|guaranteed?|overwhelming|conclusive)\b",
    re.IGNORECASE,
)

# First word of each verdict label, which the summary must contain
VERDICT_KEYWORDS = {
    "VERIFIED_REAL": "verified",
    "UNCERTAIN_NEEDS_VERIFICATION": "uncertain",
    "SUSPICIOUS_FAKE": "suspicious",
    "HIGH_RISK_SCAM": "scam",
}


def _claim_stems(text: str) -> set:
    """CLAIM_WORDS in text, without "-ly" so "strong" and "strongly" count as the same claim."""
    return {w.lower().removesuffix("ly") for w in CLAIM_WORDS.findall(text)}


def overclaims(summary: str, facts: List[str]) -> List[str]:
    """Strength or certainty words the summary uses that the facts do not."""
    return sorted(_claim_stems(summary) - _claim_stems(" ".join(facts)))


LLM_PROMPT = """You explain to a job seeker why a fake job detection system reached its verdict.
Write 3 to 5 short sentences of plain text: no headings, lists, bold or other formatting.
Use ONLY the facts below and do not add facts, advice or numbers. The only number you may
use is the confidence percentage. Start with the verdict, then the main reason, then the
less important evidence and warnings in the order given. Keep the facts' own strength words
(strongly, moderately, slightly, similar) and do not call anything else the strongest,
main or most important.

Facts:
{facts}

Explanation:"""


# Local Ollama model that writes the plain-language summary (the pipeline itself uses OLLAMA_MODEL)
EXPLANATION_MODEL: str = os.getenv("EXPLANATION_MODEL", "llama3.2")

# Attempts at an LLM summary; after a rejected one the model is told why and asked again
LLM_ATTEMPTS: int = 2

LLM_RETRY_PROMPT = """{prompt} {previous}

That explanation was rejected: {problem}
Rewrite it following the rules above.

Explanation:"""


def _summary_problem(text: str, facts: List[str], verdict: Optional[str]) -> Optional[str]:
    """Why an LLM summary cannot be shown, or None if it passes every check."""
    if not text:
        return "it was empty."
    if not is_grounded(text, facts):
        return "it quoted numbers that are not in the facts."
    extra = overclaims(text, facts)
    if extra:
        return f"it used the words {', '.join(extra)}, which claim more than the facts say."
    keyword = VERDICT_KEYWORDS.get(verdict)
    if keyword and keyword not in text.lower():
        return "it does not name the verdict."
    return None


def summarize_with_llm(explanation: Dict[str, Any], model: Optional[str] = None) -> Optional[str]:
    """
    Plain-language summary of explanation["llm_facts"] from the local Ollama model.

    A summary is rejected if it quotes a number not in the facts, claims strength or
    certainty the facts do not (see CLAIM_WORDS), or does not name the verdict; the model
    then gets one more try with the reason. Returns None if Ollama is unavailable or no
    attempt passes; the caller should then show explanation["summary"] instead.
    """
    facts = explanation.get("llm_facts") or []
    if not facts:
        return None
    prompt = LLM_PROMPT.format(facts="\n".join(f"- {f}" for f in facts))
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=model or EXPLANATION_MODEL, temperature=0)
        request = prompt
        for _ in range(LLM_ATTEMPTS):
            reply = llm.invoke(request)
            text = (getattr(reply, "content", None) or str(reply)).strip()
            # The app shows the summary as escaped text, so drop any Markdown emphasis
            text = re.sub(r"[*#`]+", "", text).strip()
            problem = _summary_problem(text, facts, explanation.get("verdict"))
            if problem is None:
                return text
            print(f"[Warning] LLM explanation rejected: {problem}")
            request = LLM_RETRY_PROMPT.format(prompt=prompt, previous=text, problem=problem)
    except Exception as e:
        print(f"[Warning] LLM explanation unavailable: {e}")
    return None
