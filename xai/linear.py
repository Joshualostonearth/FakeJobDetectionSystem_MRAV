"""
Exact term attributions for M4's M1 classifier.

clf_q is CalibratedClassifierCV(LinearSVC, sigmoid, cv=5): five LinearSVCs, each followed by
its own sigmoid. The averaged decision function

    f(x) = mean_i (w_i . x + c_i) = w_bar . x + c_bar

is linear, so contribution_j = w_bar_j * x_j is exact: sum_j contribution_j + c_bar == f(x).
q itself is mean_i sigmoid(a_i f_i(x) + b_i), which is monotone in each f_i but not exactly a
function of f; the attributions explain f (the "fraud score" before calibration).

Word n-gram features are readable as-is. char_wb n-grams live inside single whitespace tokens,
so each char n-gram's contribution is split over the tokens that contain it, in proportion to
occurrences; word bigrams are split half/half over their two tokens. Both splits conserve the
total, so token scores add up to the text part of f exactly.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

_WS = re.compile(r"\s\s+")


def linear_parts(clf_q) -> Tuple[np.ndarray, float]:
    """(w_bar, c_bar) averaged over the calibrated folds. Handles the attribute rename
    (`base_estimator` in old scikit-learn, `estimator` from 1.2)."""
    ws, cs = [], []
    for cc in clf_q.calibrated_classifiers_:
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator")
        ws.append(np.asarray(est.coef_).ravel())
        cs.append(float(np.ravel(est.intercept_)[0]))
    return np.mean(ws, axis=0), float(np.mean(cs))


def mean_decision_function(clf_q, X) -> np.ndarray:
    vals = []
    for cc in clf_q.calibrated_classifiers_:
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator")
        vals.append(est.decision_function(X))
    return np.mean(vals, axis=0)


class LinearExplainer:
    def __init__(self, detector):
        self.det = detector
        self.w, self.c = linear_parts(detector.clf_q)
        self.names = np.asarray(detector.feats.get_feature_names_out(), dtype=object)
        # FeatureUnion order: [word TF-IDF, char_wb TF-IDF]
        (self.word_name, self.word_vec), (self.char_name, self.char_vec) = detector.feats.transformer_list
        if self.char_vec.analyzer != "char_wb" or self.word_vec.analyzer != "word":
            raise ValueError("Unexpected TF-IDF layout in the bundle (expected [word, char_wb])")
        self.n_word = len(self.word_vec.vocabulary_)
        self.n_char = len(self.char_vec.vocabulary_)
        self.n_text = self.n_word + self.n_char
        self.meta_names = np.asarray(detector.ohe.get_feature_names_out(), dtype=object)
        self._token_re = re.compile(self.word_vec.token_pattern)
        self._char_an = self.char_vec.build_analyzer()

    # ── raw contributions ────────────────────────────────

    def contributions(self, row: Dict[str, Any]) -> Dict[str, Any]:
        X = self.det.m1_matrix([row])
        x = X.getrow(0)
        idx, vals = x.indices, x.data
        contrib = self.w[idx] * vals
        f = float(mean_decision_function(self.det.clf_q, X)[0])
        total = float(contrib.sum() + self.c)
        word = idx < self.n_word
        char = (idx >= self.n_word) & (idx < self.n_text)
        meta = idx >= self.n_text
        return {
            "f": f, "intercept": self.c, "reconstructed": total, "identity_error": abs(total - f),
            "idx": idx, "contrib": contrib, "word_mask": word, "char_mask": char, "meta_mask": meta,
            "word_total": float(contrib[word].sum()), "char_total": float(contrib[char].sum()),
            "meta_total": float(contrib[meta].sum()),
        }

    # ── aggregation to whitespace tokens ─────────────────

    def token_scores(self, row: Dict[str, Any], parts: Dict[str, Any]) -> Dict[str, float]:
        """
        Lower-cased whitespace token -> summed contribution over all its occurrences in the
        'full' text. Sum over tokens == word_total + char_total.
        """
        doc = row["full"].lower()            # both vectorizers lowercase by default
        ws_tokens = _WS.sub(" ", doc).split()

        scores: Dict[str, float] = defaultdict(float)
        idx, contrib = parts["idx"], parts["contrib"]

        # char_wb n-grams: per-token n-gram lists, then split each feature over occurrences
        ngram_occ: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        token_ngrams: Dict[str, Dict[str, int]] = {}
        for tok in set(ws_tokens):
            counts: Dict[str, int] = defaultdict(int)
            for g in self._char_an(tok):
                counts[g] += 1
            token_ngrams[tok] = counts
        tok_freq: Dict[str, int] = defaultdict(int)
        for tok in ws_tokens:
            tok_freq[tok] += 1
        for tok, counts in token_ngrams.items():
            for g, n in counts.items():
                ngram_occ[g].append((tok, n * tok_freq[tok]))
        char_vocab_names = self.names
        for j, cv in zip(idx[parts["char_mask"]], contrib[parts["char_mask"]]):
            g = char_vocab_names[j].split("__", 1)[1]
            occ = ngram_occ.get(g)
            if not occ:
                continue
            tot = sum(n for _, n in occ)
            for tok, n in occ:
                scores[tok] += cv * n / tot

        # word n-grams: sequence of (token, ws-token) pairs in document order
        seq: List[Tuple[str, str]] = []
        for tok in ws_tokens:
            for t in self._token_re.findall(tok):
                seq.append((t, tok))
        uni_pos: Dict[str, List[str]] = defaultdict(list)
        bi_pos: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for i, (t, wt) in enumerate(seq):
            uni_pos[t].append(wt)
            if i + 1 < len(seq):
                bi_pos[f"{t} {seq[i + 1][0]}"].append((wt, seq[i + 1][1]))
        for j, cv in zip(idx[parts["word_mask"]], contrib[parts["word_mask"]]):
            term = char_vocab_names[j].split("__", 1)[1]
            if " " in term:
                occ = bi_pos.get(term)
                if not occ:
                    continue
                for a, b in occ:
                    scores[a] += cv / (2 * len(occ))
                    scores[b] += cv / (2 * len(occ))
            else:
                occ = uni_pos.get(term)
                if not occ:
                    continue
                for wt in occ:
                    scores[wt] += cv / len(occ)
        return dict(scores)

    # ── report ───────────────────────────────────────────

    def explain(self, row: Dict[str, Any], top_k: int = 10, field_labels: Tuple[str, ...] = ()) -> Dict[str, Any]:
        parts = self.contributions(row)
        tok = self.token_scores(row, parts)
        idx, contrib = parts["idx"], parts["contrib"]

        labels = {f"{c}:" for c in field_labels}
        content_tok = {t: v for t, v in tok.items() if t not in labels}
        freq: Dict[str, int] = defaultdict(int)
        for t in _WS.sub(" ", row["full"].lower()).split():
            freq[t] += 1
        per_occ = {t: v / freq[t] for t, v in content_tok.items() if freq.get(t)}

        def top(d: Dict[str, float], positive: bool) -> List[Dict[str, Any]]:
            items = sorted(((k, v) for k, v in d.items() if (v > 0) == positive and v != 0),
                           key=lambda kv: -abs(kv[1]))[:top_k]
            return [{"term": k, "contribution": float(v)} for k, v in items]

        word_feats = {self.names[j].split("__", 1)[1]: float(v)
                      for j, v in zip(idx[parts["word_mask"]], contrib[parts["word_mask"]])}
        meta = [{"feature": str(self.meta_names[j - self.n_text]), "contribution": float(v)}
                for j, v in zip(idx[parts["meta_mask"]], contrib[parts["meta_mask"]])]
        meta.sort(key=lambda d: -abs(d["contribution"]))

        return {
            "decision_function": parts["f"],
            "intercept": parts["intercept"],
            "identity_error": parts["identity_error"],
            "totals": {"word_ngrams": parts["word_total"], "char_ngrams": parts["char_total"],
                       "metadata": parts["meta_total"]},
            "token_sum_error": abs(sum(tok.values()) - parts["word_total"] - parts["char_total"]),
            "top_fraud_terms": top(content_tok, True),
            "top_legit_terms": top(content_tok, False),
            "top_fraud_ngrams": top(word_feats, True),
            "top_legit_ngrams": top(word_feats, False),
            "metadata": meta,
            "token_scores": per_occ,  # per occurrence, for highlighting
            "token_scores_total": content_tok,
        }
