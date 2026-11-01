"""
Integrated Gradients for M4's DistilBERT hybrid gate (used only in the REVIEW zone).

Target: the fraud log-odds F = logit_fraud - logit_real, attributed to the word-embedding
layer. Baseline: the same sequence with every token except [CLS]/[SEP] replaced by [PAD]
(metadata held fixed). Completeness: sum(attributions) ~= F(x) - F(baseline); the residual is
reported. The input is tokenised without padding: padded positions are masked out, so the
model output is the same while IG gets much cheaper.

Uses captum's LayerIntegratedGradients when installed, otherwise a Riemann (midpoint) sum over
inputs_embeds, which is the same estimator.

Attention weights are deliberately not used as an explanation.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

DEFAULT_STEPS = 40
# Completeness residual above which the UI warns (relative to |F(x) - F(baseline)|)
RESIDUAL_WARN = 0.05


def _inputs(detector, row):
    import torch
    tok = detector.tokenizer
    enc = tok([row["text"]], truncation=True, max_length=512, return_tensors="pt")
    ids = enc["input_ids"].to(detector.device)
    mask = enc["attention_mask"].to(detector.device)
    meta = torch.tensor(detector.ohe.transform(detector.meta_frame([row])).toarray(),
                        dtype=torch.float32, device=detector.device)
    special = torch.tensor([tok.cls_token_id, tok.sep_token_id], device=ids.device)
    base = torch.where(torch.isin(ids, special), ids, torch.full_like(ids, tok.pad_token_id))
    return ids, mask, meta, base


def _words(tokens: List[str], scores: np.ndarray) -> List[Dict[str, Any]]:
    """Merge WordPiece '##' continuations into words (scores summed); drop [CLS]/[SEP]."""
    words: List[Dict[str, Any]] = []
    for t, s in zip(tokens, scores):
        if t in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if t.startswith("##") and words:
            words[-1]["word"] += t[2:]
            words[-1]["attribution"] += float(s)
        else:
            words.append({"word": t, "attribution": float(s)})
    return words


def integrated_gradients(detector, row: Dict[str, Any], n_steps: int = DEFAULT_STEPS) -> Dict[str, Any]:
    import torch

    model = detector.bert
    ids, mask, meta, base = _inputs(detector, row)

    def F_ids(input_ids, attention_mask, meta_t):
        logits, _ = model(input_ids, attention_mask, meta_t)
        return logits[:, 1] - logits[:, 0]

    with torch.no_grad():
        fx = float(F_ids(ids, mask, meta)[0])
        fb = float(F_ids(base, mask, meta)[0])
        _, w = model(ids, mask, meta)

    method = "captum.LayerIntegratedGradients"
    try:
        from captum.attr import LayerIntegratedGradients
        lig = LayerIntegratedGradients(F_ids, model.bert.embeddings.word_embeddings)
        attr, delta = lig.attribute(inputs=ids, baselines=base, additional_forward_args=(mask, meta),
                                    n_steps=n_steps, internal_batch_size=8, return_convergence_delta=True)
        tok_scores = attr.sum(-1)[0].detach().cpu().numpy()
        delta = float(delta[0])
    except ImportError:
        method = "manual Riemann sum (captum not installed)"
        emb = model.bert.embeddings.word_embeddings
        with torch.no_grad():
            ex, eb = emb(ids), emb(base)
        grads = torch.zeros_like(ex)
        for k in range(n_steps):
            a = (k + 0.5) / n_steps
            e = (eb + a * (ex - eb)).detach().requires_grad_(True)
            logits, _ = model(None, mask, meta, inputs_embeds=e)
            (logits[:, 1] - logits[:, 0]).sum().backward()
            grads += e.grad
        tok_scores = ((ex - eb) * grads / n_steps).sum(-1)[0].detach().cpu().numpy()
        delta = float(tok_scores.sum() - (fx - fb))

    tokens = detector.tokenizer.convert_ids_to_tokens(ids[0].tolist())
    words = _words(tokens, tok_scores)
    gap = fx - fb
    rel = abs(delta) / max(abs(gap), 1e-6)
    order = sorted(range(len(words)), key=lambda i: -words[i]["attribution"])
    return {
        "method": method,
        "n_steps": n_steps,
        "target": "fraud log-odds (logit_fraud - logit_real)",
        "baseline": "[PAD] tokens (keeping [CLS]/[SEP]), same metadata",
        "f_input": fx,
        "f_baseline": fb,
        "attribution_sum": float(tok_scores.sum()),
        "completeness_residual": delta,
        "completeness_relative": rel,
        "completeness_ok": rel <= RESIDUAL_WARN,
        "n_tokens": int(ids.shape[1]),
        "truncated": int(ids.shape[1]) >= 512,
        "words": words,
        "top_fraud_words": [words[i] for i in order[:10] if words[i]["attribution"] > 0],
        "top_legit_words": [words[i] for i in reversed(order) if words[i]["attribution"] < 0][:10],
        "gate_weights": {"text": float(w[0, 0]), "metadata": float(w[0, 1])},
    }
