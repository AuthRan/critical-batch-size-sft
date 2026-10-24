"""Gradient noise scale  B_simple = tr(Sigma) / |G|^2  (McCandlish et al. 2018, Appendix A).

Generalised to unequal micro-batch sizes by least squares on
    E|g_b|^2 = |G|^2 + tr(Sigma) / b
over rows (b, |g_b|^2) collected from micro-batches and from their accumulated batch.
Fitting the two parameters jointly avoids taking a ratio of two noisy ratios.
"""
from __future__ import annotations

import numpy as np
import torch


def fit_rows(rows) -> dict:
    """rows: iterable of (b, |g_b|^2). Returns G2, trSigma, B_simple."""
    b = np.asarray([r[0] for r in rows], dtype=float)
    y = np.asarray([r[1] for r in rows], dtype=float)
    X = np.stack([np.ones_like(b), 1.0 / b], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    G2, trS = float(coef[0]), float(coef[1])
    b_max = float(b.max())
    if G2 > 0:
        return {"G2": G2, "trSigma": trS, "B_simple": trS / G2, "flag": "ok", "b_max": b_max, "n_rows": int(len(b))}
    # The intercept |G|^2 is pinned by the largest batches; a non-positive fit means the signal is
    # buried under tr(Sigma)/b even at b_max, so the noise scale is at least of order b_max.
    return {"G2": G2, "trSigma": trS, "B_simple": float("inf"), "flag": "unresolved", "b_max": b_max, "n_rows": int(len(b))}


def collect_rows(model, big_batches, loss_fn, params=None, loss_scale: float = 1.0):
    """Collect (b, |g_b|^2) rows.

    big_batches: list of lists of (batch, b). loss_fn(model, batch) must return the SUM of token
    losses for that micro-batch and b is its number of supervised tokens. Each inner list is
    accumulated (token-weighted) into one big batch, which contributes one more row.
    loss_scale guards fp16 underflow: the loss is multiplied by it and the gradients divided back.
    """
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]
    rows: list[tuple[int, float]] = []
    was_training = model.training
    model.train()
    for big in big_batches:
        B = sum(b for _, b in big)
        acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        for batch, b in big:
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch) / b * loss_scale
            loss.backward()
            g2 = 0.0
            for p, a in zip(params, acc):
                if p.grad is None:
                    continue
                g = p.grad.float() / loss_scale
                g2 += float((g * g).sum())
                a.add_(g, alpha=b / B)
            rows.append((b, g2))
        G2 = sum(float((a * a).sum()) for a in acc)
        rows.append((B, G2))
        del acc
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return rows
