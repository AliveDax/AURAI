"""Per-component scores -> one ranking, with an explanation for every result.

Raw scores live on different scales: CLIP text-image cosines sit in a narrow band (~0.15-0.35),
image-image cosines are wider, the colour score is already ~[0, 1]. Adding raw values would make
the weights meaningless, so each ACTIVE component is first normalized across the whole catalog.
Then the weights of the active components are renormalized to sum to 1 and
    final = sum_c w_c * norm_c
Everything is one vectorized pass over the catalog; the weights are constants during a request.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import rankdata

log = logging.getLogger(__name__)


def normalize(x: np.ndarray, method: str) -> np.ndarray:
    """minmax: (x - min) / (max - min), in [0, 1]. A constant array (no information) gives 0.5.
    rank:   rank / (N - 1), in [0, 1]; robust to outliers, ties share their average rank.
    none:   raw values (an ablation only: the weights then mean little)."""
    x = np.asarray(x, dtype=float)
    if method == "none":
        return x
    if method == "minmax":
        lo, hi = x.min(), x.max()
        return np.full_like(x, 0.5) if hi - lo < 1e-12 else (x - lo) / (hi - lo)
    if method == "rank":
        return np.full_like(x, 0.5) if len(x) < 2 else (rankdata(x) - 1) / (len(x) - 1)
    raise ValueError(f"unknown normalization {method!r}")


def active_weights(raw: dict[str, np.ndarray | None], weights: dict[str, float]) -> dict[str, float]:
    """Weights of the components that have a score, renormalized to sum to 1. A component without
    input (no mood text -> no emotion score) is dropped, and that is logged, never silent."""
    active = {k: float(weights[k]) for k, v in raw.items() if v is not None and weights.get(k, 0) > 0}
    dropped = [k for k, v in raw.items() if v is None]
    if dropped:
        log.info("Inactive components (no input): %s; weights renormalized over %s", dropped, list(active))
    total = sum(active.values())
    if total <= 0:
        raise ValueError("no active score component with a positive weight")
    return {k: w / total for k, w in active.items()}


def fuse(raw: dict[str, np.ndarray | None], weights: dict[str, float], method: str
         ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Returns (final scores (N,), normalized active components, the renormalized weights used)."""
    w = active_weights(raw, weights)
    norm = {k: normalize(raw[k], method) for k in w}
    final = np.sum([w[k] * norm[k] for k in w], axis=0)
    return final, norm, w


def top_n(final: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n highest final scores, best first. A stable sort keeps the catalog order
    for exact ties, so the result is deterministic."""
    return np.argsort(-final, kind="stable")[:n]


def explain(i: int, raw: dict, norm: dict, weights: dict, final: np.ndarray) -> dict[str, dict]:
    """Per component for artwork i: raw score, normalized score, weight and its share of the final
    score (w_c * norm_c / final), so the UI can say e.g. 'color 52%, emotion 31%, style 17%'."""
    total = float(final[i])
    out = {}
    for k, w in weights.items():
        part = w * float(norm[k][i])
        out[k] = {"raw": float(raw[k][i]), "norm": float(norm[k][i]), "weight": w,
                  "share": part / total if total > 0 else 0.0}
    return out
