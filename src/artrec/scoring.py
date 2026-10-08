"""Combining the individual scores into one ranking.

Raw scores live on very different scales (CLIP cosine similarities cluster
around 0.15-0.35; our colour scores are 0-1), so each component is z-scored
across the candidate set for this query before weighting. A weight of 0.25
then really means "a quarter of the decision".
"""
from __future__ import annotations

import numpy as np

from .config import FitConfig


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    std = x.std()
    return np.zeros_like(x) if std < 1e-9 else (x - x.mean()) / std


def combine(components: dict[str, np.ndarray | None], weights: dict[str, float]) -> tuple[np.ndarray, dict]:
    """Weighted sum of z-scored components. Components that are None (optional
    input not given) are dropped and the remaining weights renormalised.
    Returns (final_scores, {name: weighted contribution array}) for explanations."""
    active = {k: v for k, v in components.items() if v is not None and weights.get(k, 0) > 0}
    if not active:
        raise ValueError("No score components available")
    total_w = sum(weights[k] for k in active)
    contributions = {k: (weights[k] / total_w) * zscore(v) for k, v in active.items()}
    final = np.sum(list(contributions.values()), axis=0)
    return final, contributions


def effective_weights(components: dict[str, np.ndarray | None], weights: dict[str, float]) -> dict[str, float]:
    active = [k for k, v in components.items() if v is not None and weights.get(k, 0) > 0]
    total = sum(weights[k] for k in active)
    return {k: weights[k] / total for k in active}


# ---------------------------------------------------------------------------
# Size and shape
# ---------------------------------------------------------------------------


def size_filter(art_w_cm: np.ndarray, art_h_cm: np.ndarray, space_w_cm: float, space_h_cm: float,
                cfg: FitConfig) -> np.ndarray:
    """True for artworks that physically fit with a margin. Unknown sizes are excluded:
    we can't tell the user a piece fits if we don't know how big it is."""
    m = cfg.min_margin_cm
    known = ~(np.isnan(art_w_cm) | np.isnan(art_h_cm))
    fits = (art_w_cm + 2 * m <= space_w_cm) & (art_h_cm + 2 * m <= space_h_cm)
    return known & fits


def fill_score(art_w_cm: np.ndarray, art_h_cm: np.ndarray, space_w_cm: float, space_h_cm: float,
               cfg: FitConfig) -> np.ndarray:
    """1.0 when the artwork fills the ideal share (2/3-3/4) of the space in its
    tighter dimension, decreasing smoothly outside that band."""
    fill = np.maximum(np.asarray(art_w_cm, float) / space_w_cm, np.asarray(art_h_cm, float) / space_h_cm)
    lo, hi = cfg.ideal_fill_low, cfg.ideal_fill_high
    below = np.exp(-0.5 * ((fill - lo) / 0.2) ** 2)
    above = np.exp(-0.5 * ((fill - hi) / 0.1) ** 2)
    return np.where(fill < lo, below, np.where(fill > hi, above, 1.0))


def aspect_score(art_aspect: np.ndarray, space_aspect: float, cfg: FitConfig) -> np.ndarray:
    """Portrait art for tall gaps, landscape art for wide gaps."""
    diff = np.abs(np.log(np.asarray(art_aspect, float) / space_aspect))
    return np.exp(-diff / cfg.aspect_tolerance)
