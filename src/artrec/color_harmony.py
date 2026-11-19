"""Colour harmony between a reference palette (the room, or the user's colour) and every artwork.
Fully independent of CLIP. Pure vectorized numpy: one pass over the whole catalog, no loops.

Design:
- Complementary, analogous, triadic, split-complementary and monochromatic hue relations are ALL
  good outcomes: harmony comes both from matching and from deliberate contrast (a bright painting
  in a dull room). The bad case is a hue relation that matches no recognised pattern.
- Neutral colours (grey, white, near-black) have no meaningful hue and go with almost anything,
  so they get a fixed hue score instead.
- Saturation and value use a plateau: similar OR moderately different is fine (a bold accent on a
  muted wall); only an extreme mismatch (pastel room, neon art) is penalised. Whether a contrast
  works depends on proportion (accent vs dominant colour, cf. the 60-30-10 rule), and we do not
  model artwork scale, so the plateau is deliberately wide.
All numbers come from config.yaml and are placeholders until checked against the survey.
"""
from __future__ import annotations

import numpy as np

from artrec.color_extract import Palette


def hue_distance(h1, h2) -> np.ndarray:
    """Circular distance between hues in degrees, in [0, 180]. 10 vs 350 -> 20."""
    d = np.abs(np.asarray(h1, dtype=float) - np.asarray(h2, dtype=float)) % 360.0
    return np.minimum(d, 360.0 - d)


def hue_score(d, ideal_angles, sigma: float) -> np.ndarray:
    """How close a hue distance is to the nearest ideal angle: a Gaussian bump around each angle,
    keep the best. 1.0 exactly on an ideal angle, near 0 halfway between two of them."""
    d = np.asarray(d, dtype=float)[..., None]
    a = np.asarray(ideal_angles, dtype=float)
    return np.exp(-0.5 * ((d - a) / sigma) ** 2).max(axis=-1)


def plateau_score(delta, plateau: float, floor: float) -> np.ndarray:
    """1.0 while |delta| <= plateau, then a straight line down to `floor` at |delta| = 1."""
    x = np.abs(np.asarray(delta, dtype=float))
    decay = 1.0 - (1.0 - floor) * (x - plateau) / (1.0 - plateau)
    return np.where(x <= plateau, 1.0, np.clip(decay, floor, 1.0))


def pair_scores(ref_hsv, ref_neutral, art_hsv, art_neutral, c: dict) -> np.ndarray:
    """Score of every (reference colour, artwork colour) pair.
    ref_*: (R, ...) one reference palette; art_*: (N, K, ...) the whole catalog. Returns (N, R, K).
    pair = w_hue * hue + w_sat * sat + w_val * val, with the hue term replaced by
    neutral_hue_score when either colour is neutral (sat/val terms still apply)."""
    rh, ah = ref_hsv[None, :, None, :], art_hsv[:, None, :, :]          # broadcast to (N, R, K, 3)
    hue = hue_score(hue_distance(rh[..., 0], ah[..., 0]), c["ideal_angles"], c["sigma"])
    either_neutral = ref_neutral[None, :, None] | art_neutral[:, None, :]
    hue = np.where(either_neutral, c["neutral_hue_score"], hue)
    sat = plateau_score(rh[..., 1] - ah[..., 1], c["plateau"], c["floor"])
    val = plateau_score(rh[..., 2] - ah[..., 2], c["plateau"], c["floor"])
    return c["w_hue"] * hue + c["w_sat"] * sat + c["w_val"] * val


def color_scores(ref: Palette, art: dict[str, np.ndarray], c: dict) -> np.ndarray:
    """Harmony of every artwork with the reference palette, shape (N,), in [0, 1].
    weighted_mean (default): sum_i sum_j cov_ref[i] * cov_art[j] * pair(i, j), so large colour areas
    count more than small ones. best_pair: the single best-matching pair (ignores coverage).
    Padding slots have coverage 0 / valid False and so never contribute."""
    pair = pair_scores(ref.hsv, ref.neutral, art["hsv"], art["neutral"], c)
    if c["aggregation"] == "best_pair":
        return np.where(art["valid"][:, None, :], pair, -np.inf).max(axis=(1, 2))
    if c["aggregation"] != "weighted_mean":
        raise ValueError(f"unknown color aggregation {c['aggregation']!r}")
    w = ref.coverage[None, :, None] * art["coverage"][:, None, :]
    # Coverages sum to 1, so w sums to 1 per artwork; dividing anyway keeps it exact after float32 rounding.
    return (pair * w).sum(axis=(1, 2)) / w.sum(axis=(1, 2))


def color_match_scores(user: Palette, art: dict[str, np.ndarray], c: dict) -> np.ndarray:
    """How much of each artwork is the user's colour or a shade of it, shape (N,), in [0, 1]
    (0.4 = about 40% of the painting is that colour family).
    - Chromatic user colour (e.g. blue): an artwork colour counts when it is chromatic too and its
      hue is close (narrow Gaussian on the hue distance). Saturation and brightness may differ a
      lot (wide Gaussians): navy and sky blue still count as shades of blue, but a dull dark brown
      counts only a little as "orange".
    - Neutral user colour (white, grey, black): an artwork colour counts when it is neutral too and
      its brightness (HSV value) is close.
    Per user colour: sum over the artwork's colours of coverage * closeness. Several user colours
    ("blue and gold") are averaged, so an artwork needs all of them to score high."""
    u, a = user.hsv[None, :, None, :], art["hsv"][:, None, :, :]          # (1, U, 1, 3), (N, 1, K, 3)
    gauss = lambda d, sigma: np.exp(-0.5 * (d / sigma) ** 2)  # noqa: E731
    chromatic_close = (gauss(hue_distance(u[..., 0], a[..., 0]), c["color_match_hue_sigma"])
                       * gauss(u[..., 1] - a[..., 1], c["color_match_sat_sigma"])
                       * gauss(u[..., 2] - a[..., 2], c["color_match_value_sigma"]))
    neutral_close = gauss(u[..., 2] - a[..., 2], c["color_match_neutral_value_sigma"])
    art_neutral = art["neutral"][:, None, :]
    close = np.where(user.neutral[None, :, None], art_neutral * neutral_close, ~art_neutral * chromatic_close)
    return (close * art["coverage"][:, None, :]).sum(axis=2).mean(axis=1)
