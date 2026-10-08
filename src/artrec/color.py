"""Colour logic. No ML model here: this is our own, fully explainable colour theory.

Pipeline:
  1. Convert pixels to CIELAB (perceptually uniform, unlike RGB).
  2. k-means in LAB to get a palette: k colours plus the share of pixels each covers.
  3. Convert palette colours to LCh (lightness, chroma, hue) for harmony rules.
  4. Score hue relationships against classic harmony templates.
  5. Neutral colours (white, grey, beige; low chroma) have no meaningful hue,
     so they are handled by a separate rule instead of the hue templates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab, lab2rgb
from sklearn.cluster import KMeans

from .config import ColorConfig

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------


@dataclass
class Palette:
    lab: np.ndarray      # (k, 3) cluster centres in CIELAB
    weights: np.ndarray  # (k,)   share of pixels per cluster, sums to 1

    @property
    def lch(self) -> np.ndarray:
        return lab_to_lch(self.lab)

    def to_hex(self) -> list[str]:
        rgb = np.clip(lab2rgb(self.lab[None, :, :])[0], 0, 1)
        return ["#%02x%02x%02x" % tuple(int(round(c * 255)) for c in px) for px in rgb]


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    lab = np.atleast_2d(lab)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([L, C, h], axis=1)


def image_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> float LAB (H, W, 3)."""
    img = image_rgb.astype(np.float64) / 255.0 if image_rgb.dtype == np.uint8 else image_rgb
    return rgb2lab(img)


def extract_palette(
    image_rgb: np.ndarray,
    k: int,
    mask: np.ndarray | None = None,
    max_pixels: int = 20_000,
    seed: int = 0,
) -> Palette:
    """Dominant colours of an image (optionally only inside `mask`)."""
    lab = image_to_lab(image_rgb).reshape(-1, 3)
    if mask is not None:
        lab = lab[mask.reshape(-1).astype(bool)]
    if len(lab) == 0:
        raise ValueError("No pixels to extract a palette from (empty mask?)")
    rng = np.random.default_rng(seed)
    if len(lab) > max_pixels:
        lab = lab[rng.choice(len(lab), max_pixels, replace=False)]
    k = max(1, min(k, len(np.unique(lab.round(1), axis=0))))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(lab)
    counts = np.bincount(km.labels_, minlength=k).astype(float)
    order = np.argsort(-counts)
    return Palette(lab=km.cluster_centers_[order], weights=counts[order] / counts.sum())


# ---------------------------------------------------------------------------
# Harmony rules
# ---------------------------------------------------------------------------

# (centre of hue difference in degrees, tolerance). Classic colour-wheel schemes.
HARMONY_TEMPLATES = {
    "analogous/monochromatic": (0.0, 30.0),
    "triadic": (120.0, 15.0),
    "split-complementary": (150.0, 12.0),
    "complementary": (180.0, 18.0),
}


def hue_difference(h1, h2):
    d = np.abs(np.asarray(h1) - np.asarray(h2)) % 360.0
    return np.minimum(d, 360.0 - d)  # 0..180


def hue_pair_score(h1, h2) -> np.ndarray:
    """How well two hues fit a known harmony scheme, in [0, 1]."""
    d = hue_difference(h1, h2)
    scores = [np.exp(-0.5 * ((d - c) / w) ** 2) for c, w in HARMONY_TEMPLATES.values()]
    return np.max(scores, axis=0)


def best_template(h1: float, h2: float) -> str:
    d = float(hue_difference(h1, h2))
    return max(HARMONY_TEMPLATES, key=lambda n: np.exp(-0.5 * ((d - HARMONY_TEMPLATES[n][0]) / HARMONY_TEMPLATES[n][1]) ** 2))


def palette_harmony(ref: Palette, art: Palette, cfg: ColorConfig) -> float:
    """Weighted average harmony over all (reference colour, artwork colour) pairs.

    - both chromatic: hue-template score
    - either neutral: neutrals pair with almost anything -> fixed base score
    """
    ref_lch, art_lch = ref.lch, art.lch
    ref_neutral = ref_lch[:, 1] < cfg.neutral_chroma
    art_neutral = art_lch[:, 1] < cfg.neutral_chroma

    pair = hue_pair_score(ref_lch[:, None, 2], art_lch[None, :, 2])
    neutral_pair = ref_neutral[:, None] | art_neutral[None, :]
    pair = np.where(neutral_pair, cfg.neutral_wall_base, pair)

    w = ref.weights[:, None] * art.weights[None, :]
    return float((pair * w).sum() / w.sum())


def lightness_contrast(wall: Palette, art: Palette) -> float:
    """Art should stand out from the wall a little. 0 = same lightness, 1 = >=30 L* apart."""
    l_wall = float((wall.lab[:, 0] * wall.weights).sum())
    l_art = float((art.lab[:, 0] * art.weights).sum())
    return min(1.0, abs(l_wall - l_art) / 30.0)


def room_harmony(art: Palette, wall: Palette, decor: Palette | None, cfg: ColorConfig) -> dict:
    """Harmony of an artwork with the room. Returns the score and its parts so
    the UI and the writeup can explain every recommendation."""
    wall_h = palette_harmony(wall, art, cfg)
    contrast = lightness_contrast(wall, art)
    wall_score = 0.85 * wall_h + 0.15 * contrast
    if decor is None:
        total = wall_score
        decor_h = None
    else:
        decor_h = palette_harmony(decor, art, cfg)
        total = cfg.wall_share * wall_score + (1 - cfg.wall_share) * decor_h
    return {"score": total, "wall": wall_h, "contrast": contrast, "decor": decor_h}


def is_neutral_palette(p: Palette, cfg: ColorConfig) -> bool:
    return float((p.lch[:, 1] * p.weights).sum()) < cfg.neutral_chroma


# ---------------------------------------------------------------------------
# Preferred colour
# ---------------------------------------------------------------------------


def _delta_e(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    a, b = np.broadcast_arrays(np.atleast_2d(lab_a), np.atleast_2d(lab_b))
    return deltaE_ciede2000(a, b)


def preferred_color_score(art: Palette, target_lab: np.ndarray, cfg: ColorConfig) -> float:
    """Share of the artwork covered by colours close to the user's colour.
    A painting that is 40% near-teal beats one with a small teal accent."""
    de = _delta_e(art.lab, target_lab[None, :])
    sim = np.exp(-0.5 * (de / cfg.preferred_sigma) ** 2)
    return float((sim * art.weights).sum())


def color_present(art: Palette, target_lab: np.ndarray, cfg: ColorConfig) -> bool:
    """Objective check used in evaluation: is the colour clearly in the artwork?"""
    de = _delta_e(art.lab, target_lab[None, :])
    return bool(((de < cfg.presence_delta_e) & (art.weights >= cfg.presence_min_share)).any())


# ---------------------------------------------------------------------------
# Parsing user colours ("teal", "#2a9d8f", "42,157,143")
# ---------------------------------------------------------------------------

NAMED_COLORS = {
    "red": "#c0392b", "crimson": "#a4161a", "maroon": "#6d1a1a", "pink": "#e89bb0",
    "orange": "#e67e22", "terracotta": "#c4673f", "peach": "#f4b183", "coral": "#ef7c6b",
    "yellow": "#f1c40f", "mustard": "#c9a227", "gold": "#c9a43b", "beige": "#d9c8a9",
    "cream": "#f2ead3", "brown": "#7b5233", "tan": "#c19a6b", "olive": "#6b7a2e",
    "green": "#3c8d40", "sage": "#9caf88", "emerald": "#1f7a4d", "mint": "#a8d8b9",
    "teal": "#2a9d8f", "turquoise": "#30c5c0", "cyan": "#3fb8d6", "sky blue": "#87bfe0",
    "blue": "#2f6db3", "navy": "#1f2f57", "indigo": "#3f3c8f", "purple": "#7d4b9b",
    "lavender": "#b9a7d6", "violet": "#8a5fbf", "white": "#f5f5f2", "grey": "#8c8c8c",
    "gray": "#8c8c8c", "charcoal": "#3a3a3a", "black": "#1a1a1a",
}


def parse_color(text: str) -> np.ndarray:
    """Return a LAB colour (3,) from a colour name, hex code or 'r,g,b'."""
    t = text.strip().lower()
    if t in NAMED_COLORS:
        t = NAMED_COLORS[t]
    if re.fullmatch(r"#?[0-9a-f]{6}", t):
        t = t.lstrip("#")
        rgb = [int(t[i:i + 2], 16) for i in (0, 2, 4)]
    elif re.fullmatch(r"\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}", t):
        rgb = [int(x) for x in t.split(",")]
    else:
        raise ValueError(f"Unrecognised colour: {text!r}. Use a name like 'teal', or a hex code.")
    return rgb2lab(np.array([[rgb]], dtype=float) / 255.0)[0, 0]
