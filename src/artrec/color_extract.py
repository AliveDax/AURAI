"""Dominant-colour palettes: k-means in LAB, reported as HSV, with neutral flags.

Why LAB for clustering: k-means uses plain Euclidean distance, and in LAB that distance roughly
follows how different two colours *look*. In RGB, two blues that look almost identical can be
far apart while a blue and a purple can be close.

Why HSV for the output: the harmony rules (color_harmony.py) talk about hue angles, saturation
and value, so each cluster centre is converted to hue in degrees [0, 360) and S, V in [0, 1].

Independent of CLIP. Used offline for the whole catalog (scripts/05_extract_palettes.py, cached)
and per request for the room photo(s).
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.color import lab2rgb, rgb2hsv, rgb2lab
from sklearn.cluster import KMeans

from artrec.config import Config

log = logging.getLogger(__name__)

# k-means restarts, fixed by design (not a tuning knob): 10 restarts with a fixed seed
# make the palette stable and reproducible.
KMEANS_N_INIT = 10


@dataclass
class Palette:
    """One image's (or a pooled set of images') palette, largest cluster first."""
    rgb: np.ndarray       # (k, 3) cluster centres in RGB [0, 1], for display
    hsv: np.ndarray       # (k, 3) hue in degrees [0, 360), saturation and value in [0, 1]
    coverage: np.ndarray  # (k,) share of pixels per cluster, sums to 1
    neutral: np.ndarray   # (k,) bool: grey / white / near-black, so the hue means nothing

    def __len__(self) -> int:
        return len(self.coverage)

    def to_hex(self) -> list[str]:
        return ["#%02x%02x%02x" % tuple(int(round(c * 255)) for c in px) for px in self.rgb]


# --------------------------------------------------------------------------- pixels

def load_small_rgb(src, max_side: int) -> np.ndarray:
    """Image (path, file-like or PIL image) -> float array (h, w, 3) in [0, 1], shrunk so the
    longest side is `max_side`. Palettes need no detail, fewer pixels make k-means fast, and the
    resize filter averages away JPEG noise and canvas texture."""
    if isinstance(src, Image.Image):
        img = src.convert("RGB")
    else:
        with Image.open(src) as f:
            f.draft("RGB", (max_side, max_side))  # JPEG: decode at reduced size, much faster
            img = ImageOps.exif_transpose(f).convert("RGB")
    img = img.copy()
    img.thumbnail((max_side, max_side))  # keeps the aspect ratio, never enlarges
    return np.asarray(img, dtype=np.float32) / 255.0


def gray_world(img: np.ndarray) -> np.ndarray:
    """Gray-world white balance: scale each channel so all three have the same mean, which removes
    a global colour cast. OFF by default (config color.white_balance): the artwork will hang under
    the same light as the room, so colours as photographed are the honest baseline."""
    means = img.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.where(means == 0, 1.0, means)
    return np.clip(img * gain, 0.0, 1.0)


# --------------------------------------------------------------------------- palette

def neutral_mask(hsv: np.ndarray, neutral_sat: float, neutral_val: float) -> np.ndarray:
    """Neutral = nearly grey (low saturation) or nearly black (low value): the hue is noise."""
    return (hsv[..., 1] < neutral_sat) | (hsv[..., 2] < neutral_val)


def rgb_to_hsv_deg(rgb: np.ndarray) -> np.ndarray:
    """(n, 3) RGB in [0, 1] -> (n, 3) HSV with the hue in degrees."""
    hsv = rgb2hsv(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    hsv[:, 0] = (hsv[:, 0] * 360.0) % 360.0  # skimage gives the hue in [0, 1]
    return hsv


def palette_from_pixels(pixels: np.ndarray, color_cfg: dict, seed: int) -> Palette:
    """(M, 3) RGB pixels in [0, 1] -> Palette of at most k colours.
    RGB -> LAB, k-means, centres back to RGB -> HSV, coverage = share of pixels per cluster,
    sorted largest first. Clusters below min_coverage (a signature, a highlight) are dropped and
    the coverage renormalized."""
    if len(pixels) == 0:
        raise ValueError("no pixels to build a palette from")
    lab = rgb2lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)

    # A flat image can have fewer distinct colours than k; asking for more clusters than that
    # only gives duplicate centres and a warning.
    k = min(color_cfg["k"], len(np.unique(pixels, axis=0)))
    km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed).fit(lab)

    coverage = np.bincount(km.labels_, minlength=k) / len(lab)
    order = np.argsort(-coverage, kind="stable")
    centres, coverage = km.cluster_centers_[order], coverage[order]
    keep = coverage >= color_cfg["min_coverage"]
    keep[0] = True  # never return an empty palette
    centres, coverage = centres[keep], coverage[keep] / coverage[keep].sum()

    # The mean of LAB points can land slightly outside the sRGB gamut; lab2rgb clips it and warns.
    # Clipping is what we want, so the warning is muted.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rgb = np.clip(lab2rgb(centres.reshape(-1, 1, 3)).reshape(-1, 3), 0.0, 1.0)
    hsv = rgb_to_hsv_deg(rgb)
    return Palette(rgb=rgb.astype(np.float32), hsv=hsv.astype(np.float32),
                   coverage=coverage.astype(np.float32),
                   neutral=neutral_mask(hsv, color_cfg["neutral_sat"], color_cfg["neutral_val"]))


def palette_from_images(images: Sequence, cfg: Config) -> Palette:
    """Palette of one image, or of several pooled together (multiple room photos). Each photo is
    shrunk to the same longest side first, so every photo contributes a similar number of pixels
    and one high-resolution photo cannot dominate."""
    c = cfg.color
    chunks = []
    for src in images:
        img = load_small_rgb(src, c["downsample_px"])
        if c["white_balance"]:
            img = gray_world(img)  # per photo: each photo can have its own colour cast
        chunks.append(img.reshape(-1, 3))
    return palette_from_pixels(np.concatenate(chunks), c, cfg.seed)


def palette_from_hex(hex_colors: Sequence[str], cfg: Config) -> Palette:
    """The user's colour(s) as a palette with equal coverage, so it can stand in for the room
    palette in the harmony score (color_reference_mode: override)."""
    rgb = np.array([[int(h.lstrip("#")[i:i + 2], 16) / 255.0 for i in (0, 2, 4)] for h in hex_colors],
                   dtype=np.float32)
    hsv = rgb_to_hsv_deg(rgb)
    cov = np.full(len(rgb), 1.0 / len(rgb), dtype=np.float32)
    return Palette(rgb, hsv.astype(np.float32), cov,
                   neutral_mask(hsv, cfg.color["neutral_sat"], cfg.color["neutral_val"]))


# --------------------------------------------------------------------------- padded arrays

def stack_palettes(palettes: Sequence[Palette], k: int) -> dict[str, np.ndarray]:
    """List of palettes (each <= k colours) -> fixed-shape arrays, so scoring is one vectorized
    pass over the catalog. Missing slots are padding: coverage 0 and valid False, so they add
    nothing to any coverage-weighted sum."""
    n = len(palettes)
    out = {"rgb": np.zeros((n, k, 3), np.float32), "hsv": np.zeros((n, k, 3), np.float32),
           "coverage": np.zeros((n, k), np.float32), "neutral": np.zeros((n, k), bool),
           "valid": np.zeros((n, k), bool)}
    for i, p in enumerate(palettes):
        m = len(p)
        out["rgb"][i, :m], out["hsv"][i, :m] = p.rgb, p.hsv
        out["coverage"][i, :m], out["neutral"][i, :m] = p.coverage, p.neutral
        out["valid"][i, :m] = True
    return out


def palette_at(arrays: dict[str, np.ndarray], i: int) -> Palette:
    """Row i of the stacked arrays back as a Palette (padding removed), e.g. for display."""
    v = arrays["valid"][i]
    return Palette(arrays["rgb"][i][v], arrays["hsv"][i][v], arrays["coverage"][i][v], arrays["neutral"][i][v])


# --------------------------------------------------------------------------- cache

def palette_settings(cfg: Config) -> dict:
    """The config values that change the cached palettes. Stored inside the cache, so a cache made
    with other settings is never silently reused."""
    c = cfg.color
    return {"k": c["k"], "downsample_px": c["downsample_px"], "min_coverage": c["min_coverage"],
            "white_balance": bool(c["white_balance"]), "seed": cfg.seed}


def palette_cache_path(cfg: Config) -> Path:
    """e.g. embeddings/palettes_k5_px192_cov0.05_wb0.npz. The settings are in the name, so an
    ablation cache (white balance on, another k) can sit next to the baseline."""
    s = palette_settings(cfg)
    name = f"palettes_k{s['k']}_px{s['downsample_px']}_cov{s['min_coverage']}_wb{int(s['white_balance'])}.npz"
    return cfg.path("embeddings_dir") / name


def save_palettes(arrays: dict[str, np.ndarray], catalog: pd.DataFrame, cfg: Config) -> Path:
    """Palette arrays + the catalog row order + the settings, in one .npz."""
    n = len(catalog)
    for name, arr in arrays.items():
        if arr.shape[0] != n:
            raise ValueError(f"palette array {name}: {arr.shape[0]} rows, catalog has {n}")
    if not arrays["valid"][:, 0].all():
        raise ValueError("some artworks have an empty palette; refusing to cache")
    path = palette_cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")  # np.savez adds .npz unless the name ends in it
    np.savez(tmp, **arrays, artwork_id=catalog["artwork_id"].to_numpy(),
             image_path=catalog["image_path"].to_numpy(dtype=str), meta=json.dumps(palette_settings(cfg)))
    tmp.replace(path)  # atomic: an interrupted run never leaves a half-written cache
    return path


def load_palettes(cfg: Config, catalog: pd.DataFrame) -> dict[str, np.ndarray]:
    """Cached palettes, refused unless they line up row-for-row with the catalog and were made with
    the current settings. The neutral flags depend only on two thresholds, so they are recomputed
    from the stored HSV (tuning those thresholds needs no new k-means run)."""
    path = palette_cache_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"No palette cache {path.name}; run scripts/05_extract_palettes.py")
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    meta = json.loads(str(data.pop("meta")))
    if meta != palette_settings(cfg):
        raise ValueError(f"{path.name} was made with {meta}, config wants {palette_settings(cfg)}; "
                         "re-run scripts/05_extract_palettes.py")
    ids, paths = data.pop("artwork_id"), data.pop("image_path")
    same = (len(ids) == len(catalog)
            and (ids == catalog["artwork_id"].to_numpy()).all()
            and (paths == catalog["image_path"].to_numpy(dtype=str)).all())
    if not same:
        raise ValueError(f"{path.name} does not match catalog.csv row order; re-run scripts/05_extract_palettes.py --force")
    data["neutral"] = neutral_mask(data["hsv"], cfg.color["neutral_sat"], cfg.color["neutral_val"]) & data["valid"]
    return data
