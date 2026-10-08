"""The artwork catalog: metadata + precomputed CLIP embeddings + colour palettes.

Built once by scripts/build_catalog.py, then loaded at query time, so a query
only has to process the room photo and the user's text.

Required metadata columns:
  art_id, image_path, title, artist, style, emotion, valence, arousal,
  therapy, description, height_cm, width_cm   (sizes may be NaN if unknown)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .color import Palette
from .config import CATALOG_PARQUET, EMBEDDINGS_NPY, PALETTES_NPZ

REQUIRED_COLUMNS = [
    "art_id", "image_path", "title", "artist", "style", "emotion", "valence",
    "arousal", "therapy", "description", "height_cm", "width_cm",
    "img_width_px", "img_height_px",
]


@dataclass
class Catalog:
    meta: pd.DataFrame          # one row per artwork, REQUIRED_COLUMNS
    embeddings: np.ndarray      # (N, D) L2-normalised CLIP image embeddings
    palette_lab: np.ndarray     # (N, K, 3)
    palette_weights: np.ndarray  # (N, K)

    def __len__(self) -> int:
        return len(self.meta)

    def palette(self, i: int) -> Palette:
        w = self.palette_weights[i]
        keep = w > 0
        return Palette(self.palette_lab[i][keep], w[keep] / w[keep].sum())

    @property
    def aspect(self) -> np.ndarray:
        """width / height, from real dimensions when known, else from the image."""
        real = self.meta["width_cm"] / self.meta["height_cm"]
        img = self.meta["img_width_px"] / self.meta["img_height_px"]
        return real.fillna(img).to_numpy(float)

    @classmethod
    def load(cls, meta_path: Path = CATALOG_PARQUET, emb_path: Path = EMBEDDINGS_NPY,
             pal_path: Path = PALETTES_NPZ) -> "Catalog":
        meta = pd.read_parquet(meta_path)
        missing = set(REQUIRED_COLUMNS) - set(meta.columns)
        if missing:
            raise ValueError(f"Catalog is missing columns: {sorted(missing)}")
        pal = np.load(pal_path)
        cat = cls(meta.reset_index(drop=True), np.load(emb_path), pal["lab"], pal["weights"])
        cat.validate()
        return cat

    def save(self, meta_path: Path = CATALOG_PARQUET, emb_path: Path = EMBEDDINGS_NPY,
             pal_path: Path = PALETTES_NPZ) -> None:
        self.validate()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta.to_parquet(meta_path, index=False)
        np.save(emb_path, self.embeddings.astype(np.float16))
        np.savez_compressed(pal_path, lab=self.palette_lab.astype(np.float32),
                            weights=self.palette_weights.astype(np.float32))

    def validate(self) -> None:
        n = len(self.meta)
        assert self.embeddings.shape[0] == n, "embeddings / metadata length mismatch"
        assert self.palette_lab.shape[0] == n and self.palette_weights.shape[0] == n, \
            "palettes / metadata length mismatch"
        self.embeddings = self.embeddings.astype(np.float32)
