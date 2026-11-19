"""Shared test helpers. Nothing here needs a GPU, CLIP weights or the dataset."""
import hashlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from artrec.color_extract import palette_from_hex, stack_palettes
from artrec.config import load_config
from artrec.recommend import Resources

D = 32  # embedding size of the fake encoder


def fake_vec(key: str) -> np.ndarray:
    """A deterministic unit vector per string, standing in for a CLIP embedding."""
    seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=D)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeEncoder:
    """Same methods as ClipEncoder. Images map to a vector by their (rounded) mean colour, texts by
    their exact string, so a test knows which artwork *should* be closest."""

    def encode_images(self, images):
        return np.stack([fake_vec("img:" + str(np.asarray(im).reshape(-1, 3).mean(0).round(-1))) for im in images])

    def encode_phrases(self, phrases, templates):
        out = []
        for p in phrases:
            e = np.mean([fake_vec(t.format(p)) for t in (templates or ["{}"])], axis=0)
            out.append(e / np.linalg.norm(e))
        return np.stack(out).astype(np.float32)


@pytest.fixture
def cfg():
    return load_config()


def solid_image(rgb, size=(60, 40)) -> Image.Image:
    return Image.new("RGB", size, tuple(rgb))


@pytest.fixture
def fake_resources(cfg):
    """A 30-artwork catalog with random embeddings and simple palettes."""
    rng = np.random.default_rng(0)
    hexes = ["#c0392b", "#2f6db3", "#e67e22", "#2a9d8f", "#f5f5f2", "#7d4b9b"]
    n = 30
    catalog = pd.DataFrame({
        "artwork_id": np.arange(n), "image_path": [f"dataset/Images/Test/{i}.jpg" for i in range(n)],
        "style": "Test", "filename": [f"{i}.jpg" for i in range(n)], "artist": "X",
        "title": [f"Art {i}" for i in range(n)], "dominant_emotion": "Calm",
    })
    emb = rng.normal(size=(n, D)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    palettes = stack_palettes([palette_from_hex([hexes[i % len(hexes)], "#f5f5f2"][: 1 + i % 2], cfg)
                               for i in range(n)], cfg.color["k"])
    return Resources(catalog, emb, palettes, FakeEncoder())
