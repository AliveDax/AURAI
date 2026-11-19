"""Frozen pretrained CLIP (open_clip): load it, encode images and text, read/write the cache.

CLIP is only a feature extractor here. It is never trained or modified: eval mode, no
gradients, everything under torch.inference_mode(). Every embedding is L2-normalized right
after encoding, so cosine similarity is a plain dot product everywhere else in the code.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from artrec.config import Config

log = logging.getLogger(__name__)

# Validity check, not a tunable: a stored row whose length is further than this from 1 means
# the cache was not normalized (or is corrupt).
NORM_TOLERANCE = 1e-3


def set_seed(seed: int) -> None:
    """Same seed for every library that has randomness."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(allow_cpu: bool = False):
    """The GPU, or a loud error. The CPU is only used when the caller explicitly allows it, and
    even then it is logged as a warning: never a silent fallback."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        log.warning("NO CUDA GPU FOUND - running CLIP on the CPU. This will be very slow.")
        return torch.device("cpu")
    raise RuntimeError("No CUDA GPU available (torch.cuda.is_available() is False). Refusing to fall back "
                       "to the CPU silently. Run scripts/00_check_env.py and install the CUDA build of torch.")


def load_image(src, draft_px: int) -> Image.Image:
    """Open an image (path, file-like upload, or an already open PIL image) as upright RGB.
    - draft(): for JPEGs the decoder skips detail we would throw away anyway (CLIP sees 336 px),
      which makes big phone photos and scans much faster to read.
    - exif_transpose(): phones store photos unrotated plus an orientation flag; without this a
      portrait photo would be analysed lying on its side."""
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    if hasattr(src, "seek"):
        src.seek(0)  # an uploaded file may have been read before (e.g. on a Streamlit rerun)
    with Image.open(src) as img:
        img.draft("RGB", (draft_px, draft_px))
        return ImageOps.exif_transpose(img).convert("RGB")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Unit-length rows. Zero rows stay zero instead of becoming NaN."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


class ClipEncoder:
    """One CLIP model, frozen, with the three operations the system needs:
    images -> embeddings, texts -> embeddings, phrases averaged over prompt templates."""

    def __init__(self, model_name: str, pretrained: str, allow_cpu: bool = False):
        import open_clip
        import torch

        self.torch = torch
        self.name, self.pretrained = model_name, pretrained
        self.device = get_device(allow_cpu)
        # OpenAI's weights were trained with the QuickGELU activation, but the plain model config
        # builds standard GELU. Loading them that way runs (with only a warning) yet gives subtly
        # wrong features, so the pretrained tag's own setting is used.
        quick_gelu = bool(open_clip.get_pretrained_cfg(model_name, pretrained).get("quick_gelu", False))
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, force_quick_gelu=quick_gelu)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)  # frozen: we only ever read features out of it
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @classmethod
    def from_config(cls, cfg: Config, allow_cpu: bool = False) -> "ClipEncoder":
        return cls(cfg.clip["model"], cfg.clip["pretrained"], allow_cpu)

    @property
    def dim(self) -> int:
        return int(self.model.visual.output_dim)

    def encode_tensor(self, x) -> np.ndarray:
        """An already preprocessed batch -> (n, D) float32 unit vectors. fp16 autocast on the GPU
        halves memory and time; the normalization is done in float32."""
        torch = self.torch
        with torch.inference_mode(), torch.autocast("cuda", enabled=self.device.type == "cuda"):
            feats = self.model.encode_image(x.to(self.device, non_blocking=True))
        return l2_normalize(feats.float().cpu().numpy()).astype(np.float32)

    def encode_images(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        """PIL images -> (n, D) unit vectors."""
        out = []
        for i in range(0, len(images), batch_size):
            batch = self.torch.stack([self.preprocess(im) for im in images[i:i + batch_size]])
            out.append(self.encode_tensor(batch))
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Texts -> (n, D) unit vectors. The open_clip tokenizer cuts every text to CLIP's 77-token
        limit (checked by scripts/03_benchmark_clip.py), so long input cannot crash this."""
        torch = self.torch
        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", enabled=self.device.type == "cuda"):
            feats = self.model.encode_text(tokens)
        return l2_normalize(feats.float().cpu().numpy()).astype(np.float32)

    def encode_phrases(self, phrases: Sequence[str], templates: Sequence[str] | None) -> np.ndarray:
        """One embedding per phrase, averaged over prompt templates ("a painting that feels {}").
        Short descriptive prompts match images better than bare words, and averaging several
        templates smooths out the quirks of any single wording. The mean of unit vectors is not
        unit length, so it is re-normalized. No templates -> the phrase is used as it is."""
        templates = list(templates) if templates else ["{}"]
        filled = [t.format(p) for p in phrases for t in templates]
        emb = self.encode_texts(filled).reshape(len(phrases), len(templates), -1)
        return l2_normalize(emb.mean(axis=1)).astype(np.float32)


class ImageFileDataset:
    """Paths -> (preprocessed tensor, row index) for a torch DataLoader. A file that fails to load
    gives None and is logged, so one bad image does not kill a long run. Module-level class so it
    can be pickled into DataLoader worker processes on Windows."""

    def __init__(self, paths: Sequence[Path], preprocess, draft_px: int):
        self.paths, self.preprocess, self.draft_px = list(paths), preprocess, draft_px

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            return self.preprocess(load_image(self.paths[i], self.draft_px)), i
        except Exception as err:  # any read/decode failure: report it, don't crash the batch
            log.warning("Could not load image %s (%s)", self.paths[i], err)
            return None, i


def collate_skip_failed(batch):
    """Stack the images that loaded; return the row indices of the ones that failed separately."""
    import torch
    ok = [(x, i) for x, i in batch if x is not None]
    x = torch.stack([x for x, _ in ok]) if ok else None
    return x, [i for _, i in ok], [i for x, i in batch if x is None]


# --------------------------------------------------------------------------- cache

def cache_stem(model: str, pretrained: str) -> str:
    """e.g. 'emb_ViT-L-14-336_openai'. The model is part of every cache name, so embeddings from
    different models can never be mixed up."""
    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "-", s)  # noqa: E731
    return f"emb_{safe(model)}_{safe(pretrained)}"


def embedding_cache_paths(cfg: Config, model: str | None = None, pretrained: str | None = None) -> tuple[Path, Path]:
    """(embedding .npy, index .csv) for a model; defaults to the model in config.yaml."""
    stem = cache_stem(model or cfg.clip["model"], pretrained or cfg.clip["pretrained"])
    d = cfg.path("embeddings_dir")
    return d / f"{stem}.npy", d / f"{stem}_index.csv"


def catalog_fingerprint(catalog: pd.DataFrame) -> str:
    """Hash of the catalog's row order. Partial checkpoints made for a different catalog must not
    be reused, because row i has to mean the same artwork everywhere."""
    text = "\n".join(f"{a}|{p}" for a, p in zip(catalog["artwork_id"], catalog["image_path"]))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def atomic_save_npy(arr: np.ndarray, path: Path) -> None:
    """Write to a temp file, then rename, so an interrupted run never leaves a half-written file
    that looks valid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr)
    try:
        os.replace(tmp, path)
    except PermissionError as err:
        raise PermissionError(f"Could not write {path}. If OneDrive is syncing this folder it may be "
                              "locking the file: pause sync and re-run.") from err


def check_embeddings(emb: np.ndarray, n_rows: int, what: str = "embeddings") -> None:
    """Raise if the matrix has the wrong row count, non-finite values or non-unit rows."""
    if emb.ndim != 2 or emb.shape[0] != n_rows:
        raise ValueError(f"{what}: shape {emb.shape}, expected ({n_rows}, D) to match catalog.csv")
    if not np.isfinite(emb).all():
        raise ValueError(f"{what}: contains NaN/inf rows")
    norms = np.linalg.norm(emb, axis=1)
    if np.abs(norms - 1).max() > NORM_TOLERANCE:
        raise ValueError(f"{what}: rows are not L2-normalized (norms {norms.min():.4f}..{norms.max():.4f})")


def save_embeddings(emb: np.ndarray, catalog: pd.DataFrame, emb_path: Path, index_path: Path) -> None:
    """Validated embeddings + an index file (artwork_id, image_path) recording which artwork each
    row belongs to, so the alignment with catalog.csv can be checked on load."""
    check_embeddings(emb, len(catalog))
    atomic_save_npy(emb.astype(np.float32), emb_path)
    catalog[["artwork_id", "image_path"]].to_csv(index_path, index=False, encoding="utf-8")


def load_embeddings(cfg: Config, catalog: pd.DataFrame, model: str | None = None,
                    pretrained: str | None = None) -> np.ndarray:
    """Cached artwork embeddings, refused unless they line up row-for-row with the catalog."""
    emb_path, index_path = embedding_cache_paths(cfg, model, pretrained)
    if not emb_path.exists() or not index_path.exists():
        raise FileNotFoundError(f"No embedding cache {emb_path.name}; run scripts/04_embed_artworks.py")
    index = pd.read_csv(index_path, keep_default_na=False)
    same = (len(index) == len(catalog)
            and (index["artwork_id"].to_numpy() == catalog["artwork_id"].to_numpy()).all()
            and (index["image_path"].to_numpy() == catalog["image_path"].to_numpy()).all())
    if not same:
        raise ValueError(f"{index_path.name} does not match catalog.csv row order; "
                         "re-run scripts/04_embed_artworks.py --force")
    emb = np.load(emb_path)
    check_embeddings(emb, len(catalog), emb_path.name)
    return emb
