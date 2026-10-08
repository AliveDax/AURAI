"""Thin wrapper around open_clip. The model is used exactly as published
(OpenAI weights, no fine-tuning). All embeddings are L2-normalised, so a dot
product is the cosine similarity."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image

from .config import CLIP_MODEL, CLIP_PRETRAINED


class ClipEncoder:
    def __init__(self, model_name: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED,
                 device: str | None = None):
        import open_clip
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.use_amp = self.device == "cuda"

    def _norm(self, x):
        return (x / x.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    def encode_images(self, images: Iterable, batch_size: int = 32) -> np.ndarray:
        """`images`: PIL images or uint8 numpy arrays."""
        torch = self.torch
        out, batch = [], []

        def flush():
            x = torch.stack(batch).to(self.device)
            with torch.no_grad(), torch.autocast("cuda", enabled=self.use_amp):
                out.append(self._norm(self.model.encode_image(x)))
            batch.clear()

        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            batch.append(self.preprocess(img.convert("RGB")))
            if len(batch) == batch_size:
                flush()
        if batch:
            flush()
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=self.use_amp):
            return self._norm(self.model.encode_text(tokens))

    def encode_prompt_ensemble(self, templates: list[str], value: str) -> np.ndarray:
        """Average several phrasings of the same idea, then re-normalise."""
        e = self.encode_texts([t.format(value) for t in templates]).mean(0)
        return e / np.linalg.norm(e)

    @property
    def dim(self) -> int:
        return int(self.model.visual.output_dim)
