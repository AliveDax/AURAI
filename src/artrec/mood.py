"""How strongly each artwork conveys each mood, as percentages that add up to 100.

Three signals each give a distribution over the moods in prompts.MOOD_PROFILES:
  clip   - CLIP image vs descriptive phrases per mood, softmax at CLIP's logit scale
  label  - EmoArt's (human-verified) emotion label: P(label | mood), where a mood spreads
           its probability over the emotions it covers (cozy = content or calm), with a
           little for other labels in the same valence/arousal quadrant
  colour - colour psychology: how well the artwork's warmth, lightness and
           colourfulness suit each mood (prompts.MOOD_PROFILES)
They are blended with MoodConfig weights. "Cozy 62%" means the artwork reads as
cozy far more than as any of the other moods; 1/len(moods) means no lean either way.

A mood outside the lexicon ("nostalgic") is added as an extra CLIP-only class.
"""
from __future__ import annotations

import numpy as np

from . import prompts
from .color import colour_mood_fit
from .config import EMOART_EMOTIONS, MoodConfig

CLIP_LOGIT_SCALE = 100.0  # CLIP's own temperature
COLOUR_TEMPERATURE = 0.08  # colour-fit differences of ~0.1 shift the distribution clearly
EMOTION_QUADRANT = {e: i // 3 for i, e in enumerate(EMOART_EMOTIONS)}  # labels come in quadrants of 3
QUADRANT_OF = {("positive", "high"): 0, ("negative", "high"): 1, ("negative", "low"): 2, ("positive", "low"): 3}


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class MoodModel:
    def __init__(self, catalog, encoder, cfg: MoodConfig):
        self.cfg = cfg
        self.encoder = encoder
        self.names = list(prompts.MOOD_PROFILES)
        emb = []
        for n in self.names:
            e = encoder.encode_texts(prompts.MOOD_PROFILES[n]["prompts"]).mean(0)
            emb.append(e / np.linalg.norm(e))
        self.embeddings = catalog.embeddings
        self.clip_logits = CLIP_LOGIT_SCALE * catalog.embeddings @ np.stack(emb).T  # (N, M)

        fit = np.array([[colour_mood_fit(catalog.palette(i), prompts.MOOD_PROFILES[n]) for n in self.names]
                        for i in range(len(catalog))])
        self.colour = softmax(fit / COLOUR_TEMPERATURE)

        meta = catalog.meta
        labels = meta["emotion"].astype(str).str.lower().to_numpy()
        quads = [QUADRANT_OF.get((str(v).lower(), str(a).lower())) for v, a in zip(meta["valence"], meta["arousal"])]
        targets = [set(prompts.mood_emotions(n)) for n in self.names]
        target_quads = [{EMOTION_QUADRANT[e] for e in t if e in EMOTION_QUADRANT} for t in targets]
        lab = np.full((len(meta), len(self.names)), 0.02)
        for i, (label, quad) in enumerate(zip(labels, quads)):
            for m in range(len(self.names)):
                if label in targets[m]:
                    lab[i, m] = 1.0 / len(targets[m])  # a mood covering two emotions gets half each
                elif quad is not None and quad in target_quads[m]:
                    lab[i, m] = 0.1
        self.label = lab / lab.sum(1, keepdims=True)

    def distribution(self, idx: np.ndarray, extra_text: str | None = None) -> tuple[list[str], np.ndarray]:
        """(mood names, P) with P of shape (len(idx), moods) and rows summing to 1."""
        c = self.cfg
        if extra_text is None:
            clip = softmax(self.clip_logits[idx])
            P = (c.clip * clip + c.label * self.label[idx] + c.colour * self.colour[idx]) / (c.clip + c.label + c.colour)
            return self.names, P
        # A mood the lexicon doesn't know: CLIP is the only signal that can judge it
        e = self.encoder.encode_prompt_ensemble(prompts.MOOD_TEMPLATES, extra_text)
        logits = np.column_stack([self.clip_logits[idx], CLIP_LOGIT_SCALE * self.embeddings[idx] @ e])
        return self.names + [extra_text], softmax(logits)

    def match(self, mood: str, idx: np.ndarray) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
        """Share of each artwork's feeling that is the requested mood (summed over the moods
        named in the text). Returns (match, requested names, all names, P)."""
        requested = prompts.mood_profiles(mood)
        if requested:
            names, P = self.distribution(idx)
        else:
            names, P = self.distribution(idx, extra_text=mood.strip())
            requested = [mood.strip()]
        cols = [names.index(n) for n in requested]
        return P[:, cols].sum(1), requested, names, P
