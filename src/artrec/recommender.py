"""End-to-end recommendation for one query.

    room photo ─┬─ wall segmentation ─ empty-space box ─┬─ A4 / manual size ─ size filter
                │                                       └─ wall palette ──────┐
                ├─ decor palette ─────────────────────────────────────────────┼─ colour harmony
                └─ surroundings (wall greyed) ─ CLIP ─ room style ─ art prompts ─ surroundings score
    description ─ CLIP text ─ content score
    colour      ─ LAB ─ preferred-colour score
    mood        ─ CLIP text ─ emotion score  (+ comfort prior from EmoArt valence labels)

All components are z-scored per query and combined with config weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import prompts
from .catalog import Catalog
from .color import parse_color, preferred_color_score, room_harmony
from .config import NEGATIVE_EMOTIONS, SETTINGS, Settings
from .measure import Measurement, a4_mask, find_a4, homography_from_a4, measure_box
from .room import Box, RoomAnalysis, analyze_room
from .scoring import aspect_score, combine, effective_weights, fill_score, size_filter

ORIENTATION_ASPECT = {"portrait": 0.75, "landscape": 1.33, "square": 1.0}


@dataclass
class Query:
    room_image: np.ndarray                    # uint8 RGB
    mood: str | None = None
    description: str | None = None
    preferred_color: str | None = None
    user_box: Box | None = None               # user-chosen placement (x, y, w, h) px
    orientation: str | None = None            # "portrait" | "landscape" | "square"
    manual_size_cm: tuple[float, float] | None = None  # (width, height) of the space
    use_a4: bool = True


@dataclass
class Recommendation:
    rank: int
    art_id: str
    row: dict
    score: float
    contributions: dict[str, float]
    details: dict = field(default_factory=dict)


@dataclass
class Result:
    recommendations: list[Recommendation]
    room: RoomAnalysis
    measurement: Measurement | None
    a4_corners: np.ndarray | None
    n_candidates: int
    weights_used: dict[str, float]
    room_style: dict[str, float]
    message: str = ""
    n_size_unknown: int = 0  # candidates kept although their size couldn't be checked


class Recommender:
    def __init__(self, catalog: Catalog, encoder, segmenter=None, settings: Settings = SETTINGS):
        self.catalog = catalog
        self.encoder = encoder
        self.segmenter = segmenter
        self.s = settings
        self.style_names = list(prompts.ROOM_STYLES)
        self.room_style_emb = encoder.encode_texts([prompts.ROOM_STYLES[n][0] for n in self.style_names])
        art = []
        for n in self.style_names:
            e = encoder.encode_texts(prompts.ROOM_STYLES[n][1]).mean(0)
            art.append(e / np.linalg.norm(e))
        self.art_style_emb = np.stack(art)
        labels = catalog.meta["emotion"].astype(str).str.lower()
        self.is_negative = labels.isin(NEGATIVE_EMOTIONS).to_numpy()

    # -- components ------------------------------------------------------------

    def room_style_probs(self, surroundings: np.ndarray) -> np.ndarray:
        img = self.encoder.encode_images([surroundings])[0]
        logits = 100.0 * (self.room_style_emb @ img)  # CLIP's standard logit scale
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def text_score(self, templates: list[str], text: str, idx: np.ndarray) -> np.ndarray:
        e = self.encoder.encode_prompt_ensemble(templates, text)
        return self.catalog.embeddings[idx] @ e

    # -- main entry --------------------------------------------------------------

    def recommend(self, q: Query, top_n: int | None = None, weights: dict | None = None) -> Result:
        top_n = top_n or self.s.top_n
        weights = weights or self.s.weights.as_dict()
        img = q.room_image
        meta = self.catalog.meta

        # 1) Reference sheet (optional) and room analysis
        corners = find_a4(img) if q.use_a4 else None
        exclude = a4_mask(corners, img.shape[:2]) if corners is not None else None
        room = analyze_room(img, self.s.color, self.segmenter, q.user_box, exclude)

        # 2) Real-world size of the space: manual entry wins, then A4
        measurement = None
        if q.manual_size_cm:
            measurement = Measurement(q.manual_size_cm[0], q.manual_size_cm[1], "manual")
        elif corners is not None and room.space_box is not None:
            measurement = measure_box(homography_from_a4(corners), room.space_box)

        # 3) Hard size filter
        art_w = meta["width_cm"].to_numpy(float)
        art_h = meta["height_cm"].to_numpy(float)
        if measurement:
            keep = size_filter(art_w, art_h, measurement.width_cm, measurement.height_cm, self.s.fit)
        else:
            keep = np.ones(len(meta), bool)
        idx = np.flatnonzero(keep)
        size_known = ~(np.isnan(art_w[idx]) | np.isnan(art_h[idx]))
        style_p = self.room_style_probs(room.surroundings_image)
        style_dict = dict(sorted(zip(self.style_names, style_p.round(3).tolist()), key=lambda kv: -kv[1]))
        if len(idx) == 0:
            return Result([], room, measurement, corners, 0, {}, style_dict,
                          "No artworks in the catalog fit this space. Try a larger space or check the measurement.")

        # 4) Component scores (None = input not given -> weight redistributed)
        comps: dict[str, np.ndarray | None] = {}
        comps["content"] = self.text_score(prompts.CONTENT_TEMPLATES, q.description, idx) if q.description else None
        comps["surroundings"] = (self.catalog.embeddings[idx] @ self.art_style_emb.T) @ style_p

        harmony_parts = [room_harmony(self.catalog.palette(i), room.wall_palette, room.decor_palette, self.s.color)
                         for i in idx]
        comps["harmony"] = np.array([h["score"] for h in harmony_parts])

        target = parse_color(q.preferred_color) if q.preferred_color else None
        comps["preferred_color"] = (
            np.array([preferred_color_score(self.catalog.palette(i), target, self.s.color) for i in idx])
            if target is not None else None
        )

        mood = q.mood or self.s.comfort.default_mood
        comps["emotion"] = self.text_score(prompts.MOOD_TEMPLATES, mood, idx)

        aspect = ORIENTATION_ASPECT.get(q.orientation or "", room.space_aspect)
        if measurement and size_known.any():
            fill = fill_score(art_w[idx], art_h[idx], measurement.width_cm, measurement.height_cm, self.s.fit)
            # Unknown sizes get the median fill of the known ones: neither rewarded nor punished
            fill = np.where(size_known, fill, np.median(fill[size_known]))
            comps["fit"] = 0.5 * fill + 0.5 * aspect_score(self.catalog.aspect[idx], aspect or 1.0, self.s.fit)
        elif aspect:
            comps["fit"] = aspect_score(self.catalog.aspect[idx], aspect, self.s.fit)
        else:
            comps["fit"] = None

        # 5) Combine + comfort prior
        final, contrib = combine(comps, weights)
        comfort = np.zeros(len(idx))
        if not prompts.mood_is_negative(q.mood):
            comfort = -self.s.comfort.negative_penalty * self.is_negative[idx]
            final = final + comfort

        # 6) Top-N with explanations
        order = np.argsort(-final)[:top_n]
        recs = []
        for rank, j in enumerate(order, 1):
            i = int(idx[j])
            c = {k: float(v[j]) for k, v in contrib.items()}
            if comfort[j]:
                c["comfort_penalty"] = float(comfort[j])
            recs.append(Recommendation(
                rank=rank, art_id=str(meta.at[i, "art_id"]), row=meta.iloc[i].to_dict(),
                score=float(final[j]), contributions=c,
                details={
                    "harmony": harmony_parts[j],
                    "palette_hex": self.catalog.palette(i).to_hex(),
                    "size_checked": bool(measurement) and bool(size_known[j]),
                    "raw": {k: float(v[j]) for k, v in comps.items() if v is not None},
                },
            ))
        n_unknown = int((~size_known).sum()) if measurement else 0
        return Result(recs, room, measurement, corners, len(idx), effective_weights(comps, weights), style_dict,
                      n_size_unknown=n_unknown)
