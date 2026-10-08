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

import re
from dataclasses import dataclass, field

import numpy as np

from . import prompts
from .catalog import Catalog
from .color import colour_mood_fit, parse_color, preferred_color_score, room_harmony
from .config import EMOART_EMOTIONS, NEGATIVE_EMOTIONS, SETTINGS, Settings
from .measure import Measurement, a4_mask, find_a4, homography_from_a4, measure_box
from .room import Box, RoomAnalysis, analyze_room
from .scoring import aspect_score, combine, effective_weights, fill_score, size_filter, zscore

ORIENTATION_ASPECT = {"portrait": 0.75, "landscape": 1.33, "square": 1.0}

# EMOART_EMOTIONS is ordered in valence/arousal quadrants of three
EMOTION_QUADRANT = {e: i // 3 for i, e in enumerate(EMOART_EMOTIONS)}
QUADRANT_OF = {("positive", "high"): 0, ("negative", "high"): 1, ("negative", "low"): 2, ("positive", "low"): 3}
SUBJECT_STOPWORDS = {"a", "an", "the", "of", "with", "and", "or", "some", "something", "any", "in", "on",
                     "painting", "paintings", "picture", "pictures", "art", "artwork", "artworks", "image",
                     "showing", "shows", "like", "kind", "style", "please", "i", "want", "me", "my"}

# Plain-language names for each score, shown next to its contribution
EXPLANATION_LABELS = {
    "content": "Matches your description",
    "surroundings": "Suits your room's style",
    "harmony": "Colours work with your wall and decor",
    "preferred_color": "Contains your colour",
    "emotion": "Matches the mood",
    "fit": "Fits the space",
    "comfort_penalty": "Less comforting mood",
}


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
    a4_corners: np.ndarray | None = None      # corners the user marked; skips detection


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
    subject_matches: int | None = None  # artworks that clearly show the requested subject


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
        # Some images (often flat abstracts) are close to almost any text in CLIP space;
        # subject scores are measured relative to this generic prompt to cancel that out
        g = encoder.encode_texts(prompts.GENERIC_ART_PROMPTS).mean(0)
        self.generic_sim = catalog.embeddings @ (g / np.linalg.norm(g))
        labels = catalog.meta["emotion"].astype(str).str.lower()
        self.is_negative = labels.isin(NEGATIVE_EMOTIONS).to_numpy()
        self.labels = labels.to_numpy()
        meta = catalog.meta
        self.quadrant = np.array([QUADRANT_OF.get((str(v).lower(), str(a).lower()), -1)
                                  for v, a in zip(meta["valence"], meta["arousal"])])
        self.text_lc = (meta["title"].fillna("").astype(str) + " . "
                        + meta["description"].fillna("").astype(str)).str.lower().to_numpy()

    # -- components ------------------------------------------------------------

    def room_style_probs(self, surroundings: np.ndarray) -> np.ndarray:
        img = self.encoder.encode_images([surroundings])[0]
        logits = 100.0 * (self.room_style_emb @ img)  # CLIP's standard logit scale
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def text_score(self, templates: list[str], text: str, idx: np.ndarray) -> np.ndarray:
        e = self.encoder.encode_prompt_ensemble(templates, text)
        return self.catalog.embeddings[idx] @ e

    def _subject_filter(self, description: str, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Artworks that clearly show what the user asked for: a strong CLIP match, or a
        moderate one that EmoArt's content description confirms ('dog' in its text).
        Returns (kept positions into idx, their content scores, number of real matches)."""
        cfg = self.s.subject
        clip = self.text_score(prompts.CONTENT_TEMPLATES, description, idx) - self.generic_sim[idx]
        z = zscore(clip)
        words = [w for w in re.findall(r"[a-z]+", description.lower()) if w not in SUBJECT_STOPWORDS and len(w) > 2]
        if words:
            pat = re.compile(r"\b(" + "|".join(re.escape(w.rstrip("s")) for w in words) + r")(s|es)?\b")
            keyword = np.array([bool(pat.search(t)) for t in self.text_lc[idx]])
        else:
            keyword = np.zeros(len(idx), bool)
        match = (z >= cfg.z_alone) | (keyword & (z >= cfg.z_with_keyword))
        n_match = int(match.sum())
        if n_match < min(cfg.min_pool, len(idx)):  # too few: take the best matches anyway
            match = np.zeros(len(idx), bool)
            match[np.argsort(-(z + keyword))[:min(cfg.min_pool, len(idx))]] = True
        keep = np.flatnonzero(match)
        return keep, clip[keep], n_match

    def mood_score(self, mood: str, idx: np.ndarray) -> tuple[np.ndarray, dict]:
        """CLIP (descriptive phrases per mood) + EmoArt emotion labels + colour psychology,
        each z-scored and blended. Moods outside the lexicon fall back to CLIP alone."""
        cfg = self.s.mood
        names = prompts.mood_profiles(mood)
        if names:
            phrases = [p for n in names for p in prompts.MOOD_PROFILES[n]["prompts"]]
            e = self.encoder.encode_texts(phrases).mean(0)
            clip = self.catalog.embeddings[idx] @ (e / np.linalg.norm(e))
        else:
            clip = self.text_score(prompts.MOOD_TEMPLATES, mood, idx)
        parts = {"clip": (clip, cfg.clip)}

        targets = prompts.mood_emotions(mood)
        if targets:
            quads = {EMOTION_QUADRANT[t] for t in targets if t in EMOTION_QUADRANT}
            lab = np.where(np.isin(self.labels[idx], targets), 1.0,
                           np.where(np.isin(self.quadrant[idx], list(quads)), 0.5, 0.0))
            parts["label"] = (lab, cfg.label)
        if names:
            fits = np.array([[colour_mood_fit(self.catalog.palette(i), prompts.MOOD_PROFILES[n]) for n in names]
                             for i in idx]).mean(1)
            parts["colour"] = (fits, cfg.colour)
        total_w = sum(w for _, w in parts.values())
        score = sum(w * zscore(v) for v, w in parts.values()) / total_w
        return score, {k: v for k, (v, _) in parts.items()}

    def _diverse_order(self, final: np.ndarray, idx: np.ndarray, top_n: int) -> np.ndarray:
        """Greedy maximal marginal relevance: each next pick is the best-scoring work after a
        penalty for looking like one already picked, at most `max_per_artist` per artist."""
        d = self.s.diversity
        ranked = np.argsort(-final)
        k = min(top_n, d.diversify_first, len(ranked))
        if k <= 1 or d.penalty <= 0:
            return ranked[:top_n]
        pool = ranked[:max(50, 10 * k)]
        emb = self.catalog.embeddings[idx[pool]]
        artists = self.catalog.meta["artist"].to_numpy()[idx[pool]]
        chosen: list[int] = []
        per_artist: dict = {}
        max_sim = np.zeros(len(pool))
        while len(chosen) < k:
            adj = final[pool] - d.penalty * np.maximum(0.0, max_sim - d.floor)
            adj[chosen] = -np.inf
            for i in np.flatnonzero(np.isfinite(adj)):
                if artists[i] and per_artist.get(artists[i], 0) >= d.max_per_artist:
                    adj[i] = -np.inf
            if not np.isfinite(adj).any():
                break
            j = int(np.argmax(adj))
            chosen.append(j)
            if artists[j]:
                per_artist[artists[j]] = per_artist.get(artists[j], 0) + 1
            max_sim = np.maximum(max_sim, emb @ emb[j])
        head = pool[chosen]
        rest = ranked[~np.isin(ranked, head)]
        return np.concatenate([head, rest])[:top_n]

    def _no_fit_message(self, art_w: np.ndarray, art_h: np.ndarray, m: Measurement) -> str:
        n_known = int((~(np.isnan(art_w) | np.isnan(art_h))).sum())
        if n_known == 0:
            return ("None of the artworks in the catalog have a recorded size, so none can be checked "
                    "against your wall. Skip the size check, or allow artworks of unknown size.")
        return (f"None of the {n_known} artworks with a recorded size fit a {m.width_cm:.0f} × "
                f"{m.height_cm:.0f} cm space (leaving {self.s.fit.min_margin_cm:.0f} cm around it). "
                "Try a larger space or check the measurement.")

    # -- main entry --------------------------------------------------------------

    def recommend(self, q: Query, top_n: int | None = None, weights: dict | None = None) -> Result:
        top_n = top_n or self.s.top_n
        weights = weights or self.s.weights.as_dict()
        img = q.room_image
        meta = self.catalog.meta

        # 1) Reference sheet (optional) and room analysis
        corners = q.a4_corners if q.a4_corners is not None else (find_a4(img) if q.use_a4 else None)
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
                          self._no_fit_message(art_w, art_h, measurement))

        # 4) Subject: if the user named one, keep only artworks that show it
        subject_matches = None
        content = None
        if q.description:
            keep_pos, content, subject_matches = self._subject_filter(q.description, idx)
            idx, size_known = idx[keep_pos], size_known[keep_pos]

        # 5) Component scores (None = input not given -> weight redistributed)
        comps: dict[str, np.ndarray | None] = {}
        comps["content"] = content
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
        comps["emotion"], mood_parts = self.mood_score(mood, idx)

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

        # 6) Combine + comfort prior
        # Missing preferred colour: its weight stays with colour (room harmony), so colour
        # keeps its share of the decision instead of being spread over everything
        weights = dict(weights)
        if comps["preferred_color"] is None:
            weights["harmony"] = weights.get("harmony", 0) + weights.get("preferred_color", 0)
            weights["preferred_color"] = 0.0
        final, contrib = combine(comps, weights)
        comfort = np.zeros(len(idx))
        if not prompts.mood_is_negative(q.mood):
            comfort = -self.s.comfort.negative_penalty * self.is_negative[idx]
            final = final + comfort

        # 7) Top-N with explanations, re-ranked for variety
        order = self._diverse_order(final, idx, top_n)
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
                    "mood": {k: float(v[j]) for k, v in mood_parts.items()},
                    "size_checked": bool(measurement) and bool(size_known[j]),
                    "raw": {k: float(v[j]) for k, v in comps.items() if v is not None},
                },
            ))
        n_unknown = int((~size_known).sum()) if measurement else 0
        return Result(recs, room, measurement, corners, len(idx), effective_weights(comps, weights), style_dict,
                      n_size_unknown=n_unknown, subject_matches=subject_matches)
