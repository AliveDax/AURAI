"""Public API: UserInput -> top-N Recommendations with a per-component explanation.

    room photo(s) --CLIP--> room embedding --cosine--> every artwork embedding .... style score
    room photo(s) --k-means--> room palette --harmony rules--> artwork palettes ... color score
    mood text --text_parse--> emotion words --CLIP text--> artworks ............. emotion score
                          +-> subject words --CLIP text--> artworks ............. description score
                          +-> colour words  (or the picker) -> the user's colour

Ranking (rank()): content has priority. With a description ("sea"), paintings that clearly show it
come first. With a user colour (color_reference_mode: two_step, the default), the first picks must
contain that colour or a shade of it; with both ("red sea"), the colour picks must also show the
subject (closest shade if no subject painting has the exact colour).

Artwork embeddings and palettes are computed once offline and cached (scripts 04 and 05). Per
request, CLIP only runs on the room photo(s) and the user's text. Components without input are
dropped and the remaining weights renormalized (scorer.py).

recommend_emotion() is a separate, emotion-only mode for the human check of CLIP's emotion
matching: no room photo, and no style, colour or description score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from artrec.clip_embed import ClipEncoder, l2_normalize, load_embeddings, load_image
from artrec.color_extract import (Palette, load_palettes, neutral_mask, palette_cache_path,
                                  palette_from_hex, palette_from_images)
from artrec.color_harmony import color_match_scores, color_scores
from artrec.config import Config, load_config
from artrec.data_loader import load_catalog
from artrec.scorer import explain, fuse, normalize
from artrec.scorer import top_n as top_indices
from artrec.text_parse import ParsedText, TextParser

log = logging.getLogger(__name__)


@dataclass
class UserInput:
    room_images: list[Any]           # >= 1, required: paths or file-like objects (uploads)
    mood_text: str | None = None     # e.g. "calming" or "a calming blue forest scene"
    color_hex: str | None = None     # explicit picker colour, overrides colours named in the text


@dataclass
class Recommendation:
    rank: int
    artwork_id: int
    image_path: str
    title: str
    artist: str
    style: str
    dominant_emotion: str            # EmoArt ground-truth label: for display only, never scored
    final_score: float
    components: dict[str, dict]      # name -> {raw, norm, weight, share}
    parsed_text: ParsedText          # for debugging: what the text parser made of the mood text
    color_match: float | None = None  # share of the artwork in the user's colour family (None = no colour given)
    picked_for_color: bool = False   # two_step: one of the first picks reserved for the user's colour
    subject_match: bool | None = None  # clearly shows the user's description (None = no description given)


@dataclass
class Resources:
    """Everything loaded once and reused across requests."""
    catalog: pd.DataFrame
    art_emb: np.ndarray              # (N, D) cached CLIP image embeddings, L2-normalized
    palettes: dict[str, np.ndarray]  # cached artwork palettes, padded arrays (N, K, ...)
    encoder: Any                     # ClipEncoder (or a stand-in with the same methods, in tests)


@dataclass
class QueryFeatures:
    """What one request turns into before scoring (computed once, re-scorable many times)."""
    parsed: ParsedText
    room_emb: np.ndarray             # (D,)
    room_palette: Palette
    user_palette: Palette | None     # the user's colour(s), if any
    text_embs: dict[str, np.ndarray] = field(default_factory=dict)  # component name -> (D,)
    generic_emb: np.ndarray | None = None  # "a painting" etc., baseline for the subject check


@dataclass
class Scored:
    raw: dict[str, np.ndarray | None]  # every component, None = inactive
    norm: dict[str, np.ndarray]        # active components, normalized over the catalog
    weights: dict[str, float]          # renormalized weights of the active components
    final: np.ndarray                  # (N,)
    color_match: np.ndarray | None = None  # (N,) share in the user's colour family; None = no user colour
    color_near: np.ndarray | None = None   # (N,) same with a wider hue tolerance: the closest shades
    subject_z: np.ndarray | None = None    # (N,) how clearly each artwork shows the description; None = no description


# Module-level lazy cache: the catalog, embeddings, palettes and the CLIP model are loaded on the
# first request only. Keyed by everything that changes what gets loaded.
_RESOURCES: dict[tuple, Resources] = {}


def load_resources(cfg: Config | None = None, encoder=None) -> Resources:
    cfg = cfg or load_config()
    key = (str(cfg.path("catalog")), cfg.clip["model"], cfg.clip["pretrained"], str(palette_cache_path(cfg)))
    if key not in _RESOURCES:
        catalog = load_catalog(cfg)
        art_emb = load_embeddings(cfg, catalog)
        palettes = load_palettes(cfg, catalog)
        log.info("Loaded %d artworks, embeddings %s, palettes k=%d", len(catalog), art_emb.shape,
                 palettes["hsv"].shape[1])
        _RESOURCES[key] = Resources(catalog, art_emb, palettes, encoder or ClipEncoder.from_config(cfg))
    return _RESOURCES[key]


def analyze(user_input: UserInput, cfg: Config, res: Resources) -> QueryFeatures:
    """Run CLIP and k-means on the room photo(s) and parse + embed the text."""
    if not user_input.room_images:
        raise ValueError("At least one room photo is required.")
    images = [load_image(src, cfg.clip["image_draft_px"]) for src in user_input.room_images]

    # Several photos of the same room: average their embeddings (then re-normalize) and pool
    # their pixels into one palette.
    room_emb = l2_normalize(res.encoder.encode_images(images).mean(axis=0))
    room_palette = palette_from_images(images, cfg)

    parsed = TextParser.from_config(cfg).parse(user_input.mood_text, user_input.color_hex)
    user_palette = palette_from_hex(parsed.colors, cfg) if parsed.colors else None

    t = cfg.text
    if t["text_mode"] not in ("split", "blend"):
        raise ValueError(f"unknown text_mode {t['text_mode']!r}")
    # split: emotion and subject get separate embeddings; blend: one embedding of the whole phrase
    texts: dict[str, tuple[str | None, list[str]]] = (
        {"emotion": (parsed.emotion_text, t["emotion_templates"]),
         "description": (parsed.description_text, t["description_templates"])}
        if t["text_mode"] == "split" else
        {"text": (parsed.blend_text, t["blend_templates"])})
    text_embs = {name: res.encoder.encode_phrases([phrase], templates)[0]
                 for name, (phrase, templates) in texts.items() if phrase}

    # The subject check in rank() needs the description on its own (in either text mode) and a
    # generic "a painting" embedding as its baseline
    generic_emb = None
    if parsed.description_text:
        if "description" not in text_embs:
            text_embs["description"] = res.encoder.encode_phrases([parsed.description_text], t["description_templates"])[0]
        generic_emb = l2_normalize(res.encoder.encode_phrases(cfg.subject["generic_prompts"], None).mean(axis=0))
    return QueryFeatures(parsed, room_emb, room_palette, user_palette, text_embs, generic_emb)


def score(q: QueryFeatures, cfg: Config, res: Resources) -> Scored:
    """Raw component scores for the whole catalog, then normalization and fusion."""
    raw: dict[str, np.ndarray | None] = {}

    # Style: image-to-image cosine between the room and every artwork. Honest note: this captures
    # the overall visual "vibe" / subject resonance of the room against each artwork. It is not
    # object detection and not art-style classification (CLIP is only moderate at fine art style
    # and tends to encode content over style).
    raw["style"] = res.art_emb @ q.room_emb

    # Colour: harmony rules on the palettes. Neutral flags follow the current config thresholds.
    c = cfg.color
    art = dict(res.palettes)
    art["neutral"] = neutral_mask(art["hsv"], c["neutral_sat"], c["neutral_val"]) & art["valid"]
    mode = c["color_reference_mode"]
    color_match = color_match_scores(q.user_palette, art, c) if q.user_palette is not None else None
    if mode == "two_step":
        # The colour score is always harmony with the room; the user's colour acts in rank() instead
        raw["color"] = color_scores(q.room_palette, art, c)
    elif mode == "override":
        # A user colour replaces the room palette as the reference
        raw["color"] = color_scores(q.user_palette or q.room_palette, art, c)
    elif mode == "blend":
        # Harmony with the room, mixed with "contains the user's colour" when one was given
        raw["color"] = color_scores(q.room_palette, art, c)
        if color_match is not None:
            share = c["blend_preference_share"]
            raw["color"] = (1 - share) * raw["color"] + share * color_match
    else:
        raise ValueError(f"unknown color_reference_mode {mode!r}")

    # Text components: cosine between the text embedding and every artwork (None = no such text)
    weights = dict(cfg.scoring["weights"])
    if cfg.text["text_mode"] == "split":
        for name in ("emotion", "description"):
            e = q.text_embs.get(name)
            raw[name] = res.art_emb @ e if e is not None else None
    else:
        # Blend: one embedding of the whole phrase. It carries the emotion weight plus, when the
        # text has a subject, the description weight, so the text keeps the same total share as
        # in split mode and the two modes can be compared fairly.
        e = q.text_embs.get("text")
        raw["text"] = res.art_emb @ e if e is not None else None
        weights["text"] = weights["emotion"] + (weights["description"] if q.parsed.description_text else 0.0)
        weights.pop("emotion"), weights.pop("description")

    final, norm, used = fuse(raw, weights, cfg.scoring["normalization"])

    # For rank(): the closest shades (wider hue tolerance), and how clearly each artwork shows the
    # description: (description similarity - generic similarity) as a z-score over the catalog.
    # Subtracting the generic similarity cancels images that are close to almost any text.
    color_near = (color_match_scores(q.user_palette, art, {**c, "color_match_hue_sigma": c["color_near_hue_sigma"]})
                  if q.user_palette is not None else None)
    subject_z = None
    if q.generic_emb is not None and "description" in q.text_embs:
        d = res.art_emb @ q.text_embs["description"] - res.art_emb @ q.generic_emb
        subject_z = (d - d.mean()) / max(float(d.std()), 1e-12)
    return Scored(raw, norm, used, final, color_match, color_near, subject_z)


def subject_pool(s: Scored, cfg: Config) -> np.ndarray | None:
    """Bool mask of the artworks that clearly show the user's description (subject_z >= min_z), or
    None without a description. If too few pass, the min_pool most subject-like ones are used, so
    there is always a pool of the closest matches to work with."""
    if s.subject_z is None:
        return None
    sub = cfg.subject
    mask = s.subject_z >= sub["min_z"]
    if mask.sum() < sub["min_pool"]:
        log.info("Only %d artworks clearly show the description; using the %d closest", mask.sum(), sub["min_pool"])
        mask = np.zeros(len(mask), bool)
        mask[top_indices(s.subject_z, sub["min_pool"])] = True
    return mask


def rank(s: Scored, n: int, cfg: Config) -> tuple[np.ndarray, int, np.ndarray | None]:
    """The n result indices (best first), how many of them were picked for the user's colour, and
    the subject mask (None without a description).

    Content has priority: whenever the text has a description ("sea"), the artworks that clearly
    show it (subject_pool) come before everything else, so "sea" gives sea paintings, not
    abstracts (unless the user asks for "abstract"). Without a description: the final score.

    With a user colour (two_step), the first color_first_n results are colour picks:
      1. Pool = the subject paintings when there is a description (a red abstract never stands in
         for "red sea"), otherwise the whole catalog.
      2. Colour picks = pool artworks with at least color_match_min_share of the user's colour
         family, ordered by
             (1 - a) * ordering score + a * share of the user's colour      (a = color_first_match_weight)
         both min-max normalized over the candidates. A painting that is mostly the colour beats one
         that only just passes, and description, mood, room style and room harmony still count.
      3. Not enough of those (no red sea painting): the remaining colour slots go to pool artworks
         in the closest shade (same ordering, with the wider-tolerance colour_near), e.g. sea
         paintings in orange or maroon.
    The rest of the list: subject paintings first (if there is a description), then everything else.
    """
    c = cfg.color
    subject = subject_pool(s, cfg)
    pool = np.flatnonzero(subject) if subject is not None else np.arange(len(s.final))

    # Ordering score inside the pool. With a description it also rewards how CLEARLY an artwork
    # shows the subject, so a real seascape beats a wavy abstract that only just passed the check:
    #   base = (1 - b) * final + b * subject confidence   (both min-max over the pool, b = subject.order_weight)
    base = s.final.copy()
    if subject is not None:
        b = cfg.subject["order_weight"]
        base[pool] = (1 - b) * normalize(s.final[pool], "minmax") + b * normalize(s.subject_z[pool], "minmax")

    def best(cands: np.ndarray, colour: np.ndarray, m: int) -> np.ndarray:
        """The m best of cands by (1 - a) * base + a * colour, both min-max normalized over cands."""
        if m <= 0 or len(cands) == 0:
            return cands[:0]
        a = c["color_first_match_weight"]
        g = (1 - a) * normalize(base[cands], "minmax") + a * normalize(colour[cands], "minmax")
        return cands[top_indices(g, m)]

    # Colour picks (two_step with a user colour only)
    first = np.array([], dtype=int)
    k = min(c["color_first_n"], n)
    if c["color_reference_mode"] == "two_step" and s.color_match is not None and k > 0:
        first = best(pool[s.color_match[pool] >= c["color_match_min_share"]], s.color_match, k)
        if len(first) < k:
            log.info("Only %d %s have the requested colour; filling with the closest shades",
                     len(first), "subject paintings" if subject is not None else "artworks")
            first = np.concatenate([first, best(np.setdiff1d(pool, first), s.color_near, k - len(first))])

    taken = set(first.tolist())
    if subject is None:
        order = top_indices(s.final, len(s.final))
    else:  # content first: subject paintings (by base) before everything else (by final)
        others = np.flatnonzero(~subject)
        order = np.concatenate([pool[top_indices(base[pool], len(pool))], others[top_indices(s.final[others], len(others))]])
    rest = [i for i in order if i not in taken][: n - len(first)]
    return np.concatenate([first, np.array(rest, dtype=int)]).astype(int), len(first), subject


def to_recommendations(s: Scored, idx: np.ndarray, res: Resources, parsed: ParsedText,
                       n_color: int = 0, subject: np.ndarray | None = None) -> list[Recommendation]:
    out = []
    for r, i in enumerate(idx, 1):
        row = res.catalog.iloc[int(i)]
        out.append(Recommendation(
            rank=r, artwork_id=int(row["artwork_id"]), image_path=row["image_path"],
            title=row["title"], artist=row["artist"], style=row["style"],
            dominant_emotion=row["dominant_emotion"], final_score=float(s.final[i]),
            components=explain(int(i), s.raw, s.norm, s.weights, s.final), parsed_text=parsed,
            color_match=float(s.color_match[i]) if s.color_match is not None else None,
            picked_for_color=r <= n_color,
            subject_match=bool(subject[i]) if subject is not None else None))
    return out


def recommend(user_input: UserInput, top_n: int | None = None, cfg: Config | None = None,
              resources: Resources | None = None) -> list[Recommendation]:
    """The whole system for one request: analyse the input, score every artwork, return the top N."""
    cfg = cfg or load_config()
    res = resources or load_resources(cfg)
    q = analyze(user_input, cfg, res)
    s = score(q, cfg, res)
    idx, n_color, subject = rank(s, top_n or cfg.scoring["top_n"], cfg)
    return to_recommendations(s, idx, res, q.parsed, n_color, subject)


def recommend_emotion(mood_text: str, top_n: int | None = None, cfg: Config | None = None,
                      resources: Resources | None = None) -> list[Recommendation]:
    """Emotion only, to check with people whether CLIP's emotion matching works.
    No room photo, and no style, colour or description score: every artwork is ranked by
    cosine(emotion text, artwork) alone, with the same emotion templates as the full system.

    The emotion text is the emotion words the text parser finds ("calm blue forest" -> "calm"),
    so colour or subject words can never leak in. If it finds none ("nostalgic" is not in the
    lexicon), the whole text is used as the emotion."""
    cfg = cfg or load_config()
    res = resources or load_resources(cfg)
    text = (mood_text or "").strip()
    if not text:
        raise ValueError("Enter an emotion.")
    found = TextParser.from_config(cfg).parse(text)
    phrase = found.emotion_text or text
    if not found.emotion_text:
        log.info("No lexicon emotion word in %r; using the whole text as the emotion", text)

    emb = res.encoder.encode_phrases([phrase], cfg.text["emotion_templates"])[0]
    raw = {"emotion": res.art_emb @ emb}
    # The only component, so its weight is 1 and the final score is just the normalized cosine
    final, norm, used = fuse(raw, {"emotion": 1.0}, cfg.scoring["normalization"])
    idx = top_indices(final, top_n or cfg.emotion_only["top_n"])
    parsed = ParsedText(raw=text, emotion_text=phrase, emotion_labels=found.emotion_labels)
    return to_recommendations(Scored(raw, norm, used, final), idx, res, parsed)
