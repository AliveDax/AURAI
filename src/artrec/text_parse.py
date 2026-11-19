"""Mood text -> colours, emotion words and a subject description.

Goal: keep colour, emotion and subject separate, so each score means what it claims.
  "calming blue forest" -> colors ['#2f6db3'], emotion 'calming', description 'forest'
1. Colour words (data/color_words.yaml) are found and removed, together with trailing words such
   as "tones" or "colour". They feed the colour score, never the emotion score. A picker colour
   wins over colours mentioned in the text.
2. Emotion words: any surface form listed in data/emotion_lexicon.yaml ("calm", "peaceful",
   "cozy", ...). The matched words, as the user wrote them, become `emotion_text`.
3. What is left, minus stop words, is the subject/genre: `description_text` ("forest"), or None.
At inference the text is NOT limited to the 12 EmoArt labels: the lexicon only decides which
words count as emotional. `emotion_labels` records which label each word belongs to, for
inspection only; it is never used in scoring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from artrec.config import Config, load_yaml

_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass
class ParsedText:
    raw: str
    colors: list[str] = field(default_factory=list)       # hex codes used by the colour score
    color_words: list[str] = field(default_factory=list)  # colour names found in the text
    emotion_text: str | None = None                       # "calming" (split mode, emotion score)
    emotion_labels: list[str] = field(default_factory=list)  # e.g. ['Calm'], inspection only
    description_text: str | None = None                   # "forest" (split mode, description score)
    blend_text: str | None = None                         # "calming forest" (blend mode, one embedding)


def _phrase_pattern(phrases) -> re.Pattern:
    """Whole-word match of any phrase, longest first ("sky blue" before "blue"), so a longer
    phrase is never cut short by a shorter one. Spaces inside a phrase match any whitespace."""
    alts = sorted({p.lower().strip() for p in phrases if p.strip()}, key=len, reverse=True)
    body = "|".join(r"\s+".join(map(re.escape, p.split())) for p in alts)
    return re.compile(rf"(?<![a-z])({body})(?![a-z])")


def normalize_hex(h: str) -> str:
    if not _HEX_RE.match(h.strip()):
        raise ValueError(f"Not a hex colour: {h!r} (expected e.g. '#2a6f97')")
    return "#" + h.strip().lstrip("#").lower()


class TextParser:
    def __init__(self, color_words: dict[str, str], lexicon: dict[str, dict],
                 stopwords, trailing_words):
        self.color_hex = {name.lower(): normalize_hex(h) for name, h in color_words.items()}
        # surface form -> EmoArt label. The label itself and its prompt word count too.
        self.emotion_of: dict[str, str] = {}
        for label, entry in lexicon.items():
            for word in [label, entry.get("prompt_word", "")] + list(entry.get("synonyms", [])):
                if word:
                    self.emotion_of.setdefault(word.lower(), label)
        self.stopwords = {w.lower() for w in stopwords}
        trailing = "|".join(map(re.escape, sorted(trailing_words, key=len, reverse=True)))
        colour_body = _phrase_pattern(self.color_hex).pattern
        # colour word, optionally followed by "tones", "shade", "colour", ...
        self.color_re = re.compile(rf"{colour_body}(?:\s+(?:{trailing})(?![a-z]))?")
        self.emotion_re = _phrase_pattern(self.emotion_of)

    @classmethod
    def from_config(cls, cfg: Config) -> "TextParser":
        return cls(load_yaml(cfg.path("color_words")), load_yaml(cfg.path("emotion_lexicon")),
                   cfg.text["stopwords"], cfg.text["color_trailing_words"])

    def parse(self, text: str | None, color_hex: str | None = None) -> ParsedText:
        raw = text or ""
        t = raw.lower()

        # 1) colours: collect, then blank them out so they reach neither emotion nor description
        color_words = []
        for m in self.color_re.finditer(t):
            name = re.sub(r"\s+", " ", m.group(1))
            if name not in color_words:
                color_words.append(name)
        t = self.color_re.sub(" ", t)

        # 2) emotion words
        emotion_words, labels = [], []
        for m in self.emotion_re.finditer(t):
            word = re.sub(r"\s+", " ", m.group(1))
            if word not in emotion_words:
                emotion_words.append(word)
            if self.emotion_of[word] not in labels:
                labels.append(self.emotion_of[word])
        t = self.emotion_re.sub(" ", t)

        # 3) the rest, minus stop words, is the subject
        content = [w for w in _WORD_RE.findall(t) if w not in self.stopwords]
        description = " ".join(dict.fromkeys(content)) or None  # dict keeps order, drops repeats
        emotion = ", ".join(emotion_words) or None

        # A picker colour wins over colours mentioned in the text
        colors = [normalize_hex(color_hex)] if color_hex else [self.color_hex[w] for w in color_words]
        blend = " ".join(p for p in (emotion, description) if p) or None
        return ParsedText(raw=raw, colors=list(dict.fromkeys(colors)), color_words=color_words,
                          emotion_text=emotion, emotion_labels=labels,
                          description_text=description, blend_text=blend)
