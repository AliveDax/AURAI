"""Parse artwork dimension strings from museum/WikiArt metadata into centimetres.

Handles formats like:
  "73 x 92 cm"
  "73.0 × 92.0 cm"
  "H. 30 x W. 25 in. (76.2 x 63.5 cm)"
  "29 1/2 x 36 1/4 in."
  "Overall: 45 x 60 cm (17 3/4 x 23 5/8 in.)"
Convention (museums): height first, then width.
Returns (height_cm, width_cm) or None.
"""
from __future__ import annotations

import re
from fractions import Fraction

_NUM = r"\d+(?:\.\d+)?(?:\s+\d+/\d+)?|\d+/\d+"
_PAIR = re.compile(
    rf"(?:H\.?\s*)?({_NUM})\s*[x×X]\s*(?:W\.?\s*)?({_NUM})(?:\s*[x×X]\s*(?:D\.?\s*)?(?:{_NUM}))?\s*(cm|mm|in\.?|inches|m)\b",
    re.IGNORECASE,
)
_TO_CM = {"cm": 1.0, "mm": 0.1, "m": 100.0, "in": 2.54, "in.": 2.54, "inches": 2.54}


def _num(s: str) -> float:
    s = s.strip()
    total = 0.0
    for part in s.split():
        total += float(Fraction(part)) if "/" in part else float(part)
    return total


def parse_dimensions(text: str | None) -> tuple[float, float] | None:
    if not text or not isinstance(text, str):
        return None
    matches = list(_PAIR.finditer(text))
    if not matches:
        return None
    # Prefer a metric measurement if present.
    metric = [m for m in matches if m.group(3).lower() in ("cm", "mm", "m")]
    m = (metric or matches)[0]
    factor = _TO_CM[m.group(3).lower()]
    h, w = _num(m.group(1)) * factor, _num(m.group(2)) * factor
    if h <= 0 or w <= 0:
        return None
    return round(h, 1), round(w, 1)
