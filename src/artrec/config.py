"""Load config.yaml into a small dataclass, so every module reads its tunables from one place.

Each section of the YAML (clip, color, text, scoring, ...) stays a plain dict: adding a new
tunable only means editing config.yaml. Paths in the YAML are relative to the project root.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

# src/artrec/config.py -> the project root is two folders up. Computed, never hard-coded,
# because the project folder has been moved before.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    seed: int
    paths: dict[str, str]
    annotation_fields: dict[str, list[str]] = field(default_factory=dict)
    clip: dict[str, Any] = field(default_factory=dict)
    color: dict[str, Any] = field(default_factory=dict)
    text: dict[str, Any] = field(default_factory=dict)
    subject: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    survey: dict[str, Any] = field(default_factory=dict)
    fit_weights: dict[str, Any] = field(default_factory=dict)
    save_sheet: dict[str, Any] = field(default_factory=dict)
    emotion_only: dict[str, Any] = field(default_factory=dict)
    wall_preview: dict[str, Any] = field(default_factory=dict)
    root: Path = PROJECT_ROOT

    def path(self, key: str) -> Path:
        """Absolute path for an entry of the `paths:` section."""
        return self.root / self.paths[key]

    def with_overrides(self, **sections: dict) -> "Config":
        """A copy with some values replaced, e.g. cfg.with_overrides(scoring={"normalization": "rank"}).
        Used by the ablation and validation scripts to try variants without editing the YAML.
        Only the keys given are changed; the rest of each section is kept."""
        new = replace(self, **{name: copy.deepcopy(getattr(self, name)) for name in sections})
        for name, values in sections.items():
            getattr(new, name).update(copy.deepcopy(values))
        return new


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def load_yaml(path: Path) -> Any:
    """Read one of the YAML data files (colour words, emotion lexicon, paraphrases)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO) -> None:
    """INFO-level logging with a short format, for the scripts and the CLI."""
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
