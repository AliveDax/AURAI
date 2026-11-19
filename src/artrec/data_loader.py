"""EmoArt annotations + images on disk -> data/catalog.csv (one row per artwork).

The annotation key paths come from config.yaml (`annotation_fields`), which were read from
the real dataset/annotation.json by scripts/01_inspect_dataset.py, not from memory.

`artwork_id` is the row index of catalog.csv, and that index defines the row order of every
cached array (embeddings, palettes). load_catalog() refuses a catalog where this is not true.
"""
from __future__ import annotations

import ast
import csv
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from artrec.config import Config

log = logging.getLogger(__name__)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# EmoArt file names start with a numeric ID: "0006080_YuriZlotnikov-Signalseries.jpg".
# Not all do (the China_images folder uses names like "FS-6883_01.jpg").
_ID_RE = re.compile(r"^(\d+)_")


# --------------------------------------------------------------------------- paths

def normalize_rel_path(p: str) -> str:
    """Annotation paths use Windows backslashes, and the same accented name can be stored in
    two Unicode forms. Forward slashes + NFC make both sides comparable."""
    return unicodedata.normalize("NFC", str(p).replace("\\", "/")).strip("/")


def numeric_id(filename: str) -> str | None:
    m = _ID_RE.match(filename)
    return m.group(1) if m else None


def index_images(dataset_dir: Path, images_dir: Path) -> tuple[dict[str, Path], list[Path]]:
    """Every image file under images_dir, keyed by its normalized path relative to dataset_dir
    ("Images/Baroque/x.jpg"), which is the format the annotation paths use.
    Two files whose names differ only in Unicode form collapse to one key; the extra ones are
    returned so the caller can report them."""
    found: dict[str, Path] = {}
    duplicates: list[Path] = []
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            key = normalize_rel_path(p.relative_to(dataset_dir).as_posix())
            if key in found:
                duplicates.append(p)
            else:
                found[key] = p
    return found, duplicates


def read_extract_log(path: Path) -> dict[str, str]:
    """extract_emoart.py renamed files whose names Windows can't store. Its log maps the original
    name (stored as Python's ascii() repr) to the name saved on disk."""
    if not path.exists():
        return {}
    mapping = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not row["saved_as"]:
                continue  # extraction failed for this file: nothing on disk to point to
            try:
                original = ast.literal_eval(row["original_name_escaped"])
            except (ValueError, SyntaxError):
                log.warning("Unreadable extract_log entry: %s", row["original_name_escaped"])
                continue
            mapping[normalize_rel_path(original)] = normalize_rel_path(row["saved_as"])
    return mapping


# --------------------------------------------------------------------------- fields

def get_nested(entry: dict, keys: list[str]) -> Any:
    """Walk nested dict keys; a missing key gives None instead of a crash."""
    obj: Any = entry
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def split_camel(s: str) -> str:
    """'YuriZlotnikov' -> 'Yuri Zlotnikov'. File names drop all spaces, so this is approximate:
    words that were lower-case in the real title stay glued to the previous word."""
    return re.sub(r"(?<=[a-z0-9)])(?=[A-Z(])", " ", s).strip()


def artist_title(filename: str) -> tuple[str, str]:
    """'0006080_YuriZlotnikov-Signalseries.jpg' -> ('Yuri Zlotnikov', 'Signalseries').
    The annotations have no artist or title, so these come from the file name, for display only."""
    stem = Path(filename).stem
    m = re.match(r"^\d+_(.*)$", stem)
    if not m:
        return "", stem
    artist, _, title = m.group(1).partition("-")
    return split_camel(artist), split_camel(title)


# --------------------------------------------------------------------------- matching

def match_annotations(
    entries: list[dict],
    disk: dict[str, Path],
    extract_log: dict[str, str],
    image_key: list[str],
) -> tuple[list[tuple[dict, str, str]], list[str], Counter]:
    """Match each annotation to one file on disk, safest method first:
    exact path -> case-insensitive path -> extract_log rename -> (style folder, numeric ID).
    Returns ([(entry, disk_key, method)], unmatched annotation paths, count per method).
    Two annotations claiming the same file is a data error, so that raises."""
    lower = {k.lower(): k for k in disk}
    by_id: dict[tuple[str, str], list[str]] = defaultdict(list)
    for k in disk:
        pid = numeric_id(Path(k).name)
        if pid:
            by_id[(Path(k).parent.name, pid)].append(k)

    claimed: dict[str, str] = {}
    matched, unmatched, methods = [], [], Counter()
    for e in entries:
        ann = normalize_rel_path(get_nested(e, image_key) or "")
        key = method = None
        if ann in disk:
            key, method = ann, "exact"
        elif ann.lower() in lower:
            key, method = lower[ann.lower()], "case_insensitive"
        elif extract_log.get(ann) in disk:
            key, method = extract_log[ann], "extract_log"
        else:
            pid = numeric_id(Path(ann).name)
            cands = by_id.get((Path(ann).parent.name, pid), []) if pid else []
            cands = [c for c in cands if c not in claimed]
            if len(cands) == 1:
                key, method = cands[0], "numeric_id"
            elif len(cands) > 1:
                log.warning("Ambiguous numeric-ID match for %s: %s", ann, cands)
        if key is None:
            unmatched.append(ann)
            continue
        if key in claimed:
            raise ValueError(f"File {key} matched by two annotations: {claimed[key]} and {ann}")
        claimed[key] = ann
        methods[method] += 1
        matched.append((e, key, method))
    return matched, unmatched, methods


def image_is_readable(path: Path) -> bool:
    """Cheap integrity check (header + structure) without decoding the whole image."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as err:  # any decode problem: the caller logs and skips the file
        log.warning("Corrupt image skipped: %s (%s)", path, err)
        return False


# --------------------------------------------------------------------------- catalog

def build_catalog(cfg: Config, check_images: bool = True) -> tuple[pd.DataFrame, dict]:
    """annotation.json + images on disk -> (catalog DataFrame, report dict for the summary)."""
    root, dataset_dir = cfg.root, cfg.path("dataset_dir")
    with open(cfg.path("annotation_file"), encoding="utf-8") as f:
        entries = json.load(f)
    disk, duplicates = index_images(dataset_dir, cfg.path("images_dir"))
    fields = cfg.annotation_fields
    matched, unmatched, methods = match_annotations(
        entries, disk, read_extract_log(cfg.path("extract_log")), fields["image_path"])

    rows, corrupt = [], []
    for entry, key, method in matched:
        path = disk[key]
        if check_images and not image_is_readable(path):
            corrupt.append(key)
            continue
        artist, title = artist_title(path.name)
        row = {
            "image_path": path.relative_to(root).as_posix(),  # forward slashes, relative to root
            "style": path.parent.name,                          # EmoArt is organised by style folder
            "filename": path.name,
            "artist": artist,
            "title": title,
        }
        for col, keys in fields.items():
            if col == "image_path":
                continue
            v = get_nested(entry, keys)
            row[col] = ";".join(map(str, v)) if isinstance(v, list) else v
        row["match_method"] = method
        rows.append(row)

    df = pd.DataFrame(rows)
    df.insert(0, "artwork_id", np.arange(len(df)))
    used = {key for _, key, _ in matched}
    report = {
        "annotations": len(entries),
        "image_files_on_disk": len(disk) + len(duplicates),
        "unicode_duplicate_files_ignored": [p.as_posix() for p in duplicates],
        "matched_by_method": dict(methods),
        "annotations_without_file": unmatched,
        "files_without_annotation": sorted(k for k in disk if k not in used),
        "corrupt_images_skipped": corrupt,
        "catalog_rows": len(df),
    }
    return df, report


def load_catalog(cfg: Config) -> pd.DataFrame:
    """Read data/catalog.csv and check that artwork_id is 0..N-1 in row order."""
    path = cfg.path("catalog")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/02_build_catalog.py")
    df = pd.read_csv(path, keep_default_na=False)
    if not np.array_equal(df["artwork_id"].to_numpy(), np.arange(len(df))):
        raise ValueError("catalog.csv: artwork_id is not 0..N-1 in row order; rebuild it with scripts/02_build_catalog.py")
    return df
