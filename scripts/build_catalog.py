"""Build the artwork catalog used at query time (run once, on the GPU machine).

    python scripts/build_catalog.py --per-style 100            # ~5,600 balanced across 56 styles
    python scripts/build_catalog.py --ids-file my_subset.txt   # an explicit list of ids
    python scripts/build_catalog.py --require-dimensions       # keep only artworks with known size
    python scripts/build_catalog.py --map emotion=annotation.emotion.category
    python scripts/build_catalog.py --metadata-only            # step 1: metadata for size lookup

Outputs (data/cache/): catalog.parquet, image_embeddings.npy, palettes.npz
"""
import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from artrec.catalog import Catalog
from artrec.color import extract_palette
from artrec.config import CACHE_DIR, EMOART_DIR, ROOT, SETTINGS
from artrec.dimensions import parse_dimensions

sys.path.insert(0, str(Path(__file__).parent))
from download_emoart import find_metadata_files, read_any  # noqa: E402

FIELDS = {
    "id": ["art_id", "image_id", "id", "_key", "uid"],
    "image": ["image", "image_path", "img_path", "file_name", "filename", "image_name", "path", "img"],
    "title": ["title", "artwork_name", "name"],
    "artist": ["artist", "artist_name", "author", "painter"],
    "style": ["style", "art_style", "category", "genre"],
    "emotion": ["dominant_emotion", "domain_emotion", "emotion_category", "emotional_category", "emotion"],
    "valence": ["valence"],
    "arousal": ["arousal"],
    "therapy": ["healing_effects", "therapeutic_potential", "art_therapy", "therapy", "therapeutic"],
    "description": ["content_description", "description", "caption"],
    "dimensions": ["dimensions", "dimension", "size"],
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def detect_columns(cols: list[str], overrides: dict[str, str]) -> dict[str, str | None]:
    lower = {c: c.lower() for c in cols}
    found: dict[str, str | None] = {}
    for field, candidates in FIELDS.items():
        if field in overrides:
            found[field] = overrides[field]
            continue
        hit = None
        for cand in candidates:  # exact match on the last dotted segment first
            hit = next((c for c in cols if lower[c].split(".")[-1] == cand), None)
            if hit:
                break
        if not hit:
            for cand in candidates:
                hit = next((c for c in cols if cand in lower[c]), None)
                if hit:
                    break
        found[field] = hit
    return found


def load_metadata() -> pd.DataFrame:
    files = find_metadata_files(EMOART_DIR)
    if not files:
        sys.exit(f"No metadata under {EMOART_DIR}. Run scripts/download_emoart.py first.")
    dfs = []
    for f in files:
        try:
            df = read_any(f)
        except Exception as e:  # noqa: BLE001
            print(f"  skipping {f.name}: {e}")
            continue
        df = flatten_keep_images(df)
        df["_source_file"] = str(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def flatten_keep_images(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten nested annotation dicts into dotted columns, but keep embedded
    image dicts ({'bytes': ..., 'path': ...}) intact."""
    if not len(df):
        return df
    first = df.iloc[0]
    img_cols = [c for c in df.columns if isinstance(first[c], dict) and "bytes" in first[c]]
    nested = [c for c in df.columns if isinstance(first[c], dict) and c not in img_cols]
    if not nested:
        return df
    flat = pd.json_normalize(df[nested].to_dict("records"), sep=".")
    return pd.concat([df.drop(columns=nested).reset_index(drop=True), flat], axis=1)


class ImageResolver:
    """Turns whatever the dataset stores (relative path, bare file name, or embedded
    bytes) into a file on disk."""

    def __init__(self, root: Path, out_dir: Path):
        self.root, self.out_dir = root, out_dir
        self._index: dict[str, Path] | None = None

    def index(self) -> dict[str, Path]:
        if self._index is None:
            self._index = {p.name: p for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXT}
        return self._index

    def __call__(self, value, art_id: str) -> Path | None:
        if isinstance(value, dict) and value.get("bytes"):
            self.out_dir.mkdir(parents=True, exist_ok=True)
            p = self.out_dir / f"{art_id}.jpg"
            if not p.exists():
                Image.open(io.BytesIO(value["bytes"])).convert("RGB").save(p, quality=92)
            return p
        if isinstance(value, dict):
            value = value.get("path")
        if not isinstance(value, str):
            return None
        for cand in (Path(value), self.root / value):
            if cand.exists():
                return cand
        return self.index().get(Path(value).name)


def split_camel(s: str) -> str:
    return re.sub(r"(?<=[a-z0-9)])(?=[A-Z(])", " ", s).strip()


def artist_title_from_filename(src) -> tuple[str, str]:
    """EmoArt file names look like '0008228_QiuYing-SpringMorningintheHanPalace.jpg'.
    Approximate: words that were lower-case in the original stay joined."""
    if not isinstance(src, str):
        return "", ""
    _, _, rest = Path(src).stem.partition("_")
    parts = rest.split("-")
    # Hyphenated first names: 'Jean-PaulRiopelle-Pavane' -> artist 'Jean-Paul Riopelle'
    if len(parts) >= 3 and not re.search(r"[a-z][A-Z]", parts[0]) and re.match(r"[A-Z][a-z]+[A-Z]", parts[1]):
        parts = [parts[0] + "-" + parts[1]] + parts[2:]
    return split_camel(parts[0]), split_camel("-".join(parts[1:]))


def text(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (list, tuple, np.ndarray)):
        return ", ".join(map(str, v))
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-style", type=int, default=None, help="sample N artworks per style")
    ap.add_argument("--ids-file", type=Path, default=None, help="text file with one art id per line")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dimensions-csv", type=Path, default=EMOART_DIR / "dimensions.csv",
                    help="art_id + (dimensions text | height_cm,width_cm)")
    ap.add_argument("--require-dimensions", action="store_true")
    ap.add_argument("--map", nargs="*", default=[], help="field=column overrides")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metadata-only", action="store_true",
                    help="only write data/cache/metadata_all.parquet (for enrich_dimensions.py), no GPU work")
    args = ap.parse_args()

    raw = load_metadata()
    overrides = dict(m.split("=", 1) for m in args.map)
    cols = detect_columns(list(raw.columns), overrides)
    print("Column mapping:")
    for k, v in cols.items():
        print(f"  {k:12s} <- {v}")
    if not cols["image"] or not cols["emotion"]:
        sys.exit("Could not find the image and/or emotion column. Run download_emoart.py --inspect "
                 "and pass e.g. --map image=<col> emotion=<col>")

    df = pd.DataFrame({
        "art_id": raw[cols["id"]].astype(str) if cols["id"] else raw.index.astype(str),
        # EmoArt stores Windows paths ("Images\\Style\\file.jpg")
        "image_src": raw[cols["image"]].map(lambda v: v.replace("\\", "/") if isinstance(v, str) else v),
    })
    for f in ["title", "artist", "style", "emotion", "valence", "arousal", "therapy", "description", "dimensions"]:
        df[f] = raw[cols[f]].map(text) if cols[f] else ""
    # EmoArt uses "Contentment" where the paper's label set says "content"
    df["emotion"] = df["emotion"].str.strip().str.lower().replace({"contentment": "content"})
    if not cols["style"]:  # fall back to the folder name, EmoArt is organised by style
        df["style"] = df["image_src"].map(lambda v: Path(v).parent.name if isinstance(v, str) else "")
    if not cols["artist"] or not cols["title"]:  # fall back to the file name
        at = df["image_src"].map(artist_title_from_filename)
        if not cols["artist"]:
            df["artist"] = at.map(lambda t: t[0])
        if not cols["title"]:
            df["title"] = at.map(lambda t: t[1])
    df = df.drop_duplicates("art_id")

    # Sizes: from the dataset if present, plus an optional enrichment CSV
    hw = df["dimensions"].map(parse_dimensions)
    df["height_cm"] = hw.map(lambda t: t[0] if t else np.nan)
    df["width_cm"] = hw.map(lambda t: t[1] if t else np.nan)
    if args.dimensions_csv.exists():
        extra = pd.read_csv(args.dimensions_csv, dtype={"art_id": str})
        if "height_cm" not in extra and "dimensions" in extra:
            parsed = extra["dimensions"].map(parse_dimensions)
            extra["height_cm"] = parsed.map(lambda t: t[0] if t else np.nan)
            extra["width_cm"] = parsed.map(lambda t: t[1] if t else np.nan)
        extra = extra.set_index("art_id")[["height_cm", "width_cm"]]
        df = df.set_index("art_id")
        df.update(extra)
        df = df.reset_index()
    known = df["height_cm"].notna() & df["width_cm"].notna()
    print(f"Artworks with known physical size: {known.sum()} / {len(df)}")
    if args.metadata_only:
        out = CACHE_DIR / "metadata_all.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["image_src"]).to_parquet(out, index=False)
        print(f"Wrote {out}. Next: python scripts/enrich_dimensions.py --catalog {out}")
        return

    # Subset
    if args.ids_file:
        ids = {line.strip() for line in args.ids_file.read_text().splitlines() if line.strip()}  # ids contain spaces
        df = df[df["art_id"].isin(ids)]
    if args.require_dimensions:
        df = df[df["height_cm"].notna() & df["width_cm"].notna()]
    if args.per_style:
        df = df.sample(frac=1, random_state=args.seed).groupby("style").head(args.per_style)
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=args.seed)
    df = df.reset_index(drop=True)
    print(f"Catalog size: {len(df)}  |  emotions: {df['emotion'].value_counts().to_dict()}")

    # Images -> palettes + CLIP embeddings
    from artrec.clip_model import ClipEncoder
    enc = ClipEncoder()
    resolver = ImageResolver(EMOART_DIR, EMOART_DIR / "images_extracted")
    k = SETTINGS.color.k_palette
    rows, labs, weights, embs, batch = [], [], [], [], []

    def flush():
        embs.append(enc.encode_images([b for b in batch], batch_size=args.batch_size))
        batch.clear()

    for r in tqdm(df.itertuples(index=False), total=len(df), desc="artworks"):
        path = resolver(r.image_src, r.art_id)
        if path is None:
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        thumb = img.copy()
        thumb.thumbnail((256, 256))
        p = extract_palette(np.asarray(thumb), k=k)
        lab, w = np.zeros((k, 3)), np.zeros(k)
        lab[:len(p.weights)], w[:len(p.weights)] = p.lab, p.weights
        labs.append(lab)
        weights.append(w)
        row = r._asdict()
        row.pop("image_src")
        try:
            row["image_path"] = str(path.resolve().relative_to(ROOT))
        except ValueError:
            row["image_path"] = str(path.resolve())
        row["img_width_px"], row["img_height_px"] = img.size
        rows.append(row)
        img.thumbnail((672, 672))
        batch.append(img)
        if len(batch) == args.batch_size * 4:
            flush()
    if batch:
        flush()

    meta = pd.DataFrame(rows).drop(columns=["dimensions"])
    cat = Catalog(meta, np.concatenate(embs), np.array(labs), np.array(weights))
    cat.save()
    print(f"Saved catalog with {len(cat)} artworks to data/cache/")


if __name__ == "__main__":
    main()
