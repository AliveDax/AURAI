"""Recover physical sizes for artworks via The Met's open-access collection API.

EmoArt has no dimensions. Many of its paintings come from The Met, which records
them. We match by title + artist and parse the size. Results are appended to
data/emoart/dimensions.csv, which build_catalog.py merges in.

    python scripts/enrich_dimensions.py --catalog data/cache/catalog.parquet
    (or any CSV/parquet with art_id, title, artist columns)

Artworks that can't be matched are listed in reports/dimensions_unmatched.csv;
for those you can add sizes by hand (e.g. from WikiArt pages) to the same CSV
using columns art_id,height_cm,width_cm.
"""
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from artrec.config import CACHE_DIR, EMOART_DIR, REPORTS_DIR
from artrec.dimensions import parse_dimensions

API = "https://collectionapi.metmuseum.org/public/collection/v1"
SEARCH = "https://collectionapi.metmuseum.org/public/collection/v1.1/search"  # v1/search retired 2026-10-01
MAX_ARTIST_OBJECTS = 300  # cap per artist for the artist-search fallback
HTTP_CACHE = CACHE_DIR / "met_http_cache.json"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def compact(s: str) -> str:
    """Titles from EmoArt file names lose the spaces before lower-case words
    ('Snowat Ishinomaki'), so compare titles with all spaces removed."""
    return norm(s).replace(" ", "")


def same_title(ours: str, theirs: str) -> bool:
    a, b = compact(ours), compact(theirs)
    if not a or not b:
        return False
    # Met titles often add a series name: "Snow at X, from the series Y"
    return a == b or (len(a) >= 8 and b.startswith(a))


class Met:
    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.cache = json.loads(HTTP_CACHE.read_text()) if HTTP_CACHE.exists() else {}

    def get(self, url: str):
        if url in self.cache:
            return self.cache[url]
        time.sleep(self.delay)  # be polite to a free public API
        r = requests.get(url, timeout=20)
        if r.status_code == 404:  # a real "no such object": safe to cache
            self.cache[url] = None
            return None
        r.raise_for_status()  # don't cache outages or retired endpoints as "no match"
        data = r.json()
        self.cache[url] = data
        return data

    def save(self):
        HTTP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        HTTP_CACHE.write_text(json.dumps(self.cache))

    def search(self, query: str, field: str, max_ids: int = 500) -> list[int]:
        ids: list[int] = []
        while len(ids) < max_ids:
            res = self.get(f"{SEARCH}?hasImages=true&{field}=true&offset={len(ids)}&limit=500"
                           f"&q={requests.utils.quote(query)}") or {}
            page = res.get("objectIDs") or []
            ids += page
            if not page or len(ids) >= (res.get("total") or 0):
                break
        return ids[:max_ids]

    def find_size(self, title: str, artist: str) -> tuple[float, float, int] | None:
        surname = norm(artist).split()[-1] if norm(artist) else ""
        candidates = self.search(title, "title")
        if surname:  # titles from file names are often too garbled for the title search
            candidates += self.search(surname, "artistOrCulture", MAX_ARTIST_OBJECTS)
        for oid in dict.fromkeys(candidates):
            obj = self.get(f"{API}/objects/{oid}")
            if not obj or not same_title(title, obj.get("title", "")):
                continue
            if surname and surname not in norm(obj.get("artistDisplayName", "")):
                continue
            for m in obj.get("measurements") or []:
                em = m.get("elementMeasurements") or {}
                if "Height" in em and "Width" in em:
                    return float(em["Height"]), float(em["Width"]), oid
            hw = parse_dimensions(obj.get("dimensions"))
            if hw:
                return hw[0], hw[1], oid
            break  # matched the object but it has no usable size
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=EMOART_DIR / "dimensions.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.catalog) if args.catalog.suffix == ".parquet" else pd.read_csv(args.catalog)
    df["art_id"] = df["art_id"].astype(str)
    done = pd.read_csv(args.out, dtype={"art_id": str}) if args.out.exists() else pd.DataFrame(columns=["art_id"])
    todo = df[~df["art_id"].isin(done["art_id"]) & df["title"].astype(str).str.len().gt(0)]

    met, found, missing = Met(), [], []
    try:
        for r in tqdm(todo.itertuples(index=False), total=len(todo)):
            hit = met.find_size(r.title, r.artist)
            if hit:
                found.append({"art_id": r.art_id, "height_cm": hit[0], "width_cm": hit[1], "source": f"met:{hit[2]}"})
            else:
                missing.append({"art_id": r.art_id, "title": r.title, "artist": r.artist})
    finally:
        met.save()

    out = pd.concat([done, pd.DataFrame(found)], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(missing).to_csv(REPORTS_DIR / "dimensions_unmatched.csv", index=False)
    print(f"Matched {len(found)} new, {len(missing)} unmatched. Total with sizes: {len(out)}")
    print("Re-run build_catalog.py to merge the sizes into the catalog.")


if __name__ == "__main__":
    main()
