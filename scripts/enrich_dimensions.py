"""Recover physical sizes for artworks from Wikidata and The Met's collection API.

EmoArt has no dimensions. Wikidata records height/width for many paintings
(P2048/P2049); The Met records them for its own holdings. We match by artist +
title. Results are appended to data/emoart/dimensions.csv, which
build_catalog.py merges in.

    python scripts/enrich_dimensions.py --catalog data/cache/metadata_all.parquet
    python scripts/enrich_dimensions.py --catalog ... --sources wikidata   # skip the slow Met lookup
    (any CSV/parquet with art_id, title, artist columns)

Wikidata is queried in batches of artists, so it scales to the whole dataset.
The Met is queried per artwork and rate-limits heavy use, so it runs second,
only for what Wikidata didn't find.

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
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_CACHE = CACHE_DIR / "wikidata_sizes_cache.json"
WIKIDATA_BATCH = 40  # artist names per SPARQL query
TITLE_LANGS = ("en", "fr", "de", "es", "it", "nl", "pt")
# Titles too generic to identify a work by themselves (compared after compact())
GENERIC_TITLES = {
    "untitled", "composition", "abstraction", "abstract", "abstractcomposition", "landscape", "portrait",
    "selfportrait", "stilllife", "study", "sketch", "nude", "seascape", "flowers", "figure", "head",
    "woman", "man", "interior", "drawing", "painting", "noname", "sanstitre", "ohnetitel", "sintitulo",
}
HEADERS = {"User-Agent": "artrec-dimension-enrichment/0.1 (research project; python-requests)"}
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


def name_variants(artist: str) -> list[str]:
    """'Hasui Kawase' -> also 'Kawase Hasui' (East Asian names are often stored surname first)."""
    parts = str(artist).split()
    out = [artist]
    if len(parts) == 2:
        out.append(f"{parts[1]} {parts[0]}")
    return out


class Wikidata:
    """Sizes of all works by a set of artists, fetched in batched SPARQL queries.
    Uses normalised quantities (psn:), which Wikidata converts to metres."""

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.cache: dict[str, list] = json.loads(WIKIDATA_CACHE.read_text()) if WIKIDATA_CACHE.exists() else {}

    def save(self):
        WIKIDATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        WIKIDATA_CACHE.write_text(json.dumps(self.cache))

    def _query(self, names: list[str]) -> list[dict]:
        values = " ".join(json.dumps(n, ensure_ascii=False) + "@en" for n in names)
        langs = ", ".join(f'"{lang}"' for lang in TITLE_LANGS)
        q = f"""
        SELECT ?name ?work ?title ?h ?w WHERE {{
          VALUES ?name {{ {values} }}
          ?artist rdfs:label|skos:altLabel ?name ; wdt:P31 wd:Q5 .
          ?work wdt:P170 ?artist ;
                p:P2048/psn:P2048/wikibase:quantityAmount ?h ;
                p:P2049/psn:P2049/wikibase:quantityAmount ?w ;
                rdfs:label ?title .
          FILTER(LANG(?title) IN ({langs}))
        }}"""
        for attempt in range(5):
            r = requests.post(WIKIDATA_SPARQL, data={"query": q, "format": "json"}, headers=HEADERS, timeout=120)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(float(r.headers.get("Retry-After", 10 * (attempt + 1))))
                continue
            r.raise_for_status()
            return [{k: v["value"] for k, v in b.items()} for b in r.json()["results"]["bindings"]]
        raise RuntimeError("Wikidata query kept failing")

    def fetch(self, artists: list[str]) -> None:
        names = sorted({v for a in artists if a for v in name_variants(a)} - set(self.cache))
        batches = [names[i:i + WIKIDATA_BATCH] for i in range(0, len(names), WIKIDATA_BATCH)]
        try:
            for batch in tqdm(batches, desc="wikidata"):
                rows = self._query(batch)
                for n in batch:
                    self.cache[n] = []
                for row in rows:
                    h, w = float(row["h"]) * 100, float(row["w"]) * 100  # metres -> cm
                    if 1 <= h <= 2000 and 1 <= w <= 2000:
                        self.cache[row["name"]].append([row["work"].rsplit("/", 1)[-1], row["title"], h, w])
                time.sleep(self.delay)  # be polite to a free public service
        finally:
            self.save()

    def find_size(self, title: str, artist: str) -> tuple[float, float, str] | None:
        works = {}
        for n in name_variants(artist):
            for qid, label, h, w in self.cache.get(n, []):
                if same_title(title, label):
                    works[qid] = (round(h, 1), round(w, 1))
        if len(works) != 1:  # none, or ambiguous ('Untitled', 'Composition', ...)
            return None
        qid, (h, w) = next(iter(works.items()))
        return h, w, qid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=EMOART_DIR / "dimensions.csv")
    ap.add_argument("--sources", nargs="+", default=["wikidata", "met"], choices=["wikidata", "met"])
    args = ap.parse_args()

    df = pd.read_parquet(args.catalog) if args.catalog.suffix == ".parquet" else pd.read_csv(args.catalog)
    df["art_id"] = df["art_id"].astype(str)
    done = pd.read_csv(args.out, dtype={"art_id": str}) if args.out.exists() else pd.DataFrame(columns=["art_id"])
    todo = df[~df["art_id"].isin(done["art_id"]) & df["title"].astype(str).str.len().gt(0)]
    # A title can only identify a work if it's specific: skip generic titles and
    # titles the same artist uses more than once in the dataset (series, 'Untitled')
    key = df["artist"].astype(str) + "|" + df["title"].map(compact)
    repeated = set(key[key.duplicated(keep=False)])
    tkey = todo["artist"].astype(str) + "|" + todo["title"].map(compact)
    specific = ~todo["title"].map(compact).isin(GENERIC_TITLES) & ~tkey.isin(repeated)
    print(f"{(~specific).sum()} of {len(todo)} artworks have a generic or repeated title and are skipped")
    todo = todo[specific]

    found = []
    if "wikidata" in args.sources:
        wd = Wikidata()
        wd.fetch(todo["artist"].astype(str).unique().tolist())
        for r in todo.itertuples(index=False):
            hit = wd.find_size(r.title, r.artist)
            if hit:
                found.append({"art_id": r.art_id, "height_cm": hit[0], "width_cm": hit[1],
                              "source": f"wikidata:{hit[2]}"})
        print(f"Wikidata: {len(found)} / {len(todo)} matched")
    found_ids = {f["art_id"] for f in found}

    missing = []
    if "met" in args.sources:
        met = Met()
        try:
            for r in tqdm(todo[~todo["art_id"].isin(found_ids)].itertuples(index=False), desc="met"):
                hit = met.find_size(r.title, r.artist)
                if hit:
                    found.append({"art_id": r.art_id, "height_cm": hit[0], "width_cm": hit[1],
                                  "source": f"met:{hit[2]}"})
                    found_ids.add(r.art_id)
        finally:
            met.save()
    missing = todo[~todo["art_id"].isin(found_ids)][["art_id", "title", "artist"]]

    out = pd.concat([done, pd.DataFrame(found)], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    missing.to_csv(REPORTS_DIR / "dimensions_unmatched.csv", index=False)
    print(f"Matched {len(found)} new, {len(missing)} unmatched. Total with sizes: {len(out)}")
    print("Re-run build_catalog.py to merge the sizes into the catalog.")


if __name__ == "__main__":
    main()
