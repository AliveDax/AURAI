"""Build the blind human survey (system-level test).

For every room photo x mood, each case shows 5 artworks in random order:
  system      x2  our full system's top picks
  clip_only   x1  ablation: CLIP scores only (no colour harmony / fit) - tests whether
                  our colour logic actually adds value
  random      x1  a random catalog artwork - the fair baseline
  mismatch    x1  the system's lowest-ranked artwork - sanity check

Respondents rate each option 1-5 for "how well does this suit this wall and mood".
The key (which option is which) is saved separately and never shown.

    python scripts/make_survey.py --moods calm cozy energetic
Outputs reports/survey/: one composite image per case (paste into Google Forms),
cases.csv and key.csv.
"""
import argparse
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from artrec.catalog import Catalog
from artrec.config import REPORTS_DIR, ROOMS_DIR, ROOT, SETTINGS
from artrec.recommender import Query, Recommender

CLIP_ONLY = {"content": 0.25, "surroundings": 0.15, "emotion": 0.25,
             "harmony": 0.0, "preferred_color": 0.0, "fit": 0.0}


def load_art(path: str, size: int) -> Image.Image:
    p = ROOT / path if not path.startswith("/") else path
    return ImageOps.contain(Image.open(p).convert("RGB"), (size, size))


def composite(room: Image.Image, arts: list[Image.Image], mood: str, box=None, size: int = 300) -> Image.Image:
    """Room on top (with the target spot outlined), the five options below, lettered A-E."""
    gap = 20
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    room = room.copy()
    if box:
        x, y, w, h = box
        ImageDraw.Draw(room).rectangle([x, y, x + w, y + h], outline=(40, 200, 90), width=max(3, room.width // 200))
    W = size * len(arts) + gap * (len(arts) + 1)
    room = ImageOps.contain(room, (W - 2 * gap, int(size * 1.6)))
    top = room.height + gap + 50
    canvas = Image.new("RGB", (W, top + size + 60), "white")
    canvas.paste(room, ((W - room.width) // 2, gap // 2))
    d = ImageDraw.Draw(canvas)
    d.text((gap, room.height + gap), f"Mood: {mood}   (green box = where the artwork goes)", fill="black", font=font)
    for i, a in enumerate(arts):
        x0 = gap + i * (size + gap)
        canvas.paste(a, (x0 + (size - a.width) // 2, top + (size - a.height) // 2))
        d.text((x0 + size // 2 - 8, top + size + 12), "ABCDE"[i], fill="black", font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moods", nargs="+", default=["calm", "cozy", "energetic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-segmenter", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from artrec.clip_model import ClipEncoder
    seg = None
    if not args.no_segmenter:
        from artrec.room import WallSegmenter
        seg = WallSegmenter()
    cat = Catalog.load()
    rec = Recommender(cat, ClipEncoder(), seg, SETTINGS)

    out = REPORTS_DIR / "survey"
    out.mkdir(parents=True, exist_ok=True)
    rooms = sorted(p for p in ROOMS_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    cases, key = [], []
    for room_path in rooms:
        room_img = np.asarray(Image.open(room_path).convert("RGB"))
        for mood in args.moods:
            case_id = f"{room_path.stem}__{mood}"
            q = Query(room_image=room_img, mood=mood, use_a4=False)
            result = rec.recommend(q, top_n=len(cat))
            full = result.recommendations
            if len(full) < 6:
                print(f"skipping {case_id}: too few candidates")
                continue
            system = [r.art_id for r in full[:2]]
            clip = next(r.art_id for r in rec.recommend(q, top_n=10, weights=CLIP_ONLY).recommendations
                        if r.art_id not in system)
            mismatch = full[-1].art_id
            used = set(system) | {clip, mismatch}
            rand = rng.choice([r.art_id for r in full[2:-1] if r.art_id not in used])
            options = [(a, "system") for a in system] + [(clip, "clip_only"), (rand, "random"), (mismatch, "mismatch")]
            rng.shuffle(options)

            by_id = cat.meta.set_index("art_id")
            arts = [load_art(by_id.at[a, "image_path"], 300) for a, _ in options]
            composite(Image.fromarray(room_img), arts, mood, result.room.space_box).save(out / f"{case_id}.jpg", quality=90)
            cases.append({"case_id": case_id, "room": room_path.name, "mood": mood})
            for letter, (a, cond) in zip("ABCDE", options):
                key.append({"case_id": case_id, "option": letter, "condition": cond,
                            "art_id": a, "title": by_id.at[a, "title"]})
            print(f"built {case_id}")

    pd.DataFrame(cases).to_csv(out / "cases.csv", index=False)
    pd.DataFrame(key).to_csv(out / "key.csv", index=False)
    print(f"\n{len(cases)} cases in {out}. Keep key.csv private until the survey closes.")


if __name__ == "__main__":
    main()
