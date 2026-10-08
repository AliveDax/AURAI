"""Two objective, non-subjective tests.

A) Preferred colour (no model involved, pure colour logic):
   for each test colour, rank the catalog by the preferred-colour score and check
   how often the top-k artworks clearly contain that colour, vs. how often a
   random artwork does.

B) Empty-space detection and measurement, against hand annotations in
   data/rooms/annotations.csv:
     file,x,y,w,h,true_width_cm,true_height_cm
   (x,y,w,h = where the art should go, drawn by hand; true sizes = tape-measured,
    leave blank if unknown). Detection quality = IoU with the hand-drawn box.
   Measurement error is computed on the hand-drawn box so it isolates the
   A4/homography step from the detection step.

    python scripts/evaluate_color_and_space.py [--no-segmenter]
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from artrec.catalog import Catalog
from artrec.color import color_present, parse_color, preferred_color_score
from artrec.config import REPORTS_DIR, ROOMS_DIR, SETTINGS
from artrec.measure import a4_mask, find_a4, homography_from_a4, measure_box
from artrec.room import analyze_room, box_iou, load_photo

TEST_COLORS = ["red", "orange", "mustard", "green", "sage", "teal", "blue", "navy", "purple", "pink", "brown", "beige"]


def eval_color(cat: Catalog, ks=(5, 10)) -> pd.DataFrame:
    cc = SETTINGS.color
    palettes = [cat.palette(i) for i in range(len(cat))]
    rows = []
    for name in TEST_COLORS:
        t = parse_color(name)
        present = np.array([color_present(p, t, cc) for p in palettes])
        scores = np.array([preferred_color_score(p, t, cc) for p in palettes])
        order = np.argsort(-scores)
        row = {"colour": name, "base_rate": present.mean()}
        for k in ks:
            row[f"present@{k}"] = present[order[:k]].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def eval_space(use_segmenter: bool) -> pd.DataFrame:
    ann_path = ROOMS_DIR / "annotations.csv"
    if not ann_path.exists():
        print(f"No {ann_path}; skipping space evaluation (see annotations_template.csv).")
        return pd.DataFrame()
    seg = None
    if use_segmenter:
        from artrec.room import WallSegmenter
        seg = WallSegmenter()
    rows = []
    for r in pd.read_csv(ann_path).itertuples(index=False):
        img = load_photo(ROOMS_DIR / r.file)
        # Boxes are annotated on the full-size upright photo; load_photo downscales
        s = img.shape[1] / ImageOps.exif_transpose(Image.open(ROOMS_DIR / r.file)).width
        corners = find_a4(img)
        exclude = a4_mask(corners, img.shape[:2]) if corners is not None else None
        room = analyze_room(img, SETTINGS.color, seg, exclude_mask=exclude)
        true_box = tuple(int(round(v * s)) for v in (r.x, r.y, r.w, r.h))
        row = {"file": r.file, "detected": room.space_box is not None,
               "iou": box_iou(room.space_box, true_box) if room.space_box else 0.0}
        if room.space_box:
            x, y, w, h = room.space_box
            cx, cy = x + w / 2, y + h / 2
            tx, ty, tw, th = true_box
            row["centre_inside_true_box"] = (tx <= cx <= tx + tw) and (ty <= cy <= ty + th)
        row["a4_found"] = corners is not None
        if corners is not None and pd.notna(r.true_width_cm):
            m = measure_box(homography_from_a4(corners), true_box)
            row.update(est_width_cm=m.width_cm, est_height_cm=m.height_cm,
                       width_err_pct=100 * (m.width_cm - r.true_width_cm) / r.true_width_cm,
                       height_err_pct=100 * (m.height_cm - r.true_height_cm) / r.true_height_cm)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-segmenter", action="store_true", help="use the heuristic wall mask instead")
    args = ap.parse_args()
    out = REPORTS_DIR / "color_space"
    out.mkdir(parents=True, exist_ok=True)

    col = eval_color(Catalog.load())
    col.to_csv(out / "preferred_colour.csv", index=False)
    print("Preferred colour presence (top-k vs random artwork):")
    print(col.round(3).to_string(index=False))

    sp = eval_space(not args.no_segmenter)
    if len(sp):
        sp.to_csv(out / "space_detection.csv", index=False)
        print("\nEmpty-space detection:")
        print(sp.round(2).to_string(index=False))
        print(f"\nMean IoU: {sp['iou'].mean():.3f}")
        if "width_err_pct" in sp:
            print(f"Mean absolute size error: width {sp['width_err_pct'].abs().mean():.1f}%, "
                  f"height {sp['height_err_pct'].abs().mean():.1f}%")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
