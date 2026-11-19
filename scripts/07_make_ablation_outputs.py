"""Step 9a: top-5 per (room, mood) for every scorer configuration, so the survey can compare
configurations and not only the final system.

    .\\venv\\Scripts\\python.exe scripts/07_make_ablation_outputs.py

Configurations: each component alone, every pair, all components (the default), and the default
with the other normalizations. Each room x mood is analysed once (CLIP runs once) and only the
cheap scoring step is repeated per configuration. Outputs in results/ablation/:
  ablation_top5.csv     one row per (room, mood, configuration, rank)
  ablation_overlap.csv  how many of the default top-5 each configuration shares
  <room>__<mood>.jpg    contact sheet: one row of top-5 thumbnails per configuration
"""
from itertools import combinations

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from artrec.config import load_config, setup_logging
from artrec.recommend import UserInput, analyze, load_resources, score
from artrec.scorer import top_n

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
THUMB = 150  # px per thumbnail in the contact sheets


def configurations(cfg) -> dict[str, dict]:
    """name -> scoring overrides. Weights of left-out components are set to 0 (= inactive)."""
    base = cfg.scoring["weights"]
    comps = ["style", "color", "emotion"]  # the survey moods are single words: no description
    out = {"all (default)": {}}
    for r in (1, 2):
        for subset in combinations(comps, r):
            out[" + ".join(subset) + (" only" if r == 1 else "")] = {
                "weights": {k: (v if k in subset else 0.0) for k, v in base.items()}}
    for method in ("minmax", "rank", "none"):
        if method != cfg.scoring["normalization"]:
            out[f"all, {method} normalization"] = {"normalization": method}
    return out


def contact_sheet(rows: pd.DataFrame, catalog, cfg, title: str) -> Image.Image:
    configs = list(dict.fromkeys(rows["config"]))
    label_w, n = 220, int(rows["rank"].max())
    sheet = Image.new("RGB", (label_w + n * THUMB, 30 + len(configs) * THUMB), "white")
    d = ImageDraw.Draw(sheet)
    d.text((5, 8), title, fill="black")
    for r, name in enumerate(configs):
        y = 30 + r * THUMB
        d.text((5, y + THUMB // 2), name, fill="black")
        for _, row in rows[rows["config"] == name].iterrows():
            img = ImageOps.contain(Image.open(cfg.root / catalog.at[row["artwork_id"], "image_path"]).convert("RGB"),
                                   (THUMB - 6, THUMB - 6))
            sheet.paste(img, (label_w + (row["rank"] - 1) * THUMB + 3, y + 3))
    return sheet


def main() -> None:
    setup_logging()
    cfg = load_config()
    res = load_resources(cfg)
    out = cfg.path("results_dir") / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    rooms = sorted(p for p in cfg.path("rooms_dir").iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not rooms:
        raise SystemExit(f"No room photos in {cfg.paths['rooms_dir']}/")
    configs = configurations(cfg)
    k = cfg.survey["top_n"]

    rows = []
    for room in rooms:
        for mood in cfg.survey["moods"]:
            q = analyze(UserInput([room], mood), cfg, res)  # CLIP runs once per room x mood
            for name, override in configs.items():
                v = cfg.with_overrides(scoring=override) if override else cfg
                s = score(q, v, res)
                for rank, i in enumerate(top_n(s.final, k), 1):
                    rows.append({"room": room.name, "mood": mood, "config": name, "rank": rank,
                                 "artwork_id": int(i), "title": res.catalog.at[i, "title"],
                                 "style": res.catalog.at[i, "style"],
                                 "dominant_emotion": res.catalog.at[i, "dominant_emotion"],
                                 "final_score": round(float(s.final[i]), 4),
                                 **{f"norm_{c}": round(float(s.norm[c][i]), 4) for c in s.norm}})
            case = pd.DataFrame([r for r in rows if r["room"] == room.name and r["mood"] == mood])
            contact_sheet(case, res.catalog, cfg, f"{room.name} / mood: {mood}").save(
                out / f"{room.stem}__{mood}.jpg", quality=88)
            print(f"done {room.name} x {mood}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_top5.csv", index=False)
    default = df[df["config"] == "all (default)"].groupby(["room", "mood"])["artwork_id"].apply(set)
    overlap = (df.groupby(["room", "mood", "config"])["artwork_id"].apply(set).reset_index()
               .assign(shared_with_default=lambda t: [len(a & default[(r, m)]) for r, m, a in
                                                      zip(t["room"], t["mood"], t["artwork_id"])]))
    summary = overlap.groupby("config")["shared_with_default"].mean().sort_values(ascending=False)
    summary.rename(f"mean shared of top-{k}").to_csv(out / "ablation_overlap.csv")
    print(f"\nMean number of the default top-{k} that each configuration also picks:\n{summary.round(2).to_string()}")
    print(f"\nSummary: {len(rooms)} rooms x {len(cfg.survey['moods'])} moods x {len(configs)} configurations "
          f"-> {out}")


if __name__ == "__main__":
    main()
