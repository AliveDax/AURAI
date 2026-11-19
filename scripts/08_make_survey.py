"""Step 9b: build the blind human survey (there is no ground truth for "room + mood + art").

    .\\venv\\Scripts\\python.exe scripts/08_make_survey.py

Mood cases, one per (room, mood): the system's top-5 plus 2 controls, shuffled:
  random    a random artwork outside the top-5 (the fair baseline)
  mismatch  deliberately bad: the lowest colour score among the bottom 10% by final score
Colour cases, one per room, to test the colour component and the plateau assumption:
  color_matched  highest colour score
  color_bold     highest colour score among "bold" (highly saturated) artworks: a bold accent;
                 the case records whether the room itself is muted
  color_clash    lowest colour score
Respondents rate every item 1-5. Files get random codes; which item is which is only in key.csv.
Outputs in results/survey/: sheets/<case>.jpg (room + lettered items, paste into Google Forms),
images/<case>/ (separate files), cases.csv, key.csv (keep private), responses_template.csv.
If results/ablation/ablation_top5.csv exists (script 07), key.csv also records which scorer
configurations had each item in their top-5, so 09 can compare configurations.
"""
import random
import shutil
import string

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from artrec.config import load_config, setup_logging
from artrec.recommend import UserInput, analyze, load_resources, score
from artrec.scorer import top_n

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
COMPONENTS = ["style", "color", "emotion", "description"]


def code(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_lowercase + string.digits, k=6))


def mean_chromatic_saturation(pal: dict) -> np.ndarray:
    """Per artwork: coverage-weighted mean saturation of its non-neutral colours (0 if none)."""
    w = pal["coverage"] * (pal["valid"] & ~pal["neutral"])
    return (pal["hsv"][..., 1] * w).sum(1) / np.maximum(w.sum(1), 1e-9)


def sheet(room_path, items, cfg, title) -> Image.Image:
    """Room photo on top, the lettered items below."""
    s = cfg.survey
    room = ImageOps.contain(Image.open(room_path).convert("RGB"), (s["room_px"], s["room_px"]))
    gap, n = 20, len(items)
    width = max(room.width + 2 * gap, n * (s["item_px"] + gap) + gap)
    canvas = Image.new("RGB", (width, room.height + s["item_px"] + 130), "white")
    canvas.paste(room, ((width - room.width) // 2, gap))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=26)
    d.text((gap, room.height + gap + 5), title, fill="black", font=font)
    top = room.height + gap + 45
    for j, (letter, img_path) in enumerate(items):
        art = ImageOps.contain(Image.open(img_path).convert("RGB"), (s["item_px"], s["item_px"]))
        x = gap + j * (s["item_px"] + gap)
        canvas.paste(art, (x + (s["item_px"] - art.width) // 2, top))
        d.text((x + s["item_px"] // 2 - 8, top + s["item_px"] + 10), letter, fill="black", font=font)
    return canvas


def main() -> None:
    setup_logging()
    cfg = load_config()
    s = cfg.survey
    rng = random.Random(cfg.seed)
    res = load_resources(cfg)
    cat = res.catalog
    out = cfg.path("results_dir") / "survey"
    shutil.rmtree(out / "images", ignore_errors=True)  # stale codes from an earlier run would confuse the key
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    rooms = sorted(p for p in cfg.path("rooms_dir").iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not rooms:
        raise SystemExit(f"No room photos in {cfg.paths['rooms_dir']}/")
    abl_path = cfg.path("results_dir") / "ablation" / "ablation_top5.csv"
    ablation = pd.read_csv(abl_path) if abl_path.exists() else None
    if ablation is None:
        print("No ablation_top5.csv (run 07 first to compare configurations in 09); continuing without it.")
    bold = mean_chromatic_saturation(res.palettes) >= s["bold_min_saturation"]

    cases, key = [], []

    def add_case(case_id, room, mood, question, picks, scored):
        """picks: [(artwork_id, condition, system_rank or None)] -> shuffled, coded, saved."""
        rng.shuffle(picks)
        folder = out / "images" / case_id
        folder.mkdir(parents=True, exist_ok=True)
        Image.open(room).convert("RGB").save(folder / "room.jpg", quality=90)
        items = []
        for letter, (i, condition, sys_rank) in zip(string.ascii_uppercase, picks):
            c = code(rng)
            src = cfg.root / cat.at[i, "image_path"]
            Image.open(src).convert("RGB").save(folder / f"{c}.jpg", quality=90)
            items.append((letter, src))
            in_top = ""
            if ablation is not None and mood:
                hit = ablation[(ablation["room"] == room.name) & (ablation["mood"] == mood) & (ablation["artwork_id"] == i)]
                in_top = ";".join(sorted(hit["config"].unique()))
            key.append({"case_id": case_id, "letter": letter, "file_code": c, "artwork_id": int(i),
                        "title": cat.at[i, "title"], "condition": condition, "system_rank": sys_rank,
                        "final_score": round(float(scored.final[i]), 4),
                        **{f"norm_{k}": (round(float(scored.norm[k][i]), 4) if k in scored.norm else np.nan)
                           for k in COMPONENTS},
                        "raw_color": round(float(scored.raw["color"][i]), 4), "in_top5_of": in_top})
        sheet(room, items, cfg, question).save(out / "sheets" / f"{case_id}.jpg", quality=88)
        cases.append({"case_id": case_id, "room": room.name, "mood": mood or "", "question": question,
                      "n_items": len(picks)})

    for room in rooms:
        for mood in s["moods"]:
            sc = score(analyze(UserInput([room], mood), cfg, res), cfg, res)
            system = [int(i) for i in top_n(sc.final, s["top_n"])]
            rest = [i for i in range(len(cat)) if i not in system]
            randoms = rng.sample(rest, s["random_controls"])
            bottom = np.argsort(sc.final)[: max(1, int(len(cat) * s["mismatch_bottom_share"]))]
            bottom = [int(i) for i in bottom if i not in randoms]
            mismatch = min(bottom, key=lambda i: sc.raw["color"][i])
            picks = ([(i, "system", r) for r, i in enumerate(system, 1)]
                     + [(i, "random", None) for i in randoms] + [(mismatch, "mismatch", None)])
            add_case(f"{room.stem}__{mood}", room, mood,
                     f"Mood: '{mood}'. How well does each artwork suit this room and this mood? (1-5)", picks, sc)

        # Colour case: no mood, so only style + colour are active; the items are chosen by colour alone
        q = analyze(UserInput([room]), cfg, res)
        sc = score(q, cfg, res)
        col = sc.raw["color"]
        matched, clash = int(np.argmax(col)), int(np.argmin(col))
        bold_ids = [int(i) for i in np.flatnonzero(bold) if i not in (matched, clash)]
        picks = [(matched, "color_matched", None), (clash, "color_clash", None)]
        if bold_ids:
            picks.append((max(bold_ids, key=lambda i: col[i]), "color_bold", None))
        muted = float((q.room_palette.hsv[:, 1] * q.room_palette.coverage).sum()) < s["muted_room_max_saturation"]
        add_case(f"{room.stem}__color", room, None,
                 f"How well do each artwork's COLOURS suit this room? (1-5)  [room muted: {muted}]", picks, sc)
        print(f"built cases for {room.name}")

    pd.DataFrame(cases).to_csv(out / "cases.csv", index=False)
    pd.DataFrame(key).to_csv(out / "key.csv", index=False)
    pd.DataFrame(columns=["respondent", "case_id", "letter", "rating"]).to_csv(out / "responses_template.csv", index=False)
    print(f"\nSummary: {len(cases)} cases, {len(key)} items -> {out}. Keep key.csv private until the survey closes. "
          "Enter answers in the format of responses_template.csv, then run 09_analyze_survey.py.")


if __name__ == "__main__":
    main()
