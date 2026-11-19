"""Step 10 (optional, last): choose the fusion weights from the survey ratings.

    .\\venv\\Scripts\\python.exe scripts/10_fit_weights.py results/survey/responses.csv

Grid search over the weight simplex (style, color, emotion, description; steps of
fit_weights.step) using the normalized component scores stored for every survey item in key.csv.
Objective (config fit_weights.objective):
  spearman   rank correlation between the fused score and the item's mean rating
  top3_mean  mean rating of the 3 items each weighting would rank highest per case
The fitted weights are printed next to the hand-set ones; config.yaml is NOT changed. With few
respondents the fit is noisy: treat it as a direction, not a final answer.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from artrec.config import load_config

COMPONENTS = ["style", "color", "emotion", "description"]


def simplex(n_parts: int, steps: int):
    """All weight vectors with n_parts entries that are multiples of 1/steps and sum to 1."""
    if n_parts == 1:
        yield (steps,)
        return
    for first in range(steps + 1):
        for rest in simplex(n_parts - 1, steps - first):
            yield (first,) + rest


def fused(norms: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Like scorer.fuse: components missing for an item (NaN) are dropped and the weights renormalized."""
    active = ~np.isnan(norms)
    wsum = (active * w).sum(1)
    return np.where(wsum > 0, np.nansum(norms * w, axis=1) / np.where(wsum > 0, wsum, 1), np.nan)


def objective(items: pd.DataFrame, f: np.ndarray, kind: str) -> float:
    ok = ~np.isnan(f)
    if kind == "spearman":
        return float(spearmanr(f[ok], items["rating"].to_numpy()[ok]).statistic)
    if kind == "top3_mean":
        t = items.assign(f=f)[ok]
        return float(t.groupby("case_id").apply(lambda g: g.nlargest(3, "f")["rating"].mean()).mean())
    raise ValueError(f"unknown objective {kind!r}")


def main() -> None:
    cfg = load_config()
    default_dir = cfg.path("results_dir") / "survey"
    ap = argparse.ArgumentParser()
    ap.add_argument("responses", type=Path)
    ap.add_argument("--key", type=Path, default=default_dir / "key.csv")
    ap.add_argument("--out-dir", type=Path, default=default_dir)
    args = ap.parse_args()
    fw = cfg.fit_weights

    key = pd.read_csv(args.key)
    resp = pd.read_csv(args.responses).merge(key, on=["case_id", "letter"])
    resp = resp[resp["condition"].isin(["system", "random", "mismatch"])]  # mood cases only
    items = resp.groupby(["case_id", "letter"]).agg(
        rating=("rating", "mean"), **{f"norm_{c}": (f"norm_{c}", "first") for c in COMPONENTS}).reset_index()
    if len(items) < 10:
        raise SystemExit(f"Only {len(items)} rated items: too few to fit weights.")
    # A component no survey item used (e.g. description, when the survey moods are single words)
    # cannot be fitted: it is left out of the search and reported as untested.
    used = [c for c in COMPONENTS if items[f"norm_{c}"].notna().any()]
    untested = [c for c in COMPONENTS if c not in used]
    norms = items[[f"norm_{c}" for c in used]].to_numpy(float)

    steps = round(1 / fw["step"])
    results = []
    for parts in simplex(len(used), steps):
        w = np.array(parts) / steps
        results.append({**dict(zip(used, w)), "objective": objective(items, fused(norms, w), fw["objective"])})
    grid = pd.DataFrame(results).sort_values("objective", ascending=False)
    hand = np.array([cfg.scoring["weights"][c] for c in used])
    hand = hand / hand.sum()  # the hand-set weights renormalized over the tested components, like the scorer does
    hand_score = objective(items, fused(norms, hand), fw["objective"])
    best = grid.iloc[0]
    if untested:
        print(f"Not tested by this survey (no item used them), weights unchanged: {untested}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.out_dir / "weight_grid.csv", index=False)
    out = {"objective": fw["objective"], "n_items": len(items), "n_ratings": len(resp), "untested": untested,
           "hand_set": {**dict(zip(used, hand.tolist())), "objective": hand_score},
           "fitted": {**{c: float(best[c]) for c in used}, "objective": float(best["objective"])}}
    (args.out_dir / "fitted_weights.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame({"hand-set": out["hand_set"], "fitted": out["fitted"]}).round(3).to_string())
    print(f"\nSummary: best {fw['objective']} {best['objective']:.3f} vs hand-set {hand_score:.3f} on "
          f"{len(items)} items. config.yaml was not changed; copy the weights there only if you trust the fit.")


if __name__ == "__main__":
    main()
