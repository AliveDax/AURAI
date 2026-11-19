"""Step 9c: analyse the survey answers.

    .\\venv\\Scripts\\python.exe scripts/09_analyze_survey.py results/survey/responses.csv

The responses CSV has one row per rating: respondent, case_id, letter, rating (1-5), the format of
results/survey/responses_template.csv. Reports:
  - mean rating per condition (system vs random vs mismatch; colour items)
  - one-sided Mann-Whitney U (all ratings) and Wilcoxon signed-rank (paired per respondent x case)
    tests of "system rated higher than the control"
  - agreement: how often a respondent's top-rated item is one of the system's ranks 1-3
  - mean rating per scorer configuration (items in that configuration's top-5, from script 07)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

from artrec.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    default_dir = cfg.path("results_dir") / "survey"
    ap = argparse.ArgumentParser()
    ap.add_argument("responses", type=Path)
    ap.add_argument("--key", type=Path, default=default_dir / "key.csv")
    ap.add_argument("--out-dir", type=Path, default=default_dir)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    resp = pd.read_csv(args.responses)
    df = resp.merge(pd.read_csv(args.key, keep_default_na=False, na_values=[""]), on=["case_id", "letter"], how="left")
    if df["condition"].isna().any():
        raise SystemExit(f"{int(df['condition'].isna().sum())} responses don't match key.csv (check case_id/letter).")
    if not df["rating"].between(1, 5).all():
        raise SystemExit("Ratings must be 1-5.")

    by_cond = df.groupby("condition")["rating"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    print("Mean rating per condition:\n" + by_cond.round(2).to_string() + "\n")

    mood = df[df["condition"].isin(["system", "random", "mismatch"])]
    sys_r = mood.loc[mood["condition"] == "system", "rating"]
    tests = []
    for other in ("random", "mismatch"):
        o = mood.loc[mood["condition"] == other, "rating"]
        if len(o) == 0:
            continue
        u = mannwhitneyu(sys_r, o, alternative="greater")
        # Paired: per respondent x case, the mean of the system items vs the control item
        pair = mood[mood["condition"].isin(["system", other])].pivot_table(
            index=["respondent", "case_id"], columns="condition", values="rating", aggfunc="mean").dropna()
        diff = pair["system"] - pair[other]
        w = wilcoxon(diff, alternative="greater") if (diff != 0).any() else None
        tests.append({"system_vs": other, "mean_diff": sys_r.mean() - o.mean(), "mannwhitney_U": u.statistic,
                      "mannwhitney_p": u.pvalue, "n_pairs": len(pair),
                      "wilcoxon_p": w.pvalue if w is not None else float("nan")})
    tests = pd.DataFrame(tests)
    print("System rated higher than the control? (one-sided)\n" + tests.round(4).to_string(index=False) + "\n")

    # Agreement: is a respondent's favourite item (ties: any of them) one of the system's ranks 1-3?
    agree = []
    for _, g in mood.groupby(["respondent", "case_id"]):
        best = g[g["rating"] == g["rating"].max()]
        agree.append(bool(((best["condition"] == "system") & (best["system_rank"] <= 3)).any()))
    n_items = mood.groupby("case_id")["letter"].nunique().mean()
    agreement = pd.DataFrame([{"share_top_pick_in_system_rank_1_3": sum(agree) / max(len(agree), 1),
                               "chance_if_one_random_favourite": 3 / n_items, "n": len(agree)}])
    print("Agreement of the human top pick with system ranks 1-3:\n" + agreement.round(3).to_string(index=False) + "\n")

    per_config = pd.DataFrame()
    if "in_top5_of" in mood and mood["in_top5_of"].notna().any():
        exploded = mood.assign(config=mood["in_top5_of"].fillna("").str.split(";")).explode("config")
        per_config = (exploded[exploded["config"] != ""].groupby("config")["rating"]
                      .agg(["mean", "count"]).sort_values("mean", ascending=False))
        print("Mean rating of each configuration's top-5 items:\n" + per_config.round(2).to_string() + "\n")

    by_cond.to_csv(args.out_dir / "ratings_by_condition.csv")
    tests.to_csv(args.out_dir / "significance_tests.csv", index=False)
    agreement.to_csv(args.out_dir / "top_pick_agreement.csv", index=False)
    if len(per_config):
        per_config.to_csv(args.out_dir / "ratings_by_configuration.csv")
    ax = by_cond["mean"].plot.bar(yerr=by_cond["std"], capsize=4, figsize=(7, 4))
    ax.set_ylabel("mean rating (1-5)")
    ax.set_title("Human ratings by condition")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(args.out_dir / "ratings_by_condition.png", dpi=130)
    print(f"Summary: {len(df)} ratings from {df['respondent'].nunique()} respondents; saved to {args.out_dir}")


if __name__ == "__main__":
    main()
