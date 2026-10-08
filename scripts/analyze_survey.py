"""Analyse survey responses.

Export the Google Form responses and reshape them to a long CSV:
    respondent,case_id,option,rating        (rating 1-5)
then run:
    python scripts/analyze_survey.py reports/survey/responses_long.csv

Reports mean rating per condition and one-sided Mann-Whitney U tests of
"system rated higher than X" for X in clip_only, random, mismatch.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from artrec.config import REPORTS_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("responses", type=Path)
    ap.add_argument("--key", type=Path, default=REPORTS_DIR / "survey" / "key.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.responses).merge(pd.read_csv(args.key), on=["case_id", "option"], how="left")
    if df["condition"].isna().any():
        raise SystemExit("Some responses don't match the key; check case_id/option values.")

    summary = df.groupby("condition")["rating"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    print("Ratings by condition:\n", summary.round(2).to_string())

    sys_r = df.loc[df.condition == "system", "rating"]
    tests = []
    for other in ["clip_only", "random", "mismatch"]:
        o = df.loc[df.condition == other, "rating"]
        if len(o):
            u, p = mannwhitneyu(sys_r, o, alternative="greater")
            tests.append({"system_vs": other, "mean_diff": sys_r.mean() - o.mean(), "U": u, "p_value": p})
    tests = pd.DataFrame(tests)
    print("\nSystem rated higher than ... (one-sided Mann-Whitney U):\n", tests.round(4).to_string(index=False))

    out = REPORTS_DIR / "survey"
    summary.to_csv(out / "ratings_by_condition.csv")
    tests.to_csv(out / "significance_tests.csv", index=False)
    ax = summary["mean"].plot.bar(yerr=summary["std"], capsize=4, figsize=(6, 4))
    ax.set_ylabel("mean rating (1-5)")
    ax.set_title("Human ratings by condition")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / "ratings_by_condition.png", dpi=150)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
