"""Component-level test: does CLIP's text side find artworks with the right emotion?

Two views, both compared against baselines (important: EmoArt is heavily
skewed, ~56% of labels are 'calm', so raw accuracy alone is misleading):

1. Retrieval  (emotion word -> ranked artworks)
   precision@k = share of the top-k carrying that label
   recall@k    = share of all artworks with that label found in the top-k
   lift@k      = precision@k / label prevalence (1.0 = no better than random)

2. Classification (artwork -> which of the 12 emotion words is closest)
   top-1 / top-3 accuracy, macro recall (every class counts equally),
   and valence balanced accuracy (positive vs negative).
   Baselines: always-'calm', and chance.

Note: EmoArt labels were produced by GPT-4o with human verification, so this
measures agreement with those labels.

    python scripts/evaluate_emotion.py
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from artrec.catalog import Catalog  # noqa: E402
from artrec.config import EMOART_EMOTIONS, NEGATIVE_EMOTIONS, REPORTS_DIR  # noqa: E402
from artrec.prompts import EMOTION_EVAL_TEMPLATES, MOOD_TO_EMOTIONS  # noqa: E402

KS = (1, 5, 10, 50)


def retrieval(scores: np.ndarray, relevant: np.ndarray) -> dict:
    order = np.argsort(-scores)
    out = {"n_relevant": int(relevant.sum()), "prevalence": float(relevant.mean())}
    for k in KS:
        hits = relevant[order[:k]].sum()
        out[f"P@{k}"] = hits / k
        out[f"R@{k}"] = hits / max(1, relevant.sum())
        out[f"lift@{k}"] = (hits / k) / max(1e-9, relevant.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPORTS_DIR / "emotion"))
    args = ap.parse_args()
    from pathlib import Path
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from artrec.clip_model import ClipEncoder
    cat, enc = Catalog.load(), ClipEncoder()
    labels = cat.meta["emotion"].astype(str).str.lower().to_numpy()
    emotions = [e for e in EMOART_EMOTIONS if (labels == e).any()]
    text = np.stack([enc.encode_prompt_ensemble(EMOTION_EVAL_TEMPLATES, e) for e in emotions])
    sims = cat.embeddings @ text.T  # (N, n_emotions)

    # 1) Retrieval per emotion label
    ret = pd.DataFrame([{"query": e, **retrieval(sims[:, j], labels == e)} for j, e in enumerate(emotions)])
    # ...and for free-text mood words users actually type (relevant = any mapped label)
    words = []
    for word, mapped in MOOD_TO_EMOTIONS.items():
        rel = np.isin(labels, mapped)
        if rel.any():
            q = enc.encode_prompt_ensemble(EMOTION_EVAL_TEMPLATES, word)
            words.append({"query": word, "maps_to": "/".join(mapped), **retrieval(cat.embeddings @ q, rel)})
    words = pd.DataFrame(words)

    # 2) Classification
    order = np.argsort(-sims, axis=1)
    pred = np.array(emotions)[order[:, 0]]
    top3 = np.array(emotions)[order[:, :3]]
    top1_acc = float((pred == labels).mean())
    top3_acc = float((top3 == labels[:, None]).any(1).mean())
    per_class = {e: float((pred[labels == e] == e).mean()) for e in emotions}
    macro = float(np.mean(list(per_class.values())))
    neg_true = np.isin(labels, list(NEGATIVE_EMOTIONS))
    neg_pred = np.isin(pred, list(NEGATIVE_EMOTIONS))
    bal_val = 0.5 * ((neg_pred[neg_true]).mean() + (~neg_pred[~neg_true]).mean()) if neg_true.any() else float("nan")
    majority = pd.Series(labels).value_counts()
    summary = pd.DataFrame([
        {"metric": "top-1 accuracy", "value": top1_acc,
         "baseline": float(majority.iloc[0] / len(labels)), "baseline_name": f"always '{majority.index[0]}'"},
        {"metric": "top-3 accuracy", "value": top3_acc, "baseline": 3 / len(emotions), "baseline_name": "chance"},
        {"metric": "macro recall (classes equal)", "value": macro, "baseline": 1 / len(emotions), "baseline_name": "chance"},
        {"metric": "valence balanced accuracy", "value": float(bal_val), "baseline": 0.5, "baseline_name": "chance"},
    ])

    ret.to_csv(out / "retrieval_by_label.csv", index=False)
    words.to_csv(out / "retrieval_by_mood_word.csv", index=False)
    summary.to_csv(out / "classification_summary.csv", index=False)
    pd.Series(per_class, name="recall").to_csv(out / "classification_per_class.csv")
    cm = pd.crosstab(pd.Series(labels, name="label"), pd.Series(pred, name="predicted"))
    cm.to_csv(out / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ret["query"], ret["lift@10"])
    ax.axhline(1.0, color="k", lw=1, ls="--", label="random (lift = 1)")
    ax.set_ylabel("lift@10 (precision / prevalence)")
    ax.set_title("Emotion retrieval vs random baseline")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out / "retrieval_lift.png", dpi=150)

    print(summary.to_string(index=False))
    print("\nRetrieval by label:\n", ret[["query", "n_relevant", "prevalence", "P@5", "P@10", "lift@10", "R@50"]]
          .round(3).to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
