"""Step 4: does CLIP's emotion matching work? EmoArt's dominant_emotion labels are the answer key.

    .\\venv\\Scripts\\python.exe scripts/06_validate_emotion.py

E1  classification: for every artwork, rank the emotion prompts by cosine. recall@1/2/3/5,
    accuracy, macro-F1, balanced accuracy, per-class recall, confusion matrix. Prompt forms
    compared: bare word / single template / template ensemble. Baselines: chance and majority class.
E2  retrieval + paraphrase robustness: rank all artworks for each paraphrase in
    data/emotion_paraphrases.yaml; precision@10/@20 against the label, mean and min per emotion.
E3  emotion vs emotion+object: "calm" vs "calm forest" as one phrase (blend) vs separate
    embeddings fused like the real system (split). Precision@k against the emotion label.
    This decides the default text.text_mode.
Note: the labels were produced by GPT-4o with verification, not by human raters, so this measures
agreement with those labels. All tables and figures go to results/.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from artrec.clip_embed import ClipEncoder, load_embeddings, set_seed  # noqa: E402
from artrec.config import load_config, load_yaml, setup_logging  # noqa: E402
from artrec.data_loader import load_catalog  # noqa: E402
from artrec.scorer import fuse  # noqa: E402


def precision_at(scores: np.ndarray, relevant: np.ndarray, k: int) -> float:
    return float(relevant[np.argsort(-scores, kind="stable")[:k]].mean())


def e1(emb, y, labels, words, enc, cfg, res):
    """Classification-style test: which of the K emotion prompts is closest to each artwork?"""
    K, ks = len(labels), cfg.validation["e1_recall_ks"]
    templates = cfg.text["emotion_templates"]
    forms = {"bare word": None, "single template": templates[:1], "template ensemble": templates}
    rows, per_class, preds = [], {}, {}
    for form, tpl in forms.items():
        text = enc.encode_phrases(words, tpl)            # (K, D)
        order = np.argsort(-(emb @ text.T), axis=1)      # (N, K), best prompt first
        pred = order[:, 0]
        row = {"form": form}
        for k in ks:
            row[f"recall@{k}"] = float((order[:, :k] == y[:, None]).any(1).mean())
        rec = [float((pred[y == c] == c).mean()) for c in range(K)]
        row.update(accuracy=row["recall@1"], balanced_accuracy=float(np.mean(rec)),
                   macro_f1=float(f1_score(y, pred, labels=range(K), average="macro", zero_division=0)))
        rows.append(row)
        per_class[form], preds[form] = rec, pred

    # Baselines. chance: a random ranking. majority: always rank the labels by how common they are.
    prev = np.bincount(y, minlength=K) / len(y)
    by_freq = np.argsort(-prev)
    rows.append({"form": "baseline: chance", **{f"recall@{k}": k / K for k in ks}, "accuracy": 1 / K,
                 "balanced_accuracy": 1 / K, "macro_f1": np.nan})
    maj = np.full_like(y, by_freq[0])
    rows.append({"form": "baseline: majority", **{f"recall@{k}": float(prev[by_freq[:k]].sum()) for k in ks},
                 "accuracy": float(prev.max()), "balanced_accuracy": 1 / K,
                 "macro_f1": float(f1_score(y, maj, labels=range(K), average="macro", zero_division=0))})
    table = pd.DataFrame(rows)
    table.to_csv(res / "e1_emotion_classification.csv", index=False)
    pc = pd.DataFrame(per_class, index=labels)
    pc.insert(0, "n", np.bincount(y, minlength=K))
    pc.to_csv(res / "e1_per_class_recall.csv")

    best = max(forms, key=lambda f: table.set_index("form").loc[f, "macro_f1"])
    cm = confusion_matrix(y, preds[best], labels=range(K))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(res / "e1_confusion_matrix.csv")
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K), labels, rotation=45, ha="right")
    ax.set_yticks(range(K), [f"{lab} (n={n})" for lab, n in zip(labels, cm.sum(1))])
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("CLIP prediction")
    ax.set_ylabel("EmoArt label (GPT-4o)")
    ax.set_title(f"E1 confusion matrix, row-normalized ({best})")
    fig.tight_layout()
    fig.savefig(res / "e1_confusion_matrix.png", dpi=130)
    plt.close(fig)
    return table, best


def e2(emb, labels_arr, labels, enc, cfg, res, paraphrases):
    """Retrieval-style test: does each phrasing of an emotion retrieve artworks with that label?"""
    ks = cfg.validation["retrieval_ks"]
    modes = {"raw text": None, "template ensemble": cfg.text["emotion_templates"]}
    rows = []
    for label in labels:
        relevant = labels_arr == label
        for phrase in paraphrases.get(label, []):
            for mode, tpl in modes.items():
                sims = emb @ enc.encode_phrases([phrase], tpl)[0]
                rows.append({"emotion": label, "paraphrase": phrase, "mode": mode, "prevalence": relevant.mean(),
                             **{f"P@{k}": precision_at(sims, relevant, k) for k in ks}})
    df = pd.DataFrame(rows)
    df.to_csv(res / "e2_paraphrase_retrieval.csv", index=False)
    k_last = f"P@{ks[-1]}"
    summ = df.groupby(["emotion", "mode"]).agg(
        n_paraphrases=("paraphrase", "size"), prevalence=("prevalence", "first"),
        **{f"mean_P@{k}": (f"P@{k}", "mean") for k in ks}, **{f"min_P@{k}": (f"P@{k}", "min") for k in ks}
    ).reset_index()
    summ[f"lift_{k_last}"] = summ[f"mean_{k_last}"] / summ["prevalence"]
    summ.to_csv(res / "e2_paraphrase_summary.csv", index=False)

    pivot = summ.pivot(index="emotion", columns="mode", values=f"mean_{k_last}").reindex(labels)
    ax = pivot.plot.bar(figsize=(10, 4))
    prev = summ.drop_duplicates("emotion").set_index("emotion")["prevalence"].reindex(labels)
    ax.scatter(range(len(labels)), prev, color="k", marker="_", s=300, zorder=3, label="random (prevalence)")
    ax.set_ylabel(f"mean {k_last} over paraphrases")
    ax.set_title(f"E2 paraphrase retrieval ({k_last} vs label prevalence)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(res / f"e2_precision_at_{ks[-1]}.png", dpi=130)
    plt.close()
    macro = summ.groupby("mode")[[c for c in summ.columns if c.startswith(("mean_", "min_", "lift_"))]].mean()
    return macro


def e3(emb, labels_arr, labels, words, enc, cfg, res, subjects):
    """Emotion only vs emotion+object as one phrase (blend) vs separate embeddings (split)."""
    k, t = cfg.validation["e3_k"], cfg.text
    w = cfg.scoring["weights"]
    rows = []
    desc = {s: emb @ enc.encode_phrases([s], t["description_templates"])[0] for s in subjects}
    for label, word in zip(labels, words):
        relevant = labels_arr == label
        emo = emb @ enc.encode_phrases([word], t["emotion_templates"])[0]
        for s in subjects:
            blend = emb @ enc.encode_phrases([f"{word} {s}"], t["blend_templates"])[0]
            # split: exactly the system's fusion, restricted to the two text components
            split, _, _ = fuse({"emotion": emo, "description": desc[s]},
                               {"emotion": w["emotion"], "description": w["description"]},
                               cfg.scoring["normalization"])
            for mode, sims in (("a) emotion only", emo), ("b) blend", blend), ("c) split", split)):
                rows.append({"emotion": label, "subject": s, "mode": mode, "query": f"{word} {s}",
                             f"P@{k}": precision_at(sims, relevant, k)})
    df = pd.DataFrame(rows)
    df.to_csv(res / "e3_emotion_object_queries.csv", index=False)
    per_emotion = df.groupby(["emotion", "mode"])[f"P@{k}"].mean().unstack().reindex(labels)
    per_emotion.to_csv(res / "e3_emotion_object_per_emotion.csv")
    macro = per_emotion.mean()
    macro.rename(f"macro_P@{k}").to_csv(res / "e3_emotion_object_summary.csv")
    ax = per_emotion.plot.bar(figsize=(10, 4))
    ax.set_ylabel(f"P@{k} for the emotion label")
    ax.set_title("E3 emotion only vs emotion+object (blend vs split)")
    plt.tight_layout()
    plt.savefig(res / "e3_emotion_vs_object.png", dpi=130)
    plt.close()
    return macro


def main() -> None:
    setup_logging()
    cfg = load_config()
    set_seed(cfg.seed)
    res = cfg.path("results_dir")
    res.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog(cfg)
    emb = load_embeddings(cfg, catalog)
    enc = ClipEncoder.from_config(cfg)

    # The label set is read from the data, most common first; prompt words come from the lexicon
    counts = catalog["dominant_emotion"].value_counts()
    labels = list(counts.index)
    lexicon = load_yaml(cfg.path("emotion_lexicon"))
    missing = [lab for lab in labels if lab not in lexicon]
    if missing:
        print(f"WARNING: labels without a lexicon entry (bare label used as prompt word): {missing}")
    words = [lexicon.get(lab, {}).get("prompt_word", lab.lower()) for lab in labels]
    labels_arr = catalog["dominant_emotion"].to_numpy()
    y = np.array([labels.index(lab) for lab in labels_arr])
    para = load_yaml(cfg.path("emotion_paraphrases"))

    print(f"Labels (GPT-4o annotations, {len(labels)}): {counts.to_dict()}\n")
    t1, best = e1(emb, y, labels, words, enc, cfg, res)
    print("E1 classification:\n" + t1.round(3).to_string(index=False) + f"\n  best prompt form by macro-F1: {best}\n")
    m2 = e2(emb, labels_arr, labels, enc, cfg, res, para["paraphrases"])
    print("E2 paraphrase retrieval (macro over emotions):\n" + m2.round(3).to_string() + "\n")
    m3 = e3(emb, labels_arr, labels, words, enc, cfg, res, para["e3_subjects"])
    print(f"E3 emotion vs emotion+object (macro P@{cfg.validation['e3_k']}):\n" + m3.round(3).to_string() + "\n")

    better = "split" if m3["c) split"] >= m3["b) blend"] else "blend"
    summary = {"model": f"{cfg.clip['model']} / {cfg.clip['pretrained']}", "labels": counts.to_dict(),
               "label_source": "GPT-4o annotations (EmoArt), not human labels",
               "e1": t1.set_index("form").round(4).to_dict(orient="index"), "e1_best_form_by_macro_f1": best,
               "e2_macro_over_emotions": m2.round(4).to_dict(orient="index"),
               "e3_macro_over_emotions": m3.round(4).to_dict(),
               "e3_better_text_mode_for_emotion_precision": better,
               "config_text_mode": cfg.text["text_mode"],
               "e3_note": "E3 only measures whether the EMOTION survives adding a subject; whether the "
                          "subject itself is found is not measured (EmoArt has no subject labels)."}
    (res / "emotion_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    acc = t1.set_index("form")
    print(f"Summary: best E1 form '{best}' gets macro-F1 {acc.loc[best, 'macro_f1']:.3f} "
          f"(majority baseline {acc.loc['baseline: majority', 'macro_f1']:.3f}) and balanced accuracy "
          f"{acc.loc[best, 'balanced_accuracy']:.3f} (chance {1 / len(labels):.3f}). E3: {better} keeps the "
          f"emotion better than the other mode (config text_mode = {cfg.text['text_mode']}). Saved to results/.")


if __name__ == "__main__":
    main()
