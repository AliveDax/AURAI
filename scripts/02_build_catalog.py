"""Step 1b: annotation.json + images on disk -> data/catalog.csv.

    .\\venv\\Scripts\\python.exe scripts/02_build_catalog.py

Reports how many annotations matched an image by each method, which matched none, files without
an annotation and corrupt images. Writes results/catalog_build_summary.json and
results/catalog_emotion_counts.csv. Rebuilding the catalog invalidates the embedding and palette
caches (they check the row order and will ask you to re-run scripts 04 and 05).
"""
import json

from artrec.config import load_config, setup_logging
from artrec.data_loader import build_catalog


def main() -> None:
    setup_logging()
    cfg = load_config()
    df, report = build_catalog(cfg)
    out = cfg.path("catalog")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    res = cfg.path("results_dir")
    res.mkdir(parents=True, exist_ok=True)
    counts = df.groupby(["dominant_emotion", "valence", "arousal"]).size().rename("n").reset_index()
    counts.sort_values("n", ascending=False).to_csv(res / "catalog_emotion_counts.csv", index=False)
    report["styles"] = int(df["style"].nunique())
    report["dominant_emotion_counts"] = df["dominant_emotion"].value_counts().to_dict()
    (res / "catalog_build_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nCatalog: {len(df)} artworks, {report['styles']} styles -> {cfg.paths['catalog']}")
    print(f"Matched by method: {report['matched_by_method']}")
    print(f"Annotations without an image: {len(report['annotations_without_file'])} "
          f"{report['annotations_without_file'][:5]}")
    print(f"Images without an annotation (not used): {len(report['files_without_annotation'])}")
    print(f"Unicode-duplicate files ignored: {len(report['unicode_duplicate_files_ignored'])}; "
          f"corrupt images skipped: {len(report['corrupt_images_skipped'])}")
    print(f"Emotion labels (from the data): {report['dominant_emotion_counts']}")


if __name__ == "__main__":
    main()
