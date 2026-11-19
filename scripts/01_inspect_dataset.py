"""Step 1a: look at the real annotation file before trusting any key name or label.

    .\\venv\\Scripts\\python.exe scripts/01_inspect_dataset.py

Prints the key structure of the first entries, the unique values of the label fields in
config.yaml (these are the true emotion labels), the path format, and how many annotation
paths point to an image on disk. Writes results/dataset_inspection.json.
"""
import json
from collections import Counter

from artrec.config import load_config
from artrec.data_loader import get_nested, index_images, normalize_rel_path


def show_keys(obj, indent: int = 2) -> None:
    """Nested keys with a short preview of each value."""
    for k, v in obj.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            show_keys(v, indent + 2)
        else:
            print(" " * indent + f"{k}: {repr(v)[:90]}")


def main() -> None:
    cfg = load_config()
    with open(cfg.path("annotation_file"), encoding="utf-8") as f:
        entries = json.load(f)
    print(f"{cfg.paths['annotation_file']}: a {type(entries).__name__} of {len(entries)} entries\n")
    for e in entries[:2]:
        print("Entry:")
        show_keys(e)
        print()

    fields = cfg.annotation_fields
    report = {"entries": len(entries), "top_level_keys": dict(Counter(k for e in entries for k in e))}
    for col in ("dominant_emotion", "valence", "arousal"):
        values = Counter(str(get_nested(e, fields[col])) for e in entries)
        report[col] = dict(values.most_common())
        print(f"{col} ({len(values)} unique): {dict(values.most_common())}")

    paths = [str(get_nested(e, fields["image_path"])) for e in entries]
    print(f"\nPath format example: {paths[0]!r}  (backslashes: {sum(chr(92) in p for p in paths)} of {len(paths)})")
    disk, dups = index_images(cfg.path("dataset_dir"), cfg.path("images_dir"))
    exact = sum(normalize_rel_path(p) in disk for p in paths)
    print(f"Images on disk: {len(disk) + len(dups)} files; annotation paths found exactly: {exact} / {len(paths)}")
    report.update(images_on_disk=len(disk) + len(dups), exact_path_matches=exact)

    out = cfg.path("results_dir") / "dataset_inspection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSummary: {len(report['dominant_emotion'])} emotion labels read from the data; saved {out.name}.")


if __name__ == "__main__":
    main()
