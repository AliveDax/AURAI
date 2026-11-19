"""Step 3: dominant-colour palette of every artwork, computed once and cached.

    .\\venv\\Scripts\\python.exe scripts/05_extract_palettes.py [--force]

Writes embeddings/palettes_k<k>_px<px>_cov<min>_wb<0|1>.npz (row i = catalog row i), a summary
JSON and a figure with example palettes in results/. A bad image is a hard stop, because the
palettes must stay row-aligned with the catalog.
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from artrec.color_extract import (load_palettes, palette_at, palette_cache_path, palette_from_images,  # noqa: E402
                                  save_palettes, stack_palettes)
from artrec.config import load_config, setup_logging  # noqa: E402
from artrec.data_loader import load_catalog  # noqa: E402

N_EXAMPLES = 16  # artworks shown in the example figure


def example_figure(catalog, arrays, cfg, out):
    rng = np.random.default_rng(cfg.seed)
    rows = rng.choice(len(catalog), N_EXAMPLES, replace=False)
    fig, axes = plt.subplots(N_EXAMPLES // 4, 8, figsize=(16, 2.2 * N_EXAMPLES // 4))
    for n, i in enumerate(rows):
        ax_img, ax_pal = axes[n // 4, 2 * (n % 4)], axes[n // 4, 2 * (n % 4) + 1]
        img = Image.open(cfg.root / catalog["image_path"].iloc[i]).convert("RGB")
        img.thumbnail((200, 200))
        ax_img.imshow(img)
        ax_img.set_title(catalog["style"].iloc[i][:22], fontsize=7)
        p = palette_at(arrays, i)
        left = 0.0
        for rgb, cov, neutral in zip(p.rgb, p.coverage, p.neutral):
            ax_pal.barh(0, cov, left=left, color=rgb, edgecolor="k" if neutral else "none")
            left += cov
        ax_pal.set_xlim(0, 1)
        for ax in (ax_img, ax_pal):
            ax.axis("off")
    fig.suptitle("Artwork palettes (bar width = coverage; outlined = neutral)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    catalog = load_catalog(cfg)
    path = palette_cache_path(cfg)
    if path.exists() and not args.force:
        try:
            load_palettes(cfg, catalog)
            print(f"Valid cache {path.name} already exists. Use --force to redo it.")
            return
        except ValueError as err:
            print(f"Existing cache is not valid ({err}); recomputing.")

    t0 = time.perf_counter()
    palettes = []
    for p in tqdm(catalog["image_path"], desc="palettes"):
        try:
            palettes.append(palette_from_images([cfg.root / p], cfg))
        except Exception as err:
            raise SystemExit(f"Could not read {p} ({err}). Fix or remove it, rebuild the catalog (02), then re-run.")
    seconds = time.perf_counter() - t0
    arrays = stack_palettes(palettes, cfg.color["k"])
    save_palettes(arrays, catalog, cfg)

    res = cfg.path("results_dir")
    res.mkdir(parents=True, exist_ok=True)
    n_colors = arrays["valid"].sum(1)
    cov, neutral = arrays["coverage"], arrays["neutral"]
    summary = {"settings": {k: cfg.color[k] for k in ("k", "downsample_px", "min_coverage", "white_balance")},
               "artworks": len(catalog), "seconds": round(seconds, 1),
               "colors_per_palette": {int(k): int(v) for k, v in zip(*np.unique(n_colors, return_counts=True))},
               "share_dominant_color_neutral": round(float(neutral[:, 0].mean()), 3),
               "mean_neutral_coverage": round(float((cov * neutral).sum(1).mean()), 3),
               "cache_file": f"{cfg.paths['embeddings_dir']}/{path.name}"}
    (res / f"palette_summary_{path.stem}.json").write_text(json.dumps(summary, indent=2))
    example_figure(catalog, arrays, cfg, res / f"palette_examples_{path.stem}.png")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary: {len(catalog)} palettes, {n_colors.mean():.2f} colours on average; "
          f"{summary['share_dominant_color_neutral']:.0%} of artworks have a neutral dominant colour.")


if __name__ == "__main__":
    main()
