"""Step 2b: CLIP image embeddings for the whole catalog, computed once and cached.

    .\\venv\\Scripts\\python.exe scripts/04_embed_artworks.py [--force]

Resumable: every `chunk_size` rows are checkpointed to embeddings/partial/<model>/, so an
interrupted run continues where it stopped. Writes embeddings/emb_<model>_<pretrained>.npy
(N x D, L2-normalized, row i = catalog row i) plus an index file, and fails loudly on any
mismatch. Skips the work if a valid cache already exists (use --force to redo it).
"""
import argparse
import json
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from artrec.clip_embed import (ClipEncoder, ImageFileDataset, cache_stem, catalog_fingerprint, collate_skip_failed,
                               embedding_cache_paths, load_embeddings, save_embeddings, set_seed)
from artrec.config import load_config, setup_logging
from artrec.data_loader import load_catalog


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if a valid cache exists")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config()
    set_seed(cfg.seed)
    catalog = load_catalog(cfg)
    c = cfg.clip
    emb_path, index_path = embedding_cache_paths(cfg)
    stem = cache_stem(c["model"], c["pretrained"])

    if emb_path.exists() and not args.force:
        try:
            load_embeddings(cfg, catalog)
            print(f"Valid cache {emb_path.name} already exists ({len(catalog)} rows). Use --force to redo it.")
            return
        except ValueError as err:
            print(f"Existing cache is not valid ({err}); recomputing.")

    # Checkpoints are only reused if they were made for this exact model and catalog order
    partial = cfg.path("embeddings_dir") / "partial" / stem
    meta = {"model": c["model"], "pretrained": c["pretrained"], "n_rows": len(catalog),
            "chunk_size": c["chunk_size"], "catalog_sha1": catalog_fingerprint(catalog)}
    meta_file = partial / "meta.json"
    if args.force or not meta_file.exists() or json.loads(meta_file.read_text()) != meta:
        shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(meta, indent=2))

    enc = ClipEncoder.from_config(cfg)
    if torch.cuda.is_available():  # VRAM stats exist only on CUDA (not on a Mac's MPS)
        torch.cuda.reset_peak_memory_stats()
    paths = [cfg.root / p for p in catalog["image_path"]]
    t0, n_new = time.perf_counter(), 0
    starts = range(0, len(paths), c["chunk_size"])
    for start in tqdm(starts, desc=f"chunks ({c['model']})"):
        end = min(start + c["chunk_size"], len(paths))
        chunk_file = partial / f"chunk_{start:06d}_{end:06d}.npy"
        if chunk_file.exists():
            continue  # done in an earlier run
        loader = DataLoader(ImageFileDataset(paths[start:end], enc.preprocess, c["image_draft_px"]),
                            batch_size=c["batch_size"], shuffle=False, num_workers=c["num_workers"],
                            collate_fn=collate_skip_failed, pin_memory=True)
        out = np.full((end - start, enc.dim), np.nan, np.float32)
        failed = []
        for x, idx, bad in loader:
            failed += bad
            if x is not None:
                out[idx] = enc.encode_tensor(x)
        if failed:
            # Embeddings must stay row-aligned with the catalog, so a bad image is a hard stop
            raise SystemExit(f"{len(failed)} images could not be read, e.g. {paths[start + failed[0]]}. "
                             "Fix or remove them, rebuild the catalog (02), then re-run.")
        np.save(chunk_file, out)
        n_new += end - start
    seconds = time.perf_counter() - t0

    emb = np.concatenate([np.load(partial / f"chunk_{s:06d}_{min(s + c['chunk_size'], len(paths)):06d}.npy")
                          for s in starts])
    save_embeddings(emb, catalog, emb_path, index_path)  # checks rows, finiteness and unit length
    shutil.rmtree(partial, ignore_errors=True)

    # Sanity check: artworks of the same style should be more similar to each other than to others
    rng = np.random.default_rng(cfg.seed)
    i, j = rng.integers(0, len(emb), (2, 20000))
    cos = (emb[i] * emb[j]).sum(1)
    same = catalog["style"].to_numpy()[i] == catalog["style"].to_numpy()[j]
    summary = {"model": c["model"], "pretrained": c["pretrained"], "rows": len(emb), "dim": int(emb.shape[1]),
               "embedded_this_run": n_new, "seconds": round(seconds, 1),
               "img_per_s": round(n_new / seconds, 1) if n_new else None,
               "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                                if torch.cuda.is_available() else None),
               "mean_cos_same_style": round(float(cos[same].mean()), 4),
               "mean_cos_different_style": round(float(cos[~same].mean()), 4),
               "cache_file": f"{cfg.paths['embeddings_dir']}/{emb_path.name}"}
    out = cfg.path("results_dir") / f"embedding_summary_{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSummary: {len(emb)} artworks embedded with {c['model']} ({emb.shape[1]}-d, unit length), "
          f"same-style pairs are more similar ({summary['mean_cos_same_style']}) than different-style "
          f"pairs ({summary['mean_cos_different_style']}).")


if __name__ == "__main__":
    main()
