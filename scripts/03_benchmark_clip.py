"""Step 2a: can this GPU run the configured CLIP model, and how fast?

    .\\venv\\Scripts\\python.exe scripts/03_benchmark_clip.py

Loads the model from config.yaml, embeds `benchmark_images` catalog images, and reports load time,
images/sec (decoding on the CPU and encoding on the GPU, separately and end to end), peak VRAM
and whether the tokenizer truncates long text to 77 tokens. On out-of-memory it steps down through
`fallback_models` and says so. Writes results/clip_benchmark.csv.
"""
import time

import numpy as np
import pandas as pd
import torch

from artrec.clip_embed import ClipEncoder, load_image, set_seed
from artrec.config import load_config, setup_logging
from artrec.data_loader import load_catalog


def benchmark(name: str, pretrained: str, paths: list, cfg) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    enc = ClipEncoder(name, pretrained)
    load_s = time.perf_counter() - t0
    weights_gb = torch.cuda.memory_allocated() / 1024**3

    t0 = time.perf_counter()
    batch = torch.stack([enc.preprocess(load_image(p, cfg.clip["image_draft_px"])) for p in paths])
    cpu_s = time.perf_counter() - t0
    enc.encode_tensor(batch[:2])  # warm-up: the first CUDA call includes one-off setup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    emb = np.concatenate([enc.encode_tensor(batch[i:i + cfg.clip["batch_size"]])
                          for i in range(0, len(batch), cfg.clip["batch_size"])])
    torch.cuda.synchronize()
    gpu_s = time.perf_counter() - t0

    tokens = enc.tokenizer(["word " * 200])  # far longer than CLIP's limit
    norms = np.linalg.norm(emb, axis=1)
    return {"model": name, "pretrained": pretrained, "status": "ok", "input_px": getattr(enc.preprocess.transforms[0], "size", None),
            "embed_dim": enc.dim, "n_images": len(paths), "batch_size": cfg.clip["batch_size"],
            "load_s": round(load_s, 1), "weights_vram_gb": round(weights_gb, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "cpu_decode_img_per_s": round(len(paths) / cpu_s, 1), "gpu_img_per_s": round(len(paths) / gpu_s, 1),
            "end_to_end_img_per_s": round(len(paths) / (cpu_s + gpu_s), 1),
            "norm_min": round(float(norms.min()), 4), "norm_max": round(float(norms.max()), 4),
            "tokenizer_len_for_200_words": int(tokens.shape[1]), "tokenizer_truncates": int(tokens.shape[1]) == 77}


def main() -> None:
    setup_logging()
    cfg = load_config()
    set_seed(cfg.seed)
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU: run scripts/00_check_env.py. The benchmark is GPU-only.")
    catalog = load_catalog(cfg)
    paths = [cfg.root / p for p in catalog.sample(cfg.clip["benchmark_images"], random_state=cfg.seed)["image_path"]]

    rows = []
    for name in [cfg.clip["model"]] + cfg.clip["fallback_models"]:
        try:
            rows.append(benchmark(name, cfg.clip["pretrained"], paths, cfg))
            break  # the first model that fits is the one to use
        except torch.cuda.OutOfMemoryError:
            print(f"OUT OF MEMORY with {name}; trying the next smaller model.")
            rows.append({"model": name, "pretrained": cfg.clip["pretrained"], "status": "out of memory"})
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = cfg.path("results_dir") / "clip_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.T.to_string(header=False))
    ok = df[df["status"] == "ok"]
    if ok.empty:
        raise SystemExit("No model fits in VRAM.")
    r = ok.iloc[0]
    if r["model"] != cfg.clip["model"]:
        print(f"\nNOTE: {cfg.clip['model']} did not fit; {r['model']} did. Set clip.model in config.yaml.")
    print(f"\nSummary: {r['model']} runs at ~{r['end_to_end_img_per_s']} img/s end to end, peak VRAM "
          f"{r['peak_vram_gb']} GB, tokenizer truncates to 77: {r['tokenizer_truncates']}. "
          f"Full catalog ({len(catalog)}) ~{len(catalog) / r['end_to_end_img_per_s'] / 60:.0f} min.")


if __name__ == "__main__":
    main()
