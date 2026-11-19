"""Step 0: check the environment. Run this first on a new machine.

    .\\venv\\Scripts\\python.exe scripts/00_check_env.py

Prints the project root (warns about OneDrive), Python, torch + GPU, every package from
requirements.txt, and whether the dataset, main.py and data/rooms/ are there.
"""
import importlib.metadata as md
import platform
import re
import sys

from artrec.config import PROJECT_ROOT, load_config

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def line(ok: bool, text: str) -> None:
    print(f"  [{'OK' if ok else '!!'}] {text}")


def main() -> None:
    cfg = load_config()
    print(f"Project root: {PROJECT_ROOT}")
    if "onedrive" in str(PROJECT_ROOT).lower():
        print("  WARNING: the project is inside OneDrive. OneDrive tries to sync the 2.6 GB dataset and the\n"
              "  caches and can lock files while scripts write them. Pause sync while working, or move the\n"
              "  folder outside OneDrive (then recreate the venv).")
    if " " in str(PROJECT_ROOT):
        print("  Note: the path contains spaces; quote it in every command.")

    print(f"\nPython {platform.python_version()} ({sys.executable})")
    try:
        import torch
        cuda = torch.cuda.is_available()
        line(True, f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
        if cuda:
            p = torch.cuda.get_device_properties(0)
            line(True, f"GPU: {p.name}, {p.total_memory / 1024**3:.1f} GB VRAM")
        else:
            line(False, "NO CUDA GPU: torch.cuda.is_available() is False. The system will NOT fall back to the CPU.\n"
                        "       Install the CUDA build: .\\venv\\Scripts\\python.exe -m pip install torch torchvision "
                        "--index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        line(False, "torch is NOT installed (install the CUDA build, see requirements.txt)")

    print("\nPackages (requirements.txt):")
    missing = []
    for raw in (PROJECT_ROOT / "requirements.txt").read_text().splitlines():
        name = re.split(r"[<>=\s#]", raw.strip())[0]
        if not name:
            continue
        try:
            line(True, f"{name} {md.version(name)}")
        except md.PackageNotFoundError:
            line(False, f"{name} MISSING")
            missing.append(name)

    print("\nData:")
    ds, imgs, ann = cfg.path("dataset_dir"), cfg.path("images_dir"), cfg.path("annotation_file")
    line(ds.exists(), f"{cfg.paths['dataset_dir']}/")
    line(ann.exists(), f"{cfg.paths['annotation_file']}")
    if imgs.exists():
        folders = [p for p in imgs.iterdir() if p.is_dir()]
        n = sum(1 for p in imgs.rglob("*") if p.suffix.lower() in IMAGE_EXT)
        line(n > 0, f"{cfg.paths['images_dir']}/: {len(folders)} style folders, {n} image files")
    else:
        line(False, f"{cfg.paths['images_dir']}/ missing (extract Images.tar.gz with extract_emoart.py)")
    line((PROJECT_ROOT / "main.py").exists(), "main.py")
    rooms = cfg.path("rooms_dir")
    n_rooms = sum(1 for p in rooms.glob("*") if p.suffix.lower() in IMAGE_EXT) if rooms.exists() else 0
    line(rooms.exists(), f"{cfg.paths['rooms_dir']}/ ({n_rooms} room photos; aim for 10-15 varied rooms)")

    print("\nSummary:", "everything needed is installed." if not missing else
          f"install the missing packages: .\\venv\\Scripts\\python.exe -m pip install {' '.join(missing)}")


if __name__ == "__main__":
    main()
