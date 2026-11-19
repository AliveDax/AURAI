"""Extract EmoArt Images.tar.gz on Windows, even when some filenames are
unreadable to Windows tar. Run from the project root:  python extract_emoart.py
Reads dataset/Images.tar.gz and extracts into dataset/.
Renamed or failed files are listed in dataset/extract_log.csv.
"""
import csv
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

ARCHIVE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/Images.tar.gz")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dataset")
LOG = OUT / "extract_log.csv"

ILLEGAL = re.compile(r'[<>:"|?*\x00-\x1f]')
RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
MAX_PART = 120  # longest allowed single file/folder name
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def to_text(name: str):
    """tarfile returns undecodable bytes as lone surrogates. Recover them as latin-1 text."""
    try:
        name.encode("utf-8")
        return name, False
    except UnicodeEncodeError:
        raw = name.encode("utf-8", "surrogateescape")
        return raw.decode("latin-1"), True


def clean_part(part: str) -> str:
    """Make one path component safe to create on Windows."""
    part = ILLEGAL.sub("_", part).rstrip(" .")
    if not part:
        part = "_"
    if part.split(".")[0].upper() in RESERVED:
        part = "_" + part
    if len(part) > MAX_PART:
        ext = Path(part).suffix[:10]
        part = part[: MAX_PART - len(ext)] + ext
    return part


def win_long(path: Path) -> Path:
    """On Windows, use the long-path prefix so paths over 260 characters still work."""
    if os.name == "nt":
        return Path("\\\\?\\" + str(path.resolve()))
    return path


def main() -> None:
    if not ARCHIVE.exists():
        sys.exit(f"Archive not found: {ARCHIVE}. Run this from the project root, or pass the archive path.")
    OUT.mkdir(parents=True, exist_ok=True)
    out_root = OUT.resolve()

    seen = set()  # lowercase target paths, Windows is case-insensitive
    log_rows = []
    n_written = n_skipped_same = n_renamed = n_failed = n_images = 0
    incomplete = False

    try:
        with tarfile.open(ARCHIVE, "r:gz", encoding="utf-8", errors="surrogateescape") as tar:
            for member in tar:
                if not member.isreg():
                    continue  # folders are created on demand; links and special files are ignored
                text, undecodable = to_text(member.name)
                parts = [p for p in PurePosixPath(text.replace("\\", "/")).parts if p not in ("", ".", "..", "/")]
                if not parts:
                    n_failed += 1
                    log_rows.append((ascii(member.name), "", "empty name after cleaning"))
                    continue
                clean = [clean_part(p) for p in parts]
                target = out_root.joinpath(*clean)

                # Two different originals can clean to the same name; keep both.
                k = 0
                while str(target).lower() in seen:
                    k += 1
                    target = target.with_name(f"{target.stem}_dup{k}{target.suffix}")
                seen.add(str(target).lower())

                reasons = []
                if undecodable:
                    reasons.append("undecodable bytes in name")
                if clean != parts:
                    reasons.append("illegal characters or name cleaned")
                if k:
                    reasons.append(f"duplicate after cleaning (dup{k})")
                if reasons:
                    n_renamed += 1
                    log_rows.append((ascii(member.name), str(target.relative_to(out_root)), "; ".join(reasons)))

                real = win_long(target)
                try:
                    if real.exists() and real.stat().st_size == member.size:
                        n_skipped_same += 1
                    else:
                        real.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as src, open(real, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        n_written += 1
                    if target.suffix.lower() in IMAGE_EXT:
                        n_images += 1
                except OSError as err:
                    n_failed += 1
                    log_rows.append((ascii(member.name), str(target.relative_to(out_root)), f"write failed: {err}"))

                done = n_written + n_skipped_same
                if done and done % 500 == 0:
                    print(f"  ...{done} files")
    except (EOFError, tarfile.ReadError) as err:
        incomplete = True
        print(f"\nWARNING: the archive ended early or is damaged ({err}).")
        print("The download is probably incomplete. Re-run the curl.exe line (it resumes), then run this again.")

    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original_name_escaped", "saved_as", "reason"])
        w.writerows(log_rows)

    print("\n--- Summary ---")
    print(f"Files written now:                 {n_written}")
    print(f"Files already there (skipped):     {n_skipped_same}")
    print(f"Image files in total:              {n_images}   (EmoArt-5k should have about 5,600)")
    print(f"Names that had to be fixed:        {n_renamed}")
    print(f"Files that FAILED:                 {n_failed}")
    print(f"Details: {LOG}")
    if incomplete:
        print("Status: INCOMPLETE (see warning above)")
    elif n_failed:
        print("Status: finished with failures, check extract_log.csv")
    else:
        print("Status: OK")


if __name__ == "__main__":
    main()