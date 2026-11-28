"""
Build eval/reference.csv: sample images from the damaged-box dataset on Roboflow.

The reference set is what "normal inputs look like" for the drift check:
a sample of the data component A was built on (see eval/README.md), NOT our
own test photos (those are the cases). Downloads the dataset export once
(cached zip in logs/), resizes a sample to max 1024 px into
eval/reference_images/, and writes eval/reference.csv.

IMPORTANT: some dataset images show people. After downloading, LOOK AT
eval/reference_images/, delete any image with a person or address, then run
`uv run python sample_reference.py --csv` to rewrite the CSV from what is left.
Commit the images and the CSV, then run `uv run python main.py reference`.

Usage:
    uv run python sample_reference.py [n]      # download + sample n images (default 40) + write csv
    uv run python sample_reference.py --csv    # only rewrite the csv from eval/reference_images/
"""

import csv
import io
import os
import random
import sys
import zipfile
from pathlib import Path

import httpx
from PIL import Image

from env import load_env

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "eval" / "reference_images"
OUT_CSV = ROOT / "eval" / "reference.csv"
ZIP_CACHE = ROOT / "logs" / "damaged-box-dataset.zip"

WORKSPACE, PROJECT, VERSION = "project-33xgh", "damaged-box-auh4b", 1
EXPORT_URL = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{VERSION}/folder"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SEED = 7  # fixed, so re-running samples the same images


def write_csv() -> None:
    images = sorted(p for p in OUT_DIR.glob("*") if p.suffix.lower() in SUFFIXES)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input"])
        writer.writerows([str(p.relative_to(ROOT))] for p in images)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}: {len(images)} inputs.")
    if len(images) < 30:
        print("WARNING: fewer than 30; eval/README.md asks for 30+.")


def download_zip(api_key: str) -> bytes:
    if ZIP_CACHE.is_file():
        print(f"Using cached export {ZIP_CACHE.relative_to(ROOT)}.")
        return ZIP_CACHE.read_bytes()
    print("Requesting the dataset export from Roboflow...")
    response = httpx.get(EXPORT_URL, params={"api_key": api_key}, timeout=120, follow_redirects=True)
    response.raise_for_status()
    link = response.json().get("export", {}).get("link")
    if not link:
        raise SystemExit(f"No export link in Roboflow's answer: {response.text[:300]}")
    print("Downloading the export zip (one-time; cached in logs/)...")
    data = httpx.get(link, timeout=600, follow_redirects=True).content
    ZIP_CACHE.parent.mkdir(exist_ok=True)
    ZIP_CACHE.write_bytes(data)
    return data


def main() -> int:
    if "--csv" in sys.argv:
        write_csv()
        return 0

    load_env()
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("ROBOFLOW_API_KEY is not set (see SETUP.md).")
        return 1
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 40

    data = download_zip(api_key)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [m for m in zf.namelist() if Path(m).suffix.lower() in SUFFIXES]
        if not names:
            raise SystemExit("The export zip has no images.")
        random.Random(SEED).shuffle(names)
        OUT_DIR.mkdir(exist_ok=True)
        kept = 0
        for name in names:
            if kept >= n:
                break
            with zf.open(name) as member:
                im = Image.open(member).convert("RGB")
            im.thumbnail((1024, 1024))
            im.save(OUT_DIR / f"ref_{kept:03d}.jpg", quality=85)
            kept += 1
    print(f"Sampled {kept} of {len(names)} images into {OUT_DIR.relative_to(ROOT)} (seed {SEED}).")

    write_csv()
    print("\nNOW: look through the folder, DELETE any image showing a person or an address,")
    print("then run `uv run python sample_reference.py --csv` and commit images + csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
