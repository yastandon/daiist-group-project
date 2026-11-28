"""
Build the drift demo's shifted inputs from our own test photos.

Takes every image in eval/inputs/, saves a darkened and a blurred copy in
eval/inputs_shifted/, and writes eval/shifted.csv pointing at the copies.
Commit both: `main.py drift-demo` runs them through the pipeline, and the
drift alert must fire on them (darker photos shift `brightness`, blurrier
photos shift the vision model's `confidence`).

Usage:
    uv run python make_shifted.py [--darken 0.35] [--blur 4]
"""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "eval" / "inputs"
OUT_DIR = ROOT / "eval" / "inputs_shifted"
OUT_CSV = ROOT / "eval" / "shifted.csv"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--darken", type=float, default=0.35, help="brightness factor (default 0.35)")
    parser.add_argument("--blur", type=float, default=4.0, help="gaussian blur radius (default 4)")
    args = parser.parse_args()

    photos = sorted(p for p in INPUTS.glob("*") if p.suffix.lower() in SUFFIXES)
    if not photos:
        print(f"No photos in {INPUTS.relative_to(ROOT)}. Add the team's test photos first.")
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for photo in photos:
        with Image.open(photo) as im:
            im = im.convert("RGB")
            dark = OUT_DIR / f"{photo.stem}_dark.jpg"
            ImageEnhance.Brightness(im).enhance(args.darken).save(dark, quality=85)
            blur = OUT_DIR / f"{photo.stem}_blur.jpg"
            im.filter(ImageFilter.GaussianBlur(args.blur)).save(blur, quality=85)
        rows += [str(dark.relative_to(ROOT)), str(blur.relative_to(ROOT))]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input"])
        writer.writerows([r] for r in rows)

    print(f"{len(photos)} photos -> {len(rows)} shifted inputs in {OUT_DIR.relative_to(ROOT)}.")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}. Commit both, then run `uv run python main.py drift-demo`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
