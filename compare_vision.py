"""
Step 1 deliverable: compare vision options for the damage classifier (component A).

Runs every image in a folder (default eval/inputs/) through:
  1. The Roboflow hosted classification model `damaged-box-auh4b/1`
     (https://universe.roboflow.com/project-33xgh/damaged-box-auh4b, CC BY 4.0).
  2. Gemini Vision zero-shot, asked for the same label plus a damage type.

It also computes the basic image features (brightness, size) that the drift
check will use later, so we can see them on real photos early.

Raw API responses are cached in logs/vision_cache.json (git-ignored), so
re-runs do not burn free-tier quota. The summary is written to
eval/vision_comparison.csv and printed.

Usage:
    uv run python compare_vision.py [folder]

Needs GEMINI_API_KEY and ROBOFLOW_API_KEY in .env (see SETUP.md).
"""

import base64
import csv
import json
import os
import sys
from pathlib import Path

import httpx
from PIL import Image

from env import load_env

ROOT = Path(__file__).resolve().parent
CACHE_FILE = ROOT / "logs" / "vision_cache.json"
OUT_CSV = ROOT / "eval" / "vision_comparison.csv"

ROBOFLOW_MODEL = "damaged-box-auh4b/1"
GEMINI_MODEL = "gemini-2.5-flash"

GEMINI_PROMPT = (
    "You are inspecting a photo for a parcel damage claim. "
    "Answer in JSON with exactly these keys: "
    '"label" ("damaged" or "undamaged"), '
    '"damage_type" (one of "crushed", "torn", "wet", "hole", "burned", "other", "none"), '
    '"confidence" (a number between 0 and 1 for your certainty in the label). '
    "Judge only the parcel or product in the photo."
)


def image_features(path: Path) -> dict:
    """Brightness and size: the numbers the drift check will watch."""
    with Image.open(path) as im:
        gray = im.convert("L")
        pixels = list(gray.getdata())
        return {
            "width": im.width,
            "height": im.height,
            "brightness": round(sum(pixels) / len(pixels) / 255, 3),
        }


def roboflow_classify(path: Path, api_key: str) -> dict:
    """Label and confidence from the hosted damaged-box model."""
    data = base64.b64encode(path.read_bytes()).decode()
    response = httpx.post(
        f"https://classify.roboflow.com/{ROBOFLOW_MODEL}",
        params={"api_key": api_key},
        content=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    # Single-label classification returns top/confidence; fall back to the
    # highest-confidence entry of predictions.
    if "top" in body:
        return {"label": body["top"], "confidence": round(float(body["confidence"]), 3), "raw": body}
    best = max(body.get("predictions", []), key=lambda p: p["confidence"])
    return {"label": best["class"], "confidence": round(float(best["confidence"]), 3), "raw": body}


def gemini_classify(path: Path, api_key: str) -> dict:
    """Zero-shot label, damage type and self-reported confidence from Gemini."""
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[suffix]
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{
                "parts": [
                    {"text": GEMINI_PROMPT},
                    {"inline_data": {"mime_type": mime, "data": base64.b64encode(path.read_bytes()).decode()}},
                ],
            }],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0},
        },
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    text = body["candidates"][0]["content"]["parts"][0]["text"]
    answer = json.loads(text)
    tokens = body.get("usageMetadata", {}).get("totalTokenCount", 0)
    return {
        "label": answer.get("label", "?"),
        "damage_type": answer.get("damage_type", "?"),
        "confidence": round(float(answer.get("confidence", 0)), 3),
        "tokens": tokens,
        "raw": body,
    }


def load_cache() -> dict:
    if CACHE_FILE.is_file():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=1), encoding="utf-8")


def main() -> int:
    load_env()
    missing = [k for k in ("ROBOFLOW_API_KEY", "GEMINI_API_KEY") if not os.environ.get(k, "").strip()]
    if missing:
        print(f"Missing key(s) in .env: {', '.join(missing)}. See SETUP.md for where to get them.")
        return 1

    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval" / "inputs"
    images = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not images:
        print(f"No images found in {folder}. Drop test photos there (no faces, no addresses) and rerun.")
        return 1

    cache = load_cache()
    rows = []
    for path in images:
        key = str(path.relative_to(ROOT))
        if key not in cache:
            print(f"querying APIs for {key} ...")
            cache[key] = {
                "roboflow": roboflow_classify(path, os.environ["ROBOFLOW_API_KEY"]),
                "gemini": gemini_classify(path, os.environ["GEMINI_API_KEY"]),
            }
            save_cache(cache)
        rf, gm = cache[key]["roboflow"], cache[key]["gemini"]
        features = image_features(path)
        # The Roboflow model says damagedpackages/undamagedpackages; normalise both to damaged/undamaged.
        rf_label = "damaged" if "un" not in rf["label"] else "undamaged"
        rows.append({
            "image": key,
            "brightness": features["brightness"],
            "roboflow_label": rf_label,
            "roboflow_confidence": rf["confidence"],
            "gemini_label": gm["label"],
            "gemini_damage_type": gm["damage_type"],
            "gemini_confidence": gm["confidence"],
            "gemini_tokens": gm["tokens"],
            "agree": rf_label == gm["label"],
        })

    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    agreement = sum(r["agree"] for r in rows) / len(rows)
    print(f"\n{len(rows)} images; models agree on {agreement:.0%}. Written to {OUT_CSV}.\n")
    for r in rows:
        mark = "==" if r["agree"] else "!="
        print(f"  {r['image']:<40} roboflow {r['roboflow_label']:<10} ({r['roboflow_confidence']:.2f}) "
              f"{mark} gemini {r['gemini_label']:<10} ({r['gemini_confidence']:.2f}, {r['gemini_damage_type']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
