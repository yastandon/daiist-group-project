"""
Component A: is the parcel in the photo damaged?

Calls the Roboflow hosted classification model `damaged-box-auh4b/1`
(https://universe.roboflow.com/project-33xgh/damaged-box-auh4b, CC BY 4.0),
the day-one choice from the Step 1 comparison (compare_vision.py).

Mirrors gemini.py: httpx (no new dependency), retries with backoff, and a
git-ignored cache keyed by the image's content, so evaluation re-runs and
`main.py reference` do not burn the free tier. Roboflow's free hosted
inference has no per-call price, so component A reports cost 0.

Also computes the image features (brightness, size) that the drift check
watches: a darker or blurrier stream of photos is our main drift scenario.
"""

import base64
import hashlib
import json
import time
from pathlib import Path

import httpx
import os
from PIL import Image

ROOT = Path(__file__).resolve().parent
CACHE_FILE = ROOT / "logs" / "vision_api_cache.json"  # logs/ is git-ignored

MODEL = "damaged-box-auh4b/1"
VERSION = f"roboflow-{MODEL}"
API_URL = f"https://classify.roboflow.com/{MODEL}"

# The model's class names, normalised to what decision.py expects.
LABELS = {"damagedpackages": "damaged", "undamagedpackages": "undamaged"}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class VisionError(RuntimeError):
    pass


def is_image(raw: str) -> bool:
    """Does this input string point to an image file on disk?"""
    path = Path(str(raw))
    return (
        len(str(raw)) < 500
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.is_file()
    )


def features(image_path: str) -> dict[str, float]:
    """Numbers that describe the photo, for the drift check."""
    with Image.open(image_path) as im:
        gray = im.convert("L")
        pixels = list(gray.getdata())
        return {
            "brightness": round(sum(pixels) / len(pixels) / 255, 3),
            "megapixels": round(im.width * im.height / 1e6, 3),
        }


def _load_cache() -> dict:
    if CACHE_FILE.is_file():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=1), encoding="utf-8")


def classify(image_path: str, retries: int = 3) -> dict:
    """Label and confidence for one photo: {"label", "confidence", "model_label", "latency_s", "cached"}.

    Raises VisionError if the key is missing or the API keeps failing; the
    caller (pipeline.component_a) turns that into a reviewed case, never a crash.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise VisionError("ROBOFLOW_API_KEY is not set (see SETUP.md).")

    data = Path(image_path).read_bytes()
    key = hashlib.sha256(data + MODEL.encode()).hexdigest()
    cache = _load_cache()
    if key in cache:
        return cache[key] | {"cached": True}

    started = time.perf_counter()
    last_error = ""
    for attempt in range(retries):
        try:
            response = httpx.post(
                API_URL,
                params={"api_key": api_key},
                content=base64.b64encode(data).decode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=60,
            )
        except httpx.HTTPError as error:
            last_error = f"{type(error).__name__}: {error}"
        else:
            if response.status_code == 200:
                break
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code not in (429, 500, 503):
                raise VisionError(last_error)
        if attempt < retries - 1:
            time.sleep(2**attempt)  # 1 s, 2 s
    else:
        raise VisionError(last_error)

    body = response.json()
    try:
        if "top" in body:
            model_label, confidence = body["top"], float(body["confidence"])
        else:
            best = max(body["predictions"], key=lambda p: p["confidence"])
            model_label, confidence = best["class"], float(best["confidence"])
    except (KeyError, IndexError, ValueError) as error:
        raise VisionError(f"Unexpected response from Roboflow: {error}") from error

    result = {
        "label": LABELS.get(model_label, model_label),
        "confidence": round(confidence, 3),
        "model_label": model_label,
        "latency_s": round(time.perf_counter() - started, 3),
    }
    cache = _load_cache()
    cache[key] = result
    _save_cache(cache)
    return result | {"cached": False}
