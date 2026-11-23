"""
Small helper for calling Gemini and getting JSON back, with token and cost counting.

Used by decision.py (component B) and judge.py (the LLM judge). Calls the REST API
with httpx (already installed with gradio), so there is no extra dependency for Render.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
CACHE_FILE = ROOT / "logs" / "llm_cache.json"  # logs/ is git-ignored

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# List prices in USD per 1M tokens (paid tier; output includes thinking tokens).
# Free-tier calls cost nothing, but we report what the same traffic would cost on the paid tier.
# Looked up 2026-11-23 from secondary sources; CHECK BEFORE REPORTING: https://ai.google.dev/pricing
# gemini-3.8-flash is on an introductory price until 2026-12-31 (reported to double from 2027-01-01).
PRICES_USD_PER_M = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
}
USD_TO_EUR = 0.86  # CHECK: exchange rate used for the report


class GeminiError(RuntimeError):
    pass


def cost_eur(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICES_USD_PER_M.get(model)
    if price is None:
        return 0.0
    usd = input_tokens / 1e6 * price["input"] + output_tokens / 1e6 * price["output"]
    return round(usd * USD_TO_EUR, 6)


def _cache_key(model: str, system: str, prompt: str, schema: dict) -> str:
    raw = json.dumps([model, system, prompt, schema], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache() -> dict:
    if CACHE_FILE.is_file():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=1), encoding="utf-8")


def generate_json(
    models: str | list[str],
    system: str,
    prompt: str,
    schema: dict,
    temperature: float = 0.0,
    use_cache: bool = False,
    retries: int = 3,
) -> dict:
    """Call Gemini and return {"answer", "model", "input_tokens", "output_tokens", "tokens", "cost_eur", "latency_s", "cached"}.

    `models` can be a list: if the first model is overloaded or out of quota after the
    retries, the next one is tried. The model that actually answered is in "model".

    `use_cache=True` (for the evaluation) reuses earlier answers to the exact same call,
    so re-running the evaluation does not burn the free daily quota.
    """
    models = [models] if isinstance(models, str) else list(models)
    key = _cache_key(models[0], system, prompt, schema)
    if use_cache:
        cache = _load_cache()
        if key in cache:
            return cache[key] | {"cached": True}

    started = time.perf_counter()
    errors = []
    for model in models:
        try:
            result = _call(model, system, prompt, schema, temperature, retries)
        except GeminiError as error:
            errors.append(f"{model}: {error}")
            continue
        result["latency_s"] = round(time.perf_counter() - started, 3)
        if use_cache:
            cache = _load_cache()
            cache[key] = result
            _save_cache(cache)
        return result | {"cached": False}
    raise GeminiError(" | ".join(errors))


def _call(model: str, system: str, prompt: str, schema: dict, temperature: float, retries: int) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiError("GEMINI_API_KEY is not set (see SETUP.md).")

    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseSchema": schema,
            # Little "thinking": the task is short, and thinking tokens are billed as output.
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }

    last_error = ""
    for attempt in range(retries):
        try:
            response = httpx.post(API_URL.format(model=model), params={"key": api_key}, json=body, timeout=60)
        except httpx.HTTPError as error:
            last_error = f"{type(error).__name__}: {error}"
        else:
            if response.status_code == 200:
                break
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code not in (429, 500, 503):
                raise GeminiError(last_error)
        if attempt < retries - 1:
            time.sleep(2 ** attempt)  # 1 s, 2 s
    else:
        raise GeminiError(last_error)

    data = response.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        answer = json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError) as error:
        raise GeminiError(f"Unexpected response from Gemini: {error}") from error

    usage = data.get("usageMetadata", {})
    input_tokens = int(usage.get("promptTokenCount", 0))
    output_tokens = int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0))
    return {
        "answer": answer,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens": input_tokens + output_tokens,
        "cost_eur": cost_eur(model, input_tokens, output_tokens),
    }
