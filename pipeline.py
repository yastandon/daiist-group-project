"""
Your product's logic: the two components and how one feeds the other.

Everything here is a PLACEHOLDER so the template runs end to end without any
key: component A is a keyword "classifier" and component B a canned "writer".
Replace both with your real components (or import them from your own modules)
but keep what `run()` returns, because app.py, evaluate.py, main.py and
check.py all read it:

    run(raw_input) -> {
        "input_summary": str,          # short, no personal data: it is logged
        "features": {name: number},    # what the drift check watches
        "outputs": {component: ...},   # each component's output, JSON-serialisable
        "versions": {component: str},  # model and prompt versions
        "review": bool,                # does this case go to a person?
        "tokens": int,                 # tokens spent on this request (0 if none)
        "cost_eur": float,             # cost of this request in euros
    }

`raw_input` is what the App tab's input gives you: text, or a file path if you
switch the input to gr.Image / gr.File with type="filepath". The `input` column
of eval/cases.csv uses the same convention.
"""

from pathlib import Path

import decision

VERSIONS = {
    "component_a": "placeholder-keywords-v0",
    "component_b": decision.VERSION,  # overwritten per request with the model that answered
}

# Review rule: a case goes to a person when ANY of these holds.
# PROPOSED values, to be agreed by the team and justified in REPORT.md with the evaluation numbers.
REVIEW_BELOW_CONFIDENCE = 0.7   # component A is unsure (same 0.7 as in prompts/decision_v1.txt)
REVIEW_ABOVE_VALUE_EUR = 100.0  # a wrong approval above this costs more than an agent's review
# ...or component B chose "escalate", said the request is out of scope, or the LLM call failed.

_NEGATIVE = {"broken", "late", "missing", "wrong", "refund", "bad", "problem", "error"}
_POSITIVE = {"thanks", "great", "perfect", "fine", "good", "works"}


def read_input(raw_input: str) -> str:
    """Text as it is; for a file path, the file's text. Replace for images."""
    path = Path(str(raw_input))
    if len(str(raw_input)) < 300 and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return str(raw_input)


def input_features(text: str) -> dict[str, float]:
    """Numbers that describe the input, for the drift check (lengths, brightness, language...)."""
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    return {
        "n_words": float(len(words)),
        "share_uppercase": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
    }


def component_a(text: str) -> dict:
    """PLACEHOLDER first component: label the input and say how sure it is."""
    words = {w.strip(".,!?;:").lower() for w in text.split()}
    negative, positive = len(words & _NEGATIVE), len(words & _POSITIVE)
    if negative == positive:
        return {"label": "unclear", "confidence": 0.5}
    label = "complaint" if negative > positive else "no_issue"
    confidence = 0.5 + 0.5 * abs(negative - positive) / (negative + positive)
    return {"label": label, "confidence": round(confidence, 3)}


def component_b(claim_text: str, a: dict, claim_value_eur: float | None = None) -> dict:
    """Component B (decision.py): Gemini drafts approve / deny / escalate and the customer reply from A's output."""
    return decision.decide(a, claim_text, claim_value_eur)


def needs_review(a: dict, b: dict, claim_value_eur: float | None = None) -> bool:
    return (
        float(a.get("confidence", 0.0)) < REVIEW_BELOW_CONFIDENCE
        or (claim_value_eur is not None and claim_value_eur > REVIEW_ABOVE_VALUE_EUR)
        or b["decision"] == "escalate"
        or not b["in_scope"]
        or "error" in b
    )


def run(raw_input: str, claim_text: str | None = None, claim_value_eur: float | None = None) -> dict:
    # TODO (with component A): raw_input becomes the photo path, and claim_text the customer's text.
    # Until then the template's text input doubles as the claim text.
    text = read_input(raw_input)
    claim = text if claim_text is None else claim_text
    a = component_a(text)
    b = component_b(claim, a, claim_value_eur)
    usage = {k: b.pop(k) for k in ("tokens", "input_tokens", "output_tokens", "cost_eur", "latency_s")}
    features = input_features(text) | {"confidence": a["confidence"]}
    return {
        "input_summary": text[:120],
        "features": features,
        "outputs": {"component_a": a, "component_b": b},
        "versions": VERSIONS | {"component_b": b["version"]},
        "review": needs_review(a, b, claim_value_eur),
        "tokens": usage["tokens"],
        "cost_eur": usage["cost_eur"],
    }
