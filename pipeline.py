"""
Damage claim triage: the two components and how one feeds the other.

Component A (vision.py): the Roboflow hosted model damaged-box-auh4b/1 labels
the parcel photo damaged / undamaged with a confidence.
Component B (decision.py): Gemini drafts approve / deny / escalate and the
customer reply from A's output and the claim text.

`run()` keeps the template's contract, because app.py, evaluate.py, main.py and
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

`raw_input` is one string (the App tab's input and the `input` column of
eval/cases.csv use the same convention):

    eval/inputs/c01.jpg                      a photo, no claim text
    eval/inputs/c01.jpg | arrived crushed    a photo plus the claim text
    the box arrived crushed                  text only: no photo, so it goes to review
"""

import decision
import vision

VERSIONS = {
    "component_a": vision.VERSION,
    "component_b": decision.VERSION,  # overwritten per request with the model that answered
}

# Review rule: a case goes to a person when ANY of these holds.
# PROPOSED values, to be agreed by the team and justified in REPORT.md with the evaluation numbers.
REVIEW_BELOW_CONFIDENCE = 0.7   # component A is unsure (same 0.7 as in prompts/decision_v1.txt)
REVIEW_ABOVE_VALUE_EUR = 100.0  # a wrong approval above this costs more than an agent's review
# ...or there is no photo, the vision call failed, component B chose "escalate",
# said the request is out of scope, or the LLM call failed.


def parse_input(raw_input: str) -> tuple[str | None, str]:
    """One input string -> (photo path or None, claim text). See the module docstring."""
    raw = str(raw_input).strip()
    left, sep, right = raw.partition("|")
    if sep and vision.is_image(left.strip()):
        return left.strip(), right.strip()
    if vision.is_image(raw):
        return raw, ""
    return None, raw


def component_a(image_path: str | None) -> dict:
    """Component A (vision.py). Never raises: no photo or a failed call become a reviewed case."""
    if image_path is None:
        return {"label": "no_photo", "confidence": 0.0}
    try:
        result = vision.classify(image_path)
    except vision.VisionError as error:
        return {"label": "vision_error", "confidence": 0.0, "error": str(error)}
    return {k: result[k] for k in ("label", "confidence", "model_label", "latency_s")}


def component_b(claim_text: str, a: dict, claim_value_eur: float | None = None) -> dict:
    """Component B (decision.py): Gemini drafts approve / deny / escalate and the customer reply from A's output.

    use_cache: identical photo + text give the identical answer anyway (temperature 0),
    and `main.py reference` and the evaluation re-run the same inputs often.
    """
    return decision.decide(a, claim_text, claim_value_eur, use_cache=True)


def input_features(image_path: str | None, claim_text: str) -> dict[str, float]:
    """Numbers that describe the input, for the drift check. Same keys for every request.

    Only photo features (and A's confidence, added in run()): the reference set is
    photos without claim text, so a text feature like word count would differ from
    real cases by construction and make the drift alert fire on normal traffic.
    """
    photo = vision.features(image_path) if image_path else {"brightness": 0.0, "megapixels": 0.0}
    return {"has_photo": 1.0 if image_path else 0.0, **photo}


def needs_review(a: dict, b: dict, claim_value_eur: float | None = None) -> bool:
    return (
        float(a.get("confidence", 0.0)) < REVIEW_BELOW_CONFIDENCE
        or (claim_value_eur is not None and claim_value_eur > REVIEW_ABOVE_VALUE_EUR)
        or b["decision"] == "escalate"
        or not b["in_scope"]
        or "error" in b
    )


def run(raw_input: str, claim_text: str | None = None, claim_value_eur: float | None = None) -> dict:
    image_path, parsed_text = parse_input(raw_input)
    claim = parsed_text if claim_text is None else claim_text
    a = component_a(image_path)
    b = component_b(claim, a, claim_value_eur)
    usage = {k: b.pop(k) for k in ("tokens", "input_tokens", "output_tokens", "cost_eur", "latency_s")}
    features = input_features(image_path, claim) | {"confidence": a["confidence"]}
    photo_name = image_path.rsplit("/", 1)[-1] if image_path else "no photo"
    return {
        "input_summary": f"[{photo_name}] {claim}"[:120],
        "features": features,
        "outputs": {"component_a": a, "component_b": b},
        "versions": VERSIONS | {"component_b": b["version"]},
        "review": needs_review(a, b, claim_value_eur),
        "tokens": usage["tokens"],
        "cost_eur": usage["cost_eur"],
    }
