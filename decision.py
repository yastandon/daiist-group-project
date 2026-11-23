"""
Component B: the decision and the customer reply.

Takes component A's output (what the vision model saw in the photo) and the
customer's claim text, and asks Gemini for a decision (approve / deny / escalate)
and a short reply to the customer.

Expected vision input (component A's output), keys used if present:
    {"label": "damaged" | "undamaged" | ..., "damage_type": "wet" | "torn" | ...,
     "severity": "low" | "medium" | "high", "confidence": 0.0-1.0}

Try it without component A: `uv run python try_decision.py`
"""

import json
from pathlib import Path

import gemini

ROOT = Path(__file__).resolve().parent

# Main model first (flash-lite: cheapest, and the free tier is less congested); the backup
# (3.8 flash) is used only if the main one is overloaded or out of quota.
# (gemini-2.5-* is closed to new API keys, so the professor's key could not use it.)
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
MODEL = MODELS[0]
PROMPT_VERSION = "decision_v1"
VERSION = f"{MODEL}+{PROMPT_VERSION}"

DECISIONS = ["approve", "deny", "escalate"]

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "in_scope": {"type": "BOOLEAN"},
        "decision": {"type": "STRING", "enum": DECISIONS},
        "reason": {"type": "STRING"},
        "reply": {"type": "STRING"},
    },
    "required": ["in_scope", "decision", "reason", "reply"],
    "propertyOrdering": ["in_scope", "decision", "reason", "reply"],
}

FALLBACK_REPLY = "Thank you for your claim. A member of our team will review it and get back to you."


def system_prompt(version: str = PROMPT_VERSION) -> str:
    return (ROOT / "prompts" / f"{version}.txt").read_text(encoding="utf-8")


def user_prompt(vision: dict, claim_text: str, claim_value_eur: float | None = None) -> str:
    seen = {k: vision[k] for k in ("label", "damage_type", "severity", "confidence") if k in vision}
    value = f"€{claim_value_eur:.2f}" if claim_value_eur is not None else "unknown"
    return (
        f"Vision model output:\n{json.dumps(seen)}\n\n"
        f"Order value: {value}\n\n"
        f"Customer's claim text:\n<claim>\n{claim_text.strip() or '(no text given)'}\n</claim>"
    )


def decide(vision: dict, claim_text: str, claim_value_eur: float | None = None, use_cache: bool = False) -> dict:
    """Component B. Never raises: if Gemini fails, it escalates and says why in `error`."""
    try:
        result = gemini.generate_json(
            MODELS, system_prompt(), user_prompt(vision, claim_text, claim_value_eur), SCHEMA, use_cache=use_cache
        )
    except gemini.GeminiError as error:
        return {
            "in_scope": True, "decision": "escalate", "reason": "LLM call failed.",
            "reply": FALLBACK_REPLY, "error": str(error),
            "model": None, "version": f"none+{PROMPT_VERSION}",
            "tokens": 0, "input_tokens": 0, "output_tokens": 0, "cost_eur": 0.0, "latency_s": 0.0,
        }
    answer = result["answer"]
    decision = answer.get("decision") if answer.get("decision") in DECISIONS else "escalate"
    return {
        "in_scope": bool(answer.get("in_scope", True)),
        "decision": decision,
        "reason": str(answer.get("reason", "")),
        "reply": str(answer.get("reply", "")).strip() or FALLBACK_REPLY,
        "model": result["model"],
        "version": f"{result['model']}+{PROMPT_VERSION}",
        "tokens": result["tokens"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "cost_eur": result["cost_eur"],
        "latency_s": result["latency_s"],
    }
