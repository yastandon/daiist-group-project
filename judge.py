"""
LLM judge for component B's free-text output (the decision and the customer reply).

    judge.grade(vision, claim_text, output)  -> {"verdict": "good"|"bad", criteria..., "comment", tokens, cost}
    judge.agreement(judge_verdicts, human_labels) -> agreement %, Cohen's kappa, confusion counts

The judge must be checked against the team's own labels (eval/human_labels.csv,
label = good / bad) before its numbers are used in REPORT.md.
"""

import json
from pathlib import Path

import gemini

ROOT = Path(__file__).resolve().parent

# The judge uses a different (stronger) model first than the one it grades, so it is
# not simply agreeing with itself; the backup is used only when the first is unavailable.
MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
PROMPT_VERSION = "judge_v1"

CRITERIA = ["decision_ok", "reply_matches_decision", "no_invented_promises", "tone_ok", "no_personal_data"]

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        **{c: {"type": "BOOLEAN"} for c in CRITERIA},
        "verdict": {"type": "STRING", "enum": ["good", "bad"]},
        "comment": {"type": "STRING"},
    },
    "required": [*CRITERIA, "verdict", "comment"],
    "propertyOrdering": [*CRITERIA, "verdict", "comment"],
}


def grade(vision: dict, claim_text: str, output: dict, use_cache: bool = True) -> dict:
    """Grade one component-B output. Raises gemini.GeminiError if no model answers."""
    system = (ROOT / "prompts" / f"{PROMPT_VERSION}.txt").read_text(encoding="utf-8")
    shown = {k: output.get(k) for k in ("in_scope", "decision", "reason", "reply")}
    seen = {k: vision[k] for k in ("label", "damage_type", "severity", "confidence") if k in vision}
    prompt = (
        f"Vision model output:\n{json.dumps(seen)}\n\n"
        f"Customer's claim text:\n<claim>\n{claim_text.strip()}\n</claim>\n\n"
        f"Assistant's output:\n{json.dumps(shown, ensure_ascii=False)}"
    )
    result = gemini.generate_json(MODELS, system, prompt, SCHEMA, use_cache=use_cache)
    answer = result["answer"]
    # The verdict follows the criteria, whatever the model wrote in "verdict".
    verdict = "good" if all(answer.get(c) is True for c in CRITERIA) else "bad"
    return {
        **{c: answer.get(c) for c in CRITERIA},
        "verdict": verdict,
        "comment": answer.get("comment", ""),
        "judge_version": f"{result['model']}+{PROMPT_VERSION}",
        "tokens": result["tokens"],
        "cost_eur": result["cost_eur"],
    }


def agreement(judge_verdicts: dict[str, str], human_labels: dict[str, str]) -> dict:
    """Compare judge verdicts with the team's labels, by case id (labels: good / bad)."""
    ids = sorted(set(judge_verdicts) & set(human_labels))
    if not ids:
        return {"n": 0}
    pairs = [(human_labels[i].strip().lower(), judge_verdicts[i].strip().lower()) for i in ids]
    n = len(pairs)
    agree = sum(h == j for h, j in pairs) / n
    # Cohen's kappa: agreement beyond what two raters would reach by chance.
    p_h = sum(h == "good" for h, _ in pairs) / n
    p_j = sum(j == "good" for _, j in pairs) / n
    chance = p_h * p_j + (1 - p_h) * (1 - p_j)
    kappa = (agree - chance) / (1 - chance) if chance < 1 else 1.0
    confusion = {f"human_{h}/judge_{j}": sum(p == (h, j) for p in pairs) for h in ("good", "bad") for j in ("good", "bad")}
    disagreements = [i for i, (h, j) in zip(ids, pairs) if h != j]
    return {"n": n, "agreement": round(agree, 3), "kappa": round(kappa, 3), **confusion, "disagreements": disagreements}
