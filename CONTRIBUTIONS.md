# Contributions

*Fill this in as you go, not on the last day. It is checked against the commits and against the peer assessment. A first name and a GitHub username are enough: this repo is public.*

## Members

| First name | GitHub username | Main areas |
|---|---|---|
| | | |
| Manu | Manucallej | LLM component (decision + reply), LLM judge, review rule, cost tracking |
| | | |
| | | |
| | | |

## Log

*One line per piece of work: the date, who did it (more than one name is fine), and what. Code done in pair programming counts for both, even if only one of you committed it.*

| Date | Who | What |
|---|---|---|
| 2026-11-23 | Manu | Took over the LLM component and judge from Lucas (agreed with the team). |
| 2026-11-23 | Manu | Component B: Gemini decision (approve / deny / escalate) and customer reply, prompt `decision_v1`, with token and € cost per request (`decision.py`, `gemini.py`, `prompts/decision_v1.txt`). |
| 2026-11-23 | Manu | LLM judge with a 5-criteria rubric and agreement with human labels (`judge.py`, `prompts/judge_v1.txt`). |
| 2026-11-23 | Manu | Wired component B and the proposed review rule into `pipeline.py`; switched from gemini-2.5 (closed to new keys) to gemini-3.5-flash-lite with gemini-3.8-flash as backup. |
