# Group Assignment Report

- **Team** (Blackboard group name):
- **Section**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A]
- **App URL** (on Render):
- **Final commit** (tag `final`):

*At most about 3,000 words, tables excluded. Members and who did what go in CONTRIBUTIONS.md.*

## Business case

**Who uses it.** The claims team of a mid-size online retailer (home goods and small electronics). When a customer reports that an order arrived damaged, they submit a photo of the item and a short description through the returns portal. A claims agent then has to decide what to do with each case.

**The decision it supports.** For every damage claim: approve (refund or replacement), deny (no visible damage or out of policy), or escalate to a senior agent. The product drafts this decision and the customer reply; the agent stays in the loop for the cases the product is not confident about.

**Baseline (what happens today).** An agent opens every claim manually: looks at the photo, checks the description against the order, decides, and writes a reply. We assume 4–6 minutes per claim end to end. At a fully loaded cost of €25/hour, that is **€1.70–2.50 per claim**, and customers wait 24–48 hours for an answer because claims queue behind other support work.

**Success metric.** Three numbers, measured on `eval/cases.csv` against the baseline:

1. **Decision agreement**: on the claims the product resolves automatically, its approve/deny decision matches the human decision in ≥ 90% of cases.
2. **Automation rate**: ≥ 50% of claims are resolved without a human (the rest go to review by the confidence/value rule).
3. **Cost and speed**: < €0.05 and < 30 seconds per claim, versus ~€2 and 24–48 hours today.

**Value in euros.** Assumptions: 100 claims/day, 22 working days/month, 5 minutes of agent time saved per auto-resolved claim, €25/hour loaded agent cost.

- Agent time saved: 100 × 50% × 5 min = ~4.2 hours/day ≈ **€2,300/month**.
- API and hosting cost at that volume: roughly €15–30/month (Gemini calls at ~€0.01/claim plus hosting), i.e. ~1% of the saving.
- Not quantified but real: faster answers on damage claims measurably reduce chargebacks and repeat contacts; wrongly approved claims cost on average ~€35 (the refund), which is why claims above a value threshold always go to a person regardless of model confidence.

The go/no-go decision in the last section returns to these numbers with the measured results.

## Architecture

*The two components and how one feeds the other: a diagram plus a short walk-through of one request from input to output. Which models or APIs, and why those. What you reused from assignments 1–3, and whose it was.*

## Data

*Where each component's data comes from, with links, licences and sizes. Confirm that there is no personal data in your inputs or logs.*

## Evaluation

### Components

*Each component on its own: the metric, the result, the test data and its size.*

### End to end

*How you wrote `eval/cases.csv` (counts by kind, two examples). Results against the baseline. Where the output is free text: the judge's rubric and your agreement with it on your own labels.*

### Cost and latency

*Per request: tokens, euros and seconds, for each component and in total. What it would cost per month at the volume in your business case.*

## Operations

*What you log. The quality signal and the drift check you monitor, the reference set, and the alert threshold with its justification. The drift demo: what you shifted, and the result with the alert firing (screenshot or numbers). The rate of false alarms on normal inputs.*

## Failures

*Failed end-to-end cases grouped by cause (which component, or the hand-off between them), with counts, and what you would fix first.*

## Decision

*Which cases go to a person: the rule, and how many cases a day that would be. Go / no-go / go with conditions: the conditions, and the evidence behind them.*

## Bonus: reinforcement learning

*Delete this section if you did not do it. The loop (feedback → reward → update → behaviour), the reward and what it could reward by accident, the exploration strategy and its cost, the evidence from real logs (and which part was simulated), the risks.*

## How to run it

*Anything beyond SETUP.md that the professor needs to run it: which keys, any download step, how long the first run takes.*

## AI use

*What you used and how, or "none".*
