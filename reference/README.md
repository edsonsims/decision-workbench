# Recorded development runs

Twenty authored synthetic cases: nine fictional clauses, ten invoice/ledger
examples and one unsupported task. Nineteen cases have a category label. These
are development fixtures, not a held-out benchmark or real professional advice.

## Default: dispatch task shapes in code

| Provider | Clause categories | Invoice queues | Categories overall | Option-order category changes |
| --- | --- | --- | --- | --- |
| Transparent rules | 7/9 | 10/10 | 17/19 | 0/20 |
| DecisionTune 1.0 / MLX | 6/9 | 2/10 | 8/19 | 1/20 |

All 20 tasks were dispatched correctly by the **deterministic** shape router.
That result is not neural routing accuracy. Each provider reached 19 specialist
and 19 reviewer decisions in the primary run; reversing options repeated those
38 calls. The neural run performed 76 successful model decisions; the rules run
performed zero neural calls. Both runs had zero tested workflow-contract failures
and flagged 18 cases for review. Every legal case required review.

The local model lost to the rule baseline on these tasks. In particular, it was
weak at applying the explicit invoice routing policy to supplied arithmetic
findings. The arithmetic itself was done by Python. A human-review gate can
contain a bad choice without making that choice accurate.

- [Rules summary](rules/summary.json) and [full traces](rules/traces.jsonl)
- [Local model summary](decisiontune-mlx/summary.json) and [full traces](decisiontune-mlx/traces.jsonl)
- [Separate actual-model TCP API smoke](api-smoke.json): one direct decision and
  one two-call review, recorded separately from the fixture-call totals above.

Each trace retains the raw choice, option scores, gate reasons and reversed-order
result. Summary provenance records data/source hashes and tested runtime/model
identity. Neural scores are uncalibrated; no calibration or broad accuracy claim
is inferred from this small set.

## Separate learned-router experiment

The first design asked the model to route a task whose JSON shape already
identified its domain. It chose `other` on all 20 cases: one correct route, with
no specialists reached. Thus zero of 19 category-labelled tasks completed with
the correct category. **Specialist accuracy was not measured by that run.**

The explicit `--routing model` mode preserves that design for reproduction. The
current frozen-code replay records 20 router decisions in the primary run and 20
more in the option-order probe, all from the actual local model:

- [Learned-router summary](model-router/summary.json)
- [Learned-router traces](model-router/traces.jsonl)

This finding motivated deterministic dispatch before evaluating the specialist
roles. The cases, classifier prompts, model and margin threshold were not changed
to improve the classifier scores. All results remain development observations.
