# Study design

## Question

Can a local typed-decision model fit into a bounded document-review workflow
without letting model output silently become business authority?

This milestone compares a transparent rule baseline with DecisionTune 1.0 on
authored microcases. It studies decisions, evidence and failure behavior. It does
not test general legal knowledge, independent financial reasoning, real customer
outcomes or model superiority.

## Method

1. Load a JSONL row containing `input` and separate `expected` labels.
2. Pass only `input` into the LangGraph workflow.
3. Dispatch validated task shapes in code, then run specialist and reviewer roles
   under a two-call path. The explicit `--routing model` comparison instead uses
   a learned router and permits up to three calls.
4. Keep literal source verification and Decimal arithmetic in code. Never give
   the finance classifier the deterministic policy-status label as an input.
5. Record the raw model choices before policy gates. A gate cannot turn a wrong
   choice into a correct classification score.
6. Repeat with reversed option insertion order, preserving labels/descriptions.
7. Save summary and trace records, runtime versions, pinned model provenance and
   hashes of every workbench source file and the dataset.

The model is loaded once per harness invocation. Recorded elapsed time includes
the workflow after loading, not model download/startup; no speed claim is made.
The option-order probe adds decision calls, which are included in total counts.

## Evidence boundaries

The legal role performs clause taxonomy on fictional text, not enforceability or
jurisdiction analysis. Quoted offsets use Python string code points, not UTF-8
byte offsets. Literal citation integrity is distinct from semantic support.

The finance role interprets precomputed counts, amounts and currency checks.
This separates a model's routing choice from numerical authority. Tenant-scoped
fixtures let the tests verify filtering before data reaches a model. This is not
an authenticated financial integration.

The rule baseline and neural model share the graph and supplied options. Shape
routing accuracy measures the programmed dispatcher, not model intelligence.
The learned-router trial is separate: if it selects `other`, the specialists are
not reached. A missing downstream category counts as an end-to-end failure, not
as an observed wrong answer by a specialist that never ran.

The providers'
scores must remain separately labelled. Fixture development and prompt design
occurred in the same study; the fixture set is not held out. Any later changes
require a new result artifact rather than silently replacing an old claim.

## Diverse follow-up studies

| Domain | Next concrete question | State |
| --- | --- | --- |
| Legal documents | Clause triage with exact citations and a mandatory review gate | Implemented baseline |
| Finance operations | Invoice/ledger discrepancies with deterministic arithmetic and model routing | Implemented baseline |
| Reliability operations | Does an incident router abstain when a runbook lacks evidence? | Planned |
| Research assistance | Can a claim reviewer distinguish an exact quote from supporting evidence? | Planned |
| Procurement | Can a typed policy gate flag order exceptions without approving a purchase? | Planned |

The next useful experiment is to freeze a new evaluation split before changing
prompts, then compare calibration, option-order stability and review coverage.
Another is a provider adapter for an independently served decision model or Jev;
no such provider was called in this milestone.

## Sources

- [DecisionTune model card, limits and license](https://huggingface.co/decision-tune/decisiontune-1.0)
- [DecisionTune implementation](https://github.com/decision-tune/decision-tune)
- [TypeSafe's Jev typed-decision interface](https://docs.typesafe.ai/introduction)
- [LangGraph graph, state and conditional edges](https://docs.langchain.com/oss/python/langgraph/graph-api)
