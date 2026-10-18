# Decision Workbench

What happens when a small decision model sits between a document, a specialist
and a reviewer? This project makes those boundaries visible and testable.

Two workflows run through a bounded LangGraph graph:

- **Contract triage:** classify a fictional clause, attach an exact source quote,
  and send the result to a human review queue.
- **Invoice reconciliation:** calculate ledger findings with Decimal, then ask
  the model which review queue those findings belong in. Currency, duplicates
  and tenant boundaries remain explicit code checks.

The roles are dispatcher → specialist → reviewer. The default dispatcher uses
the task's validated shape, followed by at most two provider calls. An explicit
learned-router variant permits up to three calls for comparison. The specialists
use typed choices, not free-form generated explanations. A deterministic
baseline and a real local neural model implement the same decision interface.

## Run the harness without downloading a model

Python 3.11 or later:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pytest
.venv/bin/python -m workbench.harness --backend rules --out .local/rules
```

The baseline runs the actual LangGraph workflow with transparent rules. It tests
orchestration and policy behavior; it is not labelled AI inference.

Each run writes a summary and per-case traces. The default option-order probe
runs each case again with the same labels in reverse order. Reports distinguish
primary calls, probe calls, neural decision attempts and successful neural decisions.

## Try the local API

```sh
.venv/bin/python -m workbench.api --backend rules --port 8080
```

In another terminal:

```sh
curl http://127.0.0.1:8080/v1/decisions -H 'Content-Type: application/json' \
  --data '{"state":{"role":"legal_analyst","document":{"text":"Keep the information confidential."}},"question":"Which clause topic is present?","options":{"confidentiality":"Protect disclosed information","payment":"Fees or payment deadlines"}}'
```

The rules provider supports the workflow's structured role inputs. Start with `--backend decisiontune
--model-path /absolute/model/path --engine mlx` to use the local model instead.

`POST /v1/decisions` returns a choice, option scores, margin and backend name.
`POST /v1/reviews` accepts a workflow task from a fixture's `input` field.
`GET /healthz` reports readiness; `/openapi.json` describes the request schemas.
The server binds to loopback, serializes provider access, limits body/state sizes,
and disables hosted tracing explicitly. It is a local API, not a Jev wire-protocol
implementation. [API tests](tests/test_api.py) include real TCP requests and
malformed-provider cases.

## Run the local decision model

[DecisionTune 1.0](https://huggingface.co/decision-tune/decisiontune-1.0) is an
Apache-2.0 model and implementation with a typed-decision interface similar in
purpose to [Jev](https://docs.typesafe.ai/introduction). It selects among supplied
options. This project does not claim API or accuracy equivalence with Jev.

The verified local run used macOS arm64 and the MLX backend. Its exact runtime
environment is in `requirements-model-macos.lock`. On Apple silicon:

```sh
python3 -m venv .venv-model
.venv-model/bin/python -m pip install -r requirements-model-macos.lock
HF_HOME="$PWD/.local/model-cache" .venv-model/bin/decision-tune download --yes --backend mlx
```

The download prints a local snapshot path. Pass that path explicitly:

```sh
.venv-model/bin/python -m workbench.harness --backend decisiontune \
  --engine mlx --model-path /absolute/path/printed/by/download --out .local/model
```

The weights occupy about 1.58 GB. They stay outside Git. The adapter checks a
pinned manifest and verifies model files before loading; the harness never
downloads weights implicitly. No API key, paid inference or hosted tracing is
used. The package also offers Torch and ONNX backends, but they are not part of
this repository's local model-verification claim.

## What the evaluation measures

The small [fixture set](examples/cases.jsonl) is authored synthetic development
data. Expected labels are kept outside runtime inputs. It is not a held-out
benchmark or evidence of performance on real contracts or accounts.

The harness records raw routing/category correctness, review flags, contract
violations, call counts, option-order changes, model identity and source/data
hashes. The [checked-in results](reference/README.md) include mistakes and raw
traces for both providers.

A source quote can match the input exactly while its classification is wrong.
Likewise, invoice arithmetic can be correct while a model selects the wrong
queue. Those are separate checks and separate results. Neural scores here are
uncalibrated softmax outputs; the margin is a routing heuristic, not a verified
probability that a decision is correct.

## Design choices

- **Small typed decision model:** inspectable options and local execution, but
  weaker open-ended reasoning and sensitivity to wording/order.
- **Bounded specialist roles:** separate prompts and auditable handoffs, with
  more calls than one classifier. Validated task shapes are dispatched in code;
  use `--routing model` to reproduce the learned-router comparison. The graph is explicitly programmed; it is not
  an unrestricted autonomous assistant.
- **Deterministic evidence tools:** exact arithmetic and literal citations can be
  tested independently, but neither establishes legal interpretation or semantic
  correctness.
- **Review gates:** low margins, invalid provider outputs, missing evidence and
  disagreement are visible outcomes. All legal cases require human review. The
  system never executes payments, provides legal advice or changes an account.

See [the study design](docs/study-design.md) for the method, limits and next
experiments. [AI assistance](AI_ASSISTANCE.md) and third-party attribution are
recorded separately.
