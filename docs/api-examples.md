# Local API examples

Start from the repository root after installing `requirements.lock`:

```sh
.venv/bin/python -m workbench.api --backend rules --port 8080
```

The examples below were exercised against an actual local HTTP server. They use
fictional input and the **rules baseline**, with no model download or inference.

## One typed choice

```sh
curl -sS http://127.0.0.1:8080/v1/decisions \
  -H 'Content-Type: application/json' \
  --data '{"state":{"role":"legal_analyst","document":{"text":"Keep the information confidential."}},"question":"Which clause topic is present?","options":{"confidentiality":"Protect disclosed information","payment":"Fees or payment deadlines"}}'
```

```json
{
  "choice": "confidentiality",
  "probabilities": {"confidentiality": 1.0, "payment": 0.0},
  "margin": 1.0,
  "backend": "rules-baseline"
}
```

Rules mode interprets the workflow's role-specific structured states. Its one-hot
scores encode a rule result, not model confidence. The question/options interface
also supports a local DecisionTune provider, whose scores have a different meaning.

## A clause through the full review graph

```sh
curl -sS http://127.0.0.1:8080/v1/reviews \
  -H 'Content-Type: application/json' \
  --data '{"id":"readme-example","tenantId":"demo-team","document":{"id":"clause-1","text":"Keep the information confidential."}}'
```

Selected fields from the response; the complete response also contains evidence,
review output, gate reasons and the per-role trace:

```json
{
  "route": "legal",
  "routingMode": "shape",
  "analysis": {
    "category": "confidentiality",
    "evidence": {
      "documentId": "clause-1",
      "start": 0,
      "end": 34,
      "quote": "Keep the information confidential.",
      "offsetUnit": "unicode_code_points"
    }
  },
  "requiresHumanReview": true,
  "paymentApproved": false,
  "modelCalls": 2
}
```

`modelCalls` counts provider invocations, including rules invocations; it does not
imply neural inference. The dispatcher uses task shape and consumes no provider
call. Legal results always require a human, even with an exact source quote.

## Use a verified local model

With the pinned Apple-silicon environment and downloaded snapshot described in
the [model reproduction guide](../reference/public-v1/README.md#run-the-models-again-on-apple-silicon):

```sh
.local/public-eval-venv/bin/python -m workbench.api --backend decisiontune \
  --model-path "$DECISIONTUNE_MODEL" --engine mlx --port 8080
```

Run that command in the same terminal as the reproduction guide, which creates
the environment and sets `DECISIONTUNE_MODEL`. This starts the same HTTP adapter
with actual local model inference. It is distinct from the
rules outputs recorded above. The server does not download weights implicitly.

## Boundaries to inspect

- `GET /healthz` reports the selected backend; `GET /openapi.json` exposes schemas.
- Unknown fields, invalid JSON, duplicate JSON keys and invalid types are rejected.
- The full request body is bounded to 256 KiB; decision state to 65,536 serialized
  UTF-8 bytes. Oversized inputs are rejected instead of silently truncated.
- Provider exceptions are sanitized; workflow gates and raw choices remain in
  the response for inspection.
- Caller-supplied tenant IDs are workflow inputs, not an authentication system.
  This loopback service has no deployment or multi-user security claim.

See [API tests](../tests/test_api.py) for real TCP and boundary cases and
[the study design](study-design.md) for what the workflow does and does not measure.
