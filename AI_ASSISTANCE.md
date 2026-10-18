# Assistance and verification

Code, synthetic fixtures, expected labels, tests and documentation were drafted
with OpenAI Codex assistance. Separate agents reviewed boundaries, tenant
filtering, Decimal arithmetic, citation offsets, budgets, telemetry and API
behavior. This was automated review; owner review and study are not claimed.

Local verification used Python 3.13.1 on macOS arm64. The 135-test suite covers
unit, property, workflow and HTTP API behavior. API tests use rules and explicit
test doubles. They are distinct from the actual DecisionTune/MLX inference runs
recorded under reference/.

The optional neural model is third-party work, loaded from a pinned, verified
local snapshot. This repository did not train it. Its published benchmark numbers
are not this project's results. No paid provider or live account was used, and
hosted tracing is disabled explicitly.

The reference fixture labels were authored during development, with no lawyer,
accountant or external annotator validation. Legal examples are fictional text
taxonomy; finance examples are synthetic document reconciliation. Model errors
and the failed learned-router experiment are retained. These are not production,
held-out research, legal-advice or financial-decision claims.
