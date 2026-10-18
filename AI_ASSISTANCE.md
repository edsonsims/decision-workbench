# Assistance and verification

Code, synthetic fixtures, synthetic expected labels, tests and documentation were drafted
with OpenAI Codex assistance. Separate agents reviewed boundaries, tenant
filtering, Decimal arithmetic, citation offsets, budgets, telemetry and API
behavior. This was automated review; owner review and study are not claimed.

Local verification used Python 3.13.1 on macOS arm64. The 276-test suite covers
unit, property, workflow, HTTP API, public-data preparation, metrics and adapter
behavior. API tests use rules and explicit
test doubles. They are distinct from the actual DecisionTune/MLX inference runs
recorded under reference/.

The neural models are third-party work, loaded from pinned, verified
local snapshots. This repository did not train them. Their published benchmark numbers
are not this project's results. No paid provider or live account was used, and
hosted tracing is disabled explicitly.

The reference fixture labels were authored during development, with no lawyer,
accountant or external annotator validation. Legal examples are fictional text
taxonomy; finance examples are synthetic document reconciliation. Model errors
and the failed learned-router experiment are retained.

The later public-data comparison uses the original CUAD and BANKING77 test
annotations. Codex selected task categories, wrote descriptions and implemented
the deterministic preparation before model inference; it did not author those
source annotations. The frozen public slice, known training overlap, unknown
contamination and transformed-task limits are recorded separately in
docs/public-evaluation.md. These are not production, contamination-free research,
legal-advice or financial-decision claims. Actual model runs are separate from
the offline tests, which include explicit test doubles.

The README visual refresh and original SVG illustrations were also drafted with Codex. The artwork is illustrative, not a screenshot or live system. Runnable README examples and their selected output fields were checked against local CLI/HTTP runs; they do not imply owner review, production use or new model-quality evidence.
