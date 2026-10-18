# Working in Decision Workbench

- Build a small, tested decision workflow, not a collection of agent names. Keep explicit state, distinct role prompts, typed decisions, bounded tool calls and an auditable trace.
- Preserve the difference between deterministic policy checks, replay fixtures and actual model inference. Never report a test double as model quality or copy another model's benchmark as our result.
- Use fictional contract clauses and synthetic invoice/ledger data. Do not import private contracts, financial accounts, employer data or personal records.
- Legal examples classify text for a human review queue. Financial examples reconcile documents and flag discrepancies; they do not approve payments, lending, investments or legal outcomes.
- Prefer permissively licensed local decision models. Keep weights outside Git, pin tested model revision and package versions, record provenance and attribution. A compatible API is not evidence of equivalent accuracy or licensing.
- No paid API calls, external tracing, credential changes or infrastructure deployment without specific owner authorization. Disable telemetry and use local traces by default.
- Run offline unit/property tests, the fixture evaluation harness and any affected model adapter checks. Report actual runtime, case counts, failures and coverage; keep expected labels outside runtime workflow input.
- Preserve owner edits and true dates. Disclose AI assistance and leave owner study/review unclaimed until reported.
