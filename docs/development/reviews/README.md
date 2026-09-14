# Engineering review archive

Historical code-review and audit artefacts, kept for provenance. **Status trackers inside the
older files are frequently stale** — read `docs_root/LEDGER.md` (or, if this repo checkout does
not carry `docs_root/`, ask for the living ledger) for what is actually open today, not the
Open/Resolved counts printed in these files.

| File | Date | Scope | Status |
|---|---|---|---|
| `REVIEW_AGENT_PLAN_20260211.md` | 2026-02-11 | Review plan | historical |
| `REVIEW_AGENT_FINDINGS_20260211.md` | 2026-02-11 | Full codebase, 6 findings | resolved (see `REMEDIATION_REPORT_20260211.md`) |
| `REMEDIATION_REPORT_20260211.md` | 2026-02-11 | Remediation of the above | 6/6 resolved |
| `DOCUMENTATION_STATUS_20250211.md` | 2026-02-11 | Docs infra | historical |
| `CODE_REVIEW_20260408.md` | 2026-04-08 | Full codebase, 37 findings | **stale tracker** — several items (DSL/CUDA, `reset_states`) were fixed without updating the file's own table; see ledger `R-001`/`D-011` |
| `BACKEND_AUDIT_20260410.md` | 2026-04-10 | GUI vs shared pipeline, 10 gaps | check ledger for current status |
| `UNITS_GAINS_AUDIT_20260410.md` | 2026-04-10 | Units/gains, 5 findings | 1 fixed at the time (F4); rest superseded by ledger F-005 |
| `PUBLICATION_READINESS_20260914.md` | 2026-09-14 | Publication-readiness audit incl. pressure-simulation parity | **current** — source of ledger findings F-001..F-022 |

For anything not covered above, the living ledger (`docs_root/LEDGER.md`) is the single source of
truth for open findings and decisions.
