Systematically implement all findings from a review document, one at a time, with tests and commits.

Read the review findings file specified in $ARGUMENTS (e.g., `reviews/REVIEW_AGENT_FINDINGS_20260408.md`). If no file is given, look for the most recent file matching `reviews/REVIEW_AGENT_FINDINGS_*.md`.

Also read `.github/copilot-instructions.md` for project standards before touching any code.

## Workflow

### Phase 1 — Plan

1. Read the findings file completely.
2. Build an ordered task list covering every finding, sorted: Critical → Major → Medium → Low.
3. Show the list to the user and confirm before starting.

### Phase 2 — Implement (one finding at a time)

For each finding:

1. **Read the affected file** at the exact location cited.
2. **Implement the fix** exactly as specified in the finding's remediation steps.
3. **Write or update a test** that would have caught this bug. Put it in `tests/unit/test_<module>.py`. Reference the finding ID in the test docstring:
   ```python
   def test_<description>():
       """Regression for ReviewFinding#<ID>. Verifies <what>."""
   ```
4. **Run the test** to confirm it passes: `pytest tests/unit/test_<module>.py::test_<name> -v`
5. **Run the full suite** to confirm no regressions: `pytest tests/ -x -q`
6. **Commit** with a conventional commit message referencing the finding:
   ```
   fix(<scope>): <description> (resolves ReviewFinding#<ID>)
   ```
7. **Update the status** in the findings file: mark the finding `✅ RESOLVED — <commit hash>`.
8. Move to the next finding.

### Blockers

If a fix cannot be implemented (requires a breaking change, is blocked by another issue, or needs architectural decisions):
- Mark the finding `⛔ BLOCKED — <reason>` in the findings file.
- Skip to the next finding.
- Report all blockers in the final summary.

Never implement a breaking change without explicit user approval. Propose a deprecation path first.

### Phase 3 — Report

After all findings are processed, write `reviews/REMEDIATION_REPORT_<YYYYMMDD>.md`:

```markdown
# Remediation Report — <date>
**Source:** <findings file>

## Summary
- Total findings: X
- Resolved: A (Y%)
- Blocked: B
- Skipped: C

## Resolved
### <ID>: <title> ✅
Commit: <hash>
Test: <test file::test name>

## Blocked
### <ID>: <title> ⛔
Reason: <why blocked>
Requested action: <what is needed>

## Coverage Delta
Before: X% — After: Y%
```

## Standards to Enforce

- Type hints on every modified function signature
- Google Style docstrings with tensor shapes and units on every modified public API
- No bare `except Exception` — use specific types
- No `assert` for validation — use `raise ValueError` with values in the message
- Conventional commit messages on every commit
- One commit per finding — do not batch multiple findings into one commit
