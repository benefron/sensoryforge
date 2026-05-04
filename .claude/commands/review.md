Perform a rigorous, comprehensive review of the SensoryForge codebase (or the scope specified in $ARGUMENTS).

You are operating as a principal-level engineer with expertise in computational neuroscience and PyTorch. Read `.github/copilot-instructions.md` first to ground yourself in the project standards.

## Scope

If $ARGUMENTS is empty, review the full codebase. Otherwise treat $ARGUMENTS as the scope (e.g., "filters only", "sensoryforge/neurons/adex.py", "since last commit").

## Review Checklist

For every file in scope, check:

**Scientific Accuracy**
- [ ] Equations match cited papers (cite must appear in docstring)
- [ ] Parameters are in biological ranges
- [ ] Units are physically correct (time ms/s, voltage mV, current mA, distance mm)

**PyTorch Conventions**
- [ ] No hand-rolled loops over neurons or spatial dimensions — must be vectorised
- [ ] Device specified explicitly on every tensor creation
- [ ] Batch dimension first: `[batch, time, ...]`
- [ ] No undocumented in-place operations in forward passes

**Documentation Quality**
- [ ] All public APIs have Google Style docstrings (Args/Returns/Raises/Example)
- [ ] Tensor shapes annotated in Args and Returns: `[batch, time, num_neurons]`
- [ ] Physical units in every Args/Returns line that carries a quantity
- [ ] Working `Example:` block in every public method

**Architecture Compliance**
- [ ] Inherits from correct base class (BaseFilter, BaseNeuron, BaseStimulus, etc.)
- [ ] `from_config()` classmethod present and functional
- [ ] `to_dict()` method present
- [ ] `reset_state()` (singular) implemented where applicable
- [ ] Registered in `sensoryforge/register_components.py`
- [ ] No bare `except Exception` — specific exception types only
- [ ] No `assert` for input validation — use `raise ValueError` with a message containing actual values

**Test Coverage**
- [ ] All public APIs have tests
- [ ] Edge cases tested (empty input, extreme values, wrong shape)
- [ ] Config round-trip test (`from_config(obj.to_dict())` produces equal object)

## Output Format

Write findings to `reviews/REVIEW_AGENT_FINDINGS_<YYYYMMDD>.md` using this structure:

```markdown
# Review Findings — <date>
**Scope:** <what was reviewed>

## Executive Summary
<count of critical/major/medium/low findings>

## Critical Issues
### C1: <title>
**File:** `path/to/file.py:L<start>-L<end>`
**Problem:** <what is wrong and why>
**Current code:** <snippet>
**Fix:** <concrete corrected code>
**Test:** <how to verify>

## Major Issues
[same format]

## Medium Issues
[same format]

## Low / Enhancements
[same format]

## Documentation Gaps
[same format]

## Testing Gaps
[same format]
```

Every finding must have an exact `file.py:L<line>` reference. Every finding must include a concrete fix. Vague findings ("improve error handling") are not acceptable — show the actual code change.

After writing the file, summarise the finding counts by severity in the conversation.
