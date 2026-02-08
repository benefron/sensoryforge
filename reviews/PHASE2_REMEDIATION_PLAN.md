# Phase 2 Remediation Plan

## Objectives
- Resolve performance and correctness issues identified in the Phase 2 deep review.
- Add missing tests for extended stimuli and YAML/pipeline integration.
- Draft a concrete GUI/CLI/YAML integration plan.
- Keep project standards: type hints, Google-style docstrings, unit tests, and documentation updates.

## Planned Work
1. Fix DSL execution to use torch-native evaluation (no numpy round-trips), preserve dtype/device, and keep autograd intact.
2. Improve CompositeGrid sampling for Poisson and hex arrangements with vectorized tensor generation.
3. Correct AdaptiveSolver backend detection; align extras and error messages with supported backends.
4. Remove duplicate YAML config keys and add duplicate-key validation in config loading.
5. Implement YAML-driven integration scaffolding in generalized pipeline (minimal but end-to-end).
6. Add unit tests for texture and moving stimuli.
7. Add integration tests for YAML-configured pipelines and solver selection.
8. Draft an integration plan document for GUI/CLI/YAML exposure.
