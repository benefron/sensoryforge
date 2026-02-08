# Phase 2 Deep Code Review (2026-02-08)

## Scope
- CompositeGrid implementation and tests
- Equation DSL implementation and tests
- Solver architecture (Euler + Adaptive)
- Extended stimuli modules
- Integration gaps (GUI, YAML, CLI, pipeline)
- Documentation and test coverage against project standards

## Critical Issues

1) DSL breaks GPU + autograd and forces CPU round-trips
- Evidence: numpy lambdify and per-call tensor->numpy->tensor conversions in [sensoryforge/neurons/model_dsl.py](sensoryforge/neurons/model_dsl.py#L435-L493).
- Impact: GPU execution is effectively disabled; autograd is broken due to numpy usage; performance collapses for large batches/time steps.
- Recommended fix:
  - Replace numpy lambdify with torch-compatible lambdify or a custom sympy->torch translator.
  - Keep all computations as torch ops to preserve device, dtype, and autograd.
  - Consider caching compiled torch functions and optional TorchScript export for speed.

2) Adaptive solver torchode path is never enabled
- Evidence: torchode import sets HAS_TORCHODE to False even when import succeeds, making torchode unreachable and forcing ImportError unless torchdiffeq is installed. See [sensoryforge/solvers/adaptive.py](sensoryforge/solvers/adaptive.py#L14-L25).
- Impact: advertised optional dependency does not work; inconsistent with extras_require.
- Recommended fix:
  - Set HAS_TORCHODE = True on successful import.
  - Implement a torchode backend path or remove torchode from extras if unsupported.

3) Default YAML config has duplicate key that silently overwrites data
- Evidence: `bayesian_estimator` is defined twice. Later block overwrites earlier block in YAML parsing. See [sensoryforge/config/default_config.yml](sensoryforge/config/default_config.yml#L197-L220).
- Impact: silent config loss and misleading defaults.
- Recommended fix:
  - Consolidate into a single `bayesian_estimator` block.
  - Add config validation to detect duplicate keys.

## Major Issues

1) CompositeGrid Poisson sampling is O(n^2) and scales poorly
- Evidence: greedy minimum distance check loops through candidates and computes distances to all selected points in Python. See [sensoryforge/core/composite_grid.py](sensoryforge/core/composite_grid.py#L315-L364).
- Impact: severe slowdown for large grids; violates the “avoid hand-rolled loops over spatial dimensions” guideline.
- Recommended fix:
  - Implement Bridson’s Poisson disk sampling with spatial hashing.
  - Use a grid-based acceleration structure or chunked `torch.cdist` with pruning.
  - Consider allowing a fast approximate mode for large densities.

2) CompositeGrid hex generation uses nested Python loops and list accumulation
- Evidence: nested loops and list append, then python list->tensor conversion. See [sensoryforge/core/composite_grid.py](sensoryforge/core/composite_grid.py#L366-L417).
- Impact: slow and memory-heavy for large grids; CPU-bound even on GPU workflows.
- Recommended fix:
  - Vectorize using torch meshgrid + row offset masks.
  - Generate all candidate coordinates in tensors, then apply bounds mask.

3) Generalized pipeline does not expose Phase 2 features
- Evidence: the pipeline is hard-wired to GridManager, Izhikevich, and legacy SA/RA filters with no CompositeGrid/DSL/solvers/extended stimuli hooks. See [sensoryforge/core/generalized_pipeline.py](sensoryforge/core/generalized_pipeline.py#L8-L175).
- Impact: Phase 2 features cannot be used end-to-end, making integration impossible without manual code changes.
- Recommended fix:
  - Add `from_config()` factory and component registries for grid, stimuli, filters, neurons, and solvers.
  - Use the new YAML schema described in the integration task.

## Moderate Issues

1) DSL precision and dtype coercion
- Evidence: lambdified outputs are always converted to float32 tensors. See [sensoryforge/neurons/model_dsl.py](sensoryforge/neurons/model_dsl.py#L452-L492).
- Impact: precision loss when users set dtype to float64 for stiff systems; also potential type mismatches.
- Recommended fix:
  - Preserve input dtype by constructing tensors with `dtype=dtype` and avoiding forced float32.

2) DSL runtime overhead inside the time loop
- Evidence: per-step python loops over state variables and repeated lambdify calls, plus tensor conversions inside the loop. See [sensoryforge/neurons/model_dsl.py](sensoryforge/neurons/model_dsl.py#L518-L640).
- Impact: high overhead for long simulations; may be acceptable for prototyping but not for production-scale runs.
- Recommended fix:
  - Precompute a vectorized update function using torch ops.
  - Fuse threshold/reset evaluation into a single kernel where feasible.

3) GUI does not yet support YAML or Phase 2 component selection
- Evidence: no YAML loading or component selection in the main window. See [sensoryforge/gui/main.py](sensoryforge/gui/main.py#L1-L92).
- Impact: GUI cannot exercise new features or pipeline configurations.
- Recommended fix:
  - Add YAML load/save workflow and a minimal set of Phase 2 selectors in the UI.
  - Gate advanced editing behind an “Advanced” modal or panel (recommended for usability).

## Test Coverage Gaps

- No dedicated tests for extended stimuli modules (`texture`, `moving`).
  - Add unit tests for `gabor_texture`, `edge_grating`, `noise_texture`, `linear_motion`, `circular_motion`, `custom_path_motion`, `MovingStimulus`.
- No integration tests for YAML-driven pipelines or CLI.
  - Add integration tests for YAML config loading, CompositeGrid instantiation, DSL neuron compilation, adaptive solver selection, and end-to-end CLI invocation.
- No performance or stress tests for large grids.
  - Add benchmark tests (optional) to catch regressions in CompositeGrid scaling.

## Documentation Gaps

- Phase 2 features have implementation docs, but no user guide sections were found.
  - Add user guide pages for CompositeGrid, DSL, solvers, and extended stimuli in docs/user_guide/ and cross-link from docs/index.md.

## Integration Plan Notes (GUI/CLI/YAML)

- Recommended UX approach:
  - Keep core options in the main UI (grid type, solver type, stimulus type).
  - Move DSL editing, solver tolerances, and custom templates to an “Advanced” modal to avoid overwhelming users.
  - Provide YAML template loader/saver as the primary advanced mechanism.

## Summary of Required Fixes
- Replace numpy-based DSL execution with torch-native expressions to restore GPU support and autograd.
- Implement a scalable Poisson/hex generation strategy for CompositeGrid.
- Fix torchode detection and either implement or remove torchode support.
- Resolve duplicate YAML keys and add config validation.
- Add missing tests for extended stimuli and YAML integration.
- Add user guide documentation for all Phase 2 modules.
