# Phase 2 Shared Context (Non-Sensitive)

## Project Snapshot
SensoryForge is a modular, extensible framework for simulating sensory encoding across modalities. The codebase is pure PyTorch: every component is a `torch.nn.Module`. The pipeline is differentiable, GPU-accelerated, and batchable.

## Phase 2 Goal (Parallelizable)
Deliver solver infrastructure, an equation DSL for neuron models, composable multi-population grids, and extended stimulus types. These tasks must not change existing behavior unless explicitly required by the task description.

## Required Coding Standards (Summary)
- Type hints are mandatory for all function signatures.
- Use Google Style docstrings with Args/Returns/Raises/Example.
- Document tensor shapes like `[batch, time, grid_h, grid_w]` and physical units (mA, mV, ms, mm).
- Avoid hand-rolled loops over neurons or spatial dimensions; use tensor broadcasting.
- Always specify and propagate `device` (cpu, cuda, mps).
- No behavior changes beyond what the task explicitly requests.
- Optional dependencies must fail gracefully with clear import errors and install hints.

## File Structure (Phase 2 Relevant)
- [sensoryforge/solvers/](sensoryforge/solvers/) (new)
- [sensoryforge/neurons/model_dsl.py](sensoryforge/neurons/model_dsl.py) (new)
- [sensoryforge/core/composite_grid.py](sensoryforge/core/composite_grid.py) (new)
- [sensoryforge/stimuli/](sensoryforge/stimuli/) (new files)
- Tests in [tests/unit/](tests/unit/)

## Acceptance Baseline
- Existing tests must still pass.
- New tests must be added for new code.
- The default behavior should remain Forward Euler unless the user opts into adaptive solvers.

## Useful Cross-Task Notes
- Keep interfaces consistent with existing module patterns.
- For any new public API, include a docstring with at least one example.
- Use ASCII unless the file already contains non-ASCII characters.

## Out of Scope
- Documentation site setup (Phase 4).
- Packaging or CI changes (Phase 5).
- Base class refactors (Phase 3).
