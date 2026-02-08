# Task: Composite Grid (Multi-Population Spatial Substrate)

## Goal
Implement `sensoryforge/core/composite_grid.py` to support multiple named receptor populations on a shared coordinate system.

## Shared Context
Read [phase2_agent_tasks/SHARED_CONTEXT.md](phase2_agent_tasks/SHARED_CONTEXT.md) first.

## Scope (Do Only These)
1. Create `sensoryforge/core/composite_grid.py` with `CompositeGrid`.
2. Add unit tests in `tests/unit/test_composite_grid.py`.

## Requirements
- Shared coordinate system with named populations.
- Population config fields: `density`, `arrangement`, optional `filter` or metadata.
- Arrangement types: `grid`, `poisson`, `hex`, `jittered_grid`.
- Density-based point generation.
- Integration hooks for use by `innervation.py` (do not change innervation in this task).

## Tests (Minimum)
- Population creation and density validation.
- Arrangement types produce expected counts and bounds.
- Coordinate consistency across populations.

## Non-Goals
- Do not modify `innervation.py` (handled separately if needed).
- Do not change docs or packaging.

## Deliverables
- `sensoryforge/core/composite_grid.py`.
- `tests/unit/test_composite_grid.py`.

## Notes
- Use type hints and Google Style docstrings.
- Favor vectorized generation where possible.
