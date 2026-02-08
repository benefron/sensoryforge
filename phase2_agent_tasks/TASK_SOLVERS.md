# Task: Solver Architecture (Base + Euler + Adaptive)

## Goal
Implement pluggable ODE solvers under `sensoryforge/solvers/` with a base interface and a default Euler solver. Add an adaptive solver wrapper that uses optional dependencies.

## Shared Context
Read [phase2_agent_tasks/SHARED_CONTEXT.md](phase2_agent_tasks/SHARED_CONTEXT.md) first.

## Scope (Do Only These)
1. Create `sensoryforge/solvers/__init__.py` with exports.
2. Create `sensoryforge/solvers/base.py` defining `BaseSolver`.
3. Create `sensoryforge/solvers/euler.py` implementing `EulerSolver`.
4. Create `sensoryforge/solvers/adaptive.py` implementing `AdaptiveSolver` that wraps optional `torchdiffeq` or `torchode`.
5. Add tests in `tests/unit/test_solvers.py`.

## Requirements
- `BaseSolver` API:
  - `step(ode_func, state, t, dt) -> new_state`
  - `integrate(ode_func, state, t_span, dt) -> trajectory`
  - `from_config(config) -> solver`
- `EulerSolver` must match current neuron forward-Euler behavior.
- `AdaptiveSolver` must raise a clear `ImportError` when optional deps are missing, with install instructions.
- Do not refactor neuron models in this task.
- Keep default solver behavior unchanged (Euler is default).

## Tests (Minimum)
- Euler step and integrate produce expected shapes and types.
- Adaptive solver raises `ImportError` when deps missing.
- `from_config` returns correct solver instance.

## Non-Goals
- Do not change neuron model code (handled elsewhere).
- Do not modify docs or packaging.

## Deliverables
- New files in `sensoryforge/solvers/`.
- `tests/unit/test_solvers.py`.

## Notes
- Keep the API stable and minimal.
- Use type hints and Google Style docstrings.
