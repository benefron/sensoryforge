# Adding a New Solver

A solver performs the numerical integration step for an ODE-driven neuron model (Izhikevich, AdEx,
MQIF, or a DSL-compiled model with `solver_config`). `EulerSolver` (forward Euler,
`sensoryforge/solvers/euler.py`) is the built-in default; `AdaptiveSolver` wraps `torchdiffeq`/
`torchode` when installed (`pip install torchdiffeq` or `torchode`; see `docs/how-to/solvers.md`).
This guide adds a new fixed-step solver the same way, registered in `SOLVER_REGISTRY`.

The complete runnable example is `docs/examples/plugin_solver.py`: it defines `HeunSolver`
(explicit trapezoidal RK2), registers it, runs the shared contract check, and integrates a known
linear ODE (`dx/dt = -x`) to show its error is lower than forward Euler's at the same step size --
the actual reason to add a second-order solver, not just a shape check. It is executed by
`tests/docs/test_docs_examples.py`.

---

## 1. The contract

All solvers inherit from `BaseSolver` (`sensoryforge/solvers/base.py`):

| Method | Signature | Purpose |
|---|---|---|
| `step(ode_func, state, t, dt)` | `[B, ...] -> [B, ...]` | One integration step. `ode_func(state, t) -> dstate/dt` has the same shape as `state`. |
| `integrate(ode_func, state, t_span, dt)` | `[B, ...] -> [B, steps+1, ...]` | Repeated `step()` over `(t_start, t_end)`; the returned trajectory's first time slice is the initial state. |
| `from_config(config)` | `dict -> cls` | Construct from a YAML dict (`{"type": ..., "dt": ...}`). |
| `to_dict()` | `-> dict` | Serialise; must include a `"type"` key (`BaseSolver`'s default does `type(self).__name__.lower().replace("solver", "")`). |
| `get_param_spec()` | `-> list[ParamSpec]` | Required on every component (G1); default `[]`. |

**Tensor conventions:** `state` is `[batch, ...]` (a neuron model's membrane variables, one row per
population member); `t` and `dt` are in **ms**, matching every other SensoryForge time value at the
user-facing API. `step()` and `integrate()` must preserve `state`'s shape exactly -- a neuron model's
`forward()` loop calls `step()` once per integration sub-step and assumes the returned tensor is safe
to feed back in unchanged.

`sensoryforge.testing.contracts.check_component("solver", cls)` checks `get_param_spec()`'s shape,
one `step()` call preserving state shape, and the full `to_dict()`/`from_config()` round trip:
every `__init__` argument must appear in `to_dict()` and `from_config(instance.to_dict())` must be a
fixed point (F-049). `BaseSolver.to_dict()` holds only `type` and `dt`, so a solver with more
arguments overrides `to_dict()` to add them, as `AdaptiveSolver` does for `method`, `rtol` and `atol`.

## 2. Register it

Two routes, same as every other kind (see `docs/how-to/extend/plugins.md`):

- **Plugin package:** `sensoryforge new-component solver MySolver --dest DIR` scaffolds an
  installable package with an entry point; no edits to this checkout.
- **In-repo:** `sensoryforge new-component solver MySolver --in-repo` writes into
  `sensoryforge/solvers/`; add one line to `register_components.py`'s `register_all()`:
  `SOLVER_REGISTRY.register("heun", HeunSolver)`.

## 3. Using it

A registered solver is used by code that creates it, `SOLVER_REGISTRY.create("heun", dt=0.05)`
(for instance inside a neuron model of your own), and `step()`/`integrate()` take it from there.

!!! warning "A config does not reach a plugin solver today"
    `SimulationEngine` builds no solver from a population's `solver_config`, `get_solver()`
    (the factory the [Solvers](../solvers.md) page shows) knows only `euler` and
    `adaptive`, and the equation DSL's `compile(solver=...)` accepts only `euler`. A config key
    that selects a registered solver is an open item, recorded in the project ledger.

Also: the scaffold's `BaseSolver.to_dict()` writes the class name in lowercase with "solver"
removed as `type` (a class `Heun` registered as `heun` gives `heun`, but `DemoRk2Solver`
registered as `demo_rk2` gives `demork2`), so override `to_dict()` when the registered name
differs.

See `docs/how-to/solvers.md` for the built-in `euler`/`adaptive` choices, and
`docs/examples/plugin_solver.py` for the worked, executed example.
