# Adding a New Solver

A solver performs the numerical integration step for an ODE-driven neuron model (Izhikevich, AdEx,
MQIF, or a DSL-compiled model with `solver_config`). `EulerSolver` (forward Euler,
`sensoryforge/solvers/euler.py`) is the built-in default; `AdaptiveSolver` wraps `torchdiffeq`/
`torchode` when installed (`pip install torchdiffeq` or `torchode`; see `docs/user_guide/solvers.md`).
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
one `step()` call preserving state shape, and a `from_config(instance.to_dict())` round trip to a
`cls` instance -- **not** full parameter-completeness (the H3 check in `_assert_to_dict_roundtrip_complete`
is wired up for the `"neuron"` kind only; see ledger F-049 and `docs/developer_guide/add_filter.md`
section 1 for the same caveat as it applies to filters).

## 2. Register it

Two routes, same as every other kind (see `docs/developer_guide/plugins.md`):

- **Plugin package:** `sensoryforge new-component solver MySolver --dest DIR` scaffolds an
  installable package with an entry point; no edits to this checkout.
- **In-repo:** `sensoryforge new-component solver MySolver --in-repo` writes into
  `sensoryforge/solvers/`; add one line to `register_components.py`'s `register_all()`:
  `SOLVER_REGISTRY.register("heun", HeunSolver)`.

## 3. Using it from a config

A population's `solver_config` (or the top-level `SimulationConfig.solver`) names the registered
solver by its `"type"` key:

```yaml
populations:
  - name: "Custom-Integrated Population"
    neuron_model: "izhikevich"
    solver_config:
      type: "heun"
      dt: 0.05
```

See `docs/user_guide/solvers.md` for the built-in `euler`/`adaptive` choices, and
`docs/examples/plugin_solver.py` for the worked, executed example.
