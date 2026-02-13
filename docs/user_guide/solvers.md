# Solvers

## Overview

SensoryForge supports pluggable ODE solvers for neuron dynamics. The default is **Forward Euler** — fast and sufficient for most models. **Adaptive solvers** (Dormand-Prince, Adams) are available for stiff systems or when higher accuracy is needed.

## When to Use Which

| Solver | Use Case | Dependencies |
|--------|----------|--------------|
| **Euler** | Default; fast; non-stiff systems | None |
| **Adaptive (dopri5)** | Stiff systems (e.g. AdEx); accuracy-critical | torchdiffeq |
| **Adaptive (adams)** | Smooth problems; long integrations | torchdiffeq |

## Forward Euler

Fixed time step, first-order method. Matches default neuron behavior.

```python
from sensoryforge.solvers import EulerSolver

solver = EulerSolver(dt=0.05)

# Use with neuron models (typically via config)
# dt is in ms; 0.05 ms is the SensoryForge default
```

## Adaptive Solver

Automatically adjusts step size to meet error tolerances. Requires `torchdiffeq`:

```bash
pip install torchdiffeq
```

```python
from sensoryforge.solvers import AdaptiveSolver

solver = AdaptiveSolver(
    method="dopri5",
    rtol=1e-5,
    atol=1e-7,
    dt=0.05,
)

# Supported methods: dopri5, dopri8, adams, bosh3
# dopri5 is recommended for most cases
```

## Factory Function

Create a solver from a configuration dictionary:

```python
from sensoryforge.solvers import get_solver

# Euler
config = {"type": "euler", "dt": 0.1}
solver = get_solver(config)

# Adaptive
config = {
    "type": "adaptive",
    "method": "dopri5",
    "rtol": 1e-5,
    "atol": 1e-7,
    "dt": 0.05,
}
solver = get_solver(config)
```

## YAML Configuration

```yaml
simulation:
  solver:
    type: euler
    dt: 0.05

# Or for adaptive:
simulation:
  solver:
    type: adaptive
    method: dopri5
    rtol: 1.0e-5
    atol: 1.0e-7
    dt: 0.05
```

## Equation DSL Integration

When compiling a neuron model from the Equation DSL, specify the solver:

```python
from sensoryforge.neurons.model_dsl import NeuronModel

model = NeuronModel(
    equations="dv/dt = (-v + I) / tau",
    threshold="v >= 1.0",
    reset="v = 0.0",
    parameters={"tau": 10.0},
    state_vars={"v": 0.0},
)

# Euler (default)
neuron = model.compile(solver="euler", dt=0.05)

# Adaptive (if torchdiffeq installed)
neuron = model.compile(solver="dopri5", dt=0.05)
```

## Performance Notes

- **Euler:** Fastest; suitable for Izhikevich, LIF, most tactile encoding.
- **Adaptive:** Slower per step but may use fewer steps for stiff systems; recommended for AdEx or when numerical stability is an issue.
- **Units:** `dt` is in milliseconds (ms) to match SensoryForge conventions.

## See Also

- [Equation DSL](equation_dsl.md)
- [Composite Grid](composite_grid.md)
- [YAML Configuration](yaml_configuration.md)
