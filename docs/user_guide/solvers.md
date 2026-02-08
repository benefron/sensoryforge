# Solvers

## Overview
SensoryForge supports pluggable ODE solvers for neuron dynamics.

## Usage

### Euler Solver
```python
from sensoryforge.solvers import EulerSolver

solver = EulerSolver(dt=0.05)
```

### Adaptive Solver
```python
from sensoryforge.solvers import AdaptiveSolver

solver = AdaptiveSolver(method="dopri5", rtol=1e-5, atol=1e-7)
```

## Configuration
```yaml
solver:
  type: adaptive
  method: dopri5
  rtol: 1.0e-5
  atol: 1.0e-7
  dt: 0.05
```

## See Also
- [Equation DSL](equation_dsl.md)
- [Composite Grid](composite_grid.md)
