# Equation DSL

## Overview

The Equation DSL lets you define neuron models using differential equations and compile them to PyTorch `nn.Module` instances. Instead of hand-writing PyTorch code, describe dynamics in standard mathematical notation — the DSL parses, validates, and compiles to efficient code.

**Benefits:**
- **Rapid prototyping** — Define models in minutes
- **Equation-first** — Write mathematics first, implementation second
- **Automatic validation** — Catch undefined variables and inconsistencies early
- **Same interface** — Compiled models work like hand-written neurons

## Quick Start

```python
from sensoryforge.neurons.model_dsl import NeuronModel

model = NeuronModel(
    equations='dv/dt = (-v + I) / tau',
    threshold='v >= 1.0',
    reset='v = 0.0',
    parameters={'tau': 10.0},
    state_vars={'v': 0.0}
)

neuron = model.compile(solver='euler', dt=0.05, device='cpu')

# Use in simulation
import torch
current = torch.randn(10, 1)  # [batch, num_neurons]
spikes, state = neuron(current)
```

## Izhikevich Example

```python
model = NeuronModel(
    equations='''
        dv/dt = 0.04*v**2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
    ''',
    threshold='v >= 30',
    reset='''
        v = c
        u = u + d
    ''',
    parameters={'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0},
    state_vars={'v': -65.0, 'u': -13.0}
)

neuron = model.compile(dt=0.05)
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `equations` | Yes | Differential equations in `dx/dt = expression` format |
| `threshold` | Yes | Spike condition, e.g. `v >= 30` |
| `reset` | Yes | State assignments when threshold crossed |
| `parameters` | No | Constants used in equations |
| `state_vars` | No | Initial values; auto-inferred if omitted |

## Supported Operations

- **Arithmetic:** `+`, `-`, `*`, `/`, `**`
- **Functions:** `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`
- **Special:** `I` for input current
- **Threshold operators:** `>=`, `>`, `<`, `<=`

## Configuration (YAML)

```yaml
neurons:
  type: dsl
  equations: |
    dv/dt = (-v + I) / tau
  threshold: "v >= 1.0"
  reset: "v = 0.0"
  parameters:
    tau: 10.0
  state_vars:
    v: 0.0
```

## Solver Options

```python
# Forward Euler (default, fast)
neuron = model.compile(solver='euler', dt=0.05)

# Adaptive (requires torchdiffeq)
neuron = model.compile(solver='dopri5', dt=0.05)
```

## See Also

- [Solvers](solvers.md) — ODE solver selection
- [Extended Stimuli](extended_stimuli.md) — Stimulus types
- [EQUATION_DSL_IMPLEMENTATION.md](../EQUATION_DSL_IMPLEMENTATION.md) — Full implementation details
