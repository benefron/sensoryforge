# Equation DSL

## Overview
The equation DSL lets you define neuron models using differential equations and compile them to PyTorch modules.

## Usage

### Basic Example
```python
from sensoryforge.neurons.model_dsl import NeuronModel

equations = """
dv/dt = -v + I
"""

model = NeuronModel(
    equations=equations,
    threshold="v >= 1.0",
    reset="v = 0.0",
    state_vars={"v": 0.0},
)

neuron = model.compile(dt=0.1, device="cpu")
```

## Configuration
```yaml
neurons:
  type: dsl
  equations: |
    dv/dt = -v + I
  threshold: "v >= 1.0"
  reset: "v = 0.0"
  state_vars:
    v: 0.0
```

## See Also
- [Solvers](solvers.md)
- [Extended Stimuli](extended_stimuli.md)
