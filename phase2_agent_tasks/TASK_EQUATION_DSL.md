# Task: Implement Equation DSL for Neuron Models

## Objective

Create a Domain-Specific Language (DSL) that allows users to define custom spiking neuron models using mathematical equations instead of hardcoding PyTorch implementations.

## Background

Currently, each neuron model (Izhikevich, AdEx, MQIF, etc.) is implemented as a separate PyTorch module with hardcoded differential equations. This approach:
- Requires Python programming knowledge
- Makes rapid prototyping difficult
- Limits accessibility for neuroscientists unfamiliar with PyTorch

The Equation DSL will allow users to define neuron models declaratively using mathematical notation, similar to Brian2 or NEST.

## Requirements

### Core Features

1. **Equation Parser**: Parse differential equations from string format
2. **Code Generator**: Generate efficient PyTorch code from parsed equations
3. **State Variables**: Support for multiple state variables (v, u, w, etc.)
4. **Parameters**: Named parameters with default values
5. **Spike Conditions**: Define when neurons spike
6. **Reset Conditions**: Define post-spike reset behavior
7. **Integration**: Support Forward Euler integration (extendable to other solvers)

### API Design

Users should be able to define a neuron model like:

```python
from sensoryforge.neurons.dsl import EquationBasedNeuron

# Define Izhikevich model using equations
izhikevich = EquationBasedNeuron(
    equations="""
    dv/dt = 0.04*v**2 + 5*v + 140 - u + I
    du/dt = a*(b*v - u)
    """,
    spike_condition="v >= threshold",
    reset_equations="""
    v = c
    u = u + d
    """,
    parameters={
        'a': 0.02,
        'b': 0.2,
        'c': -65.0,
        'd': 8.0,
        'threshold': 30.0
    },
    initial_values={
        'v': -65.0,
        'u': -13.0
    }
)

# Use like any other neuron
input_current = torch.randn(1, 100, 10)
v_trace, spikes = izhikevich(input_current)
```

### Technical Specifications

1. **Dependencies**: Use SymPy for symbolic math operations
2. **Performance**: Generated code should be comparable to handwritten implementations
3. **Validation**: Validate equations for correctness before code generation
4. **Documentation**: Full docstrings and usage examples
5. **Testing**: Comprehensive unit tests, validation against existing models

## Deliverables

- [ ] Create `sensoryforge/neurons/dsl/` module
- [ ] Implement `equation_parser.py` - Parse equations using SymPy
- [ ] Implement `code_generator.py` - Generate PyTorch code
- [ ] Implement `equation_neuron.py` - Main EquationBasedNeuron class
- [ ] Add comprehensive docstrings to all classes and methods
- [ ] Create `tests/unit/test_equation_dsl.py`
- [ ] Add examples recreating Izhikevich and AdEx using DSL
- [ ] Update `DEVELOPMENT.md` with DSL usage guide
- [ ] Ensure sympy is in requirements.txt

## Success Criteria

1. Can recreate Izhikevich model using DSL with identical behavior
2. Can recreate AdEx model using DSL with identical behavior
3. All tests pass
4. Code is well-documented with clear inline comments
5. No performance regression compared to handwritten models

## Notes

- Follow Google-style docstrings
- Use type hints throughout
- Keep code readable and maintainable
- Add inline comments for complex logic
- Ensure compatibility with existing neuron model interface
