# Task: Equation DSL for Neuron Models

## Goal
Implement `sensoryforge/neurons/model_dsl.py` providing a `NeuronModel` class that parses equations and compiles to an `nn.Module` compatible with existing neuron interfaces.

## Shared Context
Read [phase2_agent_tasks/SHARED_CONTEXT.md](phase2_agent_tasks/SHARED_CONTEXT.md) first.

## Scope (Do Only These)
1. Create `sensoryforge/neurons/model_dsl.py`.
2. Add unit tests in `tests/unit/test_model_dsl.py`.

## Requirements
- Parse equation strings with `sympy`.
- Extract state variables, derivatives, parameters.
- Support threshold conditions and reset rules.
- Provide `.compile(solver='euler', device='cpu') -> nn.Module`.
- Provide `from_config` and `to_dict` for config round-tripping.
- Validate against a hand-written Izhikevich model: DSL output matches behavior within tolerance.
- Must fail gracefully when `sympy` is missing, with install instructions.

## Tests (Minimum)
- Valid equation parsing.
- Clear errors for malformed equations.
- Compiled Izhikevich matches hand-written model.
- Serialization round-trip.

## Non-Goals
- Do not refactor existing neuron models.
- Do not change solver code (handled elsewhere).
- Do not change docs or packaging.

## Deliverables
- `sensoryforge/neurons/model_dsl.py`.
- `tests/unit/test_model_dsl.py`.

## Notes
- Use type hints and Google Style docstrings.
- Keep public API minimal and stable.
