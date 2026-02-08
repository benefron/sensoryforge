# Task: GUI, CLI, and Pipeline Integration with YAML Configuration

## Goal
Integrate the Phase 2 features (Composite Grid, Equation DSL, Extended Stimuli, Adaptive Solvers) into the existing GUI, CLI, and pipeline with full YAML configuration support for all components.

## Shared Context
Read [phase2_agent_tasks/SHARED_CONTEXT.md](phase2_agent_tasks/SHARED_CONTEXT.md) first.

## Scope (Do Only These)

### 1. CLI Enhancement
- Create or extend `sensoryforge/cli.py` to expose all Phase 2 features.
- Support YAML config file loading and override with command-line flags.
- Commands:
  - `sensoryforge run <config.yml>` — Execute pipeline from YAML.
  - `sensoryforge visualize <config.yml>` — Visualize pipeline structure and outputs.
  - `sensoryforge validate <config.yml>` — Validate YAML config without running.
  - `sensoryforge list-components` — List available filters, neurons, stimuli, solvers.

### 2. Pipeline Enhancement (generalized_pipeline.py)
- Extend `sensoryforge/core/generalized_pipeline.py` to support:
  - CompositeGrid instantiation from YAML.
  - Equation DSL model compilation from YAML.
  - Extended stimulus types (texture, moving, etc.) from YAML.
  - Adaptive solver selection and configuration from YAML.
- Maintain backward compatibility with existing single-grid, hand-written neuron pipelines.

### 3. YAML Configuration Schema
- Extend `sensoryforge/config/default_config.yml` and schemas to include:
  - `grid` section: Support `type: composite` with population definitions.
  - `neurons` section: Support `type: dsl` for equation-based models.
  - `stimuli` section: Support all stimulus types (gaussian, texture, moving).
  - `solvers` section: Support solver selection (euler, adaptive).
  - `gui` section: Parameters for GUI initialization and defaults.

### 4. GUI Integration (sensoryforge/gui/)
- Update `sensoryforge/gui/main.py` to:
  - Load configuration from YAML at startup.
  - Expose dropdowns/menus for all Phase 2 component types.
  - Allow users to switch between CompositeGrid and standard Grid.
  - Allow users to select solver type (Euler vs. Adaptive).
  - Allow users to create/edit neuron models via DSL or hand-written mode.
  - Save/load full simulation configs to/from YAML.
- Update `default_params.json` to reflect new component options.

### 5. Integration Tests
- Add integration tests in `tests/integration/test_yaml_pipeline.py`:
  - Load YAML with CompositeGrid; verify correct population setup.
  - Load YAML with DSL neuron model; verify compilation.
  - Load YAML with texture/moving stimuli; verify generation.
  - Load YAML with adaptive solver; verify solver instantiation.
  - CLI command execution end-to-end.

## Requirements

### Configuration Structure
```yaml
# Core configuration
metadata:
  name: "Example Simulation"
  version: "0.1"

# Grid (standard or composite)
grid:
  type: composite  # or 'standard'
  shape: [64, 64]
  populations:
    sa1:
      density: 0.30
      arrangement: poisson
    ra1:
      density: 0.20
      arrangement: hex
    sa2:
      density: 0.10
      arrangement: poisson

# Stimuli
stimuli:
  - type: gaussian
    config:
      sigma: 5.0
      amplitude: 1.0
  - type: texture
    config:
      pattern: "perlin"
      scale: 10
  - type: moving
    config:
      direction: [1, 0]
      speed: 2.0

# Filters
filters:
  sa1:
    type: SA
    config:
      tau_ms: 10.0
      gain: 1.0
  ra1:
    type: RA
    config:
      tau_ms: 5.0
      gain: 1.5

# Neurons (hand-written model)
neurons:
  type: izhikevich  # or 'dsl'
  config:
    a: 0.02
    b: 0.2
    c: -65
    d: 8

# Alternative: DSL neuron model
# neurons:
#   type: dsl
#   equations: |
#     dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms
#     du/dt = (a * (b*v - u)) / ms
#   threshold: "v >= 30 * mV"
#   reset: "v = c; u = u + d"
#   parameters:
#     a: 0.02
#     b: 0.2
#     c: -65
#     d: 8

# Solver
solver:
  type: euler  # or 'adaptive'
  config:
    dt: 0.001  # in seconds
    # For adaptive:
    # method: "dopri5"
    # rtol: 1.0e-5
    # atol: 1.0e-7

# GUI defaults
gui:
  auto_load_config: true
  default_stimulus_type: gaussian
  enable_visualization: true
  visualization_type: heatmap  # or 'raster'
```

### API Changes

**Pipeline.from_config(config: Dict)**
```python
def from_config(config: Dict[str, Any]) -> 'GeneralizedPipeline':
    """Load pipeline from configuration dictionary (yaml-parsed).
    
    Supports all Phase 2 components: CompositeGrid, DSL neurons, 
    extended stimuli, adaptive solvers.
    
    Args:
        config: Configuration dictionary with keys: metadata, grid, 
                stimuli, filters, neurons, solver, gui.
    
    Returns:
        Initialized GeneralizedPipeline instance.
    """
```

**CLI Usage**
```bash
# Run a simulation from YAML
sensoryforge run config.yml --duration 1000 --output result.h5

# Validate config
sensoryforge validate config.yml

# List available components
sensoryforge list-components

# Visualize pipeline
sensoryforge visualize config.yml --save output.png
```

## Tests (Minimum)

### Unit / Integration Tests
- Load YAML with CompositeGrid → verify populations created correctly.
- Load YAML with DSL neuron → verify model compiles and runs.
- Load YAML with texture/moving stimuli → verify generation.
- Load YAML with adaptive solver → verify solver is instantiated.
- YAML validation catches invalid configs (missing required fields, bad types).
- CLI `run`, `validate`, `visualize`, `list-components` all execute without error.
- GUI loads YAML config at startup and populates UI correctly.
- Save/load cycle (simulate → export config → reload → verify identical state).

## Non-Goals
- Do not refactor existing neuron, filter, or stimulus implementations.
- Do not change core Phase 2 feature APIs (CompositeGrid, DSL, solvers remain unchanged).
- Do not add new stimulus types or neuron models (use Phase 2 features as-is).
- Do not implement database or cloud storage integration.

## Deliverables
- `sensoryforge/cli.py` (new or enhanced) with commands: run, validate, visualize, list-components.
- Updates to `sensoryforge/core/generalized_pipeline.py` with `from_config()` method.
- Extended `sensoryforge/config/default_config.yml` with Phase 2 component sections.
- Schema definitions for YAML validation (optional: JSON schema file).
- Updated `sensoryforge/gui/main.py` with Phase 2 component selection and YAML I/O.
- Updated `sensoryforge/gui/default_params.json` with new component defaults.
- `tests/integration/test_yaml_pipeline.py` with comprehensive integration tests.
- Updated documentation: user guide section on YAML configuration and CLI usage.

## Notes
- Use type hints and Google Style docstrings.
- Preserve backward compatibility: existing pipelines should continue working.
- CLI should use argparse or Click for command structure.
- YAML validation can use jsonschema or pydantic for schema enforcement.
- GUI updates should not break existing functionality; add new features as optional tabs/menus.
- YAML config should be self-documenting with comments and examples.

## References
- Existing Phase 2 tasks: TASK_COMPOSITE_GRID.md, TASK_EQUATION_DSL.md, TASK_EXTENDED_STIMULI.md, TASK_SOLVERS.md.
- Current pipeline: sensoryforge/core/generalized_pipeline.py
- Current GUI: sensoryforge/gui/main.py
- Configuration: sensoryforge/config/default_config.yml
