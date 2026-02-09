# Accessing Phase 2 Features in SensoryForge

## Overview

SensoryForge v0.2.0 includes powerful Phase 2 features:
- **CompositeGrid**: Multi-population receptor mosaics
- **Equation DSL**: Custom neuron models defined via equations  
- **Extended Stimuli**: Texture (Gabor, gratings) and moving stimuli
- **Adaptive Solvers**: High-precision ODE integration (Dormand-Prince, etc.)

**Current Status**: These features are **fully functional** via the **Command-Line Interface (CLI)** and **Python API**. GUI integration is in progress.

## How to Use Phase 2 Features

### Option 1: Command-Line Interface (Recommended)

The CLI provides complete access to all Phase 2 features through YAML configuration files.

#### Quick Start

1. **Generate a configuration template** from the GUI:
   - Open SensoryForge GUI
   - Go to `File → Save Config (YAML)...`
   - Save as `my_config.yml`

2. **Edit the YAML file** to enable Phase 2 features:
   ```yaml
   grid:
     type: composite
     populations:
       sa1: {density: 10.0, arrangement: poisson}
       ra1: {density: 5.0, arrangement: hex}
   
   neurons:
     type: dsl
     equations: |
       dv/dt = 0.04*v**2 + 5*v + 140 - u + I
       du/dt = a*(b*v - u)
     threshold: "v >= 30"
     reset: "v = c; u = u + d"
     parameters: {a: 0.02, b: 0.2, c: -65.0, d: 8.0}
   
   stimuli:
     - type: texture
       config: {pattern: gabor, wavelength: 2.0}
     - type: moving
       config: {motion_type: linear, start: [-2, 0], end: [2, 0]}
   
   solver:
     type: adaptive
     config: {method: dopri5, rtol: 1e-5}
   ```

3. **Validate the configuration**:
   ```bash
   sensoryforge validate my_config.yml
   ```

4. **Run the simulation**:
   ```bash
   sensoryforge run my_config.yml --duration 1000 --output results.pt
   ```

#### CLI Commands

```bash
# Run simulation
sensoryforge run config.yml [--duration MS] [--output FILE] [--device DEVICE]

# Validate configuration
sensoryforge validate config.yml

# List available components
sensoryforge list-components

# Visualize pipeline structure
sensoryforge visualize config.yml
```

### Option 2: Python API

Use Phase 2 features programmatically:

```python
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Load configuration
config = {
    'grid': {
        'type': 'composite',
        'populations': {
            'sa1': {'density': 10.0, 'arrangement': 'poisson'},
            'ra1': {'density': 5.0, 'arrangement': 'hex'}
        }
    },
    'neurons': {
        'type': 'dsl',
        'equations': 'dv/dt = 0.04*v**2 + 5*v + 140 - u + I',
        'threshold': 'v >= 30',
        'reset': 'v = -65',
        'parameters': {'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0}
    }
}

# Create pipeline
pipeline = GeneralizedTactileEncodingPipeline.from_config(config)

# Run simulation
results = pipeline.forward(stimulus_type='texture', duration=1000)
```

### Option 3: GUI + YAML Workflow

The GUI now provides **Phase 2 feature detection** and **configuration validation**:

1. **Create/Edit YAML config** externally (using the template from `File → Save Config`)

2. **Load and validate** via GUI:
   - `File → Load Config (YAML)...`
   - Select your config file
   - GUI will validate and show detected Phase 2 features:
     - ✓ CompositeGrid detected
     - ✓ DSL neuron model detected  
     - ✓ Adaptive solver detected

3. **Run via CLI** (as suggested by the validation dialog)

## GUI Access to Phase 2 Features

### Currently Available in GUI

- **YAML Config Validation**: Load YAML files and see which Phase 2 features are detected
- **Template Generation**: Save comprehensive config templates with Phase 2 examples
- **Help System**:
  - `Help → Phase 2 Features (CLI)`: Overview of new capabilities
  - `Help → CLI Guide`: Complete CLI reference with examples

### Coming in Future Releases

Full interactive GUI controls for:
- Composite grid population editor
- Visual equation DSL editor
- Extended stimulus designer (Gabor, moving stimuli)
- Solver configuration panel

## Documentation

- **CLI Guide**: `docs/user_guide/cli.md`
- **YAML Configuration**: `docs/user_guide/yaml_configuration.md`
- **CompositeGrid**: `docs/user_guide/composite_grid.md`
- **Equation DSL**: `docs/user_guide/equation_dsl.md`
- **Extended Stimuli**: `docs/user_guide/extended_stimuli.md`
- **Adaptive Solvers**: `docs/user_guide/solvers.md`

## Examples

Example YAML configurations are available in:
- `examples/example_config.yml`: Basic configuration
- `tests/fixtures/phase2_config.yml`: Phase 2 features showcase
- Generated templates from `File → Save Config`

## Getting Help

- **In GUI**: `Help → CLI Guide` or `Help → Phase 2 Features`
- **Command Line**: `sensoryforge --help` or `sensoryforge COMMAND --help`
- **Documentation**: `docs/` directory
- **Issues**: https://github.com/benefron/sensoryforge/issues
