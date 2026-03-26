# SensoryForge Workflow: GUI → CLI

## Overview

SensoryForge is designed around a **design-then-scale** workflow:

1. **GUI** — Interactive workbench for designing experiments, tuning parameters, and observing population responses in real time
2. **CLI + YAML** — Export configurations for batch execution, parameter sweeps, and large-scale data generation
3. **Python API** — Programmatic access for custom analysis and integration

## GUI: Interactive Experimentation

The GUI provides three core tabs:

### Mechanoreceptors & Innervation
- Configure the spatial grid (size, spacing, center)
- Create multiple receptor populations (SA, RA, SA2, custom)
- Visualize receptive fields and innervation patterns
- Adjust population parameters (density, connectivity, sigma)

### Stimulus Designer
- Design stimuli interactively with real-time preview
- Supported types: Gaussian pressure, point pressure, edge
- Configure motion (static, sliding), amplitude, duration
- Preview temporal profiles (ramp-up, plateau, ramp-down)

### Spiking Neurons
- Select neuron models: Izhikevich, AdEx, MQIF, FA, SA
- Configure per-population: model type, filter (SA/RA/none), gain, noise
- Run simulations and visualize spike rasters, voltage traces
- Fine-tune model parameters via the parameter editor

### YAML Integration
- **Load Config**: `File → Load Config (YAML)` validates and previews any configuration (both canonical and legacy formats)
- **Save Config**: `File → Save Config (YAML)` exports configuration in **canonical format** (`SensoryForgeConfig`)
- **Round-Trip Workflow**: GUI → Save → CLI → Load → Same results
- **Help → Advanced Features**: Overview of all available features
- **Help → CLI Guide**: Complete CLI reference with examples

### Canonical Configuration Export

The GUI exports configurations in the **canonical format** (`SensoryForgeConfig`), which ensures:
- **GUI-CLI Parity**: Configurations saved from GUI work seamlessly with CLI
- **N-Population Support**: Export any number of populations (not limited to SA/RA/SA2)
- **Round-Trip Fidelity**: Save → Load → Same results
- **Extensibility**: Easy to add new population types and configurations

When you save a configuration from the GUI, it uses the canonical schema with `grids`, `populations`, `stimulus`, and `simulation` sections.

## CLI: Scalable Execution

Once a configuration is designed and validated in the GUI, export to YAML and use the CLI for scale:

```bash
# Validate configuration (accepts both canonical and legacy formats)
sensoryforge validate my_config.yml

# Run simulation
sensoryforge run my_config.yml --duration 1000 --output results.pt

# Run on GPU
sensoryforge run my_config.yml --device cuda --duration 5000

# List available components
sensoryforge list-components
```

### Round-Trip Workflow

The canonical format ensures seamless workflow between GUI and CLI:

1. **Design in GUI**: Configure grids, populations, stimuli interactively
2. **Save Config**: `File → Save Config (YAML)` exports canonical format
3. **Run in CLI**: `sensoryforge run config.yml` executes with same results
4. **Load Back**: GUI can load the same config for further editing

### Example: GUI-Exported Canonical Configuration

When you save from the GUI, you get canonical format:

```yaml
grids:
  - name: "Main Grid"
    arrangement: "grid"
    rows: 80
    cols: 80
    spacing: 0.15
    center_x: 0.0
    center_y: 0.0

populations:
  - name: "SA Population"
    neuron_type: "SA"
    neuron_model: "izhikevich"
    filter_method: "sa"
    innervation_method: "gaussian"
    neurons_per_row: 10
    connections_per_neuron: 28
    sigma_d_mm: 0.3
  - name: "RA Population"
    neuron_type: "RA"
    neuron_model: "izhikevich"
    filter_method: "ra"
    innervation_method: "gaussian"
    neurons_per_row: 14
    connections_per_neuron: 28
    sigma_d_mm: 0.39

stimulus:
  type: "gaussian"
  amplitude: 30.0
  sigma: 1.0

simulation:
  device: "cpu"
  dt: 0.5
```

This format is directly compatible with the CLI and supports N populations.

### Advanced Canonical Configuration Example

The GUI can export complex configurations with multiple populations:

```yaml
grids:
  - name: "Composite Receptor Grid"
    arrangement: "composite"
    # Composite grid configuration...

populations:
  - name: "SA1 Population"
    neuron_type: "SA"
    neuron_model: "izhikevich"
    filter_method: "sa"
    innervation_method: "gaussian"
    neurons_per_row: 10
  - name: "RA1 Population"
    neuron_type: "RA"
    neuron_model: "izhikevich"
    filter_method: "ra"
    innervation_method: "gaussian"
    neurons_per_row: 14
  - name: "Custom DSL Population"
    neuron_type: "Custom"
    neuron_model: "dsl"
    dsl_config:
      equations: |
        dv/dt = 0.04*v**2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
      threshold: "v >= 30"
      reset: "v = c; u = u + d"
      parameters: {a: 0.02, b: 0.2, c: -65.0, d: 8.0}

simulation:
  device: "cpu"
  dt: 0.5
  solver_config:
    type: "adaptive"
    method: "dopri5"
    rtol: 1.0e-5
    atol: 1.0e-7
```

**Note**: The GUI exports canonical format, which supports N populations. Legacy format examples are shown for backward compatibility reference.

## Python API

The Python API supports both canonical and legacy configurations:

```python
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Option 1: Load canonical config (GUI-exported format)
config = SensoryForgeConfig.from_yaml('gui_exported_config.yml')
pipeline = GeneralizedTactileEncodingPipeline.from_config(config.to_dict())
results = pipeline.forward(stimulus_type='gaussian', amplitude=30.0)

# Option 2: Load legacy config (backward compatible)
pipeline = GeneralizedTactileEncodingPipeline.from_yaml('legacy_config.yml')
results = pipeline.forward(duration_ms=1000)

# Option 3: Configure programmatically (canonical format)
config = SensoryForgeConfig(
    grids=[GridConfig(name="Grid", arrangement="grid", rows=80, cols=80, spacing=0.15)],
    populations=[
        PopulationConfig(name="SA", neuron_type="SA", neuron_model="izhikevich", 
                        filter_method="sa", innervation_method="gaussian", neurons_per_row=10),
        PopulationConfig(name="RA", neuron_type="RA", neuron_model="izhikevich",
                        filter_method="ra", innervation_method="gaussian", neurons_per_row=14),
    ],
    stimulus=StimulusConfig(type="gaussian", amplitude=30.0),
    simulation=SimulationConfig(device="cpu", dt=0.5),
)
pipeline = GeneralizedTactileEncodingPipeline.from_config(config.to_dict())
```

**Note**: For GUI-exported configs, use `SensoryForgeConfig.from_yaml()` for best compatibility and N-population support.

## Documentation

- [CLI Reference](cli.md) — Full command-line documentation
- [YAML Configuration](yaml_configuration.md) — Complete schema reference
- [Composite Grid](composite_grid.md) — Multi-population spatial substrates
- [Equation DSL](equation_dsl.md) — Custom neuron models via equations
- [Extended Stimuli](extended_stimuli.md) — Texture and moving stimuli
- [Solvers](solvers.md) — ODE solver configuration

## Example Files

- `examples/example_config.yml` — Basic configuration template
- `tests/fixtures/phase2_config.yml` — Advanced features showcase
- GUI: `File → Save Config` generates comprehensive templates
