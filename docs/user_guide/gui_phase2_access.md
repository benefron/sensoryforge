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
- **Load Config**: `File → Load Config (YAML)` validates and previews any configuration
- **Save Template**: `File → Save Config (YAML)` exports a comprehensive YAML template
- **Help → Advanced Features**: Overview of all available features
- **Help → CLI Guide**: Complete CLI reference with examples

## CLI: Scalable Execution

Once a configuration is designed and validated in the GUI, export to YAML and use the CLI for scale:

```bash
# Validate configuration
sensoryforge validate my_config.yml

# Run simulation
sensoryforge run my_config.yml --duration 1000 --output results.pt

# Run on GPU
sensoryforge run my_config.yml --device cuda --duration 5000

# List available components
sensoryforge list-components
```

### Example YAML Configuration

```yaml
pipeline:
  device: cpu
  seed: 42
  grid_size: 80
  spacing: 0.15

neurons:
  sa_neurons: 100
  ra_neurons: 196

stimuli:
  - type: gaussian
    config: {amplitude: 30.0, sigma: 1.0}

solver:
  type: euler
```

### Advanced YAML Features

```yaml
# Multi-population grid
grid:
  type: composite
  populations:
    sa1: {density: 10.0, arrangement: poisson}
    ra1: {density: 5.0, arrangement: hex}
    sa2: {density: 3.0, arrangement: poisson}

# Equation DSL neuron model
neurons:
  type: dsl
  equations: |
    dv/dt = 0.04*v**2 + 5*v + 140 - u + I
    du/dt = a*(b*v - u)
  threshold: "v >= 30"
  reset: "v = c; u = u + d"
  parameters: {a: 0.02, b: 0.2, c: -65.0, d: 8.0}

# Adaptive solver for stiff systems
solver:
  type: adaptive
  config: {method: dopri5, rtol: 1e-5}
```

## Python API

```python
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Load from YAML
pipeline = GeneralizedTactileEncodingPipeline.from_yaml('config.yml')
results = pipeline.forward(duration_ms=1000)

# Or configure programmatically
pipeline = GeneralizedTactileEncodingPipeline.from_config({
    'pipeline': {'device': 'cpu', 'grid_size': 80},
    'neurons': {'sa_neurons': 100, 'ra_neurons': 196},
})
```

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
