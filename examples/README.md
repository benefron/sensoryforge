# SensoryForge Examples

This directory contains example configurations, scripts, and notebooks demonstrating SensoryForge usage.

## Quick Start

### Using the CLI

```bash
# Validate a configuration
sensoryforge validate example_config.yml

# Run a simulation
sensoryforge run example_config.yml --duration 1000

# Save results
sensoryforge run example_config.yml --output results.pt

# List available components
sensoryforge list-components

# Visualize pipeline structure
sensoryforge visualize example_config.yml
```

### Using Python API

```python
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Load from YAML file
pipeline = GeneralizedTactileEncodingPipeline(config_path='example_config.yml')

# Or from config dict
config = {
    'pipeline': {'device': 'cpu', 'grid_size': 80},
    'neurons': {'sa_neurons': 100, 'ra_neurons': 196}
}
pipeline = GeneralizedTactileEncodingPipeline.from_config(config)

# Run simulation
results = pipeline.forward(stimulus_type='gaussian', amplitude=30.0)

# Access results
print(f"SA spikes: {results['sa_spikes'].sum().item()}")
print(f"RA spikes: {results['ra_spikes'].sum().item()}")
```

## Configuration Files

### `example_config.yml`

Basic configuration demonstrating standard pipeline usage with:
- Standard grid (80x80 receptors)
- SA, RA, and SA2 neuron populations
- Gaussian stimulus
- Izhikevich neuron model
- Euler solver

**Usage:**
```bash
sensoryforge run example_config.yml --duration 1000 --output results.pt
```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks with interactive examples:

- **Notebook examples coming soon**

## Scripts

The `scripts/` directory contains Python scripts for common tasks:

- **Script examples coming soon**

## Phase 2 Features

SensoryForge supports advanced Phase 2 features via YAML configuration:

### CompositeGrid (Multi-Population)

```yaml
grid:
  type: composite
  shape: [64, 64]
  populations:
    sa1:
      density: 100.0  # receptors per mm²
      arrangement: grid
    ra1:
      density: 70.0
      arrangement: hex
```

### Equation DSL (Custom Neurons)

```yaml
neurons:
  type: dsl
  equations: |
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms
    du/dt = (a * (b*v - u)) / ms
  threshold: "v >= 30 * mV"
  reset: |
    v = c
    u = u + d
  parameters:
    a: 0.02
    b: 0.2
    c: -65.0
    d: 8.0
```

### Extended Stimuli

```yaml
stimuli:
  # Texture stimulus
  - type: texture
    subtype: gabor
    config:
      wavelength: 0.5
      orientation: 0.0
  
  # Moving stimulus
  - type: moving
    motion_type: linear
    config:
      start: [0.0, 0.0]
      end: [2.0, 0.0]
      num_steps: 100
```

### Adaptive Solvers

```yaml
solver:
  type: adaptive
  config:
    method: dopri5
    rtol: 1.0e-5
    atol: 1.0e-7
```

**Note:** Phase 2 feature integration is coming soon. See documentation for details.

## Documentation

- **CLI Guide:** `docs/user_guide/cli.md`
- **YAML Configuration:** `docs/user_guide/yaml_configuration.md`
- **CompositeGrid:** `docs/user_guide/composite_grid.md`
- **Equation DSL:** `docs/user_guide/equation_dsl.md`
- **Extended Stimuli:** `docs/user_guide/extended_stimuli.md`
- **Solvers:** `docs/user_guide/solvers.md`

## Tips

1. **Start simple:** Use `example_config.yml` as a template
2. **Validate first:** Always run `sensoryforge validate` before running
3. **Save results:** Use `--output` to save simulation results
4. **Track configs:** Use version control for your configuration files
5. **Experiment:** Modify parameters and observe effects on encoding

## Getting Help

- Run `sensoryforge --help` for command overview
- Run `sensoryforge COMMAND --help` for command-specific help
- Check documentation in `docs/`
- Open an issue on GitHub

## Contributing Examples

Have a useful configuration or example? Please contribute!

1. Create a well-documented example
2. Test it thoroughly
3. Submit a pull request
4. Update this README

---

**SensoryForge** – Modular sensory encoding framework  
Version 0.2.0 | [GitHub](https://github.com/benefron/sensoryforge)
