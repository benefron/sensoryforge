# YAML Configuration Guide

SensoryForge uses YAML files for declarative pipeline configuration. This guide covers both the **canonical configuration schema** (recommended) and the **legacy format** (backward compatible).

## Configuration Formats

SensoryForge supports two configuration formats:

1. **Canonical Schema** (`SensoryForgeConfig`) - Recommended for new projects
   - Unified format for GUI-CLI parity
   - Supports N populations (not limited to SA/RA/SA2)
   - Round-trip fidelity (save → load → same results)
   - Exported by GUI

2. **Legacy Format** - Backward compatible
   - Hardcoded SA/RA/SA2 structure
   - Still fully supported via adapter layer
   - Use for existing projects or compatibility

## Quick Start

### Canonical Format (Recommended)

```yaml
grids:
  - name: "Main Grid"
    arrangement: "grid"
    rows: 80
    cols: 80
    spacing: 0.15

populations:
  - name: "SA Population"
    neuron_type: "SA"
    neuron_model: "izhikevich"
    filter_method: "sa"
    innervation_method: "gaussian"
    neurons_per_row: 10
  - name: "RA Population"
    neuron_type: "RA"
    neuron_model: "izhikevich"
    filter_method: "ra"
    innervation_method: "gaussian"
    neurons_per_row: 14

stimulus:
  type: "gaussian"
  amplitude: 30.0

simulation:
  device: "cpu"
  dt: 0.5
```

### Legacy Format (Backward Compatible)

```yaml
pipeline:
  device: cpu
  grid_size: 80

neurons:
  sa_neurons: 100
  ra_neurons: 196
```

Run either format:
```bash
sensoryforge run config.yml
```

## Canonical Configuration Schema

The canonical schema (`SensoryForgeConfig`) provides a unified, extensible configuration format.

### Structure

```yaml
grids:          # List of grid layers
populations:   # List of neuron populations (N populations supported)
stimulus:      # Stimulus configuration
simulation:    # Simulation settings
```

### Complete Example

```yaml
grids:
  - name: "Receptor Grid"
    arrangement: "grid"
    rows: 80
    cols: 80
    spacing: 0.15
    center_x: 0.0
    center_y: 0.0

populations:
  - name: "SA Population"
    neuron_type: "SA"
    target_grid: "Receptor Grid"
    neuron_model: "izhikevich"
    filter_method: "sa"
    innervation_method: "gaussian"
    neurons_per_row: 10
    connections_per_neuron: 28
    sigma_d_mm: 0.3
    filter_params:
      tau_r: 5.0
      tau_d: 30.0
      k1: 0.05
      k2: 3.0
    model_params:
      a: 0.02
      b: 0.2
      c: -65.0
      d: 8.0
  
  - name: "RA Population"
    neuron_type: "RA"
    target_grid: "Receptor Grid"
    neuron_model: "izhikevich"
    filter_method: "ra"
    innervation_method: "gaussian"
    neurons_per_row: 14
    connections_per_neuron: 28
    sigma_d_mm: 0.39
    filter_params:
      tau_RA: 15.0
      k3: 2.0

stimulus:
  type: "gaussian"
  amplitude: 30.0
  sigma: 0.5
  center_x: 0.0
  center_y: 0.0

simulation:
  device: "cpu"
  dt: 0.5
  duration: 1000.0
```

### Key Advantages

- **N-Population Support**: Add as many populations as needed
- **Per-Population Configuration**: Each population has its own innervation, filter, neuron, and solver config
- **GUI-CLI Parity**: GUI exports canonical format, CLI accepts it
- **Extensibility**: Easy to add new population types, arrangements, etc.

See [Configuration Schema Reference](configuration_schema.md) for complete field documentation.

## Legacy Configuration Structure

A complete configuration file has the following top-level sections:

```yaml
metadata:        # (Optional) Simulation metadata
pipeline:        # Core pipeline settings
grid:           # (Optional) Grid configuration
neurons:        # Neuron population settings
stimuli:        # (Optional) Stimulus definitions
filters:        # (Optional) Temporal filter settings
neuron_params:  # (Optional) Neuron model parameters
solver:         # (Optional) ODE solver configuration
temporal:       # (Optional) Temporal profile settings
noise:          # (Optional) Noise settings
gui:            # (Optional) GUI defaults
```

## Core Sections

### `metadata` (Optional)

Simulation metadata for tracking experiments:

```yaml
metadata:
  name: "Texture Discrimination Experiment"
  version: "1.0"
  description: "Testing responses to Gabor patterns"
  author: "Research Team"
  date: "2024-02-08"
```

### `pipeline` (Required)

Core pipeline settings:

```yaml
pipeline:
  device: cpu           # 'cpu', 'cuda', or 'mps'
  seed: 42             # Random seed (null for random)
  grid_size: 80        # Grid dimension (creates 80x80 grid)
  spacing: 0.15        # Receptor spacing in mm
  center: [0.0, 0.0]  # Grid center coordinates
```

**Default values** (from `GeneralizedTactileEncodingPipeline.DEFAULT_CONFIG`):
- `device`: `"cpu"`
- `seed`: `42`
- `grid_size`: `80`
- `spacing`: `0.15`
- `center`: `[0.0, 0.0]`

### `neurons` (Legacy Format)

Neuron population configuration (legacy format, limited to SA/RA/SA2):

```yaml
neurons:
  sa_neurons: 100   # Number of SA neurons
  ra_neurons: 196   # Number of RA neurons
  sa2_neurons: 25   # Number of SA2 neurons (optional)
  dt: 0.5          # Time step in ms
```

**Defaults:**
- `sa_neurons`: `10`
- `ra_neurons`: `14`
- `sa2_neurons`: `5`
- `dt`: `0.5`

**Note**: For N-population support, use canonical format with `populations` list.

## Grid Configuration

### Standard Grid (Single Population)

Default behavior - single receptor population:

```yaml
# Implicit standard grid (no 'grid' section needed)
pipeline:
  grid_size: 80
  spacing: 0.15
```

### Composite Grid (Multi-Population)

Multiple receptor populations with different densities and arrangements:

```yaml
grid:
  type: composite
  shape: [64, 64]
  xlim: [-5.0, 5.0]  # Spatial bounds in mm
  ylim: [-5.0, 5.0]
  populations:
    sa1:
      density: 100.0        # Receptors per mm²
      arrangement: grid     # 'grid', 'poisson', 'hex', 'jittered_grid'
    ra1:
      density: 70.0
      arrangement: hex
    sa2:
      density: 30.0
      arrangement: poisson
```

**Arrangement types:**
- `grid`: Regular rectangular lattice
- `poisson`: Random Poisson disk sampling
- `hex`: Hexagonal lattice (optimal packing)
- `jittered_grid`: Grid with random jitter

**Note:** CompositeGrid is fully supported via `CompositeReceptorGrid.add_layer()`.
The deprecated `filter` field is ignored; filter associations are configured
separately in the pipeline configuration.

## Stimuli

Define one or more stimuli for the simulation.

### Gaussian Stimulus

Static or dynamic Gaussian blob:

```yaml
stimuli:
  - type: gaussian
    config:
      sigma: 0.5        # Standard deviation in mm
      amplitude: 30.0   # Peak pressure amplitude
      center_x: 0.0     # X-coordinate of center
      center_y: 0.0     # Y-coordinate of center
      duration: 100.0   # Duration in ms (optional)
```

### Trapezoidal Stimulus

Ramp-plateau-ramp temporal profile:

```yaml
stimuli:
  - type: trapezoidal
    config:
      amplitude: 30.0
      sigma: 1.0
      center_x: 0.0
      center_y: 0.0
```

### Texture Stimuli - Phase 2

#### Gabor Pattern

Localized sinusoidal grating:

```yaml
stimuli:
  - type: texture
    subtype: gabor
    config:
      wavelength: 0.5      # Spatial wavelength in mm
      orientation: 0.0     # Orientation in radians
      sigma: 0.3          # Gaussian envelope std dev
      amplitude: 1.0
      phase: 0.0          # Phase offset (optional)
```

#### Edge Grating

Parallel edges for orientation tuning:

```yaml
stimuli:
  - type: texture
    subtype: edge_grating
    config:
      orientation: 0.0     # Radians
      spacing: 0.6        # Edge spacing in mm
      count: 5           # Number of edges
      edge_width: 0.05   # Edge sharpness
      amplitude: 1.0
```

### Moving Stimuli - Phase 2

#### Linear Motion

Stimulus moving along a straight path:

```yaml
stimuli:
  - type: moving
    motion_type: linear
    config:
      start: [0.0, 0.0]      # Starting position [x, y] in mm
      end: [2.0, 0.0]        # Ending position
      num_steps: 100         # Number of time steps
      stimulus_type: gaussian # Base stimulus to move
      stimulus_params:
        sigma: 0.3
        amplitude: 1.0
```

#### Circular Motion

Stimulus following a circular path:

```yaml
stimuli:
  - type: moving
    motion_type: circular
    config:
      center: [0.0, 0.0]     # Circle center
      radius: 1.0            # Circle radius in mm
      num_steps: 100
      start_angle: 0.0       # Starting angle (radians)
      end_angle: 6.28318     # Ending angle (2π for full circle)
      stimulus_type: gaussian
      stimulus_params:
        sigma: 0.3
```

## Filters

Temporal filtering parameters for each population:

```yaml
filters:
  # SA filter (slowly adapting)
  sa_tau_r: 5.0    # Rise time constant (ms)
  sa_tau_d: 30.0   # Decay time constant (ms)
  sa_k1: 0.05      # Gain parameter 1
  sa_k2: 3.0       # Gain parameter 2
  
  # RA filter (rapidly adapting)
  ra_tau_ra: 15.0  # RA time constant (ms)
  ra_k3: 2.0       # RA gain parameter
  
  # SA2 (simple scaling, no filter)
  sa2_scale: 0.005
```

**Defaults:** See `GeneralizedTactileEncodingPipeline.DEFAULT_CONFIG['filters']`

## Neuron Models

### Hand-Written Models (Default)

Standard neuron models (Izhikevich, AdEx, MQIF):

```yaml
neuron_params:
  # SA neuron parameters
  sa_a: 0.02
  sa_b: 0.2
  sa_c: -65.0
  sa_d: 8.0
  sa_v_init: -65.0
  sa_threshold: 30.0
  sa_a_std: 0.005      # Variability (optional)
  sa_threshold_std: 3.0
  
  # RA neuron parameters
  ra_a: 0.02
  ra_b: 0.2
  ra_c: -65.0
  ra_d: 8.0
```

### DSL Neurons (Equation-Based) - Phase 2

Define neuron models using mathematical equations:

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
  state_vars:
    v: -65.0  # Initial membrane potential
    u: -13.0  # Initial recovery variable
```

**Equation syntax:**
- Use `dv/dt` notation for differential equations
- Special variable `I` represents input current
- Parameters are substituted from `parameters` dict
- Support for standard math functions: `sin`, `cos`, `exp`, `log`, `sqrt`, etc.

**Note:** Requires `sympy` package: `pip install sensoryforge[dsl]`

## Solvers

ODE integration settings for neuron dynamics.

### Euler Solver (Default)

Simple forward Euler integration:

```yaml
solver:
  type: euler
  config:
    dt: 0.001  # Time step in seconds
```

### Adaptive Solver - Phase 2

Adaptive stepping with error control:

```yaml
solver:
  type: adaptive
  config:
    method: dopri5      # RK45, dopri5, adaptive_heun, etc.
    rtol: 1.0e-5       # Relative tolerance
    atol: 1.0e-7       # Absolute tolerance
    adjoint: false     # Use adjoint method for backprop (optional)
```

**Available methods:**
- `dopri5`: Dormand-Prince (5th order, adaptive)
- `rk4`: 4th-order Runge-Kutta
- `adaptive_heun`: 2nd-order adaptive
- `bosh3`: 3rd-order Bogacki-Shampine

**Requirements:** 
- `pip install torchdiffeq` or `pip install torchode`
- Or: `pip install sensoryforge[solvers]`

## Temporal Profile

Control stimulus temporal dynamics:

```yaml
temporal:
  t_pre: 25        # Pre-stimulus period (ms)
  t_ramp: 10       # Ramp-up duration (ms)
  t_plateau: 800   # Plateau duration (ms)
  t_post: 200      # Post-stimulus period (ms)
  dt: 0.5          # Time step (ms)
```

Total duration: `t_pre + t_ramp + t_plateau + t_ramp + t_post`

## Noise

Add noise to membrane potentials or receptor responses:

```yaml
noise:
  # Per-population membrane noise
  sa_membrane_std: 3.0
  sa_membrane_mean: 0.0
  sa_membrane_seed: 42
  
  ra_membrane_std: 3.0
  ra_membrane_mean: 0.0
  ra_membrane_seed: 43
  
  sa2_membrane_std: 3.0
  
  # Receptor noise (optional)
  use_receptor_noise: false
  receptor_std: 5.0
  receptor_seed: 123
```

## GUI Configuration

Default GUI settings:

```yaml
gui:
  auto_load_config: true
  default_stimulus_type: gaussian
  enable_visualization: true
  visualization_type: heatmap    # 'heatmap' or 'raster'
  default_grid_type: standard    # 'standard' or 'composite'
  default_neuron_type: izhikevich
  default_solver: euler
```

## Complete Example

Comprehensive configuration using all features:

```yaml
metadata:
  name: "Phase 2 Feature Showcase"
  version: "0.2"

pipeline:
  device: cpu
  seed: 42
  grid_size: 64
  spacing: 0.15
  center: [0.0, 0.0]

neurons:
  sa_neurons: 50
  ra_neurons: 70
  sa2_neurons: 20
  dt: 0.5

stimuli:
  - type: gaussian
    config:
      sigma: 0.5
      amplitude: 30.0
  - type: texture
    subtype: gabor
    config:
      wavelength: 0.5
      orientation: 0.785398  # 45 degrees
      sigma: 0.3

filters:
  sa_tau_r: 5.0
  sa_tau_d: 30.0
  ra_tau_ra: 15.0

neuron_params:
  sa_a: 0.02
  sa_b: 0.2
  sa_c: -65.0
  sa_d: 8.0
  ra_a: 0.02
  ra_b: 0.2
  ra_c: -65.0
  ra_d: 8.0

solver:
  type: euler
  config:
    dt: 0.001

temporal:
  t_pre: 25
  t_ramp: 10
  t_plateau: 500
  t_post: 100
  dt: 0.5

noise:
  sa_membrane_std: 3.0
  ra_membrane_std: 3.0

gui:
  auto_load_config: true
  default_stimulus_type: gaussian
  enable_visualization: true
```

## Validation

Validate your configuration before running:

```bash
sensoryforge validate config.yml
```

Common validation errors:
- Invalid grid type (must be 'standard' or 'composite')
- Invalid solver type (must be 'euler' or 'adaptive')
- Missing required DSL fields (equations, threshold, reset, parameters)
- Invalid arrangement type for composite grid

## Migration Guide: Legacy → Canonical

To migrate from legacy format to canonical format:

### Step 1: Convert Grid

**Legacy:**
```yaml
pipeline:
  grid_size: 80
  spacing: 0.15
  center: [0.0, 0.0]
```

**Canonical:**
```yaml
grids:
  - name: "Main Grid"
    arrangement: "grid"
    rows: 80
    cols: 80
    spacing: 0.15
    center_x: 0.0
    center_y: 0.0
```

### Step 2: Convert Populations

**Legacy:**
```yaml
neurons:
  sa_neurons: 100
  ra_neurons: 196
  sa2_neurons: 25
```

**Canonical:**
```yaml
populations:
  - name: "SA Population"
    neuron_type: "SA"
    neuron_model: "izhikevich"
    filter_method: "sa"
    innervation_method: "gaussian"
    neurons_per_row: 10  # Calculated from sa_neurons
  - name: "RA Population"
    neuron_type: "RA"
    neuron_model: "izhikevich"
    filter_method: "ra"
    innervation_method: "gaussian"
    neurons_per_row: 14  # Calculated from ra_neurons
  - name: "SA2 Population"
    neuron_type: "SA2"
    neuron_model: "izhikevich"
    filter_method: "none"
    innervation_method: "gaussian"
    neurons_per_row: 5  # Calculated from sa2_neurons
```

### Step 3: Convert Stimulus

**Legacy:**
```yaml
stimuli:
  - type: gaussian
    config:
      amplitude: 30.0
      sigma: 0.5
```

**Canonical:**
```yaml
stimulus:
  type: "gaussian"
  amplitude: 30.0
  sigma: 0.5
```

### Step 4: Convert Simulation Settings

**Legacy:**
```yaml
pipeline:
  device: cpu
```

**Canonical:**
```yaml
simulation:
  device: "cpu"
  dt: 0.5
```

**Tip**: The GUI can automatically convert legacy configs when you load them. Use `File → Load Config (YAML)` in the GUI to preview the canonical format.

## Best Practices

1. **Use canonical format for new projects**: Better extensibility and N-population support
2. **Start with defaults**: Use minimal config and override only what you need
3. **Version your configs**: Track configurations in version control
4. **Use meaningful names**: Clear population names help organize experiments
5. **Comment liberally**: YAML supports comments - use them!
6. **Validate first**: Always run `validate` before `run`
7. **Organize by experiment**: Keep related configs together
8. **GUI export**: Use GUI to export canonical configs for consistency

## See Also

- [Configuration Schema Reference](configuration_schema.md) - Complete canonical schema documentation
- [CLI Guide](cli.md) - Command-line usage
- [Pipeline API](../api_reference/pipeline.md) - Python API
- [CompositeGrid](composite_grid.md) - Multi-population grids
- [Equation DSL](equation_dsl.md) - Custom neuron models
- [Extended Stimuli](extended_stimuli.md) - Texture and moving stimuli
- [Solvers](solvers.md) - ODE integration methods
