# Configuration Schema Reference

Complete reference for the canonical `SensoryForgeConfig` schema used by SensoryForge.

## Overview

The canonical configuration schema (`SensoryForgeConfig`) provides a unified, extensible format that ensures:
- **GUI-CLI Parity**: Configurations saved from GUI work seamlessly with CLI
- **N-Population Support**: Add any number of populations (not limited to SA/RA/SA2)
- **Round-Trip Fidelity**: Save → Load → Same results
- **Extensibility**: Easy to add new population types and configurations

## Schema Structure

```yaml
grids:          # List of grid layers
populations:   # List of neuron populations (N populations supported)
stimulus:      # Stimulus configuration
simulation:    # Simulation settings
```

## GridConfig

Configuration for a single receptor grid layer.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | **Required** | Unique identifier for this grid layer |
| `arrangement` | string | `"grid"` | Grid arrangement: `"grid"`, `"poisson"`, `"hex"`, `"jittered_grid"`, `"blue_noise"` |
| `rows` | int | `None` | Number of rows (for grid arrangement) |
| `cols` | int | `None` | Number of columns (for grid arrangement) |
| `spacing` | float | `0.15` | Spacing between receptors in mm |
| `density` | float | `None` | Receptor density in receptors/mm² (for Poisson/hex) |
| `center_x` | float | `0.0` | X-coordinate of grid center in mm |
| `center_y` | float | `0.0` | Y-coordinate of grid center in mm |
| `color` | list[int] | `[66, 135, 245, 200]` | RGBA color tuple [r, g, b, a] for visualization |
| `visible` | bool | `True` | Whether this grid layer is visible in the GUI |

### Example

```yaml
grids:
  - name: "Main Receptor Grid"
    arrangement: "grid"
    rows: 80
    cols: 80
    spacing: 0.15
    center_x: 0.0
    center_y: 0.0
  
  - name: "Secondary Grid"
    arrangement: "poisson"
    rows: 40
    cols: 40
    spacing: 0.15
    density: 10.0
    center_x: 5.0
    center_y: 5.0
```

## PopulationConfig

Configuration for a single neuron population. Supports N populations with per-population configuration.

### Fields

#### Basic Identification

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | **Required** | Unique identifier for this population |
| `neuron_type` | string | `"SA"` | Type identifier: `"SA"`, `"RA"`, `"SA2"`, or custom |
| `target_grid` | string | `None` | Name of the grid layer this population connects to (uses first grid if None) |

#### Innervation Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `innervation_method` | string | `"gaussian"` | Builder: `"gaussian"`, `"one_to_one"`, `"uniform"`, `"distance_weighted"`, `"template"`, `"imported"`, or a registered plugin's name (see [Receptive Fields](receptive_fields.md)) |
| `resolvable_distance_mm` | float | `null` | `d` for the `template` builder: sigma = d/π, lattice pitch = d, neuron count derived (`neurons_per_row` ignored) |
| `innervation_params` | dict | `{}` | Extra builder parameters merged last (`template`: `k`, `normalize`, `weight_scale`, `edge_offset_mm`; `imported`: `path`; plugins: anything) |
| `connections_per_neuron` | int | `28` | Number of receptor connections per neuron |
| `sigma_d_mm` | float | `0.3` | Gaussian spread in mm (for gaussian method) |
| `distance_weight_randomness_pct` | float | `0.0` | Randomness percentage (0-100) for distance weighting |
| `use_distance_weights` | bool | `True` | Whether to use distance-based weighting |
| `far_connection_fraction` | float | `0.0` | Fraction of "far" connections (0-1) |
| `far_sigma_factor` | float | `5.0` | Sigma multiplier for far connections |
| `max_distance_mm` | float | `1.0` | Maximum connection distance in mm |
| `decay_function` | string | `"exponential"` | Distance decay function: `"exponential"`, `"linear"` |
| `decay_rate` | float | `2.0` | Decay rate parameter |
| `weight_range` | list[float] | `[0.05, 1.0]` | [min, max] weight range |
| `edge_offset` | float | `0.0` | Edge offset in mm (auto-set to spacing by default) |
| `target_layers` | list[string] | `None` | For a `"composite"` grid, the named layer subset this population/input reads (`None` = every layer) |

#### Multi-Input Populations (Phase 2, Wave M1/M2)

A population may read from more than one grid/channel, with an optional processing stage per
input. `target_grid`, `target_layers`, `innervation_method`, `sigma_d_mm`,
`connections_per_neuron`, `use_distance_weights`, `resolvable_distance_mm` and
`innervation_params` (the fields above) are **sugar** for the common single-input case:
`PopulationConfig.effective_inputs()` expands them into one implicit `PopulationInput`, and
`to_dict()` collapses a single sugar-shaped input back to those same fields, so an existing
single-input config is unaffected byte for byte. Setting `inputs` *and* any of the sugar fields
on the same population raises `ValueError`. See [Populations and Inputs](../concepts/populations_and_inputs.md).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `inputs` | list[`PopulationInput`] | `[]` | Explicit multi-input form; each entry is its own grid/channel/receptive-field builder/processing stage |
| `combine` | string | `"sum"` | How the per-input drives become one population drive: `"sum"` (element-wise, every input must produce the same neuron count `N`) or `"concat"` (neuron axis grows to `sum(N_i)`, one block per input) |

**`PopulationInput` fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `grid` | string | **Required** | Name of the grid this input samples |
| `channel` | string | `"value"` | Named channel/plane within that grid (`GridConfig.channels`) |
| `rf` | `RFBuilderConfig` | `{method: "gaussian"}` | Receptive-field builder for this input |
| `gain` | float | `1.0` | Multiplier applied to this input's drive before combining |
| `layers` | list[string] | `None` | For a composite grid, the named layer subset this input samples |
| `processing` | list[dict] | `[]` | Ordered processing-layer specs, `{method: <PROCESSING_REGISTRY name>, params: {...}}`, applied before the receptive-field bank (empty = no processing stage at all, see [Adding a Processing Layer](../extending/add_processing_layer.md)) |

**`RFBuilderConfig` fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | string | `"gaussian"` | Registered innervation/RF builder name |
| `params` | dict | `{}` | Builder parameters, merged the same way `innervation_params` is |

```yaml
populations:
  - name: "RG OnOff Population"
    neuron_type: SA
    neurons_per_row: 8
    inputs:
      - grid: "RGB Grid"
        channel: "R"
        processing:
          - method: onoff
      - grid: "RGB Grid"
        channel: "G"
        processing:
          - method: onoff
    combine: sum
```

#### Neuron Layout

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `neuron_arrangement` | string | `"grid"` | Arrangement: `"grid"`, `"poisson"`, `"hex"`, `"jittered_grid"`, `"blue_noise"` |
| `neurons_per_row` | int | `10` | Neurons per row (for grid arrangement) |
| `neuron_rows` | int | `None` | Number of rows (independent of neurons_per_row) |
| `neuron_cols` | int | `None` | Number of columns (independent of neurons_per_row) |
| `neuron_jitter_factor` | float | `0.0` | Jitter amount for jittered arrangements |

#### Neuron Model

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `neuron_model` | string | `"Izhikevich"` | Model type: `"Izhikevich"`, `"AdEx"`, `"MQIF"`, `"FA"`, `"SA"`, `"DSL"` |
| `model_params` | dict | `{}` | Model-specific parameters dict |
| `dsl_config` | dict | `None` | DSL configuration (equations, threshold, reset, parameters) |

#### Filter Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `filter_method` | string | `"none"` | Filter type: `"SA"`, `"RA"`, `"none"` |
| `filter_params` | dict | `{}` | Filter-specific parameters dict |

**Filter Parameters Examples:**

For SA filter:
```yaml
filter_params:
  tau_r: 5.0    # Rise time constant (ms)
  tau_d: 30.0   # Decay time constant (ms)
  k1: 0.05      # Gain parameter 1
  k2: 3.0       # Gain parameter 2
```

For RA filter:
```yaml
filter_params:
  tau_RA: 8.0  # RA time constant (ms)
  k3: 2.0       # RA gain parameter
```

#### Solver Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `solver_config` | dict | `None` | Solver configuration dict |

**Solver Config Example:**
```yaml
solver_config:
  type: "euler"  # or "adaptive"
  method: "dopri5"  # for adaptive
  rtol: 1.0e-5
  atol: 1.0e-7
```

#### Noise Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `noise_std` | float | `0.0` | Membrane noise standard deviation |
| `noise_mean` | float | `0.0` | Membrane noise mean |
| `noise_seed` | int | `None` | Random seed for noise |

#### Visualization

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `color` | list[int] | `[66, 135, 245, 255]` | RGBA color tuple [r, g, b, a] |
| `visible` | bool | `True` | Whether this population is visible in the GUI |

#### Simulation Control

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `True` | Whether this population is enabled for simulation |
| `input_gain` | float | `1.0` | Input gain multiplier |
| `seed` | int | `None` | Random seed for innervation generation |

### Example

```yaml
populations:
  - name: "SA Population"
    neuron_type: "SA"
    target_grid: "Main Receptor Grid"
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
    noise_std: 3.0
  
  - name: "RA Population"
    neuron_type: "RA"
    target_grid: "Main Receptor Grid"
    neuron_model: "izhikevich"
    filter_method: "ra"
    innervation_method: "gaussian"
    neurons_per_row: 14
    connections_per_neuron: 28
    sigma_d_mm: 0.39
    filter_params:
      tau_RA: 8.0
      k3: 2.0
  
  - name: "Custom DSL Population"
    neuron_type: "Custom"
    neuron_model: "dsl"
    filter_method: "none"
    innervation_method: "gaussian"
    neurons_per_row: 8
    dsl_config:
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

## StimulusConfig

Configuration for stimulus generation.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"Stimulus"` | Stimulus name/identifier |
| `type` | string | `"gaussian"` | Stimulus type: `"gaussian"`, `"texture"`, `"moving"`, `"timeline"`, `"repeated_pattern"` |
| `motion` | string | `"static"` | Motion type: `"static"`, `"moving"` |
| `amplitude` | float | `10.0` | Stimulus amplitude |
| `sigma` | float | `1.0` | Gaussian sigma (spatial spread) in mm |
| `center_x` | float | `0.0` | X-coordinate of center in mm |
| `center_y` | float | `0.0` | Y-coordinate of center in mm |
| `wavelength` | float | `0.5` | Wavelength for texture patterns in mm |
| `orientation_deg` | float | `0.0` | Orientation in degrees |
| `phase` | float | `0.0` | Phase offset |
| `motion_type` | string | `"linear"` | Motion type: `"linear"`, `"circular"` |
| `start` | list[float] | `[0.0, 0.0]` | Start position [x, y] in mm |
| `end` | list[float] | `[0.0, 0.0]` | End position [x, y] in mm |
| `duration` | float | `1000.0` | Duration in ms |

### Example

```yaml
stimulus:
  name: "Gaussian Stimulus"
  type: "gaussian"
  amplitude: 30.0
  sigma: 0.5
  center_x: 0.0
  center_y: 0.0
  duration: 1000.0
```

## SimulationConfig

Configuration for simulation settings.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device` | string | `"cpu"` | Device: `"cpu"`, `"cuda"`, `"mps"` |
| `dt` | float | `1.0` | Time step in ms |
| `duration` | float | `1000.0` | Simulation duration in ms |
| `seed` | int | `None` | Random seed (None for random) |
| `solver_config` | dict | `None` | Global solver configuration (overrides per-population) |

### Example

```yaml
simulation:
  device: "cpu"
  dt: 0.5
  duration: 1000.0
  seed: 42
```

## Complete Example

```yaml
grids:
  - name: "Main Receptor Grid"
    arrangement: "grid"
    rows: 80
    cols: 80
    spacing: 0.15
    center_x: 0.0
    center_y: 0.0

populations:
  - name: "SA Population"
    neuron_type: "SA"
    target_grid: "Main Receptor Grid"
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
    target_grid: "Main Receptor Grid"
    neuron_model: "izhikevich"
    filter_method: "ra"
    innervation_method: "gaussian"
    neurons_per_row: 14
    connections_per_neuron: 28
    sigma_d_mm: 0.39
    filter_params:
      tau_RA: 8.0
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

## Python API Usage

### Creating Config Programmatically

```python
from sensoryforge.config.schema import (
    SensoryForgeConfig,
    GridConfig,
    PopulationConfig,
    StimulusConfig,
    SimulationConfig,
)

config = SensoryForgeConfig(
    grids=[
        GridConfig(
            name="Main Grid",
            arrangement="grid",
            rows=80,
            cols=80,
            spacing=0.15,
        )
    ],
    populations=[
        PopulationConfig(
            name="SA Population",
            neuron_type="SA",
            neuron_model="izhikevich",
            filter_method="sa",
            innervation_method="gaussian",
            neurons_per_row=10,
        ),
        PopulationConfig(
            name="RA Population",
            neuron_type="RA",
            neuron_model="izhikevich",
            filter_method="ra",
            innervation_method="gaussian",
            neurons_per_row=14,
        ),
    ],
    stimulus=StimulusConfig(
        type="gaussian",
        amplitude=30.0,
        sigma=0.5,
    ),
    simulation=SimulationConfig(
        device="cpu",
        dt_ms=0.5,
    ),
)

# Save to YAML
with open('config.yml', 'w') as f:
    f.write(config.to_yaml())

# Load from YAML
config2 = SensoryForgeConfig.from_yaml('config.yml')

# Convert to dict for pipeline
pipeline_config = config2.to_dict()
```

### Round-Trip Fidelity

The canonical schema ensures perfect round-trip fidelity:

```python
# Save
config = SensoryForgeConfig(...)
yaml_str = config.to_yaml()

# Load
config2 = SensoryForgeConfig.from_yaml(yaml_str)

# Verify
assert config.grids[0].name == config2.grids[0].name
assert config.populations[0].name == config2.populations[0].name
```

## Relationship to Legacy Format

The canonical format is automatically converted to legacy format via an adapter layer in `GeneralizedTactileEncodingPipeline`. This ensures backward compatibility:

- **Legacy format** is still fully supported
- **Canonical format** is recommended for new projects
- **GUI exports** canonical format
- **CLI accepts** both formats

See [YAML Configuration Guide](yaml_configuration.md) for migration guide.

## Validation

Validate your canonical config:

```python
from sensoryforge.config.schema import SensoryForgeConfig

try:
    config = SensoryForgeConfig.from_yaml('config.yml')
    print("✓ Configuration is valid!")
except Exception as e:
    print(f"✗ Validation error: {e}")
```

Or use CLI:
```bash
sensoryforge validate config.yml
```

## See Also

- [YAML Configuration Guide](yaml_configuration.md) - Complete configuration reference with examples
- [CLI Guide](cli.md) - Command-line usage
- [GUI Workflow](gui_phase2_access.md) - GUI design → CLI scale workflow
- [Extensibility Guide](../developer_guide/extensibility.md) - Adding custom components
