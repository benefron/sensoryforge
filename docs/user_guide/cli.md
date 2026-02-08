# Command-Line Interface (CLI)

SensoryForge provides a comprehensive command-line interface for running simulations, validating configurations, and exploring available components.

## Installation

The CLI is automatically available after installing SensoryForge:

```bash
pip install sensoryforge
```

Or in development mode:

```bash
git clone https://github.com/benefron/sensoryforge.git
cd sensoryforge
pip install -e .
```

The `sensoryforge` command will be available in your terminal.

## Available Commands

### `sensoryforge run`

Run a simulation from a YAML configuration file.

**Usage:**
```bash
sensoryforge run <config.yml> [options]
```

**Options:**
- `--duration DURATION`: Simulation duration in milliseconds (default: 1000)
- `--output FILE`: Save results to file (PyTorch checkpoint `.pt` or `.pth`)
- `--device DEVICE`: Override device from config (`cpu`, `cuda`, or `mps`)

**Examples:**

Run a basic simulation:
```bash
sensoryforge run my_config.yml
```

Run with custom duration and save results:
```bash
sensoryforge run my_config.yml --duration 2000 --output results.pt
```

Override device to use GPU:
```bash
sensoryforge run my_config.yml --device cuda
```

**Output:**

Without `--output`, prints spike counts to console:
```
Loading pipeline from my_config.yml...
Running simulation (duration: 1000ms)...

Simulation completed successfully!
SA spikes: 243
RA spikes: 187
SA2 spikes: 89
```

With `--output`, saves a PyTorch checkpoint containing:
- `config`: Full configuration used
- `results`: All pipeline outputs (spikes, voltages, filtered signals, etc.)
- `pipeline_info`: Grid properties and neuron counts

### `sensoryforge validate`

Validate a YAML configuration file without running the simulation.

**Usage:**
```bash
sensoryforge validate <config.yml>
```

**Examples:**

Validate configuration:
```bash
sensoryforge validate my_config.yml
```

**Output:**

Success:
```
Validating my_config.yml...
✓ Configuration is valid!

Pipeline info:
  Device: cpu
  Grid size: (80, 80)
  SA neurons: 100
  RA neurons: 196
```

Failure:
```
Validation error: Invalid grid type: bad_type
❌ Configuration validation failed: my_config.yml
```

### `sensoryforge list-components`

List all available components (filters, neurons, stimuli, solvers).

**Usage:**
```bash
sensoryforge list-components
```

**Output:**
```
Available SensoryForge Components:
==================================================

📊 Filters:
  - SA (Slowly Adapting)
  - RA (Rapidly Adapting)
  - center_surround (for vision)

🧠 Neuron Models:
  - izhikevich (hand-written)
  - adex (Adaptive Exponential)
  - mqif (Multi-Quadratic Integrate-and-Fire)
  - dsl (Equation DSL - custom models)

🎯 Stimuli:
  - gaussian (Static Gaussian blob)
  - texture (Gabor, edge grating, perlin noise)
  - moving (Linear, circular motion)
  - trapezoidal (Ramp-plateau-ramp)
  - step (Step function)
  - ramp (Linear ramp)

⚙️  Solvers:
  - euler (Forward Euler - default)
  - adaptive (Adaptive stepping - requires torchdiffeq/torchode)

🌐 Grid Types:
  - standard (Single population)
  - composite (Multi-population mosaic)

💡 Use 'sensoryforge run --help' for usage examples
```

### `sensoryforge visualize`

Visualize pipeline structure from a configuration file.

**Usage:**
```bash
sensoryforge visualize <config.yml> [--save FILE]
```

**Options:**
- `--save FILE`: (Future) Save visualization to PNG file

**Examples:**

Display pipeline structure:
```bash
sensoryforge visualize my_config.yml
```

**Output:**
```
Loading pipeline from my_config.yml...

============================================================
PIPELINE STRUCTURE
============================================================

📍 Grid Configuration:
  Size: (80, 80)
  Spacing: 0.15 mm
  Bounds: X=(-6.0, 6.0), Y=(-6.0, 6.0)

🧠 Neuron Populations:
  SA neurons: 100
  RA neurons: 196
  SA2 neurons: 25

⚙️  Device: cpu

🎯 Configured Stimuli:
  1. gaussian
  2. texture
```

## Configuration File Format

SensoryForge uses YAML for configuration. See [YAML Configuration Guide](yaml_configuration.md) for details.

### Basic Example

```yaml
# Basic configuration
pipeline:
  device: cpu
  seed: 42
  grid_size: 80
  spacing: 0.15

neurons:
  sa_neurons: 100
  ra_neurons: 196
  dt: 0.5

stimuli:
  - type: gaussian
    config:
      sigma: 0.5
      amplitude: 30.0

solver:
  type: euler
  config:
    dt: 0.001
```

### Advanced Example (Phase 2 Features)

```yaml
# Phase 2 configuration with composite grid
metadata:
  name: "Multi-Population Simulation"
  version: "0.2"

grid:
  type: composite
  shape: [64, 64]
  populations:
    sa1:
      density: 100.0  # receptors per mm²
      arrangement: grid
      filter: SA
    ra1:
      density: 70.0
      arrangement: hex
      filter: RA

neurons:
  type: dsl  # Equation DSL
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

stimuli:
  - type: texture
    subtype: gabor
    config:
      wavelength: 0.5
      orientation: 0.0
  - type: moving
    motion_type: linear
    config:
      start: [0.0, 0.0]
      end: [2.0, 0.0]

solver:
  type: adaptive
  config:
    method: dopri5
    rtol: 1.0e-5
    atol: 1.0e-7
```

## Error Handling

The CLI provides informative error messages:

### File Not Found
```bash
$ sensoryforge run nonexistent.yml
❌ Config file not found: nonexistent.yml
```

### Invalid Configuration
```bash
$ sensoryforge validate bad_config.yml
Validation error: Invalid grid type: wrong_type
❌ Configuration validation failed: bad_config.yml
```

### Runtime Errors
```bash
$ sensoryforge run config.yml
Error running simulation: CUDA out of memory
[Traceback...]
```

## Tips and Best Practices

### 1. Validate Before Running

Always validate your configuration first:
```bash
sensoryforge validate config.yml && sensoryforge run config.yml
```

### 2. Start Small

Begin with small grid sizes and short durations for testing:
```yaml
pipeline:
  grid_size: 40  # Small for testing
  
temporal:
  t_plateau: 100  # Short duration
```

### 3. Use Version Control

Track your configuration files in git:
```bash
git add experiments/config_v1.yml
git commit -m "Add baseline configuration"
```

### 4. Organize Configurations

Keep configurations organized by experiment:
```
experiments/
  baseline/
    config.yml
    results.pt
  high_frequency/
    config.yml
    results.pt
  composite_grid/
    config.yml
    results.pt
```

### 5. Save Results

Always save important results:
```bash
sensoryforge run config.yml --output results_$(date +%Y%m%d).pt
```

## Integration with Python

The CLI uses the same Python API available for scripting:

```python
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.cli import load_config_file

# Load config from file
config = load_config_file('my_config.yml')

# Create pipeline
pipeline = GeneralizedTactileEncodingPipeline.from_config(config)

# Run simulation
results = pipeline.forward(stimulus_type='gaussian')
```

## See Also

- [YAML Configuration Guide](yaml_configuration.md) - Complete configuration reference
- [Pipeline Documentation](../api_reference/pipeline.md) - Python API reference
- [Tutorial: Your First Simulation](../tutorials/first_simulation.md) - Step-by-step guide
- [Phase 2 Features](overview.md) - CompositeGrid, DSL, extended stimuli, adaptive solvers
