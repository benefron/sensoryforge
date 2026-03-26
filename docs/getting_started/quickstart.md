# Quick Start

Get up and running with SensoryForge in minutes!

## 5-Minute Introduction

### 1. Import and Create a Pipeline

**Using Canonical Configuration (Recommended):**

```python
from sensoryforge.config.schema import SensoryForgeConfig, GridConfig, PopulationConfig, StimulusConfig, SimulationConfig
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Create canonical config
config = SensoryForgeConfig(
    grids=[
        GridConfig(
            name="Main Grid",
            arrangement="grid",
            rows=80,
            cols=80,
            spacing=0.15,  # 0.15mm between receptors
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
            connections_per_neuron=28,
            sigma_d_mm=0.3,
        ),
        PopulationConfig(
            name="RA Population",
            neuron_type="RA",
            neuron_model="izhikevich",
            filter_method="ra",
            innervation_method="gaussian",
            neurons_per_row=14,
            connections_per_neuron=28,
            sigma_d_mm=0.39,
        ),
    ],
    stimulus=StimulusConfig(
        type="gaussian",
        amplitude=30.0,
        sigma=0.5,
    ),
    simulation=SimulationConfig(
        device="cpu",  # or 'cuda' or 'mps'
        dt=0.5,  # 0.5ms time step
    ),
)

# Create pipeline from canonical config
pipeline = GeneralizedTactileEncodingPipeline.from_config(config.to_dict())
```

**Using Legacy Configuration (Backward Compatible):**

```python
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Legacy format still works
config = {
    'pipeline': {
        'device': 'cpu',
        'grid_size': 80,
        'spacing': 0.15,
    },
    'neurons': {
        'sa_neurons': 100,
        'ra_neurons': 196,
        'dt': 0.5,
    }
}

pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
```

### 2. Run a Simulation

```python
#Execute with a Gaussian stimulus
results = pipeline.forward(
    stimulus_type='gaussian',
    amplitude=30.0,  # Peak pressure
    sigma=0.5,       # Spatial spread (mm)
)

# Extract results
sa_spikes = results['sa_spikes']  # [time_steps, num_sa_neurons]
ra_spikes = results['ra_spikes']  # [time_steps, num_ra_neurons]

print(f"SA population fired {sa_spikes.sum()} spikes")
print(f"RA population fired {ra_spikes.sum()} spikes")
```

### 3. Visualize (Optional)

```python
import matplotlib.pyplot as plt

# Plot raster
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# SA spikes
times, neurons = sa_spikes.nonzero(as_tuple=True)
ax1.scatter(times.cpu() * 0.5, neurons.cpu(), s=1, c='navy', alpha=0.6)
ax1.set_ylabel('SA Neuron #')
ax1.set_title(f'SA Population Activity ({sa_spikes.sum()} spikes)')

# RA spikes
times, neurons = ra_spikes.nonzero(as_tuple=True)
ax2.scatter(times.cpu() * 0.5, neurons.cpu(), s=1, c='darkred', alpha=0.6)
ax2.set_ylabel('RA Neuron #')
ax2.set_xlabel('Time (ms)')
ax2.set_title(f'RA Population Activity ({ra_spikes.sum()} spikes)')

plt.tight_layout()
plt.show()
```

## Using the GUI

For interactive exploration, launch the graphical interface:

```bash
python -m sensoryforge.gui.main
```

The GUI provides:
- **Mechanoreceptor Tab**: Configure grid and neuron populations
- **Stimulus Tab**: Design stimulus patterns interactively  
- **Spiking Tab**: Visualize population responses in real-time
- **Protocol Suite Tab**: Run stimulus batteries for data generation

## Using the CLI

For reproducible batch processing:

```bash
# Validate a configuration
sensoryforge validate config.yml

# Run a simulation
sensoryforge run config.yml --output results.pt

# Run a parameter sweep batch
sensoryforge batch batch_config.yml --device cuda
```

## Configuration File Workflow

### Canonical Configuration Format (Recommended)

Create `my_config.yml` using the canonical schema:

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
  sigma: 0.5

simulation:
  device: "cpu"
  dt: 0.5
```

Then run:

```python
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

config = SensoryForgeConfig.from_yaml('my_config.yml')
pipeline = GeneralizedTactileEncodingPipeline.from_config(config.to_dict())
results = pipeline.forward(stimulus_type='gaussian')
```

### Legacy Configuration Format (Still Supported)

Legacy format is backward compatible:

```yaml
pipeline:
  device: cpu
  grid_size: 80
  spacing: 0.15

neurons:
  sa_neurons: 100
  ra_neurons: 196
  dt: 0.5

filters:
  sa_tau_r: 5.0
  sa_tau_d: 30.0
  ra_tau_ra: 15.0

stimuli:
  - type: gaussian
    config:
      sigma: 0.5
      amplitude: 30.0
```

```python
pipeline = GeneralizedTactileEncodingPipeline.from_yaml('legacy_config.yml')
results = pipeline.forward(stimulus_type='gaussian')
```

**Note**: The GUI exports configurations in canonical format. For new projects, use canonical format for better extensibility and N-population support.

## Common Workflows

### Desktop Prototyping

1. **GUI** → Design experiment interactively
2. **Save Config** → Export canonical YAML configuration
3. **CLI** → Run at scale on compute cluster (configs are compatible)

The GUI exports configurations in canonical format (`SensoryForgeConfig`), ensuring seamless round-trip workflow: GUI → YAML → CLI → same results.

### Python Scripting

```python
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Load pipeline
pipeline = GeneralizedTactileEncodingPipeline.from_yaml('config.yml')

# Run multiple stimuli
amplitudes = [10, 20, 30, 40, 50]
results_list = []

for amp in amplitudes:
    results = pipeline.forward(stimulus_type='gaussian', amplitude=amp)
    results_list.append(results)

# Analyze results...
```

### ML Training Dataset Generation

```bash
# Create batch configuration
sensoryforge batch training_data.yml --device cuda

# Load results for training
```

```python
import torch
from torch.utils.data import DataLoader

# Load batch results
data = torch.load('batch_results/training_data.pt')
loader = DataLoader(data['results'], batch_size=32, shuffle=True)

# Train your model
for batch in loader:
    spikes = torch.cat([batch['sa_spikes'], batch['ra_spikes']], dim=2)
    # ... your training loop
```

## Next Steps

- [Core Concepts](concepts.md) — Understand the architecture
- [First Simulation](first_simulation.md) — Detailed walkthrough
- [User Guide](../user_guide/overview.md) — Complete reference
- [Tutorials](../tutorials/quickstart_tutorial.md) — Hands-on examples
