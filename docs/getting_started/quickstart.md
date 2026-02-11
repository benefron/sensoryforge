# Quick Start

Get up and running with SensoryForge in minutes!

## 5-Minute Introduction

### 1. Import and Create a Pipeline

```python
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Create a simple pipeline
config = {
    'pipeline': {
        'device': 'cpu',  # or 'cuda' or 'mps'
        'grid_size': 80,  # 80×80 receptor grid
        'spacing': 0.15,  # 0.15mm between receptors
    },
    'neurons': {
        'sa_neurons': 100,   # Slowly adapting population
        'ra_neurons': 196,   # Rapidly adapting population
        'dt': 0.5,          # 0.5ms time step
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

Create `my_config.yml`:

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

Then run:

```python
pipeline = GeneralizedTactileEncodingPipeline.from_yaml('my_config.yml')
results = pipeline.forward(stimulus_type='gaussian')
```

## Common Workflows

### Desktop Prototyping

1. **GUI** → Design experiment interactively
2. **Save Config** → Export YAML configuration
3. **CLI** → Run at scale on compute cluster

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
