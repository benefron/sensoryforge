# Interactive Quickstart Tutorial

This tutorial walks you through using SensoryForge interactively in a Python session or Jupyter notebook.

## Prerequisites

- SensoryForge installed
- Basic Python knowledge
- (Optional) Jupyter notebook environment

## Part 1: Basic Pipeline

### Step 1: Import and Create

```python
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Minimal configuration
pipeline = GeneralizedTactileEncodingPipeline.from_config({
    'pipeline': {'device': 'cpu', 'grid_size': 40},
    'neurons': {'sa_neurons': 50, 'ra_neurons': 80}
})

print(f"Pipeline ready on {pipeline.device}")
```

### Step 2: Run with Default Stimulus

```python
# Run with trapezoidal Gaussian stimulus (default)
results = pipeline.forward(stimulus_type='gaussian')

# Examine results
print(f"SA spikes: {results['sa_spikes'].sum()}")
print(f"RA spikes: {results['ra_spikes'].sum()}")
```

### Step 3: Modify Stimulus Parameters

```python
# Try different amplitudes
amplitudes = [10, 20, 30, 40, 50]

for amp in amplitudes:
    results = pipeline.forward(stimulus_type='gaussian', amplitude=amp)
    print(f"Amplitude {amp:2d}: SA={results['sa_spikes'].sum():4.0f}, "
          f"RA={results['ra_spikes'].sum():4.0f}")
```

## Part 2: Different Stimulus Types

### Gaussian Blob

```python
results = pipeline.forward(
    stimulus_type='gaussian',
    amplitude=30.0,
    sigma=0.7,
    center_x=0.5,
    center_y=-0.5
)
```

### Step Function

```python
results = pipeline.forward(
    stimulus_type='step',
    amplitude=25.0,
    step_time=50.0  # Step at 50ms
)
```

### Ramp

```python
results = pipeline.forward(
    stimulus_type='ramp',
    initial=10.0,
    final=40.0
)
```

## Part 3: Analyzing Results

### Extract Spike Timing

```python
sa_spikes = results['sa_spikes']  # [time_steps, num_neurons]

# Get spike times for neuron 0
spike_times = sa_spikes[:, 0].nonzero(as_tuple=True)[0] * 0.5  # Convert to ms
print(f"Neuron 0 spikes at: {spike_times[:10].tolist()} ms...")
```

### Compute Inter-Spike Intervals (ISI)

```python
import torch

def compute_isi(spikes, neuron_idx):
    """Compute inter-spike intervals for a neuron."""
    spike_times = spikes[:, neuron_idx].nonzero(as_tuple=True)[0]
    if len(spike_times) < 2:
        return torch.tensor([])
    return torch.diff(spike_times.float()) * 0.5  # Convert to ms

isi = compute_isi(sa_spikes, 0)
print(f"Inter-spike intervals (ms): {isi[:10].tolist()}")
print(f"Mean ISI: {isi.mean():.2f} ms")
```

### Population Activity Over Time

```python
# Compute population activity (sum over neurons)
sa_activity = sa_spikes.sum(dim=1)  # Spikes per time step
ra_activity = ra_spikes.sum(dim=1)

# Smooth with moving average
import torch.nn.functional as F

window_size = 20
sa_smoothed = F.avg_pool1d(
    sa_activity.unsqueeze(0).unsqueeze(0).float(), 
    window_size, stride=1, padding=window_size//2
).squeeze()

ra_smoothed = F.avg_pool1d(
    ra_activity.unsqueeze(0).unsqueeze(0).float(), 
    window_size, stride=1, padding=window_size//2
).squeeze()
```

## Part 4: Visualization

### Simple Raster Plot

```python
import matplotlib.pyplot as plt

def plot_raster(spikes, title='Spike Raster', color='navy'):
    times, neurons = spikes.nonzero(as_tuple=True)
    plt.figure(figsize=(12, 4))
    plt.scatter(times.cpu() * 0.5, neurons.cpu(), s=1, c=color, alpha=0.6)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron #')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()

plot_raster(sa_spikes, 'SA Population')
plot_raster(ra_spikes, 'RA Population', color='darkred')
```

### Heatmap of Activity

```python
def plot_activity_heatmap(spikes, bin_size_ms=10):
    bin_size_steps = int(bin_size_ms / 0.5)
    num_bins = spikes.shape[0] // bin_size_steps
    
    # Reshape into bins
    binned = spikes[:num_bins * bin_size_steps].reshape(num_bins, bin_size_steps, -1)
    binned = binned.sum(dim=1)  # [num_bins, num_neurons]
    
    plt.figure(figsize=(14, 6))
    plt.imshow(binned.T.cpu(), aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Spikes per bin')
    plt.xlabel(f'Time bin ({bin_size_ms}ms)')
    plt.ylabel('Neuron #')
    plt.title('Population Activity Heatmap')
    plt.show()

plot_activity_heatmap(sa_spikes)
```

## Part 5: Parameter Exploration

### Grid Search

```python
import itertools

# Define parameter ranges
amplitudes = [20, 30, 40]
sigmas = [0.3, 0.5, 0.7]

results_grid = []

for amp, sig in itertools.product(amplitudes, sigmas):
    results = pipeline.forward(stimulus_type='gaussian', amplitude=amp, sigma=sig)
    sa_count = results['sa_spikes'].sum().item()
    ra_count = results['ra_spikes'].sum().item()
    
    results_grid.append({
        'amplitude': amp,
        'sigma': sig,
        'sa_spikes': sa_count,
        'ra_spikes': ra_count
    })

# Visualize results
import pandas as pd

df = pd.DataFrame(results_grid)
pivot = df.pivot(index='sigma', columns='amplitude', values='sa_spikes')

plt.figure(figsize=(8, 6))
import seaborn as sns
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis')
plt.title('SA Spike Count vs Parameters')
plt.ylabel('Sigma (mm)')
plt.xlabel('Amplitude')
plt.show()
```

## Part 6: Saving and Loading

### Save Pipeline Configuration

```python
import yaml

# Get pipeline config
config = pipeline.config

# Save to YAML
with open('my_pipeline_config.yml', 'w') as f:
    yaml.dump(config, f)

print("✓ Configuration saved")
```

### Save Results

```python
import torch

# Save all results
torch.save(results, 'simulation_results.pt')

# Load later
loaded_results = torch.load('simulation_results.pt')
print(f"Loaded results with {loaded_results['sa_spikes'].shape[0]} time steps")
```

## Part 7: Advanced - Custom Analysis

### Tuning Curves

```python
def compute_tuning_curve(pipeline, parameter_name, parameter_values):
    """Compute average spike count vs parameter."""
    spike_counts = []
    
    for value in parameter_values:
        results = pipeline.forward(stimulus_type='gaussian', **{parameter_name: value})
        spike_counts.append(results['sa_spikes'].sum().item())
    
    return torch.tensor(spike_counts)

# Compute amplitude tuning
amplitudes = torch.linspace(5, 50, 20)
tuning = compute_tuning_curve(pipeline, 'amplitude', amplitudes)

plt.figure(figsize=(10, 5))
plt.plot(amplitudes, tuning, 'o-', linewidth=2)
plt.xlabel('Amplitude')
plt.ylabel('Total SA Spikes')
plt.title('Amplitude Tuning Curve')
plt.grid()
plt.show()
```

### Cross-Population Correlation

```python
def population_correlation(sa_spikes, ra_spikes, bin_size_ms=10):
    """Compute correlation between SA and RA population activities."""
    bin_size_steps = int(bin_size_ms / 0.5)
    
    sa_binned = sa_spikes.sum(dim=1).unfold(0, bin_size_steps, bin_size_steps).sum(dim=1)
    ra_binned = ra_spikes.sum(dim=1).unfold(0, bin_size_steps, bin_size_steps).sum(dim=1)
    
    correlation = torch.corrcoef(torch.stack([sa_binned, ra_binned]))[0, 1]
    return correlation.item()

corr = population_correlation(sa_spikes, ra_spikes)
print(f"SA-RA correlation: {corr:.3f}")
```

## Next Steps

- [Building a Custom Pipeline](custom_pipeline.md) — Advanced configuration
- [Batch Processing Tutorial](batch_processing_tutorial.md) — Large-scale runs
- [Custom Neuron Models](custom_neurons.md) — Define your own models
- [User Guide](../user_guide/overview.md) — Complete reference
