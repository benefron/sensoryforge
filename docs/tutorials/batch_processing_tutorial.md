# Batch Processing Tutorial

This tutorial demonstrates how to use SensoryForge's batch processing system to generate large-scale datasets for machine learning.

## What You'll Learn

- Configure batch processing via YAML
- Run parameter sweeps efficiently
- Resume interrupted runs
- Export data in PyTorch and HDF5 formats
- Analyze batch results

## Prerequisites

```bash
pip install sensoryforge h5py  # h5py for HDF5 format
```

## Part 1: Basic Batch Configuration

### Create Batch Configuration

Create `my_batch.yml`:

```yaml
# Batch Processing Configuration
batch:
  output_dir: "batch_output"  # Where to save results
  format: "pytorch"            # 'pytorch' or 'hdf5'
  checkpoint_frequency: 10     # Save progress every 10 stimuli
  num_workers: 4               # Parallel workers (use with care)

# Pipeline configuration
pipeline:
  device: "cpu"
  grid_size: 32
  duration_ms: 150.0
  dt_ms: 0.5

neurons:
  sa_neurons: 60
  ra_neurons: 80

# Single stimulus
stimuli:
  - type: "gaussian"
    amplitude: 30.0
    sigma: 0.5
    repetitions: 5  # Run this 5 times with different random seeds
```

### Run Batch

```bash
# Dry run first (validates config, shows what will run)
sensoryforge batch my_batch.yml --dry-run

# Actual execution
sensoryforge batch my_batch.yml
```

Output:
```
Batch execution: 5 stimuli total
Progress: 100%|████████████| 5/5 [00:12<00:00,  2.5s/stimulus]
✓ Results saved to batch_output/results.pt
```

## Part 2: Parameter Sweeps

### Example: Amplitude vs Sigma Sweep

Create `sweep_config.yml`:

```yaml
batch:
  output_dir: "amplitude_sigma_sweep"
  format: "pytorch"
  checkpoint_frequency: 20

pipeline:
  device: "cpu"
  grid_size: 40
  duration_ms: 200.0

neurons:
  sa_neurons: 80
  ra_neurons: 100

# Parameter sweep specification
stimuli:
  - type: "gaussian"
    amplitude: [10, 20, 30, 40, 50]  # 5 values
    sigma: [0.3, 0.5, 0.7, 0.9]       # 4 values
    center_x: 0.0
    center_y: 0.0
    repetitions: 3  # 3 reps/condition
    
# Total stimuli: 5 × 4 × 3 = 60
```

### Run Sweep

```bash
sensoryforge batch sweep_config.yml --dry-run

# Output shows:
# Grid expansion: amplitude=[10, 20, 30, 40, 50], sigma=[0.3, 0.5, 0.7, 0.9]
# → 20 parameter combinations × 3 repetitions = 60 stimuli
```

## Part 3: Multi-Stimulus Batches

### Example: Touch Dataset with Multiple Stimulus Types

```yaml
batch:
  output_dir: "touch_dataset"
  format: "hdf5"  # Better for large datasets
  checkpoint_frequency: 50

pipeline:
  device: "cuda"  # Use GPU for speed
  grid_size: 64
  duration_ms: 300.0

neurons:
  sa_neurons: 150
  ra_neurons: 200

stimuli:
  # Gaussian blobs (static indentation)
  - type: "gaussian"
    amplitude: [15, 25, 35, 45]
    sigma: [0.4, 0.6, 0.8]
    repetitions: 5
  
  # Step functions (dynamic indentation)
  - type: "step"
    amplitude: [20, 30, 40]
    step_time: [50.0, 100.0, 150.0]
    repetitions: 5
  
  # Ramps (gradual loading)
  - type: "ramp"
    initial: 0.0
    final: [30, 40, 50]
    repetitions: 5

# Total: (4×3×5) + (3×3×5) + (3×5) = 60 + 45 + 15 = 120 stimuli
```

## Part 4: Working with Output Data

### PyTorch Format (.pt)

```python
import torch

# Load batch results
results = torch.load('batch_output/results.pt')

print(f"Number of stimuli: {len(results['stimuli'])}")
print(f"Keys: {results.keys()}")

# Access first stimulus results
stim_0 = results['stimuli'][0]
print(f"Stimulus 0 parameters: {stim_0['parameters']}")
print(f"SA spikes shape: {stim_0['sa_spikes'].shape}")
print(f"RA spikes shape: {stim_0['ra_spikes'].shape}")

# Extract all SA spike counts
sa_counts = [s['sa_spikes'].sum().item() for s in results['stimuli']]
print(f"SA spike counts: {sa_counts[:10]}...")
```

### HDF5 Format (.h5)

```python
import h5py
import numpy as np

# Open HDF5 file
with h5py.File('touch_dataset/results.h5', 'r') as f:
    print(f"HDF5 groups: {list(f.keys())}")
    
    # Access stimulus group
    stim_0 = f['stimulus_0000']
    print(f"Stimulus 0 datasets: {list(stim_0.keys())}")
    
    # Load spike data
    sa_spikes = stim_0['sa_spikes'][:]  # Load into memory
    ra_spikes = stim_0['ra_spikes'][:]
    
    # Read metadata
    amplitude = stim_0.attrs['amplitude']
    sigma = stim_0.attrs['sigma']
    
    print(f"Stimulus: amplitude={amplitude}, sigma={sigma}")
    print(f"SA spikes: {np.sum(sa_spikes)}")
```

### Creating PyTorch Dataset

```python
from torch.utils.data import Dataset, DataLoader

class SensoryForgeDataset(Dataset):
    """Custom dataset for SensoryForge batch results."""
    
    def __init__(self, results_path):
        self.data = torch.load(results_path)
        self.stimuli = self.data['stimuli']
    
    def __len__(self):
        return len(self.stimuli)
    
    def __getitem__(self, idx):
        stim = self.stimuli[idx]
        
        # Return spikes and parameters
        return {
            'sa_spikes': stim['sa_spikes'],
            'ra_spikes': stim['ra_spikes'],
            'amplitude': stim['parameters']['amplitude'],
            'sigma': stim['parameters']['sigma'],
        }

# Create dataset and loader
dataset = SensoryForgeDataset('batch_output/results.pt')
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Use in training loop
for batch in loader:
    sa_spikes = batch['sa_spikes']  # [batch_size, time, neurons]
    amplitude = batch['amplitude']  # [batch_size]
    # ... training code ...
```

## Part 5: Resume Interrupted Runs

### Checkpoint and Resume

```bash
# Start batch processing
sensoryforge batch large_sweep.yml

# If interrupted (Ctrl+C), resume with:
sensoryforge batch large_sweep.yml --resume

# The system automatically detects checkpoints and continues
```

**How it works:**
- Checkpoints saved to `<output_dir>/checkpoint.pt`
- Contains completed stimuli count and partial results
- Resume flag checks for checkpoint and continues from last completed stimulus

### Manual Checkpoint Inspection

```python
import torch

# Load checkpoint
checkpoint = torch.load('batch_output/checkpoint.pt')

print(f"Completed: {checkpoint['num_completed']} / {checkpoint['total_stimuli']}")
print(f"Next index: {checkpoint['next_index']}")
print(f"Config: {checkpoint['config']['batch']}")
```

## Part 6: Analyzing Sweep Results

### Visualize Parameter Effects

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load sweep results
results = torch.load('amplitude_sigma_sweep/results.pt')

# Extract parameters and spike counts
data = []
for stim in results['stimuli']:
    params = stim['parameters']
    data.append({
        'amplitude': params['amplitude'],
        'sigma': params['sigma'],
        'sa_spikes': stim['sa_spikes'].sum().item(),
        'ra_spikes': stim['ra_spikes'].sum().item(),
    })

# Convert to arrays
amplitudes = np.array([d['amplitude'] for d in data])
sigmas = np.array([d['sigma'] for d in data])
sa_counts = np.array([d['sa_spikes'] for d in data])

# Create heatmap
amp_vals = np.unique(amplitudes)
sig_vals = np.unique(sigmas)

# Average over repetitions
heatmap = np.zeros((len(sig_vals), len(amp_vals)))
for i, sig in enumerate(sig_vals):
    for j, amp in enumerate(amp_vals):
        mask = (sigmas == sig) & (amplitudes == amp)
        heatmap[i, j] = sa_counts[mask].mean()

# Plot
plt.figure(figsize=(10, 8))
plt.imshow(heatmap, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Mean SA Spike Count')
plt.xticks(range(len(amp_vals)), amp_vals)
plt.yticks(range(len(sig_vals)), sig_vals)
plt.xlabel('Amplitude')
plt.ylabel('Sigma (mm)')
plt.title('SA Response Heatmap')
plt.show()
```

### Statistical Analysis

```python
import pandas as pd
import seaborn as sns

# Create DataFrame
df = pd.DataFrame(data)

# Compute statistics per condition
stats = df.groupby(['amplitude', 'sigma']).agg({
    'sa_spikes': ['mean', 'std', 'count'],
    'ra_spikes': ['mean', 'std']
}).reset_index()

print(stats.head())

# Correlation analysis
print(f"\nAmplitude-SA correlation: {df['amplitude'].corr(df['sa_spikes']):.3f}")
print(f"Sigma-SA correlation: {df['sigma'].corr(df['sa_spikes']):.3f}")

# Violin plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.violinplot(data=df, x='amplitude', y='sa_spikes', ax=axes[0])
axes[0].set_title('SA Responses by Amplitude')

sns.violinplot(data=df, x='sigma', y='sa_spikes', ax=axes[1])
axes[1].set_title('SA Responses by Sigma')

plt.tight_layout()
plt.show()
```

## Part 7: GPU Acceleration

### Maximize GPU Utilization

```yaml
batch:
  output_dir: "gpu_batch"
  format: "pytorch"

pipeline:
  device: "cuda"  # Use GPU
  grid_size: 128  # Larger grid for GPU
  
neurons:
  sa_neurons: 300  # More neurons
  ra_neurons: 400

stimuli:
  - type: "gaussian"
    amplitude: [10, 15, 20, 25, 30, 35, 40, 45, 50]
    sigma: [0.2, 0.4, 0.6, 0.8, 1.0]
    repetitions: 10
```

**Tips for GPU efficiency:**
- Use larger grids (64×64 or 128×128)
- Increase neuron counts (> 200 per population)
- Longer simulation durations (> 200ms)
- Avoid frequent checkpointing (every 50+ stimuli)

### Monitor GPU Usage

```bash
# In another terminal during batch execution
watch -n 1 nvidia-smi
```

## Part 8: Best Practices

### Configuration Organization

```
project/
├── configs/
│   ├── base_pipeline.yml        # Shared pipeline config
│   ├── batches/
│   │   ├── gaussian_sweep.yml
│   │   ├── texture_dataset.yml
│   │   └── validation_set.yml
├── results/
│   ├── gaussian_sweep/
│   │   ├── results.pt
│   │   └── checkpoint.pt
│   └── texture_dataset/
└── analysis/
    └── analyze_sweeps.py
```

### Validation Before Large Runs

```bash
# 1. Dry run to check configuration
sensoryforge batch config.yml --dry-run

# 2. Test with small subset
# Edit config: repetitions: 1 (instead of 100)
sensoryforge batch config.yml

# 3. Verify output format
python -c "import torch; print(torch.load('batch_output/results.pt').keys())"

# 4. Run full batch
# Edit config: repetitions: 100
sensoryforge batch config.yml
```

### Resource Estimation

| Grid Size | Neurons | Duration | Memory (CPU) | Time/Stimulus |
|-----------|---------|----------|--------------|---------------|
| 32×32     | 100     | 150ms    | ~50 MB       | ~1 s          |
| 64×64     | 200     | 300ms    | ~200 MB      | ~3 s          |
| 128×128   | 500     | 500ms    | ~800 MB      | ~10 s         |

**Estimate total time:**
```
Total time ≈ (num_stimuli × time_per_stimulus) / num_workers
```

For 1000 stimuli at 3s each with 4 workers: ~12.5 minutes

## Next Steps

- [Parameter Sweeps Guide](parameter_sweeps.md) — Advanced sweep strategies
- [Custom Pipeline Tutorial](custom_pipeline.md) — Domain-specific batches
- [User Guide: Batch Processing](../user_guide/batch_processing.md) — Complete reference
- [CLI Reference](../user_guide/cli.md) — All command options
