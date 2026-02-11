# Batch Processing

Batch processing enables large-scale execution of stimulus sweeps for dataset generation, parameter exploration, and machine learning training data creation.

## Overview

The batch execution system allows you to:

- **Define parameter sweeps** - Automatically generate all combinations of parameters
- **Execute stimulus batteries** - Run hundreds or thousands of stimuli configurations
- **Generate training datasets** - Create structured data for ML model training
- **Ensure reproducibility** - Deterministic seeding for exact replication
- **Resume interrupted runs** - Checkpoint-based recovery from failures
- **Export multiple formats** - PyTorch (.pt) and HDF5 (.h5) output

## Quick Start

### 1. Create a Batch Configuration

Create a YAML file defining your parameter sweep:

```yaml
metadata:
  batch_name: "my_tactile_dataset"
  description: "Gaussian stimulus parameter sweep"

base_config:
  pipeline:
    device: cpu
    grid_size: 80
  neurons:
    sa_neurons: 100
    ra_neurons: 196

batch:
  output_dir: "./batch_results/my_dataset"
  save_format: "pytorch"  # or "hdf5"
  
  stimuli:
    - type: gaussian_sweep
      parameters:
        amplitude: [10, 20, 30, 40, 50]
        sigma: [0.3, 0.5, 0.8, 1.0, 1.5]
      repetitions: 3
```

This configuration generates **75 stimuli** (5 amplitudes × 5 sigmas × 3 repetitions).

### 2. Validate Configuration (Dry Run)

Before executing, validate your configuration:

```bash
sensoryforge batch config.yml --dry-run
```

This shows:
- Total number of stimuli that will be generated
- Example stimulus configurations
- Output directory location
- Estimated resource requirements

### 3. Execute Batch

Run the batch execution:

```bash
# On CPU
sensoryforge batch config.yml

# On GPU
sensoryforge batch config.yml --device cuda

# Custom output directory
sensoryforge batch config.yml --output ./my_results
```

### 4. Monitor Progress

During execution, you'll see:

```
Starting batch execution: my_tactile_dataset_20260211_130000
Total stimuli: 75
Output directory: batch_results/my_dataset

[1/75] Executing gaussian_a10.0_s0.3_rep0...
[2/75] Executing gaussian_a10.0_s0.3_rep1...
...
```

Progress is automatically checkpointed every stimulus.

### 5. Resume Interrupted Batches

If execution is interrupted (Ctrl+C, crash, etc.):

```bash
sensoryforge batch config.yml --resume batch_results/my_dataset/checkpoint.json
```

Already-completed stimuli are skipped.

## Parameter Sweep Expansion

### Cartesian Product

Parameter sweeps use Cartesian product to generate all combinations:

```yaml
stimuli:
  - type: gaussian_sweep
    parameters:
      amplitude: [10, 20]      # 2 values
      sigma: [0.5, 1.0]        # 2 values
      center_x: [0.0]          # 1 value
    repetitions: 3
```

Generates 2 × 2 × 1 × 3 = **12 stimuli**:

1. `amplitude=10, sigma=0.5, center_x=0, rep=0`
2. `amplitude=10, sigma=0.5, center_x=0, rep=1`
3. `amplitude=10, sigma=0.5, center_x=0, rep=2`
4. `amplitude=10, sigma=1.0, center_x=0, rep=0`
5. ...and so on

### Single Values

Parameters with single values don't multiply the sweep:

```yaml
parameters:
  amplitude: [30]  # Single value (can also write: 30.0)
  sigma: [0.5, 1.0, 1.5]
```

Generates 1 × 3 = **3 parameter combinations**.

### Multiple Sweep Types

Combine different stimulus types in one batch:

```yaml
batch:
  stimuli:
    # Gaussian sweep
    - type: gaussian_sweep
      parameters:
        amplitude: [10, 20, 30]
        sigma: [0.5, 1.0]
      repetitions: 2
    
    # Texture sweep
    - type: gabor_sweep
      parameters:
        frequency: [0.5, 1.0, 2.0]
        orientation: [0, 45, 90]
      repetitions: 1
```

Total: (3×2×2) + (3×3×1) = **21 stimuli**

## Output Formats

### PyTorch Format (.pt)

Default format, ideal for PyTorch workflows:

```python
import torch

# Load results
data = torch.load('batch_results/my_dataset/results.pt')

# Access data
metadata = data['metadata']
results = data['results']

for result in results:
    sa_spikes = result['sa_spikes']  # [T, N_sa]
    ra_spikes = result['ra_spikes']  # [T, N_ra]
    config = result['stimulus_config']
    print(f"Stimulus {config['stimulus_id']}: "
          f"{sa_spikes.sum()} SA spikes")
```

### HDF5 Format (.h5)

Scientific standard format, compatible with MATLAB, R, Python:

```python
import h5py
import numpy as np

# Load results
with h5py.File('batch_results/my_dataset/results.h5', 'r') as f:
    # Access metadata
    batch_id = f['metadata'].attrs['batch_id']
    
    # Access specific stimulus
    sa_spikes = f['responses/stim_0000/sa_spikes'][:]  # numpy array
    ra_spikes = f['responses/stim_0000/ra_spikes'][:]
    
    # Iterate all stimuli
    for stim_id in f['responses'].keys():
        spikes = f[f'responses/{stim_id}/sa_spikes'][:]
        print(f"{stim_id}: {spikes.sum()} total spikes")
```

**HDF5 Structure:**
```
results.h5
├── metadata/
│   ├── batch_id (attribute)
│   ├── num_stimuli (attribute)
│   └── config (attribute, JSON string)
├── grid/
│   ├── size (attribute)
│   └── spacing (attribute)
├── stimuli/
│   └── stim_0000/
│       ├── amplitude (attribute)
│       ├── sigma (attribute)
│       └── ... (other parameters)
└── responses/
    └── stim_0000/
        ├── sa_spikes [T, N_sa]
        ├── ra_spikes [T, N_ra]
        ├── sa2_spikes [T, N_sa2]
        └── ... (if save_intermediates=true)
```

## Configuration Reference

### Metadata Section

```yaml
metadata:
  batch_name: "descriptive_name"      # Used in batch ID
  version: "1.0"                      # Version tracking
  description: "What this batch does"
  author: "Your Name"
  date: "2026-02-11"
```

### Base Config Section

Full pipeline configuration applied to all stimuli:

```yaml
base_config:
  pipeline:
    device: cpu              # 'cpu', 'cuda', 'mps'
    seed: 42                # Base random seed
    grid_size: 80
    spacing: 0.15
  
  neurons:
    sa_neurons: 100
    ra_neurons: 196
    sa2_neurons: 25
    dt: 0.5
  
  # ... (same as single-run config)
```

See [YAML Configuration](yaml_configuration.md) for all options.

### Batch Section

```yaml
batch:
  output_dir: "./batch_results/my_batch"
  save_format: "pytorch"     # 'pytorch' or 'hdf5'
  save_intermediates: false  # true to save currents, voltages
  
  stimuli:
    - type: gaussian_sweep
      base_seed: 42           # Base seed for this sweep
      parameters:
        param_name: [value1, value2, ...]
      repetitions: 3
```

**Parameters:**

- `output_dir`: Where to save results (created if doesn't exist)
- `save_format`: Output file format
- `save_intermediates`: Include filtered currents, voltages, etc.
- `stimuli`: List of sweep configurations

**Sweep Configuration:**

- `type`: Stimulus type with `_sweep` suffix (or without)
- `base_seed`: Starting seed for this sweep (optional)
- `parameters`: Dict of parameter_name → list_of_values
- `repetitions`: Number of noise realizations per combination

## ML Training Integration

### Creating a PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SensoryForgeDataset(Dataset):
    """Dataset loader for SensoryForge batch results."""
    
    def __init__(self, batch_file):
        data = torch.load(batch_file)
        self.results = data['results']
    
    def __len__(self):
        return len(self.results)
    
    def __getitem__(self, idx):
        result = self.results[idx]
        
        # Extract stimulus (if saved)
        stimulus = result.get('stimulus', None)
        
        # Extract responses
        sa_spikes = result['sa_spikes']  # [T, N_sa]
        ra_spikes = result['ra_spikes']  # [T, N_ra]
        
        # Combine population responses
        responses = torch.cat([sa_spikes, ra_spikes], dim=1)
        
        return stimulus, responses

# Usage
dataset = SensoryForgeDataset('batch_results/my_dataset/results.pt')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for stimuli, responses in loader:
    # Train your model
    loss = model(stimuli, responses)
    loss.backward()
    optimizer.step()
```

### HDF5 Dataset Loader

```python
import h5py
import numpy as np
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    """Memory-efficient HDF5 dataset loader."""
    
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as f:
            self.num_stimuli = len(f['responses'].keys())
    
    def __len__(self):
        return self.num_stimuli
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            stim_id = f'stim_{idx:04d}'
            sa_spikes = torch.from_numpy(
                f[f'responses/{stim_id}/sa_spikes'][:]
            )
            ra_spikes = torch.from_numpy(
                f[f'responses/{stim_id}/ra_spikes'][:]
            )
        
        responses = torch.cat([sa_spikes, ra_spikes], dim=1)
        return responses

# Efficient for large datasets (doesn't load all into RAM)
dataset = HDF5Dataset('batch_results/my_dataset/results.h5')
```

## Advanced Usage

### Custom Stimulus Types

Any stimulus type supported by the pipeline works in batches:

```yaml
stimuli:
  - type: gabor_sweep
    parameters:
      frequency: [0.5, 1.0, 2.0]
      orientation: [0, 45, 90, 135]
      amplitude: [20, 30]
  
  - type: moving_gaussian_sweep
    parameters:
      velocity: [5.0, 10.0, 20.0]
      direction: [0, 90, 180, 270]
      sigma: [0.5]
```

### Accessing Metadata

```python
import json

# Batch metadata (full config + runtime info)
with open('batch_results/my_dataset/batch_metadata.json') as f:
    metadata = json.load(f)
    print(f"Batch ID: {metadata['batch_id']}")
    print(f"Num stimuli: {metadata['num_stimuli']}")

# Stimulus index (lookup table)
with open('batch_results/my_dataset/stimulus_index.json') as f:
    index = json.load(f)
    config = index['stim_0000']
    print(f"Stimulus 0: {config['type']}, "
          f"amplitude={config['amplitude']}, "
          f"sigma={config['sigma']}")
```

### Saving Intermediate Results

Enable to save filtered currents, voltages, etc:

```yaml
batch:
  save_intermediates: true
```

Then access additional data:

```python
result = results[0]
sa_currents = result['sa_currents']   # Filtered SA currents
ra_currents = result['ra_currents']   # Filtered RA currents
sa_voltages = result['sa_voltages']   # SA membrane voltages
stimulus = result['stimulus']         # Full stimulus tensor
```

**Warning:** Increases file size significantly (~10-50x depending on parameters).

## Error Handling

### Single Stimulus Failures

If one stimulus fails, the batch continues:

```
[42/100] Executing stimulus...
ERROR executing stimulus 42: Out of memory
[43/100] Executing stimulus...
```

Failed indices are recorded in checkpoint and final results.

### Resuming After Failure

```bash
sensoryforge batch config.yml --resume batch_results/my_dataset/checkpoint.json
```

Skips completed stimuli and continues from where it left off.

### Checking for Failures

```python
data = torch.load('results.pt')
failed = data['metadata'].get('failed_stimuli', [])
if failed:
    print(f"Failed stimuli: {failed}")
```

## Performance Tips

### Memory Management

**Problem:** Out of memory with large batches

**Solutions:**
1. Use HDF5 format (writes streaming, doesn't accumulate in RAM)
2. Reduce `grid_size` or neuron counts
3. Set `save_intermediates: false`
4. Process smaller batches

### GPU Acceleration

```bash
sensoryforge batch config.yml --device cuda
```

Typical speedup: 5-20x depending on batch size and grid resolution.

### Parallel Execution

Future feature - currently serial execution only.

## See Also

- [YAML Configuration](yaml_configuration.md) - Full config reference
- [CLI Reference](cli.md) - Command-line usage
- [Example Configurations](../../examples/) - Ready-to-use templates
- [Stimulus Types](extended_stimuli.md) - Available stimulus types

## Examples Repository

See `examples/batch_config.yml` for a comprehensive template with:
- Gaussian parameter sweeps
- Texture pattern sweeps
- Moving stimulus sweeps
- Single stimulus with multiple noise realizations
- Inline usage documentation
