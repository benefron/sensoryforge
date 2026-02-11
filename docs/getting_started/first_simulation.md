# Your First Simulation

A step-by-step walkthrough of creating and running your first SensoryForge simulation.

## Goal

Build a simple tactile encoding pipeline that:
1. Creates an 80×80 receptor grid
2. Defines SA and RA neuron populations
3. Applies a Gaussian pressure stimulus
4. Generates spike trains
5. Visualizes the results

## Step 1: Setup

Create a new Python file `first_simulation.py`:

```python
import torch
import matplotlib.pyplot as plt
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
```

## Step 2: Configure the Pipeline

Define your pipeline configuration:

```python
config = {
    # Core pipeline settings
    'pipeline': {
        'device': 'cpu',      # Use 'cuda' if you have a GPU
        'seed': 42,           # For reproducibility
        'grid_size': 80,      # 80×80 receptor grid
        'spacing': 0.15,      # 0.15mm between receptors
        'center': [0.0, 0.0], # Grid center coordinates
    },
    
    # Neuron populations
    'neurons': {
        'sa_neurons': 100,    # Slowly adapting neurons
        'ra_neurons': 196,    # Rapidly adapting neurons 
        'sa2_neurons': 25,    # SA2 neurons (deep pressure)
        'dt': 0.5,           # 0.5ms time step for spike generation
    },
    
    # Innervation (receptive fields)
    'innervation': {
        'receptors_per_neuron': 28,  # Convergence factor
        'sa_spread': 0.3,             # SA receptive field size (mm)
        'ra_spread': 0.39,            # RA receptive field size (mm)
    },
    
    # Temporal filters
    'filters': {
        'sa_tau_r': 5.0,     # SA rise time (ms)
        'sa_tau_d': 30.0,    # SA decay time (ms)
        'ra_tau_ra': 15.0,   # RA time constant (ms)
    },
    
    # Neuron model parameters (Izhikevich)
    'neuron_params': {
        'sa_a': 0.02,
        'sa_b': 0.2,
        'sa_c': -65.0,
        'sa_d': 8.0,
        'ra_a': 0.02,
        'ra_b': 0.2,
        'ra_c': -65.0,
        'ra_d': 8.0,
    },
    
    # Temporal profile (trapezoidal)
    'temporal': {
        't_pre': 25,         # Pre-stimulus baseline (ms)
        't_ramp': 10,        # Ramp-up duration (ms)
        't_plateau': 500,    # Stimulus plateau (ms)
        't_post': 100,       # Post-stimulus period (ms)
        'dt': 0.5,          # Time step (ms)
    },
}
```

## Step 3: Create the Pipeline

```python
print("Creating pipeline...")
pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
print(f"✓ Pipeline created")
print(f"  Device: {pipeline.device}")
print(f"  Grid size: {pipeline.grid_manager.grid_size}")
print(f"  SA neurons: {pipeline.sa_innervation.num_neurons}")
print(f"  RA neurons: {pipeline.ra_innervation.num_neurons}")
```

## Step 4: Run the Simulation

Execute with a Gaussian stimulus:

```python
print("\nRunning simulation...")
results = pipeline.forward(
    stimulus_type='gaussian',
    amplitude=30.0,       # Peak pressure (arbitrary units)
    sigma=0.5,           # Spatial spread (mm)
    center_x=0.0,        # X position (mm)
    center_y=0.0,        # Y position (mm)
    return_intermediates=True  # Include filtered currents
)

print(f"✓ Simulation complete")
```

## Step 5: Examine Results

```python
# Extract spike trains
sa_spikes = results['sa_spikes']  # [time_steps, num_sa_neurons]
ra_spikes = results['ra_spikes']  # [time_steps, num_ra_neurons]

# Print statistics
print(f"\nResults:")
print(f"  SA spikes: {sa_spikes.sum().item():.0f} ({sa_spikes.sum() / sa_spikes.shape[1]:.1f} per neuron)")
print(f"  RA spikes: {ra_spikes.sum().item():.0f} ({ra_spikes.sum() / ra_spikes.shape[1]:.1f} per neuron)")
print(f"  Duration: {sa_spikes.shape[0] * 0.5:.0f} ms")

# Examine filtered currents (if return_intermediates=True)
if 'sa_currents' in results:
    sa_currents = results['sa_currents']
    print(f"  SA current range: [{sa_currents.min():.2f}, {sa_currents.max():.2f}]")
```

## Step 6: Visualize Results

### Raster Plots

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# SA raster
times_sa, neurons_sa = sa_spikes.nonzero(as_tuple=True)
ax1.scatter(times_sa.cpu() * 0.5, neurons_sa.cpu(), 
           s=2, c='navy', alpha=0.6, marker='|')
ax1.set_ylabel('SA Neuron #', fontsize=12)
ax1.set_title(f'SA Population Activity ({sa_spikes.sum():.0f} total spikes)', 
             fontsize=14, fontweight='bold')
ax1.set_xlim([0, sa_spikes.shape[0] * 0.5])
ax1.grid(alpha=0.3)

# RA raster
times_ra, neurons_ra = ra_spikes.nonzero(as_tuple=True)
ax2.scatter(times_ra.cpu() * 0.5, neurons_ra.cpu(), 
           s=2, c='darkred', alpha=0.6, marker='|')
ax2.set_ylabel('RA Neuron #', fontsize=12)
ax2.set_xlabel('Time (ms)', fontsize=12)
ax2.set_title(f'RA Population Activity ({ra_spikes.sum():.0f} total spikes)', 
             fontsize=14, fontweight='bold')
ax2.set_xlim([0, ra_spikes.shape[0] * 0.5])
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('spike_rasters.png', dpi=150)
print("\n✓ Saved spike_rasters.png")
plt.show()
```

### Population Firing Rates

```python
# Compute firing rates (spikes per 10ms bin)
bin_size_ms = 10
bin_size_steps = int(bin_size_ms / 0.5)

sa_rate = sa_spikes.sum(dim=1).unfold(0, bin_size_steps, bin_size_steps).mean(dim=1)
ra_rate = ra_spikes.sum(dim=1).unfold(0, bin_size_steps, bin_size_steps).mean(dim=1)

time_bins = torch.arange(len(sa_rate)) * bin_size_ms

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(time_bins.cpu(), sa_rate.cpu(), 'navy', linewidth=2, label='SA Population')
ax.plot(time_bins.cpu(), ra_rate.cpu(), 'darkred', linewidth=2, label='RA Population')
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Spikes per 10ms bin', fontsize=12)
ax.set_title('Population Firing Rates', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('firing_rates.png', dpi=150)
print("✓ Saved firing_rates.png")
plt.show()
```

### Filtered Currents (Optional)

If you set `return_intermediates=True`:

```python
if 'sa_currents' in results:
    sa_currents = results['sa_currents']  # [time, neurons]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot first 10 neurons
    for i in range(min(10, sa_currents.shape[1])):
        ax.plot(torch.arange(sa_currents.shape[0]) * 0.5, 
               sa_currents[:, i].cpu(), 
               alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Current (mA)', fontsize=12)
    ax.set_title('SA Filtered Currents (first 10 neurons)', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('filtered_currents.png', dpi=150)
    print("✓ Saved filtered_currents.png")
    plt.show()
```

## Step 7: Try Different Stimuli

Modify the stimulus parameters and re-run:

```python
# Larger stimulus
results_large = pipeline.forward(
    stimulus_type='gaussian',
    amplitude=50.0,  # Stronger
    sigma=1.0,       # Wider
)

# Off-center stimulus
results_offset = pipeline.forward(
    stimulus_type='gaussian',
    amplitude=30.0,
    sigma=0.5,
    center_x=2.0,    # 2mm to the right
    center_y=-1.0,   # 1mm down
)

# Compare spike counts...
```

## Complete Script

Here's the complete first simulation script:

```python
"""first_simulation.py - Your first SensoryForge simulation"""

import torch
import matplotlib.pyplot as plt
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Configuration
config = {
    'pipeline': {'device': 'cpu', 'seed': 42, 'grid_size': 80, 'spacing': 0.15},
    'neurons': {'sa_neurons': 100, 'ra_neurons': 196, 'dt': 0.5},
    'innervation': {'receptors_per_neuron': 28, 'sa_spread': 0.3, 'ra_spread': 0.39},
    'filters': {'sa_tau_r': 5.0, 'sa_tau_d': 30.0, 'ra_tau_ra': 15.0},
    'temporal': {'t_pre': 25, 't_ramp': 10, 't_plateau': 500, 't_post': 100, 'dt': 0.5},
}

# Create pipeline
print("Creating pipeline...")
pipeline = GeneralizedTactileEncodingPipeline.from_config(config)

# Run simulation
print("Running simulation...")
results = pipeline.forward(
    stimulus_type='gaussian',
    amplitude=30.0,
    sigma=0.5,
    return_intermediates=True
)

# Extract results
sa_spikes = results['sa_spikes']
ra_spikes = results['ra_spikes']

print(f"\nResults:")
print(f"  SA: {sa_spikes.sum():.0f} spikes")
print(f"  RA: {ra_spikes.sum():.0f} spikes")

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

times_sa, neurons_sa = sa_spikes.nonzero(as_tuple=True)
ax1.scatter(times_sa * 0.5, neurons_sa, s=2, c='navy', marker='|')
ax1.set_ylabel('SA Neuron #')
ax1.set_title('SA Population Activity')

times_ra, neurons_ra = ra_spikes.nonzero(as_tuple=True)
ax2.scatter(times_ra * 0.5, neurons_ra, s=2, c='darkred', marker='|')
ax2.set_ylabel('RA Neuron #')
ax2.set_xlabel('Time (ms)')
ax2.set_title('RA Population Activity')

plt.tight_layout()
plt.savefig('first_simulation_results.png', dpi=150)
print("\n✓ Saved first_simulation_results.png")
plt.show()
```

Run it:

```bash
python first_simulation.py
```

## What You Learned

✅ How to configure a pipeline with YAML-like dictionaries  
✅ How to create and initialize a pipeline  
✅ How to run simulations with different stimuli  
✅ How to extract and analyze results  
✅ How to visualize spike trains and firing rates

## Next Steps

- [Tutorials](../tutorials/quickstart_tutorial.md) — More hands-on examples
- [User Guide](../user_guide/overview.md) — Detailed component reference
- [Batch Processing](../tutorials/batch_processing_tutorial.md) — Generate large datasets
- [Custom Neurons](../tutorials/custom_neurons.md) — Define your own neuron models
