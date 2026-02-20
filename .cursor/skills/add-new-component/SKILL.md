---
name: add-new-component
description: Step-by-step guide for adding new extensible components to SensoryForge (neurons, filters, innervation methods, stimuli, solvers, grids, processing layers)
---

# Adding New Components to SensoryForge

This skill provides a step-by-step guide for extending SensoryForge with new components following the established extensibility patterns.

## Overview

SensoryForge uses a registry-based architecture for extensibility. All components must:
1. Inherit from the appropriate base class
2. Implement `from_config()` and `to_dict()` methods
3. Register themselves in `sensoryforge/register_components.py`
4. Follow PyTorch conventions and document tensor shapes

## Step-by-Step Process

### Step 1: Identify Component Type

Determine which base class your component should inherit from:

- **Neuron Models** → `BaseNeuron` (in `sensoryforge/neurons/base.py`)
- **Filters** → `BaseFilter` (in `sensoryforge/filters/base.py`)
- **Innervation Methods** → `BaseInnervation` (in `sensoryforge/core/innervation.py`)
- **Stimuli** → `BaseStimulus` (in `sensoryforge/stimuli/base.py`)
- **Solvers** → `BaseSolver` (in `sensoryforge/solvers/base.py`)
- **Grids** → `BaseGrid` (in `sensoryforge/core/grid_base.py`)
- **Processing Layers** → `BaseProcessingLayer` (in `sensoryforge/core/processing.py`)

### Step 2: Create Component Class

Create a new file in the appropriate directory (e.g., `sensoryforge/neurons/my_neuron.py`):

```python
"""My custom neuron model implementation."""

import torch
import torch.nn as nn
from typing import Dict, Any
from sensoryforge.neurons.base import BaseNeuron


class MyNeuron(BaseNeuron):
    """Custom neuron model description.
    
    Implements [brief description of what this neuron model does].
    Based on [paper/reference if applicable].
    
    Args:
        param1: Description of param1. Units: specify.
        param2: Description of param2. Units: specify.
        dt: Time step in ms. Default: 1.0.
        noise_std: Standard deviation of noise. Units: mA. Default: 0.0.
    
    References:
        Author et al. (Year). "Paper Title". Journal. DOI.
    """
    
    def __init__(
        self,
        param1: float = 1.0,
        param2: float = 2.0,
        dt: float = 1.0,
        noise_std: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__(dt=dt)
        self.param1 = param1
        self.param2 = param2
        self.noise_std = noise_std
        self.device = torch.device(device)
        
        # Initialize any PyTorch modules/parameters here
        self.register_buffer("param1_tensor", torch.tensor(param1, device=self.device))
    
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Forward pass through neuron model.
        
        Args:
            input_current: Input current [batch, time, num_neurons] in mA
        
        Returns:
            Spike output [batch, time, num_neurons] (binary: 0 or 1)
        """
        # Implementation here
        # Must return spike tensor of same shape as input_current
        pass
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MyNeuron':
        """Create instance from configuration dictionary.
        
        Args:
            config: Configuration dict with component parameters
        
        Returns:
            MyNeuron instance
        """
        return cls(
            param1=config.get("param1", 1.0),
            param2=config.get("param2", 2.0),
            dt=config.get("dt", 1.0),
            noise_std=config.get("noise_std", 0.0),
            device=config.get("device", "cpu"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary.
        
        Returns:
            Dictionary with component type and parameters (no tensors)
        """
        return {
            "type": "my_neuron",
            "param1": float(self.param1),
            "param2": float(self.param2),
            "dt": float(self.dt),
            "noise_std": float(self.noise_std),
        }
```

### Step 3: Register Component

Add registration to `sensoryforge/register_components.py`:

```python
# At top of file
from sensoryforge.neurons.my_neuron import MyNeuron

# In register_all() function
def register_all() -> None:
    # ... existing registrations ...
    
    # Register new component
    NEURON_REGISTRY.register("my_neuron", MyNeuron)
    NEURON_REGISTRY.register("MyNeuron", MyNeuron)  # Alias for GUI compatibility
```

### Step 4: Update Canonical Schema (if needed)

If the component needs configurable parameters in YAML configs, add fields to `sensoryforge/config/schema.py`:

```python
@dataclass
class PopulationConfig:
    # ... existing fields ...
    
    # New component parameters
    my_neuron_param1: float = 1.0
    my_neuron_param2: float = 2.0
```

### Step 5: Update GUI (if applicable)

If the component should be user-selectable in the GUI:

1. **Add to dropdown/combo box** in the relevant tab (e.g., `spiking_tab.py` for neuron models)
2. **Add parameter fields** to the form if needed
3. **Update config save/load** to include new parameters

### Step 6: Write Tests

Create tests in `tests/unit/test_<component_type>.py`:

```python
import torch
from sensoryforge.neurons.my_neuron import MyNeuron
from sensoryforge.registry import NEURON_REGISTRY

def test_my_neuron_forward():
    """Test forward pass of MyNeuron."""
    neuron = MyNeuron(param1=1.0, param2=2.0, dt=1.0)
    input_current = torch.randn(1, 100, 10)  # [batch, time, num_neurons]
    spikes = neuron(input_current)
    assert spikes.shape == input_current.shape
    assert spikes.dtype == torch.float32

def test_my_neuron_registry():
    """Test that MyNeuron is registered."""
    assert NEURON_REGISTRY.is_registered("my_neuron")
    neuron_cls = NEURON_REGISTRY.get_class("my_neuron")
    assert neuron_cls == MyNeuron

def test_my_neuron_config_round_trip():
    """Test config serialization round-trip."""
    neuron = MyNeuron(param1=1.5, param2=2.5, dt=0.5)
    config = neuron.to_dict()
    neuron2 = MyNeuron.from_config(config)
    assert neuron2.param1 == neuron.param1
    assert neuron2.param2 == neuron.param2
```

### Step 7: Update Documentation

1. **Add to component list** in `docs/user_guide/components.md` (if user-facing)
2. **Add example** in `docs/examples/` if applicable
3. **Update API reference** if auto-generated

## Special Cases

### Factory Functions

Some components (like innervation methods) need factory functions due to special initialization:

```python
def create_my_innervation(**kwargs):
    """Factory function for MyInnervation."""
    receptor_coords = kwargs.pop("receptor_coords")
    neuron_centers = kwargs.pop("neuron_centers")
    device = kwargs.pop("device", "cpu")
    return MyInnervation(receptor_coords, neuron_centers, device=device, **kwargs)

INNERVATION_REGISTRY.register("my_innervation", MyInnervation, create_my_innervation)
```

### Device Handling

Always specify device explicitly:

```python
# ❌ BAD
tensor = torch.zeros(10, 10)

# ✅ GOOD
tensor = torch.zeros(10, 10, device=self.device)
```

### Tensor Shape Documentation

Always document tensor shapes and units in docstrings:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Process input.
    
    Args:
        x: Input tensor [batch, time, num_neurons] in mA
    
    Returns:
        Output tensor [batch, time, num_neurons] in mV
    """
```

## Checklist

- [ ] Component inherits from appropriate base class
- [ ] `from_config()` classmethod implemented
- [ ] `to_dict()` instance method implemented
- [ ] Component registered in `register_components.py`
- [ ] Parameters added to canonical schema (if needed)
- [ ] GUI updated (if user-facing)
- [ ] Unit tests written
- [ ] Config round-trip tests pass
- [ ] Documentation updated
- [ ] Code follows PyTorch conventions
- [ ] Tensor shapes documented
- [ ] Device handling correct

## Examples

See existing components for reference:
- **Neuron**: `sensoryforge/neurons/izhikevich.py`
- **Filter**: `sensoryforge/filters/sa_ra.py`
- **Innervation**: `sensoryforge/core/innervation.py` (GaussianInnervation)
- **Stimulus**: `sensoryforge/stimuli/gaussian.py`
- **Solver**: `sensoryforge/solvers/euler.py`
