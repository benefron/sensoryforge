# Custom Neuron Models Tutorial

This tutorial teaches you how to create custom spiking neuron models using SensoryForge's Equation DSL and hand-written implementations.

## Two Approaches

1. **Equation DSL** (recommended for neuroscientists) — Define models via equations
2. **Hand-written `nn.Module`** (recommended for ML engineers) — Maximum control and performance

## Part 1: Equation DSL Basics

### Simple Leaky Integrate-and-Fire (LIF)

```python
from sensoryforge.neurons.model_dsl import NeuronModel

# Define LIF using equations
lif = NeuronModel(
    equations='''
        dv/dt = (-v + I) / tau
    ''',
    threshold='v >= v_thresh',
    reset='v = v_reset',
    parameters={
        'tau': 10.0,        # Time constant (ms)
        'v_thresh': -50.0,  # Threshold (mV)
        'v_reset': -70.0,   # Reset voltage (mV)
    }
)

# Compile to PyTorch module
lif_module = lif.compile(solver='euler', device='cpu')

print(f"LIF neuron model ready: {lif_module}")
```

### Running the Model

```python
import torch

# Create input current
num_neurons = 10
num_steps = 1000
dt = 0.001  # 1ms time step

# Constant current
current = torch.ones(num_steps, num_neurons) * 15.0  # 15 mA

# Run simulation
spikes_list = []
voltages_list = []

for t in range(num_steps):
    lif_module.set_current(current[t])
    spikes, state = lif_module(current[t])
    
    spikes_list.append(spikes)
    voltages_list.append(state['v'])

# Stack results
all_spikes = torch.stack(spikes_list)  # [time, neurons]
all_voltages = torch.stack(voltages_list)

print(f"Total spikes: {all_spikes.sum()}")
print(f"Mean firing rate: {all_spikes.sum() / (num_neurons * num_steps / 1000):.2f} Hz")
```

## Part 2: Advanced Equation DSL

### Izhikevich Model

```python
izh = NeuronModel(
    equations='''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms
        du/dt = (a * (b*v - u)) / ms
    ''',
    threshold='v >= 30',
    reset='''
        v = c
        u = u + d
    ''',
    parameters={
        'a': 0.02,   # Recovery time scale
        'b': 0.2,    # Sensitivity
        'c': -65.0,  # Reset voltage
        'd': 8.0,    # Recovery boost
    }
)

izh_module = izh.compile(solver='euler')
```

### AdEx (Adaptive Exponential)

```python
adex = NeuronModel(
    equations='''
        dv/dt = (-g_L*(v - E_L) + g_L*Delta_T*exp((v - V_T)/Delta_T) - w + I) / C_m
        dw/dt = (a*(v - E_L) - w) / tau_w
    ''',
    threshold='v >= V_spike',
    reset='''
        v = V_reset
        w = w + b
    ''',
    parameters={
        'C_m': 200.0,      # Membrane capacitance (pF)
        'g_L': 10.0,       # Leak conductance (nS)
        'E_L': -70.0,      # Leak reversal (mV)
        'V_T': -50.0,      # Threshold (mV)
        'Delta_T': 2.0,    # Slope factor (mV)
        'V_spike': -40.0,  # Spike detection (mV)
        'V_reset': -70.0,  # Reset voltage (mV)
        'a': 2.0,          # Adaptation coupling (nS)
        'b': 100.0,        # Adaptation jump (pA)
        'tau_w': 200.0,    # Adaptation time constant (ms)
    }
)

adex_module = adex.compile(solver='dopri5')  # Use adaptive solver for stiff system
```

### Multi-Compartment Model

```python
two_compartment = NeuronModel(
    equations='''
        dv_soma/dt = (-g_L*(v_soma - E_L) + I - g_c*(v_soma - v_dend)) / C_soma
        dv_dend/dt = (-g_L*(v_dend - E_L) + g_c*(v_soma - v_dend)) / C_dend
    ''',
    threshold='v_soma >= v_thresh',
    reset='v_soma = v_reset',
    parameters={
        'C_soma': 200.0,
        'C_dend': 100.0,
        'g_L': 10.0,
        'g_c': 5.0,  # Coupling conductance
        'E_L': -70.0,
        'v_thresh': -50.0,
        'v_reset': -70.0,
    }
)

two_comp_module = two_compartment.compile()
```

## Part 3: Using DSL Models in Pipeline

### Configure via YAML

Create `custom_neuron_config.yml`:

```yaml
pipeline:
  device: "cpu"
  grid_size: 40
  duration_ms: 200.0
  dt_ms: 0.5

neurons:
  # Use custom LIF model for SA pathway
  sa_model:
    type: "dsl"  # Use Equation DSL
    equations: |
      dv/dt = (-v + I) / tau
    threshold: "v >= -50.0"
    reset: "v = -70.0"
    parameters:
      tau: 15.0
    solver: "euler"
    num_neurons: 80
  
  # Use custom Izhikevich for RA pathway
  ra_model:
    type: "dsl"
    equations: |
      dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms
      du/dt = (0.02 * (0.2*v - u)) / ms
    threshold: "v >= 30"
    reset: |
      v = -65.0
      u = u + 8.0
    parameters: {}
    solver: "euler"
    num_neurons: 100
```

### Load in Python

```python
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Load pipeline with custom neurons
pipeline = GeneralizedTactileEncodingPipeline.from_yaml('custom_neuron_config.yml')

# Run simulation
results = pipeline.forward(stimulus_type='gaussian', amplitude=30.0)

print(f"Custom SA spikes: {results['sa_spikes'].sum()}")
print(f"Custom RA spikes: {results['ra_spikes'].sum()}")
```

## Part 4: Hand-Written Neuron Models

### Create Custom Module

Create `sensoryforge/neurons/my_neuron.py`:

```python
from sensoryforge.neurons.base import BaseNeuron
import torch
import torch.nn as nn
from typing import Dict, Tuple

class QuadraticIF(BaseNeuron):
    """Quadratic integrate-and-fire neuron.
    
    Implements the equation:
        C dv/dt = a*(v - v_rest)*(v - v_crit) + I
    
    References:
        Latham et al. (2000). "Intrinsic dynamics in neuronal networks."
        J. Neurophysiol. 83(2):808-827.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Extract parameters
        self.num_neurons = config['num_neurons']
        self.dt = config.get('dt', 0.001)  # seconds
        
        # Model parameters
        self.C = config.get('C_m', 200.0)  # pF
        self.a = config.get('a', 0.5)      # coefficient
        self.v_rest = config.get('v_rest', -70.0)  # mV
        self.v_crit = config.get('v_crit', -55.0)  # mV
        self.v_thresh = config.get('v_thresh', -40.0)  # mV
        self.v_reset = config.get('v_reset', -70.0)  # mV
        
        # State variables
        self.register_buffer('v', torch.ones(self.num_neurons) * self.v_rest)
        self.register_buffer('I_input', torch.zeros(self.num_neurons))
    
    def forward(self, current: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Simulate one time step.
        
        Args:
            current: Input current [num_neurons] in mA
        
        Returns:
            spikes: Binary spike tensor [num_neurons]
            state: Dictionary with 'v' (voltage)
        """
        # Update input
        self.I_input = current
        
        # Compute derivative
        dv_dt = (self.a * (self.v - self.v_rest) * (self.v - self.v_crit) + self.I_input) / self.C
        
        # Euler integration
        self.v = self.v + dv_dt * self.dt
        
        # Check for spikes
        spikes = (self.v >= self.v_thresh).float()
        
        # Reset spiked neurons
        self.v = torch.where(spikes.bool(), torch.tensor(self.v_reset), self.v)
        
        return spikes, {'v': self.v.clone()}
    
    def reset_state(self):
        """Reset neuron state."""
        self.v.fill_(self.v_rest)
        self.I_input.zero_()
    
    def get_state(self) -> Dict:
        """Get current state."""
        return {
            'v': self.v.clone(),
            'I_input': self.I_input.clone()
        }
    
    def set_state(self, state: Dict):
        """Set state from dictionary."""
        self.v = state['v'].clone()
        self.I_input = state['I_input'].clone()
    
    @classmethod
    def from_config(cls, config: Dict) -> 'QuadraticIF':
        """Factory method for YAML instantiation."""
        return cls(config)
```

### Register Custom Model

Add to `sensoryforge/neurons/__init__.py`:

```python
from .my_neuron import QuadraticIF

__all__ = ['IzhikevichNeuronTorch', 'AdExNeuron', 'QuadraticIF', ...]
```

### Use in Pipeline

```yaml
pipeline:
  device: "cpu"
  grid_size: 40

neurons:
  sa_model:
    type: "QuadraticIF"  # Use custom class
    num_neurons: 80
    C_m: 200.0
    a: 0.5
    v_rest: -70.0
    v_crit: -55.0
    v_thresh: -40.0
    v_reset: -70.0
```

## Part 5: Testing Custom Models

### Unit Tests

Create `tests/unit/test_my_neuron.py`:

```python
import pytest
import torch
from sensoryforge.neurons.my_neuron import QuadraticIF

class TestQuadraticIF:
    """Test suite for QuadraticIF neuron."""
    
    @pytest.fixture
    def config(self):
        return {
            'num_neurons': 10,
            'dt': 0.001,
            'C_m': 200.0,
            'a': 0.5,
            'v_rest': -70.0,
            'v_crit': -55.0,
            'v_thresh': -40.0,
            'v_reset': -70.0,
        }
    
    def test_initialization(self, config):
        """Test neuron initializes correctly."""
        neuron = QuadraticIF(config)
        assert neuron.num_neurons == 10
        assert torch.allclose(neuron.v, torch.tensor(-70.0))
    
    def test_spike_generation(self, config):
        """Test neuron generates spikes with strong input."""
        neuron = QuadraticIF(config)
        
        # Apply strong current
        current = torch.ones(10) * 500.0  # Strong current
        
        # Run for multiple steps
        spike_count = 0
        for _ in range(1000):
            spikes, state = neuron(current)
            spike_count += spikes.sum().item()
        
        # Should generate spikes
        assert spike_count > 0
    
    def test_no_spikes_with_zero_input(self, config):
        """Test neuron doesn't spike spontaneously."""
        neuron = QuadraticIF(config)
        
        current = torch.zeros(10)
        
        for _ in range(1000):
            spikes, state = neuron(current)
            assert spikes.sum() == 0
    
    def test_reset_state(self, config):
        """Test state reset works."""
        neuron = QuadraticIF(config)
        
        # Modify state
        neuron.v.fill_(-50.0)
        neuron.I_input.fill_(10.0)
        
        # Reset
        neuron.reset_state()
        
        assert torch.allclose(neuron.v, torch.tensor(-70.0))
        assert torch.allclose(neuron.I_input, torch.zeros(10))
```

### Run Tests

```bash
pytest tests/unit/test_my_neuron.py -v
```

## Part 6: Biologically Realistic Models

### Fast Adapting (FA) Mechanoreceptor

```python
fa_neuron = NeuronModel(
    equations='''
        dv/dt = (-g_L*(v - E_L) + I_filtered - w) / C_m
        dw/dt = (a*(v - E_L) - w) / tau_w
        dI_filtered/dt = -I_filtered / tau_filter + I / tau_filter
    ''',
    threshold='v >= v_thresh',
    reset='''
        v = v_reset
        w = w + b
    ''',
    parameters={
        'C_m': 150.0,
        'g_L': 8.0,
        'E_L': -70.0,
        'v_thresh': -50.0,
        'v_reset': -70.0,
        'a': 3.0,          # Strong adaptation
        'b': 150.0,        # Large adaptation jump
        'tau_w': 50.0,     # Fast adaptation
        'tau_filter': 10.0, # Bandpass filtering
    }
)
```

### Slowly Adapting (SA) Mechanoreceptor

```python
sa_neuron = NeuronModel(
    equations='''
        dv/dt = (-g_L*(v - E_L) + I - w) / C_m
        dw/dt = (a*(v - E_L) - w) / tau_w
    ''',
    threshold='v >= v_thresh',
    reset='''
        v = v_reset
        w = w + b
    ''',
    parameters={
        'C_m': 200.0,
        'g_L': 10.0,
        'E_L': -70.0,
        'v_thresh': -50.0,
        'v_reset': -70.0,
        'a': 1.0,           # Weak adaptation
        'b': 50.0,          # Small adaptation jump
        'tau_w': 500.0,     # Slow adaptation
    }
)
```

## Part 7: Optimization and Best Practices

### Vectorization

**Bad** (loop over neurons):
```python
for i in range(num_neurons):
    self.v[i] = self.v[i] + dv_dt[i] * self.dt
```

**Good** (vectorized):
```python
self.v = self.v + dv_dt * self.dt
```

### GPU Compatibility

```python
class GPUReadyNeuron(BaseNeuron):
    def __init__(self, config):
        super().__init__(config)
        
        # Use register_buffer for state (auto-moves to GPU)
        self.register_buffer('v', torch.zeros(self.num_neurons))
        
        # Don't use .cuda() or .to(device) in __init__
        # Let the pipeline handle device placement
```

### Numerical Stability

```python
# Clip voltages to prevent overflow
def forward(self, current):
    dv_dt = self.compute_derivative(current)
    self.v = self.v + dv_dt * self.dt
    
    # Prevent numerical overflow
    self.v = torch.clamp(self.v, min=-100.0, max=50.0)
    
    spikes = (self.v >= self.v_thresh).float()
    return spikes, {'v': self.v}
```

### Memory Efficiency

```python
# Don't store full history in neuron class
# Let the pipeline/user handle history

class EfficientNeuron(BaseNeuron):
    def forward(self, current):
        # Process one time step
        # Return results immediately
        # Don't accumulate: self.history.append(...)
        spikes, state = self.step(current)
        return spikes, state
```

## Part 8: Advanced Features

### State-Dependent Threshold

```python
adaptive_threshold = NeuronModel(
    equations='''
        dv/dt = (-v + I) / tau_m
        dtheta/dt = (theta_inf - theta) / tau_theta
    ''',
    threshold='v >= theta',  # Dynamic threshold
    reset='''
        v = v_reset
        theta = theta + delta_theta  # Increase threshold after spike
    ''',
    parameters={
        'tau_m': 10.0,
        'tau_theta': 100.0,
        'theta_inf': -50.0,   # Baseline threshold
        'delta_theta': 5.0,   # Threshold increment
        'v_reset': -70.0,
    }
)
```

### Synaptic Input

```python
synapse_model = NeuronModel(
    equations='''
        dv/dt = (-v - E_L + g_syn*(E_syn - v) + I) / tau_m
        dg_syn/dt = -g_syn / tau_syn
    ''',
    threshold='v >= v_thresh',
    reset='v = v_reset',
    on_spike='g_syn = g_syn + w_syn',  # Increase conductance on spike
    parameters={
        'E_L': -70.0,
        'E_syn': 0.0,     # Excitatory reversal
        'tau_m': 10.0,
        'tau_syn': 5.0,
        'w_syn': 0.5,     # Synaptic weight
        'v_thresh': -50.0,
        'v_reset': -70.0,
    }
)
```

## Next Steps

- [Equation DSL Reference](../user_guide/equation_dsl.md) — Full DSL syntax
- [BaseNeuron API](../api_reference/neurons.md) — Interface documentation
- [Solver Guide](../user_guide/solvers.md) — Choosing ODE solvers
- [Contributing Guide](../contributing.md) — Submit custom models
