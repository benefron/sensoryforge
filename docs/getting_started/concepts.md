# Core Concepts

Understanding SensoryForge's architecture and design philosophy.

## Design Philosophy

SensoryForge is built around three core principles:

1. **Modality-Agnostic Architecture** — Same framework for touch, vision, audition, and custom modalities
2. **Biologically Grounded** — Components map to neuroscience concepts (receptive fields, adaptation, spike generation)
3. **Production-Ready** — GPU-accelerated, differentiable, and suitable for ML training

## Architecture Overview

```mermaid
graph TD
    A[Raw Sensory Input] --> B[Spatial Grid]
    B --> C[Receptive Fields]
    C --> D[Temporal Filtering]
    D --> E1[Population 1]
    D --> E2[Population 2]
    D --> E3[Population N]
    E1 --> F1[Neurons]
    E2 --> F2[Neurons]
    E3 --> F3[Neurons]
    F1 --> G[Spike Trains]
    F2 --> G
    F3 --> G
    
    H[Component Registry] -.-> E1
    H -.-> E2
    H -.-> E3
    H -.-> D
    H -.-> C
    
    I[Canonical Config] --> B
    I --> C
    I --> D
    I --> E1
    I --> E2
    I --> E3
```

### Key Architectural Features

- **Registry System**: All components (neurons, filters, innervation, stimuli) are registered and looked up dynamically
- **N-Population Support**: Dynamic population configuration (not limited to hardcoded SA/RA/SA2)
- **Canonical Configuration**: Single source of truth (`SensoryForgeConfig`) for GUI-CLI parity
- **Extensibility**: Easy to add new components via registry without modifying core code

### Data Flow

1. **Stimulus** → High-dimensional sensory input (e.g., pressure map, image)
2. **Grid** → Spatial substrate of receptor positions (supports multiple grid layers)
3. **Innervation** → Receptive fields connecting receptors to neurons (per-population configuration)
4. **Filtering** → Temporal dynamics applied per population (SA/RA/custom filters)
5. **Spiking** → ODE-based neuron models generate spike trains (per-population neuron models)
6. **Output** → Sparse, event-based neural representation (results keyed by population name)

**Note**: The pipeline supports N populations, each with its own innervation, filter, neuron model, and solver configuration. Results are returned as a dictionary keyed by population name.

## Key Components

### 1. Spatial Grid

**Purpose:** Define the spatial arrangement of sensory receptors.

**Types:**
- **Standard Grid** — Regular 2D array
- **Composite Grid** — Multi-population mosaic (e.g., SA1/RA1/SA2 in touch, L/M/S cones in vision)
- **Custom Arrangements** — Hexagonal, Poisson, jittered

**Example (Canonical Format):**
```python
from sensoryforge.config.schema import SensoryForgeConfig, GridConfig

# Standard grid
config = SensoryForgeConfig(
    grids=[
        GridConfig(
            name="Main Grid",
            arrangement="grid",
            rows=80,
            cols=80,
            spacing=0.15,  # 0.15mm spacing
        )
    ],
    # ... populations, stimulus, simulation
)

# Composite grid (multi-population)
config = SensoryForgeConfig(
    grids=[
        GridConfig(
            name="Composite Grid",
            arrangement="composite",
            # Composite grid configuration...
        )
    ],
    # ... populations, stimulus, simulation
)
```

**Example (Legacy Format - Still Supported):**
```python
# Legacy format still works via adapter
config = {
    'pipeline': {
        'grid_size': 80,
        'spacing': 0.15,
    }
}
```

### 2. Innervation (Receptive Fields)

**Purpose:** Define how receptors connect to downstream neurons.

**Key Concepts:**
- **Convergence:** Multiple receptors → single neuron
- **Overlap:** Receptive fields can overlap spatially
- **Weighting:** Gaussian, distance-weighted, or custom

**Types:**
- **Gaussian Innervation** — Smooth Gaussian weighting
- **Distance-Weighted** — Inverse distance falloff
- **Flat Innervation** — For irregular (non-grid) receptor layouts

**Example:**
```yaml
innervation:
  receptors_per_neuron: 28
  sa_spread: 0.3   # Receptive field size (mm)
  ra_spread: 0.39
```

### 3. Temporal Filters

**Purpose:** Model temporal dynamics of sensory pathways.

**Dual-Pathway Processing:**
- **SA (Slowly Adapting)** — Sustained response to static stimuli
- **RA (Rapidly Adapting)** — Transient response to changes

**Biological Analogy (Touch):**
- SA: Pressure magnitude (Merkel discs, Ruffini endings)
- RA: Velocity/vibration (Meissner corpuscles, Pacinian corpuscles)

**Example:**
```yaml
filters:
  sa_tau_r: 5.0    # Rise time constant (ms)
  sa_tau_d: 30.0   # Decay time constant (ms)
  ra_tau_ra: 15.0  # RA time constant (ms)
```

### 4. Neuron Models

**Purpose:** Convert filtered currents into spike trains.

**Available Models:**
- **Izhikevich** — Efficient, captures diverse spike patterns
- **AdEx** — Adaptive exponential integrate-and-fire
- **MQIF** — Multi-quadratic integrate-and-fire
- **Custom (DSL)** — Define via equations

**Example:**
```python
# Hand-written model
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch

# Equation DSL model
from sensoryforge.neurons.model_dsl import NeuronModel

model = NeuronModel(
    equations='''
        dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms
        du/dt = (a * (b*v - u)) / ms
    ''',
    threshold='v >= 30',
    reset='v = c; u = u + d',
    parameters={'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0}
)
```

### 5. Stimuli

**Purpose:** Generate sensory input patterns.

**Categories:**
- **Static:** Gaussian blobs, textures
- **Moving:** Linear, circular motion
- **Temporal:** Step, ramp, trapezoidal profiles
- **Composite:** Multiple stimuli combined

**Example:**
```python
# Gaussian stimulus
results = pipeline.forward(
    stimulus_type='gaussian',
    amplitude=30.0,
    sigma=0.5,
    center_x=0.0,
    center_y=0.0
)

# Gabor texture
results = pipeline.forward(
    stimulus_type='gabor',
    frequency=1.0,
    orientation=45,
    amplitude=25.0
)
```

### 6. ODE Solvers

**Purpose:** Integrate neuron model differential equations.

**Available Solvers:**
- **Euler** — Fast, sufficient for most cases (default)
- **Adaptive (Dormand-Prince)** — High accuracy for stiff systems

**Example:**
```yaml
solver:
  type: euler  # or 'adaptive'
  config:
    dt: 0.001  # 1ms time step
```

## Coordinate Systems

### Spatial Coordinates

- **Origin:** Grid center at (0, 0)
- **Units:** Millimeters (mm)
- **Axes:** X (horizontal), Y (vertical)

### Temporal Coordinates

- **Units:** Milliseconds (ms) for user APIs
- **Internal:** Seconds (s) for ODE integration
- **Time Step:** Configurable (default: 0.5ms for spikes, 0.001s for ODE)

## Tensor Shapes

Understanding tensor dimensions is crucial:

### Stimulus Tensor
```
[batch, time, height, width] or [batch, height, width]
```

### Receptive Field Weights
```
[num_neurons, num_receptors]
```

### Filtered Currents
```
[batch, time, num_neurons]
```

### Spike Trains
```
[time_steps, num_neurons]  # Binary (0 or 1)
```

## Pipeline Modes

### 1. Interactive (GUI)

- Real-time visualization
- Parameter tuning
- Exploratory analysis

### 2. Scripted (Python API)

- Reproducible pipelines
- Custom analysis
- Integration with ML workflows

### 3. Batch (CLI + YAML)

- Large-scale simulations
- Parameter sweeps
- Dataset generation

## Reproducibility

SensoryForge ensures reproducibility through:

1. **Deterministic Seeding** — Fixed random seeds
2. **YAML Configs** — Declarative pipeline specification
3. **Checkpoint/Resume** — Batch execution can be resumed
4. **Metadata Tracking** — All runs record full configuration

**Example:**
```yaml
pipeline:
  seed: 42  # Reproducible random number generation
```

## Differentiability

All components are PyTorch `nn.Module` instances, enabling:

- **Gradient Computation** — Backprop through entire pipeline
- **ML Integration** — Train encoding parameters end-to-end
- **Adjoint Method** — Memory-efficient gradients through ODEs (via torchdiffeq)

## Extensibility Points

SensoryForge uses a **registry-based architecture** for extensibility. All components are registered and can be extended without modifying core code:

1. **Grid Layouts** — Custom spatial arrangements (register in `GRID_REGISTRY`)
2. **Innervation Patterns** — Custom receptive field generators (register in `INNERVATION_REGISTRY`)
3. **Filters** — Custom temporal dynamics (register in `FILTER_REGISTRY`)
4. **Neuron Models** — Via DSL or hand-written `nn.Module` (register in `NEURON_REGISTRY`)
5. **Stimuli** — Custom stimulus generators (register in `STIMULUS_REGISTRY`)
6. **Solvers** — Custom ODE integrators (register in `SOLVER_REGISTRY`)

### Registry Pattern

```python
from sensoryforge.registry import NEURON_REGISTRY
from sensoryforge.neurons.base import BaseNeuron

class MyCustomNeuron(BaseNeuron):
    # Implementation...
    pass

# Register your custom component
NEURON_REGISTRY.register("my_custom_neuron", MyCustomNeuron)

# Use it in configs
config = {
    'populations': [{
        'neuron_model': 'my_custom_neuron',  # Automatically found via registry
        # ... other config
    }]
}
```

See [Extensibility Guide](../developer_guide/extensibility.md) for detailed instructions.

## Units Reference

| Quantity | Units | Example |
|----------|-------|---------|
| Space | mm | `spacing: 0.15` |
| Time | ms | `t_plateau: 500` |
| Current | mA | Internal |
| Voltage | mV | `v_init: -65.0` |
| Angle | degrees | `orientation: 45` |
| Frequency | cycles/mm | `frequency: 1.0` |

## Next Steps

- [First Simulation](first_simulation.md) — Build your first pipeline
- [User Guide](../user_guide/overview.md) — Detailed component reference
- [Tutorials](../tutorials/quickstart_tutorial.md) — Hands-on examples
