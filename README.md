# SensoryForge

[![Tests](https://github.com/benefron/sensoryforge/workflows/Tests/badge.svg)](https://github.com/benefron/sensoryforge/actions)
[![Documentation](https://github.com/benefron/sensoryforge/workflows/Documentation/badge.svg)](https://benefron.github.io/sensoryforge)
[![PyPI version](https://badge.fury.io/py/sensoryforge.svg)](https://badge.fury.io/py/sensoryforge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**An extensible playground for simulating sensory encoding and generating population activity across modalities**

SensoryForge is a GPU-accelerated, PyTorch-based toolkit for exploring sensory encoding schemes. Originally developed for tactile simulation with SA/RA dual-pathway processing, the architecture is fully **modality-agnostic** and supports vision, audition, custom modalities, and multi-modal fusion.

Designed for **neuroscientists**, **neuromorphic engineers**, and **ML researchers**, SensoryForge lets you:
- **Design experiments interactively** in the GUI — like a neuroscientist tuning an experiment at the bench
- **Scale simulations** via YAML configuration and the CLI for batch runs and parameter sweeps
- **Generate artificial datasets** for training and evaluating downstream systems
- **Test neuromorphic concepts** that can later be implemented in hardware

---

## ✨ Key Features

- 🧠 **Biologically Inspired:** Grounded in neuroscience principles (receptive fields, dual pathways, spiking dynamics)
- 🚀 **GPU Accelerated:** Built on PyTorch for efficient tensor operations
- 🔧 **Highly Extensible:** Add custom filters, neurons, stimuli, and entire modalities
- 🖥️ **Interactive GUI:** Design and test experiments visually before scaling
- 🌐 **Modality Agnostic:** Same framework for touch, vision, audition, and fabricated modalities
- 🔬 **Adaptive ODE Solvers:** Optional torchdiffeq/torchode integration for stiff systems
- 📐 **Equation DSL:** Define custom neuron models via equations — no coding required
- 📦 **Production Ready:** pip-installable with proper testing and CI/CD

---

## 🚀 Quick Start

### Installation

```bash
pip install sensoryforge
```

### Basic Example

**Using Canonical Configuration (Recommended):**

```python
from sensoryforge.config.schema import SensoryForgeConfig, GridConfig, PopulationConfig, StimulusConfig, SimulationConfig
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Create canonical config
config = SensoryForgeConfig(
    grids=[
        GridConfig(name="Main Grid", arrangement="grid", rows=80, cols=80, spacing=0.15)
    ],
    populations=[
        PopulationConfig(
            name="SA Population",
            neuron_type="SA",
            neuron_model="izhikevich",
            filter_method="sa",
            innervation_method="gaussian",
            neurons_per_row=10,
        ),
        PopulationConfig(
            name="RA Population",
            neuron_type="RA",
            neuron_model="izhikevich",
            filter_method="ra",
            innervation_method="gaussian",
            neurons_per_row=14,
        ),
    ],
    stimulus=StimulusConfig(type="gaussian", amplitude=30.0, sigma=0.5),
    simulation=SimulationConfig(device="cpu", dt=0.5),
)

# Create pipeline
pipeline = GeneralizedTactileEncodingPipeline.from_config(config.to_dict())

# Run simulation
results = pipeline.forward(stimulus_type='gaussian', amplitude=30.0, sigma=0.5)

# Extract spike trains
sa_spikes = results['sa_spikes']  # [time_steps, num_sa_neurons]
ra_spikes = results['ra_spikes']  # [time_steps, num_ra_neurons]
print(f"SA: {sa_spikes.sum()} spikes, RA: {ra_spikes.sum()} spikes")
```

**Using Legacy Configuration (Backward Compatible):**

```python
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Legacy format still works
pipeline = GeneralizedTactileEncodingPipeline.from_yaml('legacy_config.yml')
results = pipeline.forward(stimulus_type='gaussian', amplitude=30.0)
```

### Advanced Example: Custom Components via Registry

```python
from sensoryforge.core.grid import GridManager
from sensoryforge.core.innervation import create_sa_innervation, create_ra_innervation
from sensoryforge.filters.sa_ra import SAFilterTorch, RAFilterTorch
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
import torch

# Create spatial grid
grid_manager = GridManager(grid_size=80, spacing=0.15, device='cpu')

# Create SA and RA innervation
sa_innervation = create_sa_innervation(
    grid_manager, 
    neurons_per_row=10,
    connections_per_neuron=28,
    sigma_d_mm=0.3
)
ra_innervation = create_ra_innervation(
    grid_manager,
    neurons_per_row=14,
    connections_per_neuron=28,
    sigma_d_mm=0.39
)

# Temporal filtering
sa_filter = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=1.0)
ra_filter = RAFilterTorch(tau_RA=15.0, k3=2.0, dt=1.0)

# Spiking neurons
sa_neurons = IzhikevichNeuronTorch(dt=1.0, a=0.02, b=0.2, c=-65.0, d=8.0)
ra_neurons = IzhikevichNeuronTorch(dt=1.0, a=0.02, b=0.2, c=-65.0, d=8.0)

# Apply to stimulus [batch, time, height, width]
stimulus = torch.randn(1, 100, 80, 80)

# Process through pipeline
sa_input = sa_innervation(stimulus)
ra_input = ra_innervation(stimulus)
sa_filtered = sa_filter(sa_input)
ra_filtered = ra_filter(ra_input)
sa_spikes = sa_neurons(sa_filtered)[1]  # Returns (v_trace, spikes)
ra_spikes = ra_neurons(ra_filtered)[1]
```

---

## 📖 Documentation

- **[Installation Guide](https://benefron.github.io/sensoryforge/getting_started/installation/)** - Set up your environment
- **[Quick Start](https://benefron.github.io/sensoryforge/getting_started/quickstart/)** - Run your first simulation
- **[User Guide](https://benefron.github.io/sensoryforge/user_guide/overview/)** - Detailed documentation
- **[Tutorials](https://benefron.github.io/sensoryforge/tutorials/basic_pipeline/)** - Step-by-step guides
- **[API Reference](https://benefron.github.io/sensoryforge/api_reference/core/)** - Complete API documentation
- **[Extending SensoryForge](https://benefron.github.io/sensoryforge/extending/plugins/)** - Create custom components

---

## 🎯 Use Cases

### Neuroscience Research
- Simulate peripheral sensory encoding (touch, vision, audition)
- Test hypotheses about receptive field organization
- Explore dual-pathway processing (sustained vs. transient)
- Generate synthetic neural data matching biological statistics

### Neuromorphic Engineering
- Prototype spiking encoding schemes for hardware implementation
- Test event-based sensor designs (DVS, tactile arrays)
- Optimize encoding parameters for downstream processing
- Validate encoding-decoding pipelines

### Machine Learning & Data Generation
- Generate large-scale artificial spike-train datasets
- Create training data for spike-based classifiers
- Benchmark decoding algorithms
- Parameter sweeps via CLI for systematic exploration

### Multi-Modal Fusion
- Cross-modal encoding experiments
- Audio-tactile, visuo-tactile sensor fusion
- Custom/fabricated modality design

---

## 🧩 Architecture

SensoryForge uses a modular pipeline architecture:

```
Raw Stimulus → Grid → Innervation → Filters → Neurons → Spikes
                ↓
         [Receptive Fields]
                ↓
        ┌───────┴───────┐
        ↓               ↓
   Sustained Path   Transient Path
   (SA-like)        (RA-like)
        ↓               ↓
   Rate Coding    Spike Timing
```

### Core Components

- **Grid:** Spatial substrate (2D arrays, Poisson distributions, hexagonal, etc.)
- **Innervation:** Receptive field patterns (Gaussian, distance-weighted, sparse)
- **Filters:** Temporal dynamics (SA/RA, ON/OFF, custom dynamics)
- **Neurons:** Spiking models (Izhikevich, AdEx, MQIF, custom models)
- **Stimuli:** Input generation (Gaussian, textures, moving patterns, custom)

---

## 🔧 Extensibility

SensoryForge uses a **registry-based architecture** for easy extension. All components are registered and can be extended without modifying core code.

### Create a Custom Filter

```python
from sensoryforge.filters.base import BaseFilter
import torch

class MyCustomFilter(BaseFilter):
    """Custom temporal filter."""
    
    def __init__(self, threshold: float = 0.5, dt: float = 1.0):
        super().__init__(dt=dt)
        self.threshold = threshold
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply custom filtering.
        
        Args:
            x: Input tensor [batch, time, num_neurons] in mA
        
        Returns:
            Filtered output [batch, time, num_neurons] in mA
        """
        return torch.relu(x - self.threshold)
    
    @classmethod
    def from_config(cls, config: dict) -> 'MyCustomFilter':
        """Create from config dict."""
        return cls(
            threshold=config.get('threshold', 0.5),
            dt=config.get('dt', 1.0),
        )
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            'type': 'my_custom_filter',
            'threshold': self.threshold,
            'dt': self.dt,
        }
```

### Register and Use

```python
from sensoryforge.register_components import register_all
from sensoryforge.registry import FILTER_REGISTRY

# Register your custom component
FILTER_REGISTRY.register("my_custom_filter", MyCustomFilter)

# Ensure all components are registered
register_all()

# Use in config
config = {
    'populations': [{
        'filter_method': 'my_custom_filter',  # Automatically found via registry
        'filter_params': {'threshold': 0.3},
    }]
}
```

See the [Extensibility Guide](docs/developer_guide/extensibility.md) for detailed tutorials and the [Add Component Skill](.cursor/skills/add-new-component/SKILL.md) for step-by-step instructions.

---

## 🔬 Adaptive ODE Solvers

SensoryForge uses forward Euler by default, with optional adaptive solvers for stiff neuron models:

```python
from sensoryforge.solvers.adaptive import AdaptiveODESolver
from sensoryforge.neurons import IzhikevichNeuron

# Use Dormand-Prince (RK45) for accuracy-critical simulations
solver = AdaptiveODESolver(method='dopri5', rtol=1e-5, atol=1e-7)
neuron = IzhikevichNeuron(config, solver=solver)

# Or use the default forward Euler (no extra dependencies)
neuron = IzhikevichNeuron(config)  # solver='euler' by default
```

Install adaptive solver backends:
```bash
pip install torchdiffeq  # Dormand-Prince, adjoint method
pip install torchode     # GPU-parallel batched solving
```

See the [Solvers Guide](https://benefron.github.io/sensoryforge/user_guide/solvers/) for details.

---

## 📊 Examples

### Jupyter Notebooks

- [Quick Start](examples/notebooks/01_quickstart.ipynb) — Complete encoding workflow

### Python Scripts

- [Example Pipeline](examples/scripts/example_pipeline.py) — Basic touch encoding
- [Generalized Pipeline Demo](examples/scripts/generalized_pipeline_demo.py) — YAML-driven pipeline

---

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest

# Specific module
pytest tests/unit/test_filters.py -v

# With coverage
pytest --cov=sensoryforge --cov-report=html
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Installation

```bash
git clone https://github.com/benefron/sensoryforge.git
cd sensoryforge
pip install -e ".[dev]"
```

### Code Quality

We maintain high standards:
- Type hints required
- Tests required (minimum 80% coverage)
- Google-style docstrings
- Black code formatting
- Conventional commits

---

## 📄 Citation

If you use SensoryForge in your research, please cite:

```bibtex
@article{sensoryforge2026,
  title={SensoryForge: A Modular, Extensible Framework for Simulating Sensory Encoding Across Modalities},
  author={Your Name},
  journal={Journal of Open Source Software},
  year={2026},
  doi={10.xxxxx/joss.xxxxx}
}
```

---

## 📋 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🔗 Links

- **Documentation:** https://benefron.github.io/sensoryforge
- **GitHub:** https://github.com/benefron/sensoryforge
- **PyPI:** https://pypi.org/project/sensoryforge
- **Issue Tracker:** https://github.com/benefron/sensoryforge/issues

---

## 🙏 Acknowledgments

This project builds on foundational work in:
- Computational neuroscience (SA/RA encoding, dual-pathway processing)
- PyTorch ecosystem
- torchdiffeq and torchode (adaptive ODE solvers)
- Open-source scientific software community

---

## 📬 Contact

- **Maintainer:** [Your Name] ([your.email@example.com](mailto:your.email@example.com))
- **Discussions:** [GitHub Discussions](https://github.com/benefron/sensoryforge/discussions)
- **Issues:** [GitHub Issues](https://github.com/benefron/sensoryforge/issues)

---

**Built with ❤️ for the neuroscience and robotics communities**
