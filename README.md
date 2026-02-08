# SensoryForge

[![Tests](https://github.com/benefron/sensoryforge/workflows/Tests/badge.svg)](https://github.com/benefron/sensoryforge/actions)
[![Documentation](https://github.com/benefron/sensoryforge/workflows/Documentation/badge.svg)](https://benefron.github.io/sensoryforge)
[![PyPI version](https://badge.fury.io/py/sensoryforge.svg)](https://badge.fury.io/py/sensoryforge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Modular, extensible framework for simulating sensory encoding across modalities**

SensoryForge is a GPU-accelerated, PyTorch-based toolkit for exploring sensory encoding schemes inspired by neuroscience. Originally developed for tactile simulation with SA/RA dual-pathway processing, the architecture is fully **modality-agnostic** and supports vision, audition, and multi-modal fusion.

---

## ✨ Key Features

- 🧠 **Biologically Inspired:** Grounded in neuroscience principles (receptive fields, dual pathways, spiking dynamics)
- 🚀 **GPU Accelerated:** Built on PyTorch for efficient tensor operations
- 🔧 **Highly Extensible:** Plugin system for custom filters, neurons, and stimuli
- 🌐 **Modality Agnostic:** Same framework for touch, vision, audition, and more
- 🔬 **Adaptive ODE Solvers:** Optional torchdiffeq/torchode integration for stiff systems
- 📚 **Comprehensive Documentation:** Tutorials, examples, and API reference
- 📦 **Production Ready:** pip-installable with proper testing and CI/CD

---

## 🚀 Quick Start

### Installation

```bash
pip install sensoryforge
```

### Basic Example

```python
from sensoryforge import SensoryPipeline
from sensoryforge.stimuli import GaussianStimulus

# Load pipeline from configuration
pipeline = SensoryPipeline.from_config('config.yml')

# Create a simple Gaussian stimulus
stimulus = GaussianStimulus(
    center=(50, 50),  # Center position (mm)
    sigma=10,         # Spread (mm)
    amplitude=1.0,    # Peak intensity
    duration_ms=100   # Duration (ms)
)

# Run encoding pipeline
results = pipeline.encode(stimulus)

# Extract spike trains
spikes = results['spikes']  # [batch, time, num_neurons]
print(f"Generated {spikes.sum()} spikes across {spikes.shape[-1]} neurons")
```

### Touch Encoding Example

```python
from sensoryforge.core import SpatialGrid, create_sa_innervation, create_ra_innervation
from sensoryforge.filters import SAFilter, RAFilter
from sensoryforge.neurons import IzhikevichNeuron

# Create spatial grid
grid = SpatialGrid(size_mm=(100, 100), resolution_mm=1.0)

# Create SA and RA neuron populations
sa_innervation = create_sa_innervation(grid, num_neurons=100, sigma_mm=5.0)
ra_innervation = create_ra_innervation(grid, num_neurons=100, sigma_mm=3.0)

# Temporal filtering
sa_filter = SAFilter(tau_ms=10.0, gain=1.0)
ra_filter = RAFilter(tau_ms=5.0, gain=1.5)

# Spiking neurons
sa_neurons = IzhikevichNeuron(num_neurons=100, neuron_type='regular_spiking')
ra_neurons = IzhikevichNeuron(num_neurons=100, neuron_type='fast_spiking')

# Apply to stimulus
stimulus = torch.randn(1, 100, 100, 100)  # [batch, time, height, width]
sa_current = sa_filter(stimulus @ sa_innervation.T)
ra_current = ra_filter(stimulus @ ra_innervation.T)

sa_spikes, _ = sa_neurons(sa_current)
ra_spikes, _ = ra_neurons(ra_current)
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

### Touch Sensing
- Robotic tactile sensors
- Prosthetic feedback systems
- Texture classification
- Contact detection and tracking

### Vision
- Event-based cameras (DVS)
- ON/OFF pathway encoding
- Motion detection
- Edge enhancement

### Multi-Modal
- Audio-tactile fusion
- Cross-modal learning
- Sensor fusion for robotics

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

SensoryForge is designed for easy extension through base classes and a plugin system.

### Create a Custom Filter

```python
from sensoryforge.filters.base import BaseFilter
import torch

class MyCustomFilter(BaseFilter):
    """Custom temporal filter."""
    
    def __init__(self, config):
        super().__init__(config)
        self.threshold = config.get('threshold', 0.5)
    
    def forward(self, x, dt=None):
        """Apply custom filtering."""
        return torch.relu(x - self.threshold)
    
    def reset_state(self):
        """Reset internal state."""
        pass
```

### Register and Use

```python
from sensoryforge.plugins.registry import registry

# Auto-discover plugins
registry.discover_plugins(Path('sensoryforge/plugins'))

# Use custom filter
FilterClass = registry.get_filter('MyCustomFilter')
my_filter = FilterClass({'threshold': 0.3})
```

See the [Extension Guide](https://benefron.github.io/sensoryforge/extending/filters/) for detailed tutorials.

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

- [Basic Pipeline](examples/notebooks/01_basic_pipeline.ipynb) - Complete encoding workflow
- [Custom Components](examples/notebooks/02_custom_components.ipynb) - Creating extensions
- [Touch Encoding](examples/notebooks/03_touch_encoding.ipynb) - SA/RA dual pathways
- [Vision Encoding](examples/notebooks/04_vision_encoding.ipynb) - Event cameras
- [Multi-Modal](examples/notebooks/05_multimodal.ipynb) - Combining modalities

### Python Scripts

- [Simple Touch](examples/scripts/simple_touch.py) - Minimal working example
- [Texture Classification](examples/scripts/texture_classification.py) - Practical application
- [Parameter Sweep](examples/scripts/parameter_sweep.py) - Batch experiments

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
