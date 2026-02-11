# SensoryForge Documentation

**An extensible playground for generating population activity in response to multiple stimuli and modalities.**

SensoryForge is a GPU-accelerated, PyTorch-based toolkit for simulating sensory encoding. It allows neuroscientists, neuromorphic engineers, and ML researchers to design encoding experiments interactively (GUI), scale them via YAML/CLI, and generate artificial datasets for downstream analysis.

---

## Workflow

1. **GUI** — Design experiments like a neuroscientist at the bench: configure grids, create stimuli, tune neuron models, and observe population responses in real time.
2. **CLI/YAML** — Export configurations and run large-scale simulations, parameter sweeps, or batch data generation.
3. **Python API** — Programmatic access for custom analysis, integration, and scripting.

---

## Quick Links

- [User Guide](user_guide/overview.md) — Full documentation of all components
- [CLI Reference](user_guide/cli.md) — Command-line interface usage
- [YAML Configuration](user_guide/yaml_configuration.md) — Declarative pipeline setup

## Core Components

| Component | Description |
|-----------|-------------|
| **Grid** | Spatial substrate (2D arrays, Poisson, hexagonal, composite multi-population) |
| **Innervation** | Receptive field patterns (Gaussian, distance-weighted, sparse) |
| **Filters** | Temporal dynamics (SA/RA dual-pathway, custom) |
| **Neurons** | Spiking models: Izhikevich, AdEx, MQIF, FA, SA, Equation DSL |
| **Stimuli** | Gaussian bumps, textures (Gabor, gratings), moving patterns |
| **Solvers** | Forward Euler (default), adaptive (Dormand-Prince via torchdiffeq) |
| **CompositeGrid** | Multi-population receptor mosaics on shared coordinates |

## Feature Guides

- [Composite Grid](user_guide/composite_grid.md) — Multi-population spatial substrates
- [Equation DSL](user_guide/equation_dsl.md) — Define neuron models via equations
- [Solvers](user_guide/solvers.md) — ODE solver selection and configuration
- [Extended Stimuli](user_guide/extended_stimuli.md) — Texture and moving stimuli
- [GUI Workflow](user_guide/gui_phase2_access.md) — GUI design → CLI scale workflow

## Getting Started

```python
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

# Load from YAML config
pipeline = GeneralizedTactileEncodingPipeline.from_yaml('config.yml')
results = pipeline.forward(duration_ms=1000)
```

Or launch the GUI:
```bash
python sensoryforge/gui/main.py
```
