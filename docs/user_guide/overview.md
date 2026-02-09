# User Guide Overview

## Vision

SensoryForge is an extensible playground for generating population activity in response to multiple stimuli and modalities. It is designed for:

- **Neuroscience research** — simulate peripheral encoding, test hypotheses about receptive fields and dual-pathway processing
- **Neuromorphic engineering** — prototype spiking encoding schemes for hardware implementation
- **ML & data generation** — generate artificial spike-train datasets for training and benchmarking
- **Education** — understand sensory encoding concepts through interactive experimentation

## Workflow

| Stage | Tool | Purpose |
|-------|------|---------|
| **Design** | GUI | Interactively configure grids, stimuli, neuron models; observe responses in real time |
| **Scale** | CLI + YAML | Run large-scale simulations, parameter sweeps, batch data generation |
| **Analyze** | Python API + Notebooks | Custom analysis, visualization, integration with ML pipelines |

## Core Components

### Spatial Grid & Innervation
Configure the receptor sheet — grid dimensions, spacing, and per-population receptive fields. Supports regular grids, Poisson disk, hexagonal, and jittered arrangements via CompositeGrid.

### Temporal Filters
SA (slowly adapting) and RA (rapidly adapting) filters model sustained and transient response pathways. Custom filters can be added.

### Spiking Neuron Models
Choose from Izhikevich, AdEx, MQIF, FA, SA models, or define custom dynamics via the Equation DSL. All models are `torch.nn.Module` subclasses with GPU support.

### Stimulus Library
Gaussian pressure bumps, texture patterns (Gabor, gratings, noise), moving stimuli (taps, slides, trajectories), and legacy stimulus generators.

### ODE Solvers
Forward Euler (default, fast) or adaptive solvers via torchdiffeq (Dormand-Prince, Adams, etc.) for stiff systems.

### CompositeGrid
Multi-population spatial substrates — model SA1/RA1/SA2 receptor mosaics in touch, L/M/S cone types in vision, or any custom population mix.

## Feature Guides

- [Composite Grid](composite_grid.md) — Multi-population spatial substrates
- [Equation DSL](equation_dsl.md) — Define neuron models via equations
- [Solvers](solvers.md) — ODE solver selection and configuration
- [Extended Stimuli](extended_stimuli.md) — Texture and moving stimuli
- [YAML Configuration](yaml_configuration.md) — Declarative pipeline setup
- [CLI Reference](cli.md) — Command-line interface
- [GUI Workflow](gui_phase2_access.md) — Interactive design → batch execution
