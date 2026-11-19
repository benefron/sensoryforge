# Installation

## Requirements

- Python 3.10 or later
- PyTorch 1.12.0 or later
- (Optional) CUDA-capable GPU for acceleration

## Install SensoryForge

Not yet published to PyPI; install from source. This one install is for running
SensoryForge **and for extending it**: a new neuron, filter, stimulus, solver, grid or receptive-field builder is a
small separate package that SensoryForge discovers, so you do not need to fork or clone
SensoryForge to add one.

```bash
python -m venv .venv && source .venv/bin/activate     # or a conda environment
pip install "git+https://github.com/benefron/sensoryforge.git"
sensoryforge list-components
```

Optional extras go in brackets, in the same line:

```bash
pip install "sensoryforge[hdf5] @ git+https://github.com/benefron/sensoryforge.git"
```

| Extra | Adds |
|---|---|
| `hdf5` | HDF5 batch output |
| `solvers` | adaptive ODE solvers (torchdiffeq) |
| `dsl` | the equation DSL (sympy) |
| `gui` | the PyQt5 GUI |

Combine them with a comma: `sensoryforge[gui,dsl]`.

## Extend SensoryForge

Add a component without touching SensoryForge's own files:

```bash
sensoryforge new-component neuron MyNeuron   # also: filter, stimulus, solver, grid, innervation
cd sensoryforge-my-neuron
pip install -e .                             # installs your plugin, not SensoryForge
pip install pytest
pytest                                       # the shared contract checks
sensoryforge list-components                 # your component is listed
```

`new-component` writes a standalone package: its own `pyproject.toml`, a `register()`
function that SensoryForge finds through the `sensoryforge.components` entry-point
group, and a contract test. It never writes into SensoryForge's installed files. Read
[Plugin Packages](../developer_guide/plugins.md) and the
[Extensibility Overview](../developer_guide/extensibility.md); each kind has its own guide
([neuron](../developer_guide/add_neuron.md), [filter](../developer_guide/add_filter.md),
[stimulus](../developer_guide/add_stimulus.md)).

`new-component` covers neurons, filters, stimuli, solvers and grids. For a receptive-field
builder or a processing layer, start from the
[plugin template](https://github.com/benefron/sensoryforge/tree/main/examples/plugin_template)
in the repository: a complete plugin package with one of each.

## Run something first

You do not need an example file to start. Four presets ship in the package:

```bash
sensoryforge list-presets
sensoryforge run --preset tactile_sa1_ra1 --duration 100
```

The `examples/` folder in the repository (more configs, a notebook, scripts) is not in the
installed package. Fetch the file you need, for example:

```bash
curl -O https://raw.githubusercontent.com/benefron/sensoryforge/main/examples/canonical_config.yml
```

## Develop SensoryForge itself

To change SensoryForge's own code (fix a bug, change the engine, add to the core), work in
a clone. Fork the repository on GitHub first if you will send a pull request.

```bash
git clone https://github.com/<you>/sensoryforge.git
cd sensoryforge
conda env create -f environment.yml && conda activate sensoryforge   # or a venv
pip install -e ".[dev]"
pytest tests/unit -q
```

[`AGENTS.md`](https://github.com/benefron/sensoryforge/blob/main/AGENTS.md) lists the
commands, the places that are dangerous to change and the rules; `CONTRIBUTING.md` covers
style and pull requests. Never run `pip install -e .` from a git worktree: it rewrites one
pointer that every checkout shares. Use `PYTHONPATH=<worktree>` there.

## Verify Installation

Test your installation:

```bash
# Check CLI is available
sensoryforge --help

# List available components
sensoryforge list-components

# Launch GUI
python -m sensoryforge.gui.main
```

Or in Python:

```python
import sensoryforge
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

print(f"SensoryForge version: {sensoryforge.__version__}")

# Test basic pipeline creation
pipeline = GeneralizedTactileEncodingPipeline.from_config({
    'pipeline': {'device': 'cpu', 'grid_size': 20},
    'neurons': {'sa_neurons': 10, 'ra_neurons': 10}
})

print("✓ SensoryForge installed successfully!")
```

## GPU Support

### CUDA (NVIDIA)

If you have an NVIDIA GPU with CUDA support:

```bash
# Verify PyTorch sees your GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Configure SensoryForge to use GPU in your YAML config or Python code:

```yaml
pipeline:
  device: cuda  # Use GPU
```

### Apple Silicon (M1/M2/M3)

For Apple Silicon Macs, use MPS backend:

```yaml
pipeline:
  device: mps  # Use Apple Metal Performance Shaders
```

Verify MPS support:

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Ensure you're in the correct environment
pip list | grep sensoryforge

# Reinstall if needed
pip uninstall sensoryforge
pip install "git+https://github.com/benefron/sensoryforge.git"
```

### GUI Issues

If the GUI doesn't launch:

```bash
# Ensure PyQt5 is installed
pip install PyQt5>=5.15.0

# On macOS, you may need:
pip install --upgrade pyqt5 pyqtgraph
```

### Missing Dependencies

If you encounter missing optional dependencies:

```bash
# For HDF5 support
pip install h5py

# For adaptive solvers
pip install torchdiffeq

# For full functionality
pip install -r requirements.txt
```

## Next Steps

- [Quick Start](quickstart.md) — Your first SensoryForge simulation
- [Core Concepts](concepts.md) — Understanding the architecture
- [First Simulation](first_simulation.md) — Step-by-step tutorial
