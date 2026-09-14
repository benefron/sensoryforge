# Installation

## Requirements

- Python 3.10 or later
- PyTorch 1.12.0 or later
- (Optional) CUDA-capable GPU for acceleration

## Installation Methods

Not yet published to PyPI; install from source.

### From Source

```bash
# Clone the repository
git clone https://github.com/benefron/sensoryforge.git
cd sensoryforge

# Install in development mode
pip install -e .
```

### With Optional Dependencies

For full functionality, including HDF5 batch output and adaptive ODE solvers:

```bash
# From a source checkout, install specific optional dependencies
pip install -e ".[hdf5]"      # For HDF5 batch output
pip install -e ".[solvers]"   # For adaptive ODE solvers
pip install -e ".[dsl]"       # For the equation DSL
pip install -e ".[gui]"       # For the PyQt5 GUI
```

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
pip install -e .
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
