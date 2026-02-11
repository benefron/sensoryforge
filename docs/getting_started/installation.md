# Installation

## Requirements

- Python 3.8 or later
- PyTorch 1.12.0 or later
- (Optional) CUDA-capable GPU for acceleration

## Installation Methods

### From PyPI (Recommended)

```bash
pip install sensoryforge
```

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
# Install with all optional dependencies
pip install sensoryforge[full]

# Or install specific optional dependencies
pip install h5py>=3.0          # For HDF5 batch output
pip install torchdiffeq>=0.2.0 # For adaptive ODE solvers
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
pip install sensoryforge
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
