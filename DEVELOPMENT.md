# Development Setup Guide

## Setting Up the Conda Environment

SensoryForge requires a conda environment with all necessary dependencies. Follow these steps to get set up:

### 1. Create the Conda Environment

From the project root directory, run:

```bash
conda env create -f environment.yml
```

This will create a conda environment named `sensoryforge` with all required packages.

### 2. Activate the Environment

```bash
conda activate sensoryforge
```

### 3. Install SensoryForge in Development Mode

Install the package in editable/development mode so changes to the code are immediately reflected:

```bash
pip install -e .
```

### 4. Verify Installation

Test that everything is working:

```bash
python -c "import sensoryforge; print(f'SensoryForge v{sensoryforge.__version__} imported successfully!')"
```

### 5. Launch the GUI

The GUI is the primary experimentation tool:

```bash
python sensoryforge/gui/main.py
```

### 6. Use the CLI

For batch execution and scalability:

```bash
sensoryforge run examples/example_config.yml --duration 1000
sensoryforge validate examples/example_config.yml
sensoryforge list-components
```

## Alternative: Pip-Only Installation

If you prefer to use pip without conda, you can install dependencies from `requirements.txt`:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then proceed with step 3 above.

## Running Tests

With the environment activated, run pytest:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=sensoryforge tests/
```

## Building Documentation

Generate documentation with:

```bash
mkdocs build
```

Or serve documentation locally:

```bash
mkdocs serve
```

## Troubleshooting

### ModuleNotFoundError: No module named 'sensoryforge'

Make sure you've activated the conda environment and installed the package in development mode:

```bash
conda activate sensoryforge
pip install -e .
```

### PyTorch GPU Issues

If you have a different CUDA version or want CPU-only PyTorch, modify the environment.yml:

**For CPU-only:**
```yaml
- pytorch::pytorch
```

**For different CUDA version (e.g., 12.1):**
```yaml
- pytorch::pytorch::*=*cuda*
- pytorch::pytorch-cuda=12.1
```

### Adaptive ODE Solvers

For adaptive ODE solvers (recommended for stiff neuron models like AdEx):

```bash
pip install torchdiffeq torchode
```

These are optional — forward Euler is the default solver and works fine for most models.

### Equation DSL

For the equation-based neuron model DSL:

```bash
pip install sympy
```

## IDE Setup

### VS Code

1. In VS Code, select the Python interpreter from the conda environment:
   - Open Command Palette (Cmd+Shift+P)
   - Type "Python: Select Interpreter"
   - Choose the `sensoryforge` conda environment

2. (Optional) Install the Jupyter extension for notebook support

### PyCharm

1. Go to PyCharm → Preferences → Project → Python Interpreter
2. Click the gear icon → Add
3. Select "Conda Environment" and find the `sensoryforge` environment
4. Click OK

## Updating Dependencies

If you update `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

The `--prune` flag removes packages that are no longer in the file.

## Deactivating the Environment

When finished working on SensoryForge:

```bash
conda deactivate
```
