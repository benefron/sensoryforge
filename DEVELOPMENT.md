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

Note: You'll need a `setup.py` or `pyproject.toml` file in the project root. If these don't exist, you can create one (see below).

### 4. Verify Installation

Test that everything is working:

```bash
python -c "import sensoryforge; print('SensoryForge imported successfully!')"
```

### 5. Run the GUI

With the environment activated, run:

```bash
python sensoryforge/gui/main.py
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

## Creating a setup.py

If you need to create a `setup.py` for development installation, here's a template:

```python
from setuptools import setup, find_packages

setup(
    name="sensoryforge",
    version="0.2.0",
    description="Modular, extensible framework for simulating sensory encoding across modalities",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "torchaudio>=0.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "PyQt5>=5.15.0",
        "brian2>=2.4.2",
        "PyYAML>=6.0",
        "tqdm>=4.60",
        "matplotlib>=3.5",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "docs": [
            "mkdocs>=1.3.0",
            "mkdocs-material>=8.0.0",
            "mkdocs-include-markdown-plugin>=3.0.0",
        ],
    },
)
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

### Brian2 C++ Compilation

Brian2 requires a C++ compiler. On macOS, ensure Xcode Command Line Tools are installed:

```bash
xcode-select --install
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
