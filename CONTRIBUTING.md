# Contributing to SensoryForge

## Setting up a development environment

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate sensoryforge
pip install -e .
```

### Pip only

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Verify the install

```bash
python -c "import sensoryforge; print(f'SensoryForge v{sensoryforge.__version__} imported successfully!')"
sensoryforge list-components
```

### Optional extras

SensoryForge splits optional functionality into extras (see `pyproject.toml`):

```bash
pip install -e ".[gui]"      # PyQt5 GUI (python -m sensoryforge.gui.main)
pip install -e ".[hdf5]"     # HDF5 batch export
pip install -e ".[solvers]"  # Adaptive ODE solvers (torchdiffeq)
pip install -e ".[dsl]"      # Equation-based neuron model DSL (sympy)
pip install -e ".[docs]"     # mkdocs + mkdocstrings, for building the docs site
```

Forward Euler is the default solver and works fine for most models; the adaptive solvers are
recommended for stiff neuron models like AdEx.

### PyTorch GPU / CPU

`environment.yml` installs `pytorch::pytorch` (CUDA if available). For CPU-only, or a specific CUDA
version, edit that dependency line before creating the environment (see comments there), or install
CPU-only torch directly: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

### IDE setup

- **VS Code:** Command Palette → "Python: Select Interpreter" → choose the `sensoryforge` conda
  environment (or your venv). Install the Jupyter extension for notebook support if needed.
- **PyCharm:** Preferences → Project → Python Interpreter → gear icon → Add → Conda Environment →
  select `sensoryforge`.

### Updating dependencies

```bash
conda env update -f environment.yml --prune
```

## Running tests

The test suite is split into a `gui` marker (Qt-backed tests, order-dependent, run in one process
via `pytest -m gui`) and everything else:

```bash
pytest -m "not gui"   # fast, no display required
pytest -m gui          # Qt tests; set QT_QPA_PLATFORM=offscreen in CI or headless environments
pytest --cov=sensoryforge tests/   # with coverage
```

See `CLAUDE.md` for the full command reference (linting, type checking, docs) and the current
architecture overview.

## Adding a new component

Components (neurons, filters, innervation methods, stimuli, solvers, grids) are registered by
string name in `sensoryforge/register_components.py` and discovered dynamically — see:

- `docs/developer_guide/add_neuron.md`
- `docs/developer_guide/add_filter.md`
- `docs/developer_guide/add_stimulus.md`
- `docs/developer_guide/extensibility.md` for the registry pattern itself

Every new component needs: a base-class subclass implementing `forward()`, `reset_state()`,
`from_config()`, `to_dict()`; a `sensoryforge/register_components.py` entry; Google-style docstrings with tensor
shapes and physical units; and a unit test.

## Commit conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`,
`refactor:`, `test:`, `build:`, `ci:`, `perf:`, `style:`. Scope is optional, e.g. `feat(filters): ...`.

### The living ledger

This repo tracks decisions and open findings in `docs_root/LEDGER.md` (gitignored except that one
file) using the [Lore pattern](https://arxiv.org/abs/2603.15566): commit trailers as the atomic unit
of institutional knowledge. When your commit records a decision, a finding, or closes/opens a
tracked issue, add the matching trailer:

```
Decision: <one line — a choice that was made and why>
Finding:  <one line — something discovered that isn't yet a tracked issue>
Opens:    F-0NN <one line describing the new issue>
Closes:   F-0NN
```

Only write `Closes:` once you've reproduced the original failing scenario and shown it now passes —
do not write "fixed" in a commit message or the ledger before that. After committing, run
`.claude/hooks/ledger-sync.sh` to fold new trailers into `docs_root/LEDGER.md`, and commit that file
separately as `chore: ledger sync after <task>`. See `docs_root/LEDGER.md`'s own header for the full
entry format and status vocabulary.

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
