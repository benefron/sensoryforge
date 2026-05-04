# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

### Environment Setup

```bash
# Recommended (conda)
conda env create -f environment.yml
conda activate sensoryforge
pip install -e .

# Or pip-only
pip install -e ".[dev]"

# Optional extras
pip install torchdiffeq torchode   # adaptive ODE solvers
pip install sympy                  # equation DSL
```

### Tests

```bash
pytest tests/                                                  # all tests
pytest tests/unit/test_filters.py -v                          # single module
pytest tests/unit/test_filters.py::TestSAFilter::test_name -v # single test
pytest --cov=sensoryforge --cov-report=html                   # with coverage
```

### Lint & Type Checking

```bash
black sensoryforge/                          # format
flake8 sensoryforge/                         # style
mypy --disallow-untyped-defs sensoryforge/   # types
pydocstyle --convention=google sensoryforge/ # docstrings
```

### Docs

```bash
mkdocs serve   # local preview at http://localhost:8000
mkdocs build   # static build
```

### Running the GUI

```bash
python sensoryforge/gui/main.py
```

### CLI

```bash
sensoryforge run examples/example_config.yml --duration 1000
sensoryforge validate examples/example_config.yml
sensoryforge list-components
```

---

## Architecture

### Data Flow

Every simulation follows this shape-annotated pipeline:

```
Stimulus  [batch, time, H, W]
    ↓  Innervation (Gaussian receptive fields → weighted sum)
    ↓  [batch, time, N_neurons]
    ↓  Filter (SAFilterTorch or RAFilterTorch — temporal dynamics)
    ↓  [batch, time, N_neurons]  in mA
    ↓  Neuron (Izhikevich / AdEx / MQIF / DSL-compiled)
Spikes    [batch, time, N_neurons]  bool
```

- **Time unit:** ms at user-facing APIs; seconds in internal ODE integration
- **Spatial unit:** mm throughout
- **Batch dimension is always first:** `[batch, ...]`
- **No hand-rolled loops over neurons or spatial dims** — always vectorise with tensor broadcasting

### Execution Engines

There are two pipeline classes. **`SimulationEngine` is the canonical path** for all new development:

| Class | File | Use When |
|---|---|---|
| `SimulationEngine` | `core/simulation_engine.py` | **Canonical configs** (N populations, `SensoryForgeConfig`) — all new code |
| `GeneralizedTactileEncodingPipeline` | `core/generalized_pipeline.py` | Legacy configs; max 3 populations; also used as a stimulus generator inside `BatchExecutor` |

**Routing in the CLI and Batch executor:** canonical configs (has `grids` list + `populations` list, no `pipeline` key) are automatically routed through `SimulationEngine`. Legacy configs use `GeneralizedTactileEncodingPipeline`.

**GUI tabs** call `SimulationEngine._run_pop_from_drive()` (a shared static backend method) directly, after computing innervation-weighted drive locally.

`SimulationEngine.composite grids` — `_build_grids()` raises `NotImplementedError` for `arrangement == "composite"`. Do not rely on `SimulationEngine` for composite configs yet.

### Configuration: Canonical vs Legacy

**Canonical format** (preferred — use `SensoryForgeConfig` dataclass):

```python
from sensoryforge.config.schema import SensoryForgeConfig, GridConfig, PopulationConfig
config = SensoryForgeConfig(grids=[GridConfig(...)], populations=[PopulationConfig(...)])
pipeline = GeneralizedTactileEncodingPipeline.from_config(config.to_dict())
```

**Legacy format** (still supported):

```python
config = {'pipeline': {'device': 'cpu'}, 'neurons': {'sa_neurons': 100}, ...}
pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
```

The GUI and CLI both produce canonical format YAML. `SensoryForgeConfig` handles round-trip fidelity: `from_dict()` / `to_dict()` / `from_yaml()` / `to_yaml()`.

### Registry System

All components (neurons, filters, innervation, stimuli, solvers, grids) are registered by string name and created dynamically. This is the extensibility backbone.

```python
from sensoryforge.registry import NEURON_REGISTRY, FILTER_REGISTRY
NEURON_REGISTRY.register("my_neuron", MyNeuronClass)
```

`sensoryforge/register_components.py` calls `register_all()` which registers all built-in components. This is called at import time in both pipeline classes — every new component must be added here.

### Adding a New Component

Every component must:
1. Inherit from the appropriate base class (`BaseFilter`, `BaseNeuron`, `BaseStimulus`)
2. Implement `forward()`, `reset_state()`, `from_config()`, `to_dict()`
3. For stimuli: implement `get_param_spec()` returning a list of `ParamSpec` objects for GUI auto-discovery
4. Be registered in `register_components.py`
5. Have Google-style docstrings with tensor shapes (`[batch, time, N]`) and physical units (`mA`, `mV`, `ms`, `mm`)
6. Have a corresponding unit test

See `docs/developer_guide/add_stimulus.md`, `add_neuron.md`, `add_filter.md` for step-by-step guides.

### ParamSpec and get_param_spec()

`BaseStimulus` defines `get_param_spec() → list[ParamSpec]` (default `[]`).  
`ComponentRegistry.get_param_spec(name)` delegates to the registered class.  
The GUI Stimulus Designer uses this for auto-generated parameter spinboxes.

```python
from sensoryforge.stimuli.base import ParamSpec

@classmethod
def get_param_spec(cls):
    return [
        ParamSpec("amplitude", dtype="float", default=1.0,
                  min_val=0.0, max_val=500.0, unit="mA"),
    ]
```

### GUI Structure

The GUI (`sensoryforge/gui/main.py`) is a PyQt5 `QMainWindow` with **five tabs**:

1. **MechanoreceptorTab** — spatial grid config, receptor population setup, receptive field visualisation
2. **StimulusDesignerTab** — interactive stimulus creation with live preview
3. **SpikingNeuronTab** — neuron model config, run simulation, view spike raster; uses `SimulationEngine._run_pop_from_drive()` as shared backend
4. **VisualizationTab** — post-simulation analysis with dark theme pyqtgraph DockArea panels (drag/float/split)
5. **BatchTab** — parameter sweep execution and SLURM script export

The GUI reads/writes `SensoryForgeConfig`. Export to YAML → run via CLI for batch scaling.

**Project management:** `ExperimentManager` (`core/experiment_manager.py`) owns a project directory (`stimuli/`, `results/`, `figures/`). `SensoryForgeWindow` holds one instance and pushes it to all tabs via `set_experiment_manager()`.

**Expert mode:** Each tab has a `chk_expert_mode` `QCheckBox` pinned at the top of the control panel. When unchecked (Basic mode, default), advanced widgets are hidden via `w.setVisible(False)` on each widget in `self._expert_only_widgets` (MechanoreceptorTab) or `self._expert_only_widgets_spiking` (SpikingNeuronTab). State is persisted via `QSettings` keys `"gui/mechanoreceptor_tab/expert_mode"` and `"gui/spiking_tab/expert_mode"`.

**Per-column neuron toggle (MechanoreceptorTab):** `chk_square_neurons` checkbox + `spin_neurons_per_col` spinbox in Population Settings. When checked (default), `neuron_cols` is forced equal to `neuron_rows`. When unchecked, `spin_neurons_per_col` becomes visible and `neuron_cols` is set independently. Mirrors the receptor grid's `chk_square_grid` pattern.

**CSV Population Import/Export (MechanoreceptorTab):** The `_CSVPopulationModule` dataclass (`mechanoreceptor_tab.py`) mimics the `FlatInnervationModule` interface (`neuron_centers`, `innervation_weights`, `receptor_coords`, `num_neurons`) so CSV-imported populations work with the existing `_update_innervation_graphics_flat` code path. Export writes `neuron_positions.csv` (x,y mm), `innervation_weights.csv` (N×M), and `manifest.json` to a user-selected folder. Import reads the same folder, validates receptor count against the current grid (falls back to zero-fill if mismatched), and attaches the stub as `pop.flat_module`. The `csv_folder: Optional[str]` field on `NeuronPopulation` marks whether a population uses CSV data and prevents regeneration during `_generate_populations()`.

### Backend / Frontend Contract

When the GUI runs a simulation:

1. `SpikingNeuronTab._simulate_population()` computes innervation-weighted drive from stimulus frames using the local `InnervationModule`.
2. The drive tensor `[1, T, N]` is passed to `SimulationEngine._run_pop_from_drive(drive, filter_module, neuron_model, ...)`.
3. The static method applies filter → gain → noise → neuron and returns `{"spikes": ..., "drive": ..., "filtered": ..., "voltages": ...}`.
4. The tab collects per-population results and emits `simulation_finished(sim_results, ...)`.
5. `VisualizationTab.set_simulation_results(...)` receives the data.

This ensures the GUI simulation path and the `SimulationEngine.run()` path produce identical outputs for the same filter + neuron + gain settings.

### Known Technical Debt

These are open issues documented in `reviews/CODE_REVIEW_20260408.md`:

- **`SimulationEngine` composite grids** — `_build_grids()` raises `NotImplementedError` for `arrangement == "composite"`. Do not rely on `SimulationEngine` for composite configs yet.
- **Equation DSL (`model_dsl.py`) is numpy-only** — DSL-compiled neuron models do not support CUDA or autograd. They work on CPU only. This is tracked as C-2 in the review.
- **`reset_states()` vs `reset_state()`** — `SAFilterTorch` exposes `reset_states()` (plural), breaking polymorphism with the `BaseFilter` interface. New filters must use `reset_state()` (singular).
- **`BatchExecutor` stimulus shape** — `_canonical_to_legacy_config` sets `grid_size = rows*cols`, which `GeneralizedTactileEncodingPipeline` treats as per-side count. The executor works around this with bilinear interpolation, but the root bug in `_canonical_to_legacy_config` is unresolved.
- **`input_gain` unit mismatch** — The SA/RA filter parameters (`k1=0.05`, etc.) were calibrated by Pierzowski (1995) for stimulus inputs in N/mm². SensoryForge uses mA as its stimulus amplitude unit. The mismatch means the filter output is ~50× smaller than expected for a "1 mA" stimulus. The default `input_gain` in `PopulationConfig` and the SpikingNeuronTab spinbox is **50** to compensate. Do not set `input_gain=1` with default filter parameters — the neuron will receive sub-threshold current. See `docs/user_guide/units_and_gains.md`.

---

## Code Conventions

### Docstrings

Google style, mandatory for all public APIs. Always include tensor shapes and units:

```python
def forward(self, x: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
    """Apply SA filter to stimulus.

    Args:
        x: Input stimulus [batch, time, num_neurons] in mA
        dt: Time step in seconds

    Returns:
        Filtered currents [batch, time, num_neurons] in mA

    Example:
        >>> f = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=1.0)
        >>> out = f(torch.randn(2, 100, 64))
    """
```

### Input Validation

Use explicit `raise ValueError`, never `assert`:

```python
# Wrong — stripped by python -O
assert x.dim() == 3

# Right
if x.dim() != 3:
    raise ValueError(f"Expected 3-D input [batch, time, neurons], got shape {list(x.shape)}")
```

### Exception Handling

Catch specific exceptions, never bare `except Exception`. In the CLI, surface tracebacks via `--verbose`.

### Commits

Follow Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:`, `perf:`, `style:`. Scope is optional: `feat(filters): ...`.

---

## Documentation Layout

- `docs/` — user-facing docs (ships publicly with MkDocs)
- `docs_root/` — internal notes, research, working plans (not published)
- `reviews/` — engineering review artifacts and remediation plans
