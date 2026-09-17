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
sensoryforge run examples/canonical_config.yml --duration 1000
sensoryforge validate examples/canonical_config.yml
sensoryforge list-components
```

---

## Architecture

### Data Flow

Every simulation follows this shape-annotated pipeline:

```
Stimulus  [batch, time, H, W]  (or [batch, time, C, H, W] with C > 1 channels, Wave L2)
    ↓  sampled at each receptor's own (x, y) mm position → responses [batch, time, M]
    ↓  (a regular grid whose resolution matches the frame takes a bit-identical reshape
    ↓   fast path instead; every other arrangement is truly sampled, Wave L3 / F-010)
    ↓  ReceptiveFieldBank (weights [N, M] from a registered builder: gaussian, uniform,
    ↓  one_to_one, distance_weighted, template, imported, or a plugin's)
    ↓  [batch, time, N_neurons]
    ↓  Filter (SAFilterTorch or RAFilterTorch — temporal dynamics)
    ↓  [batch, time, N_neurons]  in mA
    ↓  Neuron (Izhikevich / AdEx / MQIF / DSL-compiled)
Spikes [batch, time, N_neurons] bool   — or, for a DSL model with no threshold (Phase 2, Wave N):
State  [batch, time, N_neurons] float  — an analog (non-spiking) readout, see below
```

- **Time unit:** ms at user-facing APIs; seconds in internal ODE integration
- **Spatial unit:** mm throughout
- **Batch dimension is always first:** `[batch, ...]`
- **No hand-rolled loops over neurons or spatial dims** — always vectorise with tensor broadcasting
- **Coordinates are `(x, y)` in mm everywhere inside SensoryForge**; for `ReceptorGrid(grid_size=(rows, cols))` the first meshgrid index is x (`indexing="ij"`), so receptor `k = i * cols + j`. Convert at the boundary when importing pressure-simulation's `[y, x]` centres (the `imported` builder does).

### Analog readouts (Phase 2, Wave N)

A DSL neuron model (`NeuronModel`, `neurons/model_dsl.py`) may omit `threshold`/`reset`: with no
threshold, `compile()` integrates the equations every step and `forward()` returns `(state_trace,
None)` instead of `(v_trace, spikes)`. `SimulationEngine._run_pop_from_drive` then labels the
result `"state"` instead of `"spikes"` (bin-end samples, same reduction as `"voltages"`), and
`_build_populations` builds a DSL population from `PopulationConfig.dsl_config`;
`PopulationConfig.readout` (`"auto"`/`"spiking"`/`"analog"`) can force the interpretation, raising
when incompatible with the `dsl_config`. The Spiking tab plots the state trace (labelled with the
state variable's name) in place of the spike raster for such a population. Spiking populations
(models with a threshold) are unaffected. See `docs/user_guide/analog_readouts.md`.

### Receptive fields (Phase 2, Wave I)

Every population's receptive fields are one `ReceptiveFieldBank` (`core/rf_bank.py`: buffers `weights [N, M]`, `neuron_centers [N, 2]`, `receptor_coords [M, 2]`, a `provenance` dict; `save()`/`load()` as `.pt`). Banks are built by registered builders — `BaseInnervation` subclasses in `INNERVATION_REGISTRY` whose `build()` returns a bank — through one function, `innervation.build_population_bank()`, used by `SimulationEngine`, the three legacy pipelines and the GUI. `template` (`core/rf_builders/template.py`, D-020: sigma = d/pi, pitch = d, derived neuron count) and `imported` (`rf_builders/imported.py`) derive their own neuron centres (`DERIVES_NEURON_CENTERS = True`). `PopulationConfig.innervation_params` is merged last into the builder parameters; `BaseInnervation.filter_params()` keeps the keys each builder takes. `SimulationEngine.builder_params()` passes `max_sigma_distance=0` on the grid path (that path never had a cutoff; gaussian weights are bit-identical to earlier releases, pinned by `tests/fixtures/rf_engine_golden_weights.pt`). `InnervationModule`/`FlatInnervationModule` are deprecated wrappers over a bank (removal in Phase 4). See `docs/user_guide/receptive_fields.md` and `docs/developer_guide/add_rf_builder.md`.

### Execution Engines

There are two pipeline classes. **`SimulationEngine` is the canonical path** for all new development:

| Class | File | Use When |
|---|---|---|
| `SimulationEngine` | `core/simulation_engine.py` | **Canonical configs** (N populations, `SensoryForgeConfig`) — all new code |
| `GeneralizedTactileEncodingPipeline` | `core/generalized_pipeline.py` | Legacy configs; max 3 populations; also used as a stimulus generator inside `BatchExecutor` |

**Routing in the CLI and Batch executor:** canonical configs (has `grids` list + `populations` list, no `pipeline` key) are automatically routed through `SimulationEngine`. Legacy configs use `GeneralizedTactileEncodingPipeline`.

**GUI tabs** call `SimulationEngine._run_pop_from_drive()` (a shared static backend method) directly, after computing innervation-weighted drive locally.

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
1. Inherit from the appropriate base class (`BaseFilter`, `BaseNeuron`, `BaseStimulus`, `BaseSolver`, `BaseGrid`, `BaseInnervation`)
2. Implement `forward()`, `reset_state()`, `from_config()`, `to_dict()`
3. Implement `get_param_spec()` returning a list of `ParamSpec` objects (required on every component since G1, not just stimuli)
4. Round-trip every `__init__` parameter through `to_dict()`/`from_config()` — `to_dict()` must include every constructor argument (except any listed in `_TO_DICT_EXCLUDE_PARAMS`), and `from_config(instance.to_dict())` must be a fixed point. **This full-completeness check (H3) is currently enforced for neurons only** — `sensoryforge.testing.contracts._check_neuron` is the only one of the six `_check_<kind>` functions that calls `_assert_to_dict_roundtrip_complete`; filters, stimuli, grids, solvers and innervation get the other two contract checks (basic `from_config`/`to_dict` round-trip presence, `get_param_spec()` returning `ParamSpec` objects, one forward-pass shape check) but not full parameter-completeness yet. Several shipped built-ins would fail it today if it were applied to them (`SAFilterTorch`, `RAFilterTorch`, the grid arrangement classes, `EdgeGrating`) — not yet enforced for other kinds (see ledger F-049).
5. Be registered — two routes, pick one:
   - **Plugin package (default, for third parties):** `sensoryforge new-component <kind> <Name> [--dest DIR]` scaffolds an installable `sensoryforge-<name>/` package with a `pyproject.toml` entry point (`sensoryforge.components` group), discovered automatically at import time — no edits to this checkout at all. See `docs/developer_guide/plugins.md`.
   - **In-repo (for contributing to SensoryForge itself):** `sensoryforge new-component <kind> <Name> --in-repo` writes into `sensoryforge/`, `tests/`, `docs/` directly; you then add one line to `register_components.py`'s `register_all()` by hand (printed out by the command).
6. Have Google-style docstrings with tensor shapes (`[batch, time, N]`) and physical units (`mA`, `mV`, `ms`, `mm`)
7. Have a corresponding unit test — `sensoryforge.testing.contracts.check_component(kind, cls)` runs the shared shape/round-trip/param-spec checks and is what both routes' generated tests call

Component names are matched case-insensitively (H1, F-046) — `"MyFilter"` and `"myfilter"` collide as the same registration, so don't rely on case to avoid a name clash with a built-in.

See `docs/developer_guide/add_stimulus.md`, `add_neuron.md`, `add_filter.md`, `plugins.md` for step-by-step guides.

### ParamSpec and get_param_spec()

Every base class defines `get_param_spec() → list[ParamSpec]` (default `[]`), required on all components since G1.
`ComponentRegistry.get_param_spec(name)` delegates to the registered class.  
The GUI Stimulus Designer (and other tabs) use this for auto-generated parameter spinboxes.

```python
from sensoryforge.stimuli.base import ParamSpec

@classmethod
def get_param_spec(cls):
    return [
        ParamSpec("amplitude", dtype="float", default=1.0,
                  min_val=0.0, max_val=500.0, unit="mA",
                  choices=None, help="Peak stimulus amplitude.",
                  group="Amplitude", advanced=False),
    ]
```

`ParamSpec` also carries `choices` (enum/dropdown values), `help` (longer-form text vs. `tooltip`), `group` (UI section label), and `advanced` (hidden unless Expert mode) — all added in G1.

### GUI Structure

The GUI (`sensoryforge/gui/main.py`) is a PyQt5 `QMainWindow` with **six tabs**:

0. **CircuitTab** (`gui/tabs/circuit_tab.py`, Phase 3) — the node-graph editor and the entry point. Node classes in `gui/circuit/nodes.py` map one-to-one onto the config dataclasses; `gui/circuit/serialise.py` converts graph ↔ `SensoryForgeConfig` losslessly (round-trip tested over every example and preset) and keeps node positions advisorily in `<config>.layout.json`. The inspector (`gui/circuit/inspector.py`) renders parameters from `get_param_spec()`, so a plugin component gets a settings form with no GUI code. A new *node type* or custom preview is still an in-repo change: `NODE_CLASSES` and the preview dispatch are hard-coded tables.
1. **MechanoreceptorTab** — spatial grid config, receptor population setup, receptive field visualisation
2. **StimulusDesignerTab** — interactive stimulus creation with live preview
3. **SpikingNeuronTab** — neuron model config, run simulation, view spike raster; uses `SimulationEngine._run_pop_from_drive()` as shared backend
4. **VisualizationTab** — post-simulation analysis with dark theme pyqtgraph DockArea panels (drag/float/split)
5. **BatchTab** — parameter sweep execution and SLURM script export

The GUI reads/writes `SensoryForgeConfig`. Export to YAML → run via CLI for batch scaling.

**Project management:** `ExperimentManager` (`core/experiment_manager.py`) owns a project directory (`stimuli/`, `results/`, `figures/`). `SensoryForgeWindow` holds one instance and pushes it to all tabs via `set_experiment_manager()`.

**Expert mode:** Each tab has a `chk_expert_mode` `QCheckBox` pinned at the top of the control panel. When unchecked (Basic mode, default), advanced widgets are hidden via `w.setVisible(False)` on each widget in `self._expert_only_widgets` (MechanoreceptorTab) or `self._expert_only_widgets_spiking` (SpikingNeuronTab). State is persisted via `QSettings` keys `"gui/mechanoreceptor_tab/expert_mode"` and `"gui/spiking_tab/expert_mode"`.

**Per-column neuron toggle (MechanoreceptorTab):** `chk_square_neurons` checkbox + `spin_neurons_per_col` spinbox in Population Settings. When checked (default), `neuron_cols` is forced equal to `neuron_rows`. When unchecked, `spin_neurons_per_col` becomes visible and `neuron_cols` is set independently. Mirrors the receptor grid's `chk_square_grid` pattern.

**Populations hold banks (MechanoreceptorTab):** `NeuronPopulation.bank` is a `ReceptiveFieldBank` built by `instantiate()` (grid lattice; `grid_shape` set so `innervation_weights` reads as `[N, rows, cols]` for the heatmap) or `instantiate_flat()` (composite / flat coordinates; `grid_shape=None`), with the same builder parameters `SimulationEngine.builder_params()` uses (GUI-engine parity). The method combo offers `template` with a resolvable-distance spinbox; `lbl_population_info` shows the derived neuron count.

**CSV Population Import/Export (MechanoreceptorTab):** `export_population_csv(pop, folder)` writes `neuron_positions.csv` (x,y mm), `innervation_weights.csv` (N×M), `bank.pt` and `manifest.json`; `import_population_csv(pop, folder)` builds the bank with the `imported` builder on the current grid's receptor coordinates (a receptor-count mismatch raises, no zero-fill), marks the population `innervation_method="imported"` with the path in `innervation_params`, and sets `csv_folder`, which prevents regeneration during `_generate_populations()`. The dialog handlers wrap these two methods.

### Backend / Frontend Contract

When the GUI runs a simulation:

1. `SpikingNeuronTab._simulate_population()` computes the drive from stimulus frames with the population's `ReceptiveFieldBank` (frames flattened to `[1, T, H*W]`).
2. The drive tensor `[1, T, N]` is passed to `SimulationEngine._run_pop_from_drive(drive, filter_module, neuron_model, ...)`.
3. The static method applies filter → gain → noise → neuron and returns `{"spikes": ..., "drive": ..., "filtered": ..., "voltages": ...}`.
4. The tab collects per-population results and emits `simulation_finished(sim_results, ...)`.
5. `VisualizationTab.set_simulation_results(...)` receives the data.

This ensures the GUI simulation path and the `SimulationEngine.run()` path produce identical outputs for the same filter + neuron + gain settings.

### Known Technical Debt

**The living ledger (`docs_root/LEDGER.md`) is the current source of truth for open findings.**

**Open as of 2026-09-17, and the hazards to know before changing things:**

- **Never run `pip install -e .` from a git worktree** (F-053). The conda environment is shared, and an editable install rewrites one global pointer, silently repointing every other checkout; subprocess-launched code then imports the wrong tree while tests still pass.
- **Golden fixtures have only run on macOS arm64** (F-071): `tests/integration/test_pressure_sim_parity.py`, `test_stimulus_parity.py` and `tests/fixtures/rf_engine_golden_weights.pt`. A failure on another platform may be floating-point rounding; diagnose before loosening a test or changing code. `tests/validation/test_reproducibility.py` is already platform-aware (F-068).
- **Peak memory from the watchdog is noise below about 2x** (F-056); never compare the figure across runs. Use `benchmarks/` for performance, whose CI guard is calibrated against a reference kernel (F-067) and catches only regressions of about 3x or more.
- **Continuous integration has never run on GitHub** for this repository, and the docs site has never been deployed.
- **Comparison with published afferent data is qualitative only** (F-070).
- **Voltage clamp divergence from pressure-simulation** under strongly negative drive (F-037); **flake8 debt beyond the CI subset** (F-036); **GUI tests disable the cyclic garbage collector** because of a pyqtgraph segfault (F-035); **the Circuit validator misses a sum-combine with mismatched neuron counts** (F-062); **Circuit previews duplicate the other tabs' drawing code** (F-063).

The items below were open as of 2026-09-14 (ledger `F-0NN` IDs); see
`docs/development/reviews/` for the historical audits they came from, and note several items that
audit once listed here (DSL/CUDA support, `reset_states`) were already fixed — see ledger `R-001`,
`D-011`.

- **`input_gain` unit mismatch** — The SA/RA filter parameters (`k1=0.05`, etc.) were calibrated by Parvizi-Fard et al. (2021, J. Neurophysiol.) for stimulus inputs in N/mm² (τ_RA follows Kandel, Principles of Neural Science, Ch. 21). SensoryForge uses mA as its stimulus amplitude unit. The mismatch means the filter output is ~50× smaller than expected for a "1 mA" stimulus. The default `input_gain` in `PopulationConfig` and the SpikingNeuronTab spinbox is **50** to compensate. Do not set `input_gain=1` with default filter parameters — the neuron will receive sub-threshold current. See `docs/user_guide/units_and_gains.md`.
- **Legacy `neurons.sa_neurons`/`ra_neurons` mean neurons-**per-row**, not a total count** — `InnervationModule` squares it. A config whose dense weight tensor would exceed 2e8 elements raises `ValueError`; smaller mistakes still build silently. Canonical configs are unaffected. (F-023)

**Resolved 2026-09-15, Phase 2 Wave I:** receptor grids take a `seed` and random arrangements are reproducible (F-050); `innervation_method` is honoured on ordinary grids and every population's receptive fields are a `ReceptiveFieldBank` built by a registered builder (F-051, D-020).

**Resolved 2026-09-16, Phase 2 Wave L:** `SimulationEngine` samples the stimulus at every
receptor's real `(x, y)` coordinate (bilinear interpolation via `torch.nn.functional.grid_sample`)
instead of assuming receptor index equals stimulus pixel index; a regular `"grid"` arrangement
whose resolution matches the frame stays on a bit-identical reshape fast path. Every population's
receptive-field bank is now built on the target grid's real receptor coordinates for every
arrangement, not only `"grid"` (hex/poisson/jittered/blue-noise no longer fall back to a synthetic
lattice). `SimulationEngine._build_grids()` builds `arrangement == "composite"` grids from
`GridConfig.layers`, and `GridConfig.coords_file` imports an `[M, 2]` CSV/`.pt` of receptor
coordinates directly. `GridConfig.channels` names a grid's sensor planes (single-channel default
unchanged) and `StimulusConfig.channel` names which plane a stimulus drives; see
`docs/concepts/units_and_shapes.md`, `docs/concepts/sensor_arrays.md`. Closes F-010.

**Resolved 2026-09-16 to 2026-09-17, Phases 2 to 4** (see the ledger for each record): stimuli dispatch through `STIMULUS_REGISTRY` via `sensoryforge/stimuli/render.py`, with legacy defaults preserved for names both paths know and stepped stimuli such as `moving` driven frame by frame (F-052, F-057); multi-input populations, `combine` sum/concat and a `PROCESSING_REGISTRY` with `onoff` (Wave M; the processing kind is contract-checked, F-058); the data bundle carries `neuron_modules/` and a tagged stimulus payload so pressure-simulation's viewer can run it (F-054, F-055); `sensoryforge run` reports analog populations (F-060); the Stimulus Designer reloads its own config (F-064); a lattice-size warning fires only for sizes the user set (F-069); the benchmark CI guard is calibrated against a reference kernel (F-067); and the reproducibility check is exact on the reference's platform, tolerant elsewhere (F-068).

**Resolved 2026-09-16, Phase 2 Wave N:** a DSL model's `threshold`/`reset` are optional, giving an analog (non-spiking) readout (`(state_trace, None)`); the shared backend labels this `"state"` instead of `"spikes"`; `SimulationEngine` builds DSL populations from `dsl_config` (previously `TypeError`/`ValueError: Unknown neuron model`); the Spiking tab plots the state trace for such a population. See "Analog readouts" above and `docs/user_guide/analog_readouts.md`.

**Resolved 2026-09-14** (kept here briefly so agents don't re-propose them; see ledger for the full
decision records): `SAFilterTorch` no longer rectifies by default (F-001); the canonical→legacy
adapter no longer squares grid size or neuron counts (F-012, F-025); filter and Izhikevich defaults
are resolved by `sensoryforge/config/defaults.py` for the GUI, the engine, the canonical adapter,
`TactileEncodingPipelineTorch` and `CombinedSARAFilter` (F-026, F-032), and τ_RA is 8 ms everywhere;
overriding a single Izhikevich parameter keeps the neuron-type preset as the base instead of
dropping it (F-031); D-Q1 is decided -- RA filter gain k3 = 2.0 everywhere, resolver-owned (F-030);
`from_yaml` accepts long one-line text (F-027); presets are keyword-only and tested (F-028);
internal records are excluded from the docs build (F-029); seeded innervation runs on MPS/CUDA with CPU-identical wiring (F-038); CLI and batch stimuli honour `dt_ms` and `--duration` for every stimulus type, and the legacy `neurons.dt`/`temporal.dt` keys follow each other (F-024, F-039, F-040); GUI exports write the simulated step (F-041); `dt_ms` must be a whole multiple of `integrate_dt_ms` (F-042).

**Resolved 2026-09-15** (same purpose, next day's fixes): `examples/canonical_config.yml` and `examples/canonical_batch_config.yml` are schema-valid canonical examples, and the legacy `examples/example_config.yml`/`batch_config.yml` (and the matching docs snippets) now use per-row neuron counts and run (F-043); the stimulus Δt spinbox snaps to a multiple of the integration step on `editingFinished`, `SpikingNeuronTab._run_simulation` catches `ValueError` alongside `RuntimeError`, and `gui/main.py` installs a `sys.excepthook` that shows unhandled exceptions in a `QMessageBox` (F-044); concrete neuron models round-trip every constructor parameter through `to_dict()`/`from_config()`, not just `dt` (F-045); component-name registry lookups are now case-insensitive, so a plugin neuron/filter/etc. no longer needs an exact-case match against `SimulationEngine`'s lowercased `neuron_model` (F-046); `sensoryforge new-component` refuses to write under a `site-packages`/`dist-packages` directory and its default mode generates a standalone, installable entry-point plugin package instead of writing into the core checkout, with `--in-repo` preserving the old contributor workflow (F-047). The Phase 1 task list, with the review
of Wave A, is `docs/development/handover/phase1_tasks.md`.

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
- `docs/development/reviews/` — engineering review and audit artefacts (tracked; see its README for
  which are current vs. historical)
- `docs_root/` — internal notes, research, working plans; gitignored except `docs_root/LEDGER.md`,
  the living ledger of decisions/findings (see it for what is actually open today, not the stale
  Open/Resolved counts inside older review files)
