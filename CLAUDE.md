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
when incompatible with the `dsl_config`. The GUI's Results screen plots the state trace (labelled with the
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

**The GUI** runs through `SimulationEngine.run()` itself (`gui/execution/run_controller.py`), never a private path.

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

### GUI Structure (GUI v2)

`python sensoryforge/gui/main.py` (a shim onto `sensoryforge.gui.app.main`) opens `SensoryForgeApp` (`gui/app.py`): a left **stage list** — Sensors · Stimulus · Populations · Run & Results · Batch — over a `QStackedWidget` of screens (`gui/screens/`, registered in `screens/__init__.py::SCREEN_FACTORIES`), a **pipeline strip** (`widgets/pipeline_strip.py`, one row per population, one chip per stage, status dot per chip), and a **run bar** (`widgets/run_bar.py`: device, duration, dt, seed, Run, Cancel). The node graph (Circuit tab) and the six old tabs were deleted in Phase 3 (2026-09-21).

- **One config.** `gui/session.py::Session` holds exactly one `SensoryForgeConfig`. Screens write with `session.set_by_path("populations.1.filter_params.tau_r", v)` and listen to `configChanged(path)` / `configReplaced()`. File > Save/Open config is plain `to_yaml`/`from_yaml_file` (which honours `plugins:`). A project (`gui/project.py`) is a folder with `config.yml` and `runs/<bundle>/`.
- **Validation.** `Session.errors` is `gui/validation.py::validate(config)`, recomputed on every change, keyed by stage (`grids.<i>`, `stimulus`, `populations.<i>.rf|combine|filter|neuron|readout`, ...). It builds receptive fields, filters and neurons with the engine's own code (`core/simulation_engine.py::build_filter`/`build_neuron`, and `SimulationEngine` for receptive fields, cached). The strip colours the failing chip, the run bar disables Run and says why, and each screen shows its own problems (`widgets/problem_list.py`).
- **Forms.** Every parameter form is `widgets/param_form.py::ParamForm`, generated from `get_param_spec()`. A form must show the value that runs: stimuli use `stimuli/render.py::effective_defaults`, neurons and filters the `config/defaults.py` resolvers, receptive-field inputs `SimulationEngine.builder_params()`; the tests re-run a config with every displayed value written explicitly and require identical output. Unset optional numbers show *auto*; values below 1e-3 use a scientific box.
- **Runs.** `gui/execution/run_controller.py::RunController` (a QThread worker) renders with `stimuli/render.py::render_for_config` and calls `SimulationEngine.run()`. Sweeps (`execution/sweep_controller.py`) run each job as a `sensoryforge` subprocess. `tests/integration/test_gui_engine_equality.py` pins GUI == engine == CLI.
- **Plots.** Built only by `widgets/plot_factory.py`; units go in the label text (never pyqtgraph `units=`, which SI-prefixes "1000 ms" as "1 kms"); pens are cosmetic; pyqtgraph signals connect only through `plot_factory.connect` (F-035). Build each plot once per screen lifetime. `widgets/figure_export.py` saves PNG/SVG.
- **Advanced toggle.** One toolbar checkbox (`gui/advanced`, via `gui_settings()`); a screen that has advanced rows defines `set_advanced(on)`, which the shell calls.
- **Receptive-field CSV folders.** `core/rf_builders/imported.py::write_csv_folder(bank, folder)` writes `neuron_positions.csv`, `innervation_weights.csv`, `bank.pt`, `manifest.json` (button on the Populations RF bench); the `imported` builder reads them back.

### Known Technical Debt

**The living ledger (`docs_root/LEDGER.md`) is the current source of truth for open findings.**

**Open as of 2026-09-21, and the hazards to know before changing things:**

- **Never run `pip install -e .` from a git worktree** (F-053). The conda environment is shared, and an editable install rewrites one global pointer, silently repointing every other checkout; subprocess-launched code then imports the wrong tree while tests still pass.
- **Compare against a golden fixture with `sensoryforge.testing.golden.assert_matches_golden`, not `torch.equal`** (F-071). Fixtures were generated on macOS arm64; Linux x86_64 reproduces their structure exactly but rounds values up to one float32 step differently. Comparisons between two results computed in the same process stay bit-exact.
- **Peak memory from the watchdog is noise below about 2x** (F-056); never compare the figure across runs. Use `benchmarks/` for performance, whose CI guard is calibrated against a reference kernel (F-067) and catches only regressions of about 3x or more.
- **GUI preferences go through `sensoryforge.gui.settings.gui_settings()`**, never a direct `QSettings(...)` (F-072); the test suite redirects it to a temporary directory, and a test forbids direct construction. CI runs on GitHub (Linux and macOS) and the docs deploy to https://benefron.github.io/sensoryforge/.
- **Comparison with published afferent data is qualitative only** (F-070).
- **Stimulus types disagree on amplitude scale by about 30x at their defaults** (F-083): a default `gabor` or tactile stimulus drives few or no spikes at `input_gain` 50.
- **`GridConfig.density` is never read** (F-081); every arrangement is sized by rows x cols x spacing.
- **The test suite runs with the cyclic GC on and collects after every `gui` test** (conftest autouse fixture). Collecting a destroyed window's pyqtgraph cycles mid-test can destroy a live ViewBox; do not discard and rebuild pyqtgraph widgets at runtime.
- **Voltage clamp divergence from pressure-simulation** under strongly negative drive (F-037, settled: the tactile recipes never reach the floor, and `tests/integration/test_recipe_calibration.py` fails if one comes within 20 mV of it); **flake8 debt beyond the CI subset** (F-036).

The items below were open as of 2026-09-14 (ledger `F-0NN` IDs); see
`docs/development/reviews/` for the historical audits they came from, and note several items that
audit once listed here (DSL/CUDA support, `reset_states`) were already fixed — see ledger `R-001`,
`D-011`.

- **`input_gain` unit mismatch** — The SA/RA filter parameters (`k1=0.05`, etc.) were calibrated by Parvizi-Fard et al. (2021, J. Neurophysiol.) for stimulus inputs in N/mm² (τ_RA follows Kandel, Principles of Neural Science, Ch. 21). SensoryForge uses mA as its stimulus amplitude unit. The mismatch means the filter output is ~50× smaller than expected for a "1 mA" stimulus. The default `input_gain` in `PopulationConfig` (shown on the GUI's Populations screen) is **50** to compensate; the shipped tactile recipes calibrate each population's gain against P5 instead (`tactile_sa1_ra1` SA 220 / RA 61, `tactile_sa1_ra1_adex` SA 55 / RA 86, `scripts/calibrate_recipe_gains.py`, D-ea0f017). Do not set `input_gain=1` with default filter parameters — the neuron will receive sub-threshold current. See `docs/user_guide/units_and_gains.md`.
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

**Resolved 2026-09-16, Phase 2 Wave N:** a DSL model's `threshold`/`reset` are optional, giving an analog (non-spiking) readout (`(state_trace, None)`); the shared backend labels this `"state"` instead of `"spikes"`; `SimulationEngine` builds DSL populations from `dsl_config` (previously `TypeError`/`ValueError: Unknown neuron model`); the GUI's Results screen plots the state trace for such a population. See "Analog readouts" above and `docs/user_guide/analog_readouts.md`.

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
