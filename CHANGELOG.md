# Changelog

All notable user-facing changes to SensoryForge are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); dates are commit dates from
`docs_root/LEDGER.md`, the project's decision/finding record.

## [Unreleased]

Not yet published to PyPI; install from source (see `CONTRIBUTING.md`).

### Changed (behaviour — re-run any saved results after upgrading)

- **`innervation_method` is now honoured on ordinary grids** (F-051). `SimulationEngine`, the CLI
  and the batch runner used to build every population with the Gaussian sampler regardless of the
  configured method, so `uniform`, `one_to_one` and `distance_weighted` gave the same weights as
  `gaussian`. They now build the configured method (each equal to the builder's own `build()`);
  `gaussian` weights are unchanged bit for bit for a given seed.
- **Receptive fields are one component, the `ReceptiveFieldBank`.** The engine, the legacy
  pipelines and the GUI hold a bank per population (`engine.populations[i]["bank"]`, with
  `weights [N, M]`, `neuron_centers`, `receptor_coords` and a `provenance` dict) and feed it
  flattened receptor responses. `InnervationModule` and `FlatInnervationModule` remain as
  deprecated wrappers (they build a bank internally and emit `DeprecationWarning`; removal in
  Phase 4). `GeneralizedTactileEncodingPipeline.sa_innervation` etc. are now banks.
- **GUI CSV import no longer zero-fills on a receptor-count mismatch**; it reports the two
  counts and leaves the population untouched. Imported populations can now be simulated.

- **SA filter no longer rectifies by default.** `SAFilterTorch.clip_to_positive` now defaults to
  `False` (sign-preserving SA, matching the sign convention used to recover velocity direction from
  SA activity). Pass `clip_to_positive=True` to restore the old rectified behaviour.
- **RA filter time constant (τ_RA) is 8 ms everywhere**, not 15 or 30 ms. This affects
  `RAFilterTorch`, `CombinedSARAFilter`, the GUI, the CLI/legacy pipeline, and every example config.
- **RA populations default to the fast-spiking Izhikevich preset** (`a=0.1, b=0.2, c=-65, d=2`)
  instead of regular-spiking (`a=0.02, d=8`), in the GUI, `SimulationEngine`, and the legacy
  pipeline alike. Overriding any single one of `a`/`b`/`c`/`d` on a population now keeps the rest of
  the neuron-type preset as the base instead of silently reverting them to regular-spiking.
- **RA filter gain k3 is 2.0**, not 100. On the default ramp/trapezoidal stimulus, fast-spiking RA
  firing drops from roughly 314 Hz (the old GUI default) to roughly 69 Hz. If you have tuned
  `input_gain` presets against the old k3=100 behaviour, they will need retuning.
- **Filter and neuron defaults are now resolved by a single function**
  (`sensoryforge.config.defaults.resolve_filter_params` / `resolve_neuron_params`), used by the GUI,
  `SimulationEngine`, and the legacy pipeline. Previously the GUI and the CLI could silently build
  different models from the same config.
- **Default innervation weights are analytic Gaussian; the stochastic uniform-random builder
  remains available as the control arm.** Pass `use_distance_weights=False` to restore it.
- **`requires-python` is now `>=3.10`** (a neuron module already used 3.10-only syntax). The PyQt5
  GUI is now an optional extra: `pip install -e ".[gui]"`.
- **Noise is applied after the filter and gain**, not before, in `TactileEncodingPipelineTorch`
  (matching `SimulationEngine._run_pop_from_drive` and pressure-simulation's runner).
- **`simulation.dt` is now `simulation.dt_ms`.** The old `dt` key (YAML/`from_dict`) and the
  `SimulationConfig(dt=...)` constructor keyword are still accepted as a deprecated alias (emits
  `DeprecationWarning`; passing both `dt` and a different `dt_ms` raises `ValueError`).
- **Neurons integrate at `integrate_dt_ms` (0.05 ms by default), not at the record step.** The
  drive is held constant across `dt_ms / integrate_dt_ms` sub-steps per record bin (matching
  pressure-simulation's runner exactly), which multiplies neuron-stage runtime by that factor --
  20x at the default `dt_ms = 1.0`.
- **`SimulationEngine` results' `"spikes"` are integer sub-step counts per record bin** (use
  `> 0` for a binary raster), of length `T` matching `drive`/`filtered`/`voltages` -- not booleans
  of length `T+1` as before.
- **Innervation no longer touches the global RNG**, drawing every seeded random tensor on CPU via
  a per-instance `torch.Generator`, then moving the result to the target device -- this also means
  wiring for a given seed is now identical across CPU/MPS/CUDA (previously it crashed on MPS/CUDA).

### Fixed

- The canonical config adapter no longer squares neuron counts or receptor grid size. A canonical
  population of N neurons per row now builds N² neurons consistently across the legacy pipeline and
  `SimulationEngine` (previously the legacy pipeline could build 16x the intended count); the
  README's 80x80 quick-start no longer risks exhausting memory.
  A legacy hand-written config whose neuron count reads like a total rather than a per-row value
  now raises a clear error instead of silently allocating a huge weight tensor.
- `SensoryForgeConfig.from_yaml` no longer raises `OSError` on a one-line YAML/JSON string longer
  than the filesystem's path-length limit.
- `IzhikevichNeuronTorch`'s `preset` parameter is now keyword-only, fixing a positional-argument
  footgun.
- Package config/GUI parameter files are located via `importlib.resources`, so the package now
  works correctly when installed as a wheel and run outside the source checkout (previously several
  code paths assumed the current working directory was the repository root).
- `sensoryforge run --duration` now reaches every stimulus type, including the default
  "trapezoidal" (previously excluded); the canonical adapter resolves one `dt_ms` for both neuron
  integration and every stimulus generator (previously stimuli silently used a disconnected
  0.1 ms default regardless of `simulation.dt_ms`).
- A GUI-exported config now records the time step the GUI actually simulated at, instead of always
  writing `simulation.dt_ms: 1.0`.
- `SimulationConfig`/`SimulationEngine` now reject a record step (`dt_ms`) that isn't a whole
  multiple of the neuron integration step (`integrate_dt_ms`), instead of silently rescaling
  neuron time.

### Added

- **Seeded receptor grids** (F-050): `GridConfig.seed`, `ReceptorGrid(seed=...)`, every registered
  arrangement class and `CompositeReceptorGrid.add_layer(seed=...)`. The `jittered_grid`,
  `blue_noise` and `poisson` jitter comes from a per-instance generator, so the same seed gives
  the same layout on every device, building a grid leaves the global RNG untouched, and
  `from_config(to_dict())` reproduces the coordinates.
- **`template` receptive-field builder** (`innervation_method: template`,
  `resolvable_distance_mm: d`): sigma = d/pi, lattice pitch = d, neuron count derived from the
  receptor area, `k` nearest receptors with analytic Gaussian weights and unit-L2 rows by
  default. Deterministic. `PopulationConfig` gains `resolvable_distance_mm` and
  `innervation_params` (extra builder parameters, merged last).
- **`imported` receptive-field builder** (`innervation_params: {path: ...}`): the GUI's CSV export
  folder, a saved bank or pressure-simulation population `.pt`, or a `ConstructedRF`-style `.npz`
  (centres converted from `[y, x]`, columns re-ordered to SensoryForge's receptor order).
  Provenance records the source path and a SHA-256.
- **GUI:** `template` in the innervation-method combo with a resolvable-distance field and a
  derived neuron-count label; CSV export also writes `bank.pt`; bundle exports write each
  population's bank with `ReceptiveFieldBank.save`.
- `BaseInnervation.build()`, `filter_params()`, `builder_config()`; `build_population_bank()`;
  `ComponentRegistry.name_for()`; `sensoryforge validate` accepts any registered innervation
  method (plugins included).
- Docs: `user_guide/receptive_fields.md`, `developer_guide/add_rf_builder.md`, the executed
  example `docs/examples/rf_builder_plugin.py`, and `examples/canonical_template_config.yml`.
- **The data bundle** (Phase 2, Wave J; F-011, F-013): `sensoryforge.io.bundle.write_bundle()` /
  `load_bundle()` write a self-contained run directory (`config.json` -- a superset of
  pressure-simulation's `1.0.0` "mechanoreceptor bundle" format its viewer reads unchanged --
  one `population_NN_<NAME>.pt` per population, `stimuli/stimulus.json`, and `data.h5` with
  `/stimulus/frames`, `/time_ms`, and per-population `/populations/<name>/{drive, filtered,
  spikes}`). `sensoryforge run` gains `--bundle DIR`; `SimulationEngine.run()` gains a
  keyword-only `bundle_dir` (and `stimulus_config`/`seed`) to write one directly.
  `BatchExecutor` now writes one bundle per stimulus, under
  `<output_dir>/<batch_id>/stim_%04d/`, for canonical configs (replacing the old monolithic
  consolidated `.pt`/`.h5`, which lacked neuron/receptor coordinates and `dt` and dropped
  list-valued stimulus params, F-013); `save_format` now defaults to `"hdf5"`. `sensoryforge
  batch` gains `--task-index N` to run exactly one stimulus (for a SLURM array task), and
  `generate_slurm_script()` now emits a working `sensoryforge batch ... --task-index
  $SLURM_ARRAY_TASK_ID --output ...` call (it used to emit `sensoryforge run --stimulus-index
  --format`, three flags that didn't exist anywhere, F-011). `h5py` is now required for this
  (`pip install -e '.[hdf5]'`). See `docs/user_guide/bundles.md` and the executed example
  `docs/examples/read_bundle.py`.

- `pyproject.toml` (PEP 621) replaces `setup.py`; optional extras `gui`, `hdf5`, `solvers`, `dsl`,
  `dev`, `docs`.
- `pytest -m gui` / `pytest -m "not gui"` markers separate the Qt-backed GUI test suite from the
  rest; both now run reliably in a single process each (previously the Qt suite could crash mid-run
  or at interpreter exit).
- `.github/workflows/tests.yml`: CI across Python 3.10/3.11 on Linux and macOS, plus lint/format/docs
  checks.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`.

### Extensibility (Waves F–H)

- **Docs now build with `mkdocs build --strict` and 0 warnings**; the full nav (developer guide,
  user guide, tutorials, API reference via `mkdocstrings`) is wired up and internal review/handover
  records are excluded from the published site (F1).
- **`sensoryforge list-components` and `validate` read the live registries**, not a hardcoded list,
  so a registered plugin or in-repo component shows up immediately (F2).
- **`examples/canonical_config.yml` and `examples/canonical_batch_config.yml`** are schema-valid,
  runnable canonical examples; the legacy `examples/example_config.yml`/`batch_config.yml` (and the
  matching docs snippets) were fixed to use per-row neuron counts so they run instead of exhausting
  memory (F3).
- **GUI time-step spinboxes can no longer crash the session on an invalid value**: the stimulus Δt
  spinbox snaps to a multiple of the integration step on `editingFinished`, `SpikingNeuronTab`
  catches `ValueError` alongside `RuntimeError`, and an installed `sys.excepthook` shows unhandled
  exceptions in a `QMessageBox` instead of a silent Qt abort (F4).
- **`get_param_spec()` is required on every component base class**, not just stimuli
  (`BaseNeuron`, `BaseFilter`, `BaseSolver`, `BaseGrid`, `BaseInnervation`); `ParamSpec` gained
  `choices`, `help`, `group`, and `advanced` fields for richer GUI auto-discovery (G1).
- **Third-party plugin discovery**: a distribution can advertise components via a
  `sensoryforge.components` `importlib.metadata` entry-point group, loaded automatically at import
  time; a config's `plugins:` YAML list can additionally import modules directly
  (`sensoryforge/plugins.py`) (G2).
- **Grid arrangements are real classes**, not string placeholders, so `poisson`/`hex`/`jittered`/
  `blue_noise` arrangements share the same registry/contract pattern as every other component kind
  (G3).
- **Contract tests run over every registered component**: `tests/contract/test_component_contracts.py`
  sweeps every neuron/filter/stimulus/solver/grid/innervation registration and checks
  `get_param_spec()` shape, one canonical-shape `forward()` pass, and a `from_config(to_dict())`
  round trip; the same three checks are exposed as a reusable
  `sensoryforge.testing.contracts.check_component(kind, cls)` (G4).
- **`sensoryforge new-component <kind> <name>` scaffold generator**: default mode writes a
  standalone, installable `sensoryforge-<name>/` plugin package (with a `pyproject.toml` entry
  point and a generated `tests/test_contract.py`); `--in-repo` preserves the original
  contributor workflow of writing into `sensoryforge/`, `tests/`, `docs/` directly (G5).
- **Registry component-name lookups are case-insensitive**: `"Izhikevich"` and `"izhikevich"` refer
  to the same registration; a genuine collision between two different classes under the same
  case-folded name still raises `ValueError` (H1, F-046).
- **`sensoryforge new-component` refuses to write next to an installed package**: it never targets
  a `site-packages`/`dist-packages` directory, so it is safe to run against a wheel install; the
  default plugin-package mode is now the primary documented route for third parties (H2, F-047).
- **Every built-in neuron model round-trips all of its constructor parameters** through
  `to_dict()`/`from_config()`, not only `dt` — checked by
  `sensoryforge.testing.contracts.check_component("neuron", cls)` (H3, F-045).
- **Every YAML config loader honours a config's `plugins:` list identically**: the CLI, the GUI's
  "Load YAML Configuration" action, `BatchExecutor.from_yaml`, and `SensoryForgeConfig.from_yaml_file`
  all route through one shared `sensoryforge.config.yaml_utils.load_config_file` (H4, F-048).
- **Documentation for the extension path**: `docs/developer_guide/plugins.md` documents the
  entry-point plugin route end to end; `extensibility.md`, `add_neuron.md`, `add_filter.md`, and
  `add_stimulus.md` describe both the plugin-package and in-repo routes, the `get_param_spec()`
  requirement, and the `to_dict()`/`from_config()` completeness contract; `docs/examples/plugin_filter.py`
  is a runnable worked example (define, register, and run a filter), executed by
  `tests/docs/test_docs_examples.py` (H5).

[Unreleased]: https://github.com/benefron/sensoryforge/commits/main
