# Changelog

All notable user-facing changes to SensoryForge are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); dates are commit dates from
`docs_root/LEDGER.md`, the project's decision/finding record.

## [Unreleased]

Nothing yet.

## [1.0.0] - Unreleased (prepared 2026-09-17)

Not yet released: not tagged in git, not published to PyPI, and continuous integration has never
run on GitHub for this repository. Install from source (see `CONTRIBUTING.md`). Replace this
heading's date when the release is actually tagged. This is the first version-1.0.0 changelog entry,
covering everything below back through the project's Phase 0 start,
compiled from the git history and the wave handover records
(`docs/development/handover/`) rather than summarised from memory. See
"Known limitations" at the end of this entry for what is honestly still
open going into this release.

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

Found while reviewing Phases 2 to 4, each by checking a claim against running code rather than
reading it:

- **A `moving` stimulus did not move** (F-057). Rendered through the stimulus registry it returned
  the same frame for every time sample, because the registered class advances on `step()` and the
  renderer called it once. Stepped stimuli are now driven frame by frame, and calls using the
  older flat parameters (`amplitude`, `sigma`, `start`, `end`) are routed to the generator that
  understands them.
- **Registry-rendered stimuli changed what existing configs produced.** A `gaussian` stimulus with
  no explicit `sigma` took the component's default of 0.2 mm instead of the previous 1.0 mm, 25
  times weaker on a standard grid; defaults are now preserved for every stimulus name both paths
  know, and `--duration` gives the same frame count as before.
- **`sensoryforge run` crashed on any config with an analog population** (F-060), after the run
  had already succeeded and written its bundle; it now reports the analog state range.
- **The data bundle could be loaded but not run by pressure-simulation's viewer** (F-054): it wrote
  no `neuron_modules/`, which the viewer needs to enable its Run button. Its stimulus payload was
  also an untagged dictionary that pressure-simulation's generator would silently read as a
  default blob (F-055); every payload now carries a `schema_version` and `kind`.
- **The Stimulus Designer could not load its own saved configuration** (F-064); integer spin boxes
  rejected the float values its loader passed them.
- **The Circuit tab lost node positions on every reload** (F-065); they are now saved beside the
  config in `<config>.layout.json`, and any missing or malformed layout file is ignored rather
  than blocking the config from loading.
- **Running a graph silently ignored stimulus settings the chosen stimulus does not accept**
  (F-061); it now warns, naming only the values the user actually set.
- **A warning about an ignored neuron-lattice size fired on every run of the shipped tactile
  preset** (F-069), about a value nobody had set; it now fires only for sizes the user set.
- **The reproducibility check demanded exact spike counts on every platform** (F-068). It is exact
  on the platform the reference was recorded on and tolerant elsewhere, where floating-point
  rounding can legitimately flip a spike at threshold.

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

- **A benchmark harness** (Phase 4, Wave T; F-022, second half): `benchmarks/run_benchmarks.py`
  sweeps grid size, neuron count and device, timing build and run phases separately and reporting
  a median plus observed spread per cell (not a single sample), with peak resident memory and full
  environment/version metadata recorded alongside every number. The published, regenerable table
  is `docs/reference/benchmarks.md`; `.github/workflows/benchmarks.yml` runs a fast cell on every
  push/PR and fails when the engine is more than 3x slower than the committed baseline
  (`benchmarks/check_regression.py`). The comparison is calibrated rather than in raw
  milliseconds (F-067): each run also times a fixed reference kernel in the same process, and the
  guard compares engine time relative to it, so a CI runner slower than the laptop the baseline
  came from does not fail the guard with nothing changed. On this machine's efficiency cores the
  engine's raw time rose 4.56x while the calibrated factor was 1.05x. It does not catch regressions
  under about 3x, and its calibration across processor architectures is estimated, not measured.
- **The pressure-simulation recipe** (Phase 2, Wave K; F-052): `sensoryforge/stimuli/render.py`'s
  `render_stimulus()` is now the CLI's and `BatchExecutor`'s single stimulus-dispatch point (K1) —
  it checks `STIMULUS_REGISTRY` before falling back to `GeneralizedTactileEncodingPipeline`'s
  legacy chain, so `composite`, `edge_grating`, `gabor` and any plugin-registered stimulus are now
  runnable from a config file, not just from code. Two disclosed behaviour changes come with this:
  a registered stimulus's time axis is now the same half-step-guarded
  `arange(0, duration_ms + 0.5*dt_ms, dt_ms)` pressure-simulation uses (one more sample than
  `duration_ms/dt_ms` at the default `dt_ms=1.0`), and `"gaussian"` now uses the registered
  `GaussianStimulus` class's own defaults (`amplitude=1.0, sigma=0.2`) rather than the legacy
  pipeline's (`amplitude=30, sigma=1.0`, held constant for the whole duration) when a config
  doesn't set them explicitly.
  Four pressure-simulation stimuli are ported (K2, `sensoryforge/stimuli/tactile.py`,
  registered as `ramp_gaussian`, `moving_edge`, `braille`, `drifting_grating`), each a
  transcription (not a reimplementation) of the corresponding pressure-simulation generator,
  verified bit-for-bit against a fixture exported by importing that repository directly (K3,
  `scripts/regenerate_stimulus_golden.py`, `tests/integration/test_stimulus_parity.py`).
  A new `sensoryforge/presets/` package ships two canonical-config presets as plain YAML data
  (K4): `tactile_sa1_ra1` (80x80 grid at 0.15 mm, `template` receptive fields at
  `resolvable_distance_mm=0.40`, 900 neurons/population) and `tactile_stochastic_control` (D-019's
  named control arm: `gaussian`/`use_distance_weights=false` instead of `template`).
  `sensoryforge list-presets` lists them; `sensoryforge run --preset NAME [config.yml]` runs one,
  optionally overridden by a config file. `examples/pressure_simulation_recipe.py` (K5) runs the
  whole recipe end to end (preset + all four stimuli) and writes one Wave J bundle per stimulus
  under `examples/output/pressure_simulation_recipe/` (`--quick` shortens durations for a fast
  smoke run; the real recipe runs in under 10s on CPU). See
  `docs/concepts/pressure_simulation_use_case.md`, `docs/user_guide/presets.md`, and
  `docs/developer_guide/add_stimulus.md`.
- **Analog (non-spiking) DSL readouts** (Phase 2, Wave N; F-010 DSL half): `NeuronModel`'s
  `threshold`/`reset` are now optional — with no threshold, `compile()` integrates the equations
  every step and returns `(state_trace, None)` instead of `(v_trace, spikes)` (N1). The shared
  backend kernel (`SimulationEngine._run_pop_from_drive`) labels this case `"state"` instead of
  `"spikes"` in its result dict (N2). `SimulationEngine._build_populations` builds a DSL neuron
  (`neuron_model: dsl`) from `PopulationConfig.dsl_config` and compiles it, instead of raising
  `TypeError`/`ValueError: Unknown neuron model`; the new `PopulationConfig.readout` (`"auto"` by
  default) can force `"spiking"`/`"analog"`, raising when the `dsl_config` can't support it (N3).
  The Spiking tab's raster panel plots the state trace, labelled with the state variable's name,
  in place of the spike scatter for an analog population (N4). Spiking populations are bit-for-bit
  unaffected. See `docs/user_guide/analog_readouts.md` and `docs/examples/analog_dsl.py`.
- **Multi-input populations and processing layers** (Phase 2, Wave M; with Waves L and N, closes
  the last of F-010): a population can read from several grids or channels at once.
  `PopulationConfig.inputs` is a list of `PopulationInput` (`grid`, `channel`, `rf` as an
  `RFBuilderConfig`, `gain`, `layers`, `processing`), combined by `PopulationConfig.combine`:
  `"sum"` (every input must produce the same neuron count) or `"concat"` (the neuron axis grows,
  one block per input). The existing single-input fields remain as shorthand that expands into
  one input, and every existing config still round-trips to byte-identical YAML. An optional
  per-input processing stage runs before the receptive-field bank, from the new
  `PROCESSING_REGISTRY`; `onoff` (`OnOffLayer`), a centre-surround difference of Gaussians
  emitting separate ON and OFF planes, is the first built-in layer. The `vision_onoff_rgb` preset
  and `examples/vision_rgb_onoff.py` demonstrate an RGB sensor array feeding both combine modes.
  Sum is tested as exactly equal to a single bank on the summed input; each concat block against
  its own single-input run; and ON against OFF with stimuli of opposite sign. See
  `docs/concepts/populations_and_inputs.md` and `docs/extending/add_processing_layer.md`.
- **Real receptor sampling and composite grids in `SimulationEngine`** (Phase 2, Wave L; closes
  F-010): `SimulationEngine` no longer assumes receptor index equals stimulus pixel index. It now
  samples the stimulus at each receptor's own `(x, y)` mm position with bilinear interpolation
  (`torch.nn.functional.grid_sample`), taking the old row-major reshape only as a fast path when
  the receptor arrangement is a regular `"grid"` whose resolution matches the stimulus frame
  exactly (bit-identical to earlier releases; the Wave E golden parity test and the Wave I golden
  weights fixture are unaffected). Every population's receptive-field bank is now built on the
  target grid's real receptor coordinates for every arrangement (`hex`/`poisson`/`jittered_grid`/
  `blue_noise` no longer fall back to a synthetic regular lattice with a warning). `SimulationEngine`
  can now build `arrangement: composite` grids from `GridConfig.layers` (each layer density-driven
  or given explicit/imported coordinates, in declaration order -- the receptor-index contract,
  recorded in the grid's `provenance`); `GridConfig.coords_file` imports an `[M, 2]` CSV/`.pt` of
  receptor coordinates directly; `PopulationConfig.target_layers` restricts a population to a named
  subset of a composite grid's layers. `GridConfig.channels` names a grid's sensor planes
  (single-channel default unaffected) and `StimulusConfig.channel` names which plane a stimulus
  drives -- the schema-side half of the `[batch, time, C, H, W]` channel axis Wave K's
  `render_stimulus` produces. See `docs/concepts/sensor_arrays.md`,
  `docs/concepts/units_and_shapes.md`, `docs/extending/add_grid_arrangement.md` and the executed
  example `docs/examples/grid_arrangement_plugin.py`.
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
  one `population_NN_<NAME>.pt` per population, `stimuli/stimulus.json`, `neuron_modules/`, and
  `data.h5` with
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

### Added (Phase 2, Wave I — receptive fields as one component)

- **`ReceptiveFieldBank`** (`sensoryforge/core/rf_bank.py`): the single object every population's
  receptive fields are, replacing per-method ad hoc weight tensors -- buffers `weights [N, M]`,
  `neuron_centers [N, 2]`, `receptor_coords [M, 2]` and a `provenance` dict; `save()`/`load()` as a
  `.pt` file, including pressure-simulation's own population-file layout.
- **`BaseInnervation.build()`** on all four existing innervation methods (`gaussian`, `uniform`,
  `one_to_one`, `distance_weighted`) returns a bank through one shared function,
  `build_population_bank()`, used by `SimulationEngine`, all three legacy pipelines and the GUI.
- **The `template` and `imported` receptive-field builders** (documented above under "Added") and
  **seeded receptor grids** (`GridConfig.seed`, F-050) are this wave's other two deliverables.

### Added (Phase 3 — the Circuit graph GUI)

- **The Circuit tab and graph-config serialisation** (Wave O): a node-graph editor tab that builds
  and edits a `SensoryForgeConfig` as a graph of grid/population/readout nodes and edges, with a
  lossless graph-to-config-to-graph round trip proved against five canonical example files and all
  three shipped presets (`tests/` graph round-trip suite; two node types intentionally carry more
  state than the original spec so the round trip stays lossless given the schema's single-stimulus,
  no-owner-of-simulation-settings shape). The graph validator catches dangling terminals, a readout
  with no filter, and a drive with neither a receptive-field bank nor a combine, but not yet a
  combine whose inputs disagree on neuron count (ledger F-062).
- **The inspector and the plugin palette** (Wave P): every node's parameters are rendered from
  `get_param_spec()` alone, including for a plugin component this checkout has never imported --
  proved by writing a real on-disk `*.dist-info`/`entry_points.txt` next to a real importable
  package and letting the unmodified `importlib.metadata` entry-point scan find it (no in-process
  registry call), the same technique `examples/plugin_template/tests/test_entry_point_discovery.py`
  uses in this release (see V1 below). A component with an empty parameter spec now shows a visible
  "no configurable parameters" notice instead of a blank panel. The Circuit tab draws its own small
  sensor-array/stimulus/receptive-field previews rather than reusing the Mechanoreceptor and
  Stimulus Designer tabs' plot widgets, a known duplication (ledger F-063).
- **Graph parameter sweeps and multi-channel selectors** (Wave Q), plus **3,819 lines of dead code
  removed**: four unused modules (including the whole neuron-explorer test file) deleted after a
  full-repository grep across docs, examples, scripts, packaging and the mkdocs nav. A sweep now
  reads its swept value back out of each run's own `config.json` and raises if a value was not
  varied, rather than only checking that the expected number of bundles exist. Sweep targets are
  chosen from combo boxes, not by clicking a node on the canvas (the Batch tab does not share a live
  selection with the Circuit tab); the Visualization tab's multi-channel selector is complete and
  tested as a capability, but nothing in the repository yet feeds it a real multi-channel tensor.
- **The interactive walkthrough, the concepts pages, and the Phase 3 close-out** (Wave R): a
  tutorial executed end to end (build a graph, export it, run it from the command line, re-import
  and get the same graph back) rather than written from reading the code, with real, regeneratable
  offscreen screenshots. Node classes and the inspector's preview dispatch remain hard-coded tables,
  so a plugin gets a settings form for free but a new node *type* or custom preview still needs an
  in-repo change (documented plainly in the extending guide rather than implied otherwise).

### Added (Phase 4 — validation, benchmarks, documentation, release)

- **Validation against closed-form and published references** (Wave S, F-022 first half;
  `tests/validation/`): SA/RA filter step and ramp responses checked against their analytic
  solutions across three time steps; the `template` builder's weights checked against the Gaussian
  evaluated at the same distances and its resolvable-distance arithmetic; the `grid_sample`
  stimulus-sampling path checked against an analytic Gaussian recovered at hex receptor positions;
  Izhikevich RS/FS rate-vs-current behaviour checked against Izhikevich (2003)'s published claims.
  Every test's docstring records a perturbation that makes it fail. A genuine TouchSim comparison
  could not be produced (`touchsim` is not installed and Wave S is forbidden from adding it, F-053)
  -- `tests/validation/test_touchsim_sanity.py` and `docs/examples/validation_touchsim.ipynb`
  instead check the correct qualitative SA-sustains/RA-adapts adaptation class against a committed,
  cited-literature reference (`tests/fixtures/reference/README.md` records exactly what this is and
  is not; ledger F-070 records this as an accepted, open limitation). `scripts/reproduce_figure.py`
  / `scripts/reproduce_env.sh` reproduce the pressure-simulation recipe's summary statistics from a
  fresh install, exactly on spike counts and within a stated tolerance on mean rates, checked
  in-process by `tests/validation/test_reproducibility.py`; this proof and its committed reference
  are exact-match on this machine only, not yet across platforms (ledger F-071 records the same
  macOS-arm64-only caveat for the pre-existing golden-parity tests).
- **The benchmark harness, the published table, and a CI performance guard** (Wave T, F-022 second
  half; see "A benchmark harness" above under Phase 4 F-022) -- also documented in
  `docs/reference/benchmarks.md`, generated by `benchmarks/generate_table.py`, never hand-edited.
- **The complete documentation tree, with every example executed** (Wave U): concepts,
  getting-started, user-guide, developer-guide and API-reference sections with no remaining
  "coming in Phase N" stubs; `pytest docs/examples` runs every `docs/examples/*.py` script; the
  quick-start and validation notebooks execute under `nbmake` in CI
  (`.github/workflows/tests.yml`); a GitHub Pages deploy workflow
  (`.github/workflows/deploy-docs.yml`) publishes the strict build on pushes to the default branch
  (unverified as of this release -- continuous integration has never executed on GitHub for this
  repository). This wave also found and fixed real documentation-vs-code drift beyond its own
  brief: a schema reference page naming fields that never existed on the dataclasses it described,
  a quickstart notebook calling a function that no longer exists, and a shared contract-test skip
  entry that was silently skipping the real `IdentityLayer` processing component as a side effect of
  a filter-registry placeholder skip (closing F-058).
- **The plugin template package and the JOSS paper draft** (Wave V, V1/V2): `examples/plugin_template/`
  is a complete, standalone, installable plugin adding one receptive-field builder
  (`radial_falloff`) and one processing layer (`gain_threshold`), verified through the real
  `importlib.metadata` entry-point mechanism and a genuine `pip install` of its built wheel into a
  disposable virtual environment, with its components then visible in `sensoryforge list-components`
  (which, until this release, never listed `PROCESSING_REGISTRY` at all -- a processing-layer plugin
  was invisible to that command even once correctly registered). `paper/paper.md` and
  `paper/paper.bib` are a Journal of Open Source Software draft, scoped to the forward-model tool
  this repository actually is (decoding stays out of SensoryForge; pressure-simulation consumes its
  export bundle contract), with every claim citing the test, benchmark or executed example that
  proves it.

### Known limitations (going into 1.0.0)

These are open findings in `docs_root/LEDGER.md`, listed here rather than left implicit, per this
project's own validation guardrail ("a validation that cannot fail is not a validation" applies
equally to a changelog that hides what still fails):

- **No quantitative comparison against a published afferent model exists** (F-070): the TouchSim
  comparison above is qualitative only.
- **Golden-parity fixtures (`tests/integration/test_pressure_sim_parity.py`,
  `test_stimulus_parity.py`, `tests/fixtures/rf_engine_golden_weights.pt`) have only ever run on
  macOS arm64** (F-071); a failure on another platform needs diagnosis, not an assumption of
  regression.
- **The memory watchdog cannot detect a regression below roughly a factor of two** and its numbers
  are never comparable run to run or machine to machine (F-056).
- **The CI performance guard's 3.0x regression factor is calibrated from local experiments, not
  from a run on GitHub's own runners** -- continuous integration has never executed on GitHub for
  this repository as of this release.
- **364 flake8 style violations exist beyond the CI-gated subset** (E9/F63/F7/F82 only; F-036).
- **SensoryForge's neurons clamp voltage at a floor where pressure-simulation's do not** (F-037),
  which unrectified SA (resolved separately, see above) makes reachable for strongly negative
  drive.
- **`pytest -m gui` segfaults under Python's cyclic garbage collector** in a `pyqtgraph`
  callback chain (F-035); `tests/conftest.py` disables GC for any session that collects a GUI test
  (not for non-GUI-only sessions) to avoid it, which also
  means the harness can no longer detect this crash class if it recurs.
- **The Circuit tab's graph validator does not catch a combine whose inputs disagree on neuron
  count** (F-062), and **draws its own preview widgets rather than reusing the Mechanoreceptor and
  Stimulus Designer tabs'** (F-063), so the two can drift.

<!-- Version links to the v1.0.0 tag and release are added when that tag exists (D-010: no links
to infrastructure that does not exist yet). -->
