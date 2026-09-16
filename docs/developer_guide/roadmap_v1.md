# SensoryForge v1 roadmap — general sensory-encoding simulator, pressure-simulation as the first use case

## How this roadmap is executed

Each phase is broken into waves, and each wave is specified in a handover document that an
implementation agent works from. The roadmap says what and why; the handovers say exactly how, with
the verified facts, the file paths and the proof each task owes.

| Phase | Handover | Waves | State as of 2026-09-16 |
|---|---|---|---|
| 0 and 1 | `docs/development/handover/phase1_tasks.md` | A to H | Closed, except one CI run on GitHub |
| 2 | `docs/development/handover/phase2_tasks.md` | I to N | Complete; all six waves merged and exit criteria verified |
| 3 | `docs/development/handover/phase3_tasks.md` | O to R | Specified |
| 4 | `docs/development/handover/phase4_tasks.md` | S to V | Specified |

Phase 2 onwards is orchestrated: waves are developed on their own branches in git worktrees, reviewed,
then merged into the integration branch `phase2`. `main` stays at the end of Phase 1 until Phase 2 is
complete and reviewed.

## Context

SensoryForge (`~/sensoryforge`) becomes the general, clean-slate simulator: **sensor arrays with
channels → receptive fields (biological or designed) → sensory neurons (filters + spiking, or analog
readouts via the equation DSL) → visualisation → batch data generation for learning / analysis.**
Pressure-simulation (`~/Documents/pressure simulation`, Paper B, SGA-KF) is the first use case: it
supplies the *recipe* (declared resolvable distance `d`, stimulus ensemble, scoring by mutual
information) and consumes the generated data; SensoryForge builds the grid and the receptive fields
from that recipe and generates the data.

The audit of 2026-09-14 (`reviews/PUBLICATION_READINESS_20260914.md`, ledger F-001…F-022) found:
a critical OOM in the canonical→legacy adapter, an installed package that does not run, no CI,
five scientific divergences from pressure-simulation, and an engine that ignores non-grid layouts,
cannot run DSL neurons, and exports neither weights nor coordinates. Exploration for this plan added:
pressure-simulation's analytic design toolkit is a **skeleton** (nothing builds the deterministic
template RF, nothing serialises a design), and its decoder needs the **continuous filtered response
`y`**, `W_float` and neuron centres per population — none of which SensoryForge writes today.

Decisions taken with the user (2026-09-14): analog readouts = DSL models without a spike condition;
layers = named channels on one substrate; decoding stays out of SensoryForge (stable export contract
only); release-readiness + parity first, generalisation second; SensoryForge owns the RF/grid
builders, pressure-simulation gives the recipe and does the scoring; one `ReceptiveFieldBank` with
pluggable builders; export = bundle directory that is a superset of pressure-simulation's
"mechanoreceptor bundle"; GUI = node-graph canvas.

Ledger trailer for the first implementation commit:
`Decision: SensoryForge is the general clean-slate sensory-encoding simulator (sensor channels → receptive fields → sensory neurons → spiking or analog readout → batch data); pressure-simulation is a use case that supplies the recipe (d, ensemble, MI scoring) and consumes the generated bundle.`

---

## Target architecture (the spine every phase serves)

```
SensorArray (geometry: grid|hex|poisson|imported coords; channels: ["pressure"] | ["R","G","B"] …)
   │  stimulus  [batch, T, C, H, W]   (C=1 stays [batch,T,H,W]-compatible)
   ▼
ReceptiveFieldBank per population input  (weights [N,M], neuron_centers [N,2], receptor_coords [M,2], provenance)
   builders: gaussian_stochastic (today's default, becomes the control arm) · gaussian_analytic
             (use_distance_weights=True) · template (d → σ=d/π, Δ=d, N=A/Δ², one kernel translated,
             K-nearest, analytic weights) · one_to_one · uniform · imported (CSV / .pt / design file)
   ▼  drive_i [batch, T, N]  ──► combine (sum | weighted sum | concat→N·k)  ──► drive
Filter (sa | ra | none | future ON/OFF)  ──► y  [batch, T, N]   (the continuous response B designs on)
   ▼  gain · noise
Readout: spiking neuron (Izhikevich/AdEx/MQIF/DSL) → spikes [batch, T, N] counts per record bin
         analog DSL model (no threshold) → state [batch, T, N]
   ▼
Bundle (config.json + population_NN.pt + stimuli/*.json + data.h5)  ──►  pressure-simulation / ML
```

Config (canonical `SensoryForgeConfig`, `sensoryforge/config/schema.py`) grows by:
- `GridConfig.channels: list[str] = ["value"]`, `GridConfig.coords_file: Optional[str]`.
- `PopulationConfig.inputs: list[PopulationInput]` where `PopulationInput = {grid, channel, rf: RFBuilderConfig, gain}`; existing `target_grid / innervation_method / sigma_d_mm / connections_per_neuron / use_distance_weights` stay as sugar that expands to one input (`from_dict` does the expansion, `to_dict` writes the short form when there is exactly one input). `PopulationConfig.combine: "sum"|"concat"`.
- `RFBuilderConfig = {method, params, source}`; `template` params: `resolvable_distance_mm` **or** explicit `{sigma_mm, pitch_mm, k}`; `imported` params: `{path}`.
- `PopulationConfig.readout: "spiking"|"analog"` (inferred from the neuron model: DSL without `threshold` ⇒ analog).
- `SimulationConfig.dt_ms` (record/bin step) and `SimulationConfig.integrate_dt_ms` (neuron sub-step, default 0.05) — replaces the single `dt`.
- `StimulusConfig.channel: Optional[str]`.

---

## Cross-cutting requirement A — the extensibility contract

SensoryForge is an open-source framework first. Every stage of the spine is a **plugin point**
with the same shape, so a researcher can add a grid arrangement, a stimulus, an RF builder, a
combine op, a processing layer, a filter, a neuron/readout, a solver, an exporter or a GUI node
without editing core files.

- **One contract for all component kinds** (`sensoryforge/registry.py`, extend `ComponentRegistry`):
  a base class per kind (`BaseGridArrangement`, `BaseStimulus`, `BaseRFBuilder`, `BaseCombine`,
  `BaseProcessingLayer`, `BaseFilter`, `BaseNeuron` (spiking or analog), `BaseSolver`,
  `BaseExporter`) each with `from_config(cfg)`, `to_dict()`, `get_param_spec() → list[ParamSpec]`
  (today only stimuli have it: `stimuli/base.py`), `reset_state()`, and a documented tensor
  contract in the docstring. `ParamSpec` grows `choices`, `help`, `group`, `advanced` so the GUI
  and the CLI `--help` render from it.
- **Discovery, not registration by hand.** `register_components.py` keeps the built-ins; third
  parties register via Python entry points (`[project.entry-points."sensoryforge.components"]` in
  their `pyproject.toml`) loaded with `importlib.metadata` at `register_all()`, or via a
  `plugins:` list of import paths in the YAML config for scripts. Registering twice is idempotent
  (already true, `registry.py:71-77`). Replace the `GRID_REGISTRY.register("grid", str)` placeholders
  (`register_components.py:154-158`) with real arrangement classes.
- **`sensoryforge new-component <kind> <name>` CLI scaffold**: writes the class file with the
  contract stubs, a unit test that exercises `from_config/to_dict/get_param_spec` and one forward
  pass with the kind's canonical tensor shapes, a docs page under `docs/extending/`, and the
  entry-point line. A `tests/contract/test_component_contracts.py` parametrised over every
  registered component (built-in or plugin) enforces the contract in CI — a plugin author can run
  the same test against their package.
- **GUI follows the registry.** Node-canvas nodes and inspector widgets are generated from
  `get_param_spec()`; a new plugin appears in the palette with no GUI code (the Stimulus Designer
  already does this for stimuli; generalise that path).
- **Template repository** `sensoryforge-plugin-template` (cookiecutter-style, in `examples/plugin_template/`
  and published as its own GitHub template): a minimal package adding one filter and one RF builder,
  with tests and docs, proving the whole loop.

## Cross-cutting requirement B — documentation as a deliverable

Docs are built with `mkdocs-material` + `mkdocstrings` (API reference generated from the Google
docstrings; `mkdocs build --strict` in CI so drift fails the build). Structure of `docs/`:

1. **Concepts** — the spine diagram, units and shapes (one page), sensor arrays and channels,
   receptive fields (biological vs designed, the `d → σ, Δ, N` chain), filters, readouts
   (spiking vs analog), the bundle format, "how pressure-simulation uses SensoryForge".
2. **Getting started** — install (source + extras), 5-minute CLI run, 5-minute GUI tour, first
   plugin in 10 minutes.
3. **User guide** — CLI, YAML schema (generated from the dataclasses), GUI node canvas, batch and
   SLURM, bundles and loading them in PyTorch / numpy / pressure-simulation, presets.
4. **Extending SensoryForge** — one guide per plugin kind (grid arrangement, stimulus, RF builder,
   combine, processing layer, filter, neuron, analog readout, solver, exporter, GUI node), each a
   worked example whose code lives in `docs/examples/*.py` and is executed as a test in CI
   (`pytest docs/examples`), so every guide is proven to run. Plus the plugin-template walkthrough
   and the contract test.
5. **Reference** — API (mkdocstrings), config schema, bundle schema (`schema_version 2.0.0`),
   CLI, changelog, citation, glossary (imports pressure-simulation's six-σ disambiguation).
6. **Developer guide** — architecture, testing strategy (unit / contract / integration / gui /
   parity), release process, ledger conventions (`docs_root/LEDGER.md`, trailers).

Docstring standard enforced by `pydocstyle --convention=google` in CI; every public class states
tensor shapes and units. `docs/` is published to GitHub Pages by CI on `main`.

## Phase 0 — decide and record (no code; ½ day)

Write `Decision:` trailers (one commit: `docs: record v1 scope decisions`) closing the parity findings:
- F-001 → `clip_to_positive=False` default; oscillation guard stays `v_floor` + sub-stepping.
- F-002 → τ_RA = 8 ms everywhere (class, YAML, `CombinedSARAFilter`), restore the Kandel docstring.
- F-003 → default builder `gaussian_analytic`; `gaussian_stochastic` kept as named control arm.
- F-004 → Izhikevich presets `RS/FS/IB/CH/LTS` in `neurons/izhikevich.py`; RA populations default `FS`.
- F-005 → Parvizi-Fard et al. 2021 + Kandel Ch.21; delete "Pierzowski".
- F-009 → replace `docs_root/SCIENTIFIC_HYPOTHESIS.md` with a scope note pointing at B's `PAPER_A_SIMULATOR.md`.
- F-021 → `reviews/` stays tracked but moves to `docs/development/` with a status banner; `devo_reports/` archived to `docs_root/archive/` (gitignored).
Mirror F-001/F-002 in pressure-simulation the same day (its `encoding/filters_torch.py`), with its own ledger trailer.

## Phase 1 — a package that installs, runs and is trusted (≈3 days)

**1a. Stop the OOM (F-012).** `sensoryforge/core/generalized_pipeline.py:351` → `grid_size = (rows, cols)`; delete the bilinear-resize workaround in `batch_executor.py:417-425`; regression test: canonical 20×20 through the adapter yields 400 receptors. Run the two OOM-killed integration files under `scratchpad/memwatch.sh 4000`.

**1b. Packaging (F-014).** `pyproject.toml` (PEP 621, setuptools, `requires-python >=3.10`, extras `gui`, `hdf5`, `solvers`, `dsl`, `dev`, `docs`; PyQt5/pyqtgraph move to `gui`), `package_data` for `gui/default_params.json` and `config/default_config.yml`; replace the cwd-relative path in `core/pipeline.py:86,392,422` and `examples/scripts/example_pipeline.py:7` with `importlib.resources`; one author string; delete `setup.py` after `pip install .` and `python -m build` both work in a fresh venv.

**1c. Tests you can trust (F-016, F-017).** `pytest.ini` (`testpaths`, `-p no:cacheprovider`, `filterwarnings`, marker `gui`); `tests/unit/test_stimulus_tab_gui.py` mocks via a `monkeypatch.setitem(sys.modules, …)` fixture; `pytest_sessionfinish` `os._exit` guard for Qt (or `pytest-forked` for the `gui` marker); `SensoryForgeConfig.from_yaml` accepts a path or text (`from_yaml_file` added, docstrings fixed); the five stale `dt=0.5` step-count tests updated to the schema's `dt_ms`; `test_invalid_innervation_method_raises_error` fixed by validating the method name against `INNERVATION_REGISTRY` in `SimulationEngine._build_populations`.

**1d. CI (F-015).** `.github/workflows/tests.yml`: ubuntu + macOS × py3.10/3.11; job A `pytest -m "not gui"`, job B `QT_QPA_PLATFORM=offscreen pytest -m gui --forked`; `black --check`, `flake8` (line length 88 in config), `mkdocs build --strict`. `CITATION.cff`, `CHANGELOG.md` (from this plan's phases), `CONTRIBUTING.md` (absorbs `DEVELOPMENT.md`), `CODE_OF_CONDUCT.md`.

**1e. Engine parity with pressure-simulation (F-001…F-008).** Apply Phase-0 decisions in `filters/sa_ra.py`, `config/default_config.yml`, `neurons/izhikevich.py`, `core/innervation.py`; fix the argument-less `CombinedSARAFilter()` in `core/pipeline.py:145`; add sub-stepping in `SimulationEngine._run_pop_from_drive` (integrate at `integrate_dt_ms`, record counts per `dt_ms`); make innervation use a per-instance `torch.Generator` like `filters/noise.py:41`; one noise topology (filter → gain → noise) in both `core/pipeline.py` and the engine. Parity test `tests/integration/test_pressure_sim_parity.py`: golden `spikes`, `y`, `W_float` exported from B's `encoding/encode_runner.py` for one config + seed, checked at tolerance 0 (RNG order must match: replace the batched multinomial with the same per-neuron draw *or* export B's wiring and compare from the weights onward — do the latter first, the former if B agrees to adopt the batched draw).

**1f. Docs quick fixes + docs infrastructure (F-020).** Add the developer guide, `units_and_gains.md`, `gui_walkthrough.md`, `configuration_schema.md` to `mkdocs.yml` nav; fix the 8 broken links; remove the PyPI install lines; rewrite `sensoryforge/config/README.md`; `list-components` generated from the registries (F-018). Install `mkdocstrings`, `mkdocs build --strict` and `pydocstyle` in CI now so Phases 2–4 cannot regress the docs; create the `docs/` skeleton of requirement B with stubs marked "coming in Phase N" (strict build tolerates stubs, not broken links).

**1g. Extensibility baseline (requirement A, part 1).** `get_param_spec()` on all base classes (default `[]`), entry-point discovery in `register_all()`, real grid-arrangement classes in `GRID_REGISTRY`, `tests/contract/test_component_contracts.py` over every registered component, `sensoryforge new-component` scaffold. This lands before Phase 2 so the new RF builders, combine ops and readouts are written *as plugins* against the contract, proving it.

Exit criterion: `pip install .` in a clean venv → `sensoryforge run examples/example_config.yml` and the GUI both work; CI green; parity test green.

## Phase 2 — the general core (≈2 weeks)

**2a. `ReceptiveFieldBank` (new `sensoryforge/core/rf_bank.py`).** One `nn.Module`: buffers `weights [N,M]`, `neuron_centers [N,2]`, `receptor_coords [M,2]`, `provenance: dict`; `forward(receptor_responses [batch,(T),M]) → [batch,(T),N]` (reuse `FlatInnervationModule.forward`, `core/innervation.py:1358`); `to_dict/from_config`; `save(path)/load(path)` writing the `.pt` keys B reads (`innervation_weights`, `neuron_centers`, plus `receptor_coords`). Builders as functions in `sensoryforge/core/rf_builders/` registered in `INNERVATION_REGISTRY` (replace the factory closures at `register_components.py:106-141`):
- `gaussian_stochastic` — today's `create_gaussian_innervation` (`innervation.py:880-975`), per-instance generator.
- `gaussian_analytic` — same selection, weights = Gaussian at distance (today's `use_distance_weights=True` branch, `innervation.py:952-962`).
- `template` — `d → σ=d/π, Δ=d, N=⌊A/Δ²⌋`; lattice via `create_neuron_centers(arrangement=…)` (`innervation.py:752`); one Gaussian kernel, K nearest receptors per neuron, analytic weights, unit-L2 rows optional (`normalize: none|l2|sum`). Explicit `{sigma_mm, pitch_mm, k}` also accepted. This is the builder B's blueprint specifies and does not have.
- `one_to_one`, `uniform` — ported from `innervation.py:311-530`.
- `imported` — CSV folder (today's `_on_import_population_csv` format, `mechanoreceptor_tab.py:3078`), `.pt` bundle file, or B's `ConstructedRF`-shaped `.npz` (`H`, `centers`, `sigma`, `pitch`).
Delete `InnervationModule`, `FlatInnervationModule`, `_CSVPopulationModule` once the GUI and engine use the bank; `GeneralizedTactileEncodingPipeline` adapts through the bank (keeps legacy YAML working).

**2b. Sensor arrays with channels.** `GridConfig.channels`; `ReceptorGrid` / `CompositeReceptorGrid` (`core/grid.py:97`, `core/composite_grid.py:96`) gain `channels` and `coords_file` (uses `add_layer_with_coords`, `composite_grid.py:177`); stimulus tensors gain an optional channel axis (`BaseStimulus.forward` unchanged per plane; `StimulusConfig.channel` selects the plane; builder composites per channel); `SimulationEngine._stimulus_to_receptors` implemented properly: bilinear sampling of each channel plane at `receptor_coords` (torch `grid_sample`), so pixel index ≠ receptor index and hex/poisson/imported layouts are correct (closes the "silently wrong" half of F-010). Composite grids run through the engine (each `GridConfig` = one layer of one `CompositeReceptorGrid`; delete the `NotImplementedError` at `simulation_engine.py:98`).

**2c. Multi-input populations.** `SimulationEngine._build_populations` builds one bank per `PopulationInput`, combines drives per `combine`; `_run_pop_from_drive` unchanged. `ProcessingPipeline` (`core/processing.py:129`) is wired as the optional per-input transduction stage (identity default; `on_off` centre-surround added as the first non-trivial layer — this is the vision demo).

**2d. Analog readouts.** `neurons/model_dsl.py`: `threshold`/`reset` optional (`:119`, `:411`, `:289`); without them `forward` returns `(state_trace, None)`; `BaseNeuron` docstring updated to allow `spikes=None`; `_run_pop_from_drive` labels output `"state"` when no spikes; engine can instantiate DSL models (`NeuronModel.from_config(dsl_config).compile(device)`, fixing `simulation_engine.py:260-264`, closes the DSL half of F-010); GUI plots a trace panel instead of a raster for analog populations.

**2e. Bundle export (F-013, F-011) — the contract with pressure-simulation.** New `sensoryforge/io/bundle.py`:
```
<run>/config.json                 schema_version 2.0.0, kind "sensoryforge_bundle" (superset of "mechanoreceptor_bundle" 1.0.0:
                                  grid{rows,cols,spacing_mm,center_mm}, populations[{name,neuron_type,parameters,tensors}])
      population_NN_<name>.pt     {innervation_weights [N,M], neuron_centers [N,2], receptor_coords [M,2], provenance}
      stimuli/*.json              B's generate_stimulus_from_json payloads where expressible; else SensoryForge stimulus config
      data.h5                     /stimulus/frames [T,C,H,W] · /time_ms [T] · attrs dt_ms, integrate_dt_ms
                                  /populations/<name>/{drive, y (filtered), spikes (int8 counts) | state}  [T,N]
                                  /meta: full SensoryForgeConfig YAML, seed, versions
```
Written by `SimulationEngine.run(..., bundle_dir=…)`, by the GUI auto-save (`spiking_tab.py:2300`), and by `BatchExecutor` (one bundle per stimulus, HDF5 default; `.pt` monolith removed). Binary spikes for B = `counts > 0`. `sensoryforge export-bundle` CLI. `generate_slurm_script` rewritten to emit `sensoryforge batch --task-index $SLURM_ARRAY_TASK_ID` (add that flag). Regression: B's `GUIs/ebkf_viewer.py:646-700` loader reads a SensoryForge bundle unmodified (test copies the loader logic).

**2f. Presets and the pressure-sim recipe.** `sensoryforge/presets/` YAML fragments: `tactile_sa1_ra1.yml` (Kandel-grounded SA1 RS / RA1 FS, τ's, σ's, `template` builder with `resolvable_distance_mm: 0.40`), `tactile_stochastic_control.yml`, `vision_onoff_rgb.yml` (2c demo). `examples/pressure_simulation_recipe.py`: build grid + RFs from `d`, run the four B stimuli (`ramp_gaussian`, `moving_edge`, `braille_H`, `grating` — ported from B's `_ebkf_pres_movies.py:174-240` into `stimuli/`), write a bundle. This script is the reproducible demo for Paper A and the data source for Paper B.

## Phase 3 — node-graph GUI (≈2 weeks)

Use `pyqtgraph.flowchart` (already a dependency, verified importable in the env: `Flowchart`, `Node`). New tab **Circuit** (`gui/tabs/circuit_tab.py`) with custom `Node` subclasses mapping 1:1 onto config dataclasses: `SensorArrayNode` (GridConfig, channels as output terminals), `StimulusNode` (per channel), `RFBankNode` (builder + params, in: channel, out: drive), `CombineNode`, `FilterNode`, `ReadoutNode` (spiking/analog), `RecordNode` (bundle). The flowchart state serialises to/from `SensoryForgeConfig` (`gui/main.py:456,582` converters extended) so CLI and GUI stay one config. Existing tabs: Grid & Innervation becomes the *inspector* for the selected node (its widgets are reused, not rewritten); Stimulus Designer and Visualization unchanged except channel selectors; Batch tab builds a sweep from the live graph (fixes the file-only limitation) and emits progress. Delete the unwired `protocol_*`/`neuron_explorer` modules (F-019) unless a node needs them.

Each Phase 2 component ships with its `docs/extending/` guide and executed example (requirement B §4) in the same commit; the plugin template repo is created at the end of Phase 2 using the `template` RF builder and the `on_off` processing layer as its worked examples.

## Phase 4 — validation, docs, paper artefacts (≈1.5 weeks, overlaps 3)

Benchmark script (grid size × neurons × device → wall-clock, memory) → `docs/reference/benchmarks.md`; TouchSim comparison notebook for a probe indentation (executed, outputs saved); executed quick-start notebook; the full `docs/` tree of requirement B completed (concepts, user guide, all extending guides executed in CI, reference, developer guide); GitHub Pages deploy workflow; CHANGELOG 1.0.0; JOSS paper draft `paper/paper.md` scoped per B's `PAPER_A_SIMULATOR.md` (JOSS reviewers check exactly: install, docs, tests, extensibility, community guidelines); pressure-simulation ledger upgraded to the current hooks (`/ledger-init` in B) so both repos report in `/ledger-status`.

---

## Ledger discipline for execution

Every phase commit carries `Closes: F-0NN` / `Decision:` / `Finding:` trailers; the sync hook keeps `docs_root/LEDGER.md` current. New findings discovered mid-phase are opened with `Opens:` on the commit that found them, never fixed silently. `.claude/rules/engine-parity.md` is updated when Phase 1e lands.

## Verification

- Phase 1: fresh venv `pip install .[gui,hdf5,dev]` → `sensoryforge run`, `sensoryforge batch --dry-run`, `python -m sensoryforge.gui.main` (offscreen smoke); `pytest` green locally and in CI on both OSes; `memwatch.sh` peak < 1 GB for the whole integration suite; parity test against B golden files passes.
- Phase 2: unit tests per builder (weights shape, row norms, K exact, template determinism, imported round-trip); `grid_sample` sampling test (hex layout recovers a known Gaussian to <1%); multi-input test (two channels, sum vs concat shapes); analog DSL test (no threshold → state only); bundle round-trip test + B-loader compatibility test; `examples/pressure_simulation_recipe.py` produces a bundle B's viewer opens.
- Phase 3: `pytest -m gui` for node↔config round-trip (build graph → config → graph → identical), plus manual run of the GUI end-to-end.
- Phase 4: `mkdocs build --strict`, notebooks executed by CI (`nbmake`), benchmark table regenerated.
- Extensibility (every phase): `pytest tests/contract` passes for all registered components; the plugin template installs into a clean venv, its components appear in `sensoryforge list-components` and in the GUI palette, and `pytest docs/examples` executes every extending guide.
