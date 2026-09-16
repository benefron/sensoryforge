# Phase 2 handover — the general core

Prepared 2026-09-15. The approved plan is `docs/developer_guide/roadmap_v1.md` ("Phase 2 — the general
core"). Phase 1 is closed except for one CI run on GitHub (see
`docs/development/handover/phase1_tasks.md` section 1j). Open findings are in `docs_root/LEDGER.md`;
the session-start hook injects a digest.

---

## Kickoff prompt (paste to the agent)

> You are implementing Phase 2 of `docs/developer_guide/roadmap_v1.md` in `~/sensoryforge`, on `main`.
> Work on `main` directly, not in a worktree. Your task list is
> `docs/development/handover/phase2_tasks.md`. Before starting, run `git log --oneline -20`, read
> sections 1 to 3 of that file and section 2 ("Guardrails") of
> `docs/development/handover/phase1_tasks.md` in full, and recreate the memory watchdog from the
> appendix of `phase1_tasks.md` in your scratchpad. Then do Wave I in order (I1 to I8). One task = one
> commit with the ledger trailers the task names, each trailer on a single line. Before you mark a
> task done, run its "Done when" checks and paste their output into your report. New tests must fail
> on `a513dfc`. List every task you completed with its commit hash. Do not push. Stop and report when
> Wave I is finished.

---

## 1. Why Phase 2, and in what order

SensoryForge's purpose is to be a clean-slate simulator of sensor arrays, receptive fields, sensory
neurons and their readouts, and to generate data from any such design. Pressure-simulation is the first
use case: it supplies a *recipe* (a declared resolvable distance `d`, a stimulus ensemble, mutual-
information scoring) and consumes the generated data; SensoryForge builds the grid and receptive fields
from that recipe and produces the bundles.

The waves are ordered to deliver that use case first, then the generality:

| Wave | Roadmap item | Delivers |
|---|---|---|
| **I** | 2a, plus F-050 and F-051 | Receptive fields as one component (`ReceptiveFieldBank`) with pluggable builders, including the designed `template` builder and an `imported` builder; seeded receptor grids |
| **J** | 2e, F-011, F-013 | The data bundle: the contract with pressure-simulation and with learning pipelines |
| **K** | 2f | The pressure-simulation recipe end to end: presets, ported stimuli, a bundle its viewer opens |
| **L** | 2b, F-010 (part) | Sensor arrays with channels, true receptor sampling, composite grids in the engine |
| **M** | 2c | Multi-input populations and processing layers (ON/OFF) |
| **N** | 2d, F-010 (part) | Analog readouts and DSL neurons in the engine |

Waves I to K are specified below. Waves L to N are outlined in section 5 and will be detailed after
Wave I's review, because they build on the bank API that Wave I defines.

---

## 2. Facts the agent must know before touching receptive fields

Verified on `a513dfc` during the Phase 1 close-out.

- **F-051 — innervation methods are ignored on ordinary grids.** `SimulationEngine._build_populations`
  builds `InnervationModule` for non-composite grids and never passes `innervation_method`;
  `InnervationModule` has no such parameter. Measured: for a 12×12 grid with 3 neurons per row and
  seed 5, `gaussian`, `uniform`, `one_to_one` and `distance_weighted` produce bit-identical weights.
  The CLI and batch runner use this path. Only the composite/"flat" path honours the method.
- **F-050 — receptor grids are not reproducible.** `ReceptorGrid` and `CompositeReceptorGrid` take no
  seed; `jittered_grid`, `blue_noise` and `poisson` draw from the global RNG (`torch.randn_like`/
  `torch.rand_like` in `core/grid.py` and `core/composite_grid.py`). Two identical builds differ, and a
  build changes the global RNG state. `from_config(to_dict())` therefore cannot reproduce a Poisson grid.
- **Receptor ordering (keep it).** For `ReceptorGrid(grid_size=(rows, cols))`, `get_coordinates()`
  returns `xx, yy` of shape `[rows, cols]` built with `indexing="ij"`: **the first index is x**, the
  second is y. `get_receptor_coordinates()[k] == (xx[i, j], yy[i, j])` with `k = i * cols + j`, and
  stimuli `[T, rows, cols]` are flattened row-major the same way before `matmul` with weights
  `[N, rows*cols]`. Pressure-simulation's runner uses the same flattening (`stimulus.view(T, H*W)`).
- **Pressure-simulation's designed-RF output is only specified.** `decoding/modules/rf_analysis/
  rf_space_bandwidth.py` is a skeleton (18 unimplemented sections). Its intended `ConstructedRF` has
  `sigma` (mm), `pitch` Δ (mm), `centers (N, 2)` in mm **in `[y, x]` order**, and `H (N, N_grid)` with
  unit-L2 rows. The design chain is `d → f_c = 1/(2d) → σ = d/π, Δ = d, N = A/Δ²`; the realization is
  one Gaussian template translated over the neuron lattice, truncated to the K nearest receptors, with
  analytic weights. With `d = 0.40 mm` on a 16×16 grid at 0.15 mm (side 2.4 mm), σ ≈ 0.1273 mm and N = 36.
- **Pressure-simulation's bundle format** (read by `GUIs/ebkf_viewer.py` `_on_load_bundle`, written by
  `scripts/generation/_gen_small_sa_bundle.py`): `config.json` with `schema_version "1.0.0"`, `kind
  "mechanoreceptor_bundle"`, `grid {rows, cols, spacing_mm, center_mm, device}`, `populations [{name,
  neuron_type, color, parameters{...}, tensors: "population_01_SA_6.pt", visible}]`; each population
  `.pt` holds `innervation_weights` (`(N, H, W)` or `(N, N_grid)`; the loader also accepts `weights` or
  `W`) and `neuron_centers (N, 2)`; plus `stimuli/*.json` and `neuron_modules/*.json`.
- **Current innervation code** (`sensoryforge/core/innervation.py`, 1,736 lines): builder-like strategy
  classes `BaseInnervation`, `GaussianInnervation`, `UniformInnervation`, `OneToOneInnervation`,
  `DistanceWeightedInnervation` (each with `compute_weights()`, `to_dict`/`from_config` complete since
  H6), registered in `INNERVATION_REGISTRY`; plus `create_neuron_centers`, `create_innervation_map_tensor`,
  the grid-based `InnervationModule` and the coordinate-based `FlatInnervationModule`, used by
  `SimulationEngine`, `GeneralizedTactileEncodingPipeline`, `TactileEncodingPipelineTorch`,
  `notebook_pipeline.py` and the GUI (`mechanoreceptor_tab.py`, including the plot-only
  `_CSVPopulationModule` that cannot simulate).

---

## 3. Phase 2 guardrails (in addition to Phase 1 section 2)

1. **Behaviour preservation is measured, not assumed.** Before replacing a code path, record its output
   for a fixed seed (weights, spikes) in a test; the replacement must reproduce it exactly unless the
   task says the behaviour changes, and then the changelog says so.
2. **Golden parity with pressure-simulation must stay green** (`tests/integration/test_pressure_sim_parity.py`).
   If a refactor changes how weights are injected, update the test's injection, never its tolerances.
3. **Every new component kind is a registered, contract-checked plugin point.** New builders go through
   `sensoryforge.testing.contracts.check_component`; an external package must be able to add one.
4. **Coordinates are `(x, y)` in mm everywhere inside SensoryForge.** Convert at the boundary when
   importing pressure-simulation's `[y, x]` centres, and test the conversion.
5. **Keep deprecated wrappers for one phase.** When a public class is replaced (for example
   `InnervationModule`), keep a thin wrapper that emits `DeprecationWarning` and delegates, so user
   scripts keep working until Phase 4.

---

## 4. Wave I — receptive fields as one component

### I1. Seeded, reproducible receptor grids (F-050)
- **Files:** `sensoryforge/core/grid.py` (`ReceptorGrid`), `sensoryforge/core/composite_grid.py`,
  `sensoryforge/core/grid_arrangements.py`, `sensoryforge/config/schema.py` (`GridConfig`),
  `sensoryforge/core/simulation_engine.py` (`_build_grids`).
- **Do:** add `seed: Optional[int] = None` to `ReceptorGrid`, every arrangement class and
  `CompositeReceptorGrid.add_layer`; draw jitter from a per-instance CPU `torch.Generator` (same pattern
  as `_seeded_generator` in `core/innervation.py`), then move to the device; include `seed` in `to_dict()`;
  add `GridConfig.seed` and pass it through the engine.
- **Done when:** tests show two builds with the same seed give identical coordinates for `jittered_grid`,
  `blue_noise` and `poisson`, different seeds differ, building a grid leaves `torch.get_rng_state()`
  unchanged, and `from_config(to_dict())` reproduces coordinates; the contract check covers `seed`;
  the reproducibility tests fail on `a513dfc`.
- **Trailers:** `Closes: F-050`.

### I2. `ReceptiveFieldBank`
- **Files:** new `sensoryforge/core/rf_bank.py`, new `tests/unit/test_rf_bank.py`.
- **Do:** an `nn.Module` holding buffers `weights [N, M]` (float32), `neuron_centers [N, 2]` and
  `receptor_coords [M, 2]` (mm, `(x, y)`), and a `provenance: dict` (builder name, builder `to_dict()`,
  seed, source path if imported, SensoryForge version). Provide `forward(receptor_responses)` accepting
  `[batch, M]` or `[batch, time, M]` and returning `[batch, N]` / `[batch, time, N]` (raise `ValueError`
  naming the shapes on mismatch); `num_neurons`, `num_receptors`; `save(path)` writing a `.pt` dict with
  keys `innervation_weights` (`[N, M]`), `neuron_centers`, `receptor_coords`, `provenance`; classmethod
  `load(path)` accepting that file and pressure-simulation's population files (`innervation_weights` or
  `weights` or `W`, `[N, H, W]` or `[N, M]`; `receptor_coords` optional, then required as an argument).
  Google docstrings with shapes and units.
- **Done when:** unit tests cover forward shapes and values against a hand-computed `matmul`, save/load
  round trip (bit-identical buffers and provenance), loading a pressure-simulation-style `[N, H, W]` file,
  device moves, and every `ValueError`.
- **Trailers:** none.

### I3. Existing innervation methods build banks
- **Files:** `sensoryforge/core/innervation.py` (`BaseInnervation` and the four method classes),
  `sensoryforge/register_components.py`, `sensoryforge/testing/contracts.py` (`_check_innervation`).
- **Do:** add `build(receptor_coords, neuron_centers, device=None) -> ReceptiveFieldBank` to
  `BaseInnervation` (default: `compute_weights()` wrapped with provenance). Register the method classes
  themselves in `INNERVATION_REGISTRY` if the factory closures are still used, so the registry key maps
  to a contract-checked class. Before changing anything, record in a test the weights each method
  produces today through `FlatInnervationModule` for a fixed seed and layout; `build()` must reproduce
  them exactly.
- **Done when:** the recorded-weights test passes for all four methods; the contract check calls
  `build()` and checks the bank's shapes and provenance; the test that `build()` exists fails on `a513dfc`.
- **Trailers:** none.

### I4. `template` builder — designed receptive fields
- **Files:** new `sensoryforge/core/rf_builders/template.py` (or a class in `innervation.py` if that fits
  the existing layout better; state which), registration, `tests/unit/test_rf_template_builder.py`.
- **Do:** registered name `template`. Parameters: either `resolvable_distance_mm` (`d`; then
  `sigma_mm = d / π`, `pitch_mm = d`) or explicit `sigma_mm` and `pitch_mm` (exactly one form; raise
  otherwise); `k` (nearest receptors per neuron, default 28); `normalize` in `{"none", "l2", "sum"}`
  (default `"l2"`, matching pressure-simulation's unit-L2 rows); `weight_scale` (default 1.0);
  `edge_offset_mm` (default `pitch_mm / 2`). Neuron centres: a square lattice at `pitch_mm` over the
  receptor bounding box extended by half a receptor spacing on each side, inset by `edge_offset_mm`,
  row-major in the `(x, y)` ordering of section 2; the number of neurons is derived, and any
  `neurons_per_row` in the population config is ignored with a warning. Weights: for each neuron the `k`
  nearest receptors (ties broken by lower receptor index) get `exp(-r² / (2 σ²))`, all others 0, then
  normalization and scale. Deterministic; no seed.
- **Done when:** tests show: `d = 0.40` gives `sigma_mm ≈ 0.12732` and `pitch_mm = 0.40`; on a 16×16 grid at
  0.15 mm the builder creates 36 neurons; every row has exactly `k` non-zeros; with `normalize="l2"` every
  row has unit norm; an interior neuron's non-zero weights equal the analytic Gaussian values before
  normalization; two interior neurons have identical sorted weight templates (translation invariance);
  two builds are bit-identical; both parameter forms together raise.
- **Trailers:** `Decision: the template receptive-field builder derives sigma = d/pi and pitch = d from one resolvable distance d, truncates to the k nearest receptors with analytic Gaussian weights and unit-L2 rows by default`

### I5. `imported` builder — receptive fields from files
- **Files:** new builder (same location choice as I4), registration, `tests/unit/test_rf_imported_builder.py`.
- **Do:** registered name `imported`, parameter `path`. Accept (a) the GUI's CSV export folder
  (`neuron_positions.csv`, `innervation_weights.csv`, `manifest.json`), (b) a `.pt` file readable by
  `ReceptiveFieldBank.load`, and (c) an `.npz` shaped like pressure-simulation's `ConstructedRF`
  (`H [N, N_grid]`, `centers [N, 2]` in `[y, x]`, optional `sigma`, `pitch`), converting centres to
  `(x, y)`. The receptor count must match the target grid; raise `ValueError` naming both counts
  (no silent zero-fill). Provenance records the absolute source path and a SHA-256 of the file(s).
- **Done when:** tests round-trip each of the three formats; the `[y, x]` conversion is tested with an
  asymmetric layout where swapping would be detectable; a mismatched receptor count raises.
- **Trailers:** none.

### I6. The engine and pipelines use banks (F-051)
- **Files:** `sensoryforge/core/simulation_engine.py`, `sensoryforge/core/generalized_pipeline.py`,
  `sensoryforge/core/pipeline.py`, `sensoryforge/core/notebook_pipeline.py`,
  `sensoryforge/core/innervation.py` (`InnervationModule`, `FlatInnervationModule`),
  `tests/integration/test_pressure_sim_parity.py`.
- **Do:** build every population's receptive fields with `INNERVATION_REGISTRY.get_class(method)`
  (`from_config` of the population's innervation parameters) and `build()`, for both grid and flat
  paths, and feed flattened receptor responses to the bank. Keep `InnervationModule` and
  `FlatInnervationModule` as deprecated wrappers that construct a bank internally and keep their public
  attributes (`innervation_weights`, `neuron_centers`, `num_neurons`). For non-grid receptor arrangements,
  keep today's behaviour but emit a warning naming F-010 (real receptor sampling is Wave L). Update the
  golden parity test to inject weights into the bank.
- **Done when:** a test shows the four methods now give four different weight matrices through
  `SimulationEngine` on a regular grid and each equals the builder's own `build()` output; a
  single-pixel stimulus at `(i, j)` drives exactly the neurons whose weight column `i * cols + j` is
  non-zero; golden parity, GUI-engine parity and all suites pass; the F-051 test fails on `a513dfc`.
- **Trailers:** `Closes: F-051`.

### I7. The GUI uses banks
- **Files:** `sensoryforge/gui/tabs/mechanoreceptor_tab.py` (population generation, CSV import/export,
  `_CSVPopulationModule`), `sensoryforge/gui/tabs/spiking_tab.py` (`_simulate_population`).
- **Do:** populations hold a `ReceptiveFieldBank`; CSV import uses the `imported` builder and can now be
  simulated; CSV export writes the same folder format plus the bank `.pt`; add `template` to the method
  combo box with a `resolvable_distance_mm` field; remove `_CSVPopulationModule`.
- **Done when:** gui-marked tests show an imported CSV population simulates and produces spikes, a
  `template` population shows the derived neuron count, and an export followed by an import gives
  bit-identical weights; the GUI suite passes in one process.
- **Trailers:** none.

### I8. Document receptive fields
- **Files:** new `docs/user_guide/receptive_fields.md`, new `docs/developer_guide/add_rf_builder.md`,
  new `docs/examples/rf_builder_plugin.py`, `mkdocs.yml`, `CHANGELOG.md`, `CLAUDE.md` (Data Flow and
  Architecture sections).
- **Do:** explain biological versus designed receptive fields, the `d → σ, Δ, N` chain with the 16×16 /
  d = 0.40 / N = 36 example, the six builders, the coordinate and ordering conventions of section 2, and
  how to add a builder as a plugin (the executed example defines, registers and simulates one). Changelog
  entries for I1, I4 to I7 and the F-051 behaviour change.
- **Done when:** `mkdocs build --strict` passes; `pytest tests/docs` executes the new example.
- **Trailers:** none.

### Wave I exit
All suites pass in one process each; golden parity and contract tests pass; F-050 and F-051 closed;
`sensoryforge run` on a canonical config with `innervation_method: template` and
`resolvable_distance_mm: 0.40` succeeds from a wheel installed outside the repo.

---

## 5. Waves J to N (outline; detailed after Wave I's review)

### Wave J — the data bundle (F-013, F-011)

The bundle is the contract with pressure-simulation and with any learning pipeline. Section 2 of this
file records the format its viewer reads and the receptor ordering that weights follow.

#### J1. `sensoryforge/io/bundle.py` — writer and reader
- **Files:** new `sensoryforge/io/__init__.py`, `sensoryforge/io/bundle.py`, `tests/unit/test_bundle_io.py`.
- **Do:** `write_bundle(bundle_dir, config, engine, results, stimulus, *, stimulus_config=None, seed=None, overwrite=False) -> Path` writing:

  | Path | Contents |
  |---|---|
  | `config.json` | `schema_version "2.0.0"`, `kind "sensoryforge_bundle"`, plus every field pressure-simulation's 1.0.0 loader reads: `grid {rows, cols, spacing_mm, center_mm, device}` and `populations [{name, neuron_type, color, parameters{neurons_per_row, connections_per_neuron, sigma_d_mm, weight_min, weight_max, seed, edge_offset}, tensors: "population_01_<NAME>.pt", visible}]`; plus `sensoryforge_version`, the full canonical config, and `bundle_created` (ISO-8601 UTC) |
  | `population_NN_<NAME>.pt` | exactly `ReceptiveFieldBank.save()` output: `innervation_weights [N, M]`, `neuron_centers [N, 2]`, `receptor_coords [M, 2]`, `provenance`, plus `grid_shape [rows, cols]` so a consumer can reshape to `[N, rows, cols]` |
  | `stimuli/stimulus.json` | the stimulus config dict (empty dict if unknown) |
  | `data.h5` | `/stimulus/frames` `[T, H, W]` or `[T, C, H, W]` float32 gzip-4; `/time_ms [T]`; `/populations/<name>/{drive, filtered, spikes|state}` `[T, N]` (spikes as int16 counts, gzip-4); root attributes `dt_ms`, `integrate_dt_ms`, `seed`, `sensoryforge_version`; `/meta` attributes `config_yaml` and `provenance_json` |

  `load_bundle(bundle_dir) -> Bundle` returns a dataclass with `config` (`SensoryForgeConfig`), `banks` (name → `ReceptiveFieldBank`), `stimulus`, `time_ms`, `populations` (name → dict of arrays), `meta`. Both functions take `str | Path`. Raise `ValueError` naming the file when `schema_version` is missing or its major version is not 2.
- **Done when:** a round-trip test writes a two-population run and reads it back with tensors bit-identical and config equal; `h5py` datasets have the documented names, shapes, dtypes and attributes; `[T, C, H, W]` stimuli round-trip; a missing/incompatible `schema_version` raises; the test fails on the wave's base commit.
- **Trailers:** none.

#### J2. The engine and CLI write bundles
- **Files:** `sensoryforge/core/simulation_engine.py` (`run`), `sensoryforge/cli.py` (`cmd_run`), `tests/integration/test_bundle_cli.py`.
- **Do:** `SimulationEngine.run(..., bundle_dir=None)` writes a bundle when given (it already has the config, banks and results; it must pass `return_intermediates=True` internally when a bundle is requested so `drive` and `filtered` exist). Add `sensoryforge run --bundle DIR`. Keep `--output` working.
- **Done when:** a CLI test runs a canonical config with `--bundle`, then `load_bundle` reads it and the spike array matches the `--output` `.pt` from the same seed; fails on the base commit.
- **Trailers:** none.

#### J3. Batch writes one bundle per stimulus, and SLURM works (F-011, F-013)
- **Files:** `sensoryforge/core/batch_executor.py`, `sensoryforge/cli.py` (`cmd_batch`, `create_parser`), `tests/unit/test_batch_executor_bundles.py`.
- **Do:** `BatchExecutor` writes `<output_dir>/<batch_id>/stim_%04d/` bundles, with `batch_metadata.json` and `stimulus_index.json` at the batch root; HDF5 becomes the default output format and the monolithic consolidated `.pt` is removed (`format: pytorch` now means per-bundle `.pt` payloads only). Add `sensoryforge batch --task-index N` running exactly one stimulus index, and make `generate_slurm_script` emit an array job that calls it (with `--output`), so every flag it emits exists.
- **Done when:** a batch of three stimuli produces three readable bundles with distinct spike arrays; `--task-index 1` reproduces bundle 1 exactly; a test asserts every flag in the generated SLURM script is accepted by `create_parser()` (parse the script's `sensoryforge ...` line and feed it to the parser); fails on the base commit.
- **Trailers:** `Closes: F-011`, `Closes: F-013`.

#### J4. pressure-simulation can read our bundles
- **Files:** `tests/integration/test_bundle_pressure_sim_compat.py`.
- **Do:** re-implement pressure-simulation's loader steps in the test (do not import that repo): read `config.json`, for each population entry load its `tensors` file, accept `innervation_weights`/`weights`/`W`, reshape `[N, M]` to `[N, rows, cols]` using `grid_shape`, and check `neuron_centers` shape; then rebuild the drive as that repo does (`stimulus.view(T, H*W) @ W.T`) and assert it equals the bundle's stored `drive` to 1e-6.
- **Done when:** the test passes on a freshly written bundle and fails if `config.json` drops any 1.0.0 field (parametrise one deletion).
- **Trailers:** none.

#### J5. Document the bundle
- **Files:** new `docs/user_guide/bundles.md`, new `docs/examples/read_bundle.py`, `mkdocs.yml`, `CHANGELOG.md`.
- **Do:** document the layout table, units and dtypes, how to load in torch/numpy/pandas, how spikes-as-counts differ from a binary raster, and how pressure-simulation reads it. The executed example writes a small bundle and reads it back.
- **Done when:** `mkdocs build --strict` passes and `pytest tests/docs` executes the example.
- **Trailers:** none.

#### Wave J exit
All suites pass in one process each; golden parity and contract tests pass; F-011 and F-013 closed;
from a wheel installed outside the repo, `sensoryforge run --bundle` and `sensoryforge batch` produce
bundles that `load_bundle` reads.

### Wave K — the pressure-simulation recipe
- Presets under `sensoryforge/presets/`: `tactile_sa1_ra1.yml` (SA regular-spiking, RA fast-spiking,
  τ values, `template` builder with `resolvable_distance_mm: 0.40`) and `tactile_stochastic_control.yml`.
- Port pressure-simulation's run stimuli as registered stimuli: `ramp_gaussian`, `moving_edge`,
  `braille_H`, `drifting_grating` (`experiments/ncn2026/_harness.py`, `scripts/ebkf/_ebkf_pres_movies.py`),
  with a golden test against arrays exported from pressure-simulation, like E5.
- `examples/pressure_simulation_recipe.py`: build the grid and receptive fields from `d`, run the four
  stimuli, write bundles; the Wave J loader test opens them.

### Wave L — sensor channels and receptor sampling (F-010, part)
`GridConfig.channels`, stimulus tensors with an optional channel axis, `StimulusConfig.channel`, bilinear
sampling of each channel at `receptor_coords` (`torch.nn.functional.grid_sample`) so hex, Poisson and
imported layouts are correct, and composite grids in `SimulationEngine` (remove its
`NotImplementedError`).

### Wave M — multi-input populations and processing layers
`PopulationConfig.inputs: list[{grid, channel, rf, gain}]` with the current single-input fields as sugar,
`combine: "sum" | "concat"`, one bank per input, and `ProcessingPipeline` wired as a per-input stage with
an `on_off` centre-surround layer as the first non-trivial processing plugin.

### Wave N — analog readouts and DSL neurons in the engine (F-010, part)

A population may read out a continuous state instead of spikes. This is the "or not spiking, if we go
the DSL path" half of the project's purpose, and it is independent of Waves J to M.

#### N1. DSL models without a spike condition
- **Files:** `sensoryforge/neurons/model_dsl.py`, `tests/unit/test_dsl_analog.py`.
- **Do:** make `threshold` and `reset` optional in `NeuronModel.__init__`, `_validate_model` and
  `from_config` (today all three require them). With no threshold the compiled module integrates the
  equations and returns `(state_trace, None)`; `get_param_spec()` and `to_dict()` round-trip the optional
  fields. Keep the Euler-only restriction and its error message.
- **Done when:** a leaky-integrator DSL model (`dv/dt = (-(v - v_rest) + R*I) / tau_m`, no threshold)
  compiles and returns a state trace of shape `[batch, steps+1, features]` with `spikes is None`, its
  values match a hand-written Euler integration to 1e-6, a model *with* a threshold still spikes exactly
  as before (compare against a recorded array), and `to_dict()/from_config()` round-trip both; fails on
  the base commit.
- **Trailers:** none.

#### N2. The shared backend labels analog output
- **Files:** `sensoryforge/core/simulation_engine.py` (`_run_pop_from_drive`), `sensoryforge/neurons/base.py`
  (docstring: `spikes` may be `None`), `sensoryforge/testing/contracts.py`, `tests/unit/test_analog_readout.py`.
- **Do:** when a neuron returns `None` spikes, the result dict carries `"state"` `[batch, T, N]` (bin-end
  samples, the same reduction voltages already use) and no `"spikes"` key; spiking models are unchanged.
  `SimulationEngine.run` propagates whichever key exists.
- **Done when:** a spiking population still returns `spikes` with identical values to the base commit for a
  fixed seed, an analog population returns `state` and no `spikes`, and both shapes are `[1, T, N]`.
- **Trailers:** none.

#### N3. The engine builds DSL neurons (F-010, DSL half)
- **Files:** `sensoryforge/core/simulation_engine.py` (`_build_populations`), `sensoryforge/config/schema.py`
  (`PopulationConfig.readout`), `tests/integration/test_engine_dsl.py`.
- **Do:** when `neuron_model` resolves to the DSL model, build it as `NeuronModel.from_config(pop_cfg.dsl_config)`
  then `.compile(dt=integrate_dt_ms, device=...)`, instead of calling the class with `dt=`/`noise_std=`
  (which raises today). Add `PopulationConfig.readout: str = "auto"` (`"auto"` infers analog when the DSL
  config has no threshold; `"spiking"`/`"analog"` force it, raising if impossible). Give a clear `ValueError`
  when `neuron_model` is DSL and `dsl_config` is missing.
- **Done when:** a canonical config with a DSL leaky integrator runs end to end through `SimulationEngine`
  and returns `state`; a DSL model with a threshold returns `spikes`; the missing-`dsl_config` error is
  tested; every case fails on the base commit (today it raises `TypeError`/`ValueError: Unknown neuron model`).
- **Trailers:** `Opens:` a finding for whatever of F-010 remains (composite grids and non-grid receptor
  sampling are Wave L) if you touch that code; otherwise none.

#### N4. The GUI shows analog populations
- **Files:** `sensoryforge/gui/tabs/spiking_tab.py`, `tests/unit/test_gui_analog.py` (gui-marked).
- **Do:** when a population's result has `state` instead of `spikes`, plot the state trace in place of the
  raster and label the axis with the state variable's name; the raster panel stays for spiking populations.
  Do not redesign the tab.
- **Done when:** a gui test builds an analog population, simulates, and asserts the trace panel has data and
  no raster points; the GUI suite passes in one process.
- **Trailers:** none.

#### N5. Document analog readouts
- **Files:** new `docs/user_guide/analog_readouts.md`, new `docs/examples/analog_dsl.py`, `mkdocs.yml`,
  `CHANGELOG.md`, `CLAUDE.md` (Data Flow shows spiking or analog readout).
- **Done when:** `mkdocs build --strict` passes; `pytest tests/docs` executes the example.
- **Trailers:** none.

#### Wave N exit
All suites pass in one process each; a canonical config with one spiking and one analog population runs
from a wheel installed outside the repo; spiking results are unchanged from the base commit.

---

## 6. Phase 2 exit criteria

- Waves I to N complete with their exit checks.
- A canonical config with two channels, a `template` RF input and an `imported` RF input combined into one
  population, an analog DSL readout, and a spiking population runs from a wheel installed outside the repo
  and writes a bundle that `load_bundle` reads and pressure-simulation's viewer logic opens.
- Golden parity with pressure-simulation passes; all suites, black, flake8 and `mkdocs build --strict` pass.
- Ledger: F-010, F-011, F-013, F-050, F-051 closed.
