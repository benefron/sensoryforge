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
- `sensoryforge/io/bundle.py` with `write_bundle(run_dir, config, engine, results, stimulus)` and
  `load_bundle(run_dir)`.
- Layout, a superset of pressure-simulation's format: `config.json` (`schema_version "2.0.0"`, `kind
  "sensoryforge_bundle"`, plus the 1.0.0 fields its viewer reads), `population_NN_<name>.pt`
  (`ReceptiveFieldBank.save` output), `stimuli/*.json`, and `data.h5` (`/stimulus/frames [T, C, H, W]`,
  `/time_ms [T]`, attributes `dt_ms`, `integrate_dt_ms`; `/populations/<name>/{drive, filtered, spikes
  (int counts) | state}` `[T, N]`; `/meta` with the full config YAML, seed and versions).
- Writers: `SimulationEngine.run(..., bundle_dir=...)`, GUI auto-save, and `BatchExecutor` (one bundle
  per stimulus; HDF5 becomes the default; the monolithic `.pt` is removed). `sensoryforge export-bundle`.
- SLURM: `generate_slurm_script` emits `sensoryforge batch --task-index $SLURM_ARRAY_TASK_ID` and that
  flag exists (F-011).
- Acceptance includes a test that runs pressure-simulation's viewer loader logic on a SensoryForge bundle.
  `h5py` becomes required for this wave (`pip install -e ".[hdf5]"` in the environment and in CI).

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
Optional `threshold`/`reset` in `neurons/model_dsl.py`, outputs labelled `state` when there are no spikes,
the engine instantiating DSL models (`from_config` then `compile`), and GUI trace plots for analog
populations.

---

## 6. Phase 2 exit criteria

- Waves I to N complete with their exit checks.
- A canonical config with two channels, a `template` RF input and an `imported` RF input combined into one
  population, an analog DSL readout, and a spiking population runs from a wheel installed outside the repo
  and writes a bundle that `load_bundle` reads and pressure-simulation's viewer logic opens.
- Golden parity with pressure-simulation passes; all suites, black, flake8 and `mkdocs build --strict` pass.
- Ledger: F-010, F-011, F-013, F-050, F-051 closed.
