# Phase 2 handover — the general core

Prepared 2026-09-15. The approved plan is `docs/developer_guide/roadmap_v1.md` ("Phase 2 — the general
core"). Phase 1 is closed except for one CI run on GitHub (see
`docs/development/handover/phase1_tasks.md` section 1j). Open findings are in `docs_root/LEDGER.md`;
the session-start hook injects a digest.

---

## How Phase 2 is run

Phase 2 is orchestrated: an overseeing session dispatches each wave to an implementation agent working
in its own git worktree, reviews the result, and merges it into the integration branch `phase2`. `main`
stays at the end of Phase 1 until Phase 2 is complete and reviewed.

- **Integration branch:** `phase2` (forked from `main` at `060363a`).
- **Each wave** is developed on its own branch in a worktree, then merged into `phase2` after review.
- **Every agent** reads sections 1 to 3 of this file and section 2 ("Guardrails") of
  `docs/development/handover/phase1_tasks.md`, recreates the memory watchdog from that file's appendix,
  commits one task per commit with single-line ledger trailers, never pushes, and reports each task's
  commit hash with its "Done when" output.
- **Waves J and N run in parallel** (disjoint files). K follows J. L follows N, because both
  rewrite `SimulationEngine`. M follows L. Phase 3 and Phase 4 follow Phase 2's exit check.
- **All six waves are fully specified** in sections 4 and 5; there is no further outline stage.
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
6. **Peak resident memory is a smoke alarm, not a gauge** (F-056). The watchdog samples once a
   second and sums the process and its children, so the same code measures anywhere from roughly
   800 MB to 1,500 MB run to run. Measured on 2026-09-16: commit 83b735d gave 873 MB on one run and
   1,517 MB on another, while the Wave L merge above it gave 1,415 MB. Report the number, but do not
   read a regression into anything short of the guardrail's doubling, and never compare a number
   from your run against one from someone else's. Wave T of Phase 4 builds the harness that can
   actually answer this.
7. **`Finding:` opens a numbered ledger entry; it is not a progress note.** Use it only to record a
   *new* problem, with a self-contained one-line description someone can act on a year from now.
   Annotating which part of an existing finding a commit addresses belongs in the commit body, not a
   trailer. Wave L wrote `Finding: F-010 (part 1 of 4)` on three commits and opened three empty
   findings that had to be deleted by hand at the merge.
8. **Never run `pip install -e .` from a worktree** (F-053). The conda environment is shared by every
   worktree, and an editable install rewrites one global pointer. A parallel wave that reinstalls
   repoints every other checkout at its own code, and the damage is invisible: in-process pytest keeps
   working because the working directory is on `sys.path`, while anything launched as a subprocess
   from another directory silently imports the wrong tree and still reports green. If you believe you
   need a reinstall, say so in your report instead and let the orchestrating session do it.

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

## 5. Waves J to N

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

Wave K makes SensoryForge able to reproduce, on its own, the exact stimulus ensemble that
pressure-simulation runs its decoder on, and to ship that whole run as a preset plus one example
script. It depends on Wave J (bundles) and must start from the commit where Wave J is merged.

**Fact K-a (verified 2026-09-16).** Both repositories build coordinate meshgrids with
`torch.meshgrid(x, y, indexing="ij")` (`sensoryforge/core/grid.py:80`, pressure-simulation
`encoding/grid_torch.py:39`). Frame element `[i, j]` is at `(x[i], y[j])` in both: the first frame
axis is x, the second is y. There is no transpose to undo. A port that swaps the axes will fail K3.

**Fact K-b (verified 2026-09-16, opened as F-052).** `GeneralizedTactileEncodingPipeline.generate_stimulus`
(`sensoryforge/core/generalized_pipeline.py:1030-1073`) dispatches stimulus names through a
hard-coded if/elif chain that knows nine names. `STIMULUS_REGISTRY` holds `gaussian`, `static`,
`moving`, `composite`, `timeline`, `repeated_pattern`, `texture`, `gabor`, `edge_grating`. The CLI
calls that chain even for canonical configs (`sensoryforge/cli.py:222`), so `composite`,
`edge_grating`, `gabor` and `static` cannot be run from a config file at all, and a third-party
stimulus plugin can be registered but never executed. Fixing this is K1 and is a prerequisite for
the four ported stimuli being reachable.

**Fact K-c (verified 2026-09-16).** A stimulus class may return either one frame `[H, W]`
(`GaussianStimulus`) or a whole sequence `[T, H, W]` (`MovingStimulus.forward`,
`sensoryforge/stimuli/moving.py:324-333`). Both are valid; the renderer in K1 normalises them.

#### K1. One stimulus renderer, dispatching through the registry (F-052)

New module `sensoryforge/stimuli/render.py`:

```python
def render_stimulus(
    stimulus_type: str,
    params: dict,
    xx: torch.Tensor,          # [H, W] mm
    yy: torch.Tensor,          # [H, W] mm
    dt_ms: float,
    duration_ms: float | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:   # frames [T, H, W], time_ms [T]
```

Rules:

- If `stimulus_type` is in `STIMULUS_REGISTRY`, build it with `from_config(params)` and call
  `forward(xx, yy)`. A `[H, W]` result is expanded to `[T, H, W]` with the temporal envelope below;
  a `[T, H, W]` result is returned as is, and `duration_ms` (when given) truncates it or right-pads
  with zeros. Never silently resample.
- Otherwise fall back to `GeneralizedTactileEncodingPipeline.generate_stimulus` so every legacy name
  (`trapezoidal`, `step`, `ramp`, `custom`) keeps working unchanged.
- The temporal envelope for single-frame stimuli is pressure-simulation's
  (`encoding/encode_runner.py:45-64`): `ramp_up_ms` linear rise, `plateau_ms` hold at 1.0,
  `ramp_down_ms` linear fall, zero after `ramp_up + plateau + ramp_down`, scaled by `amplitude`.
  Defaults must reproduce today's behaviour for existing configs.
- **Corrected 2026-09-16.** An earlier version of this line said to build the axis as
  `arange(0.0, duration_ms + 0.5 * dt_ms, dt_ms)`, borrowing pressure-simulation's half-step guard.
  That was wrong here and it shipped a real inconsistency: `--duration 100 --dt 1.0` gave 101 frames
  for a registered stimulus and 100 for a legacy one. The guard is correct only where the field means
  *the time of the last sample*, which is what pressure-simulation's `total_ms` means and how the
  bundle payload uses it. `duration_ms` means a duration. So:
  `n_frames = round(duration_ms / dt_ms)` and `time_ms = arange(n_frames) * dt_ms`, matching the
  legacy path. Say in the docstring which convention a given field follows.
- **Defaults must be preserved for names the legacy chain also knows.** Five registered names overlap
  it — `gaussian`, `moving`, `repeated_pattern`, `texture`, `timeline` — and their component defaults
  are not the legacy generator's. `GaussianStimulus` defaults sigma to 0.2 mm where the legacy
  generator used 1.0 mm, which on a 40x40 grid at 0.15 mm is 20 times narrower and 25 times weaker,
  silently, for every shipped config that says `type: gaussian` with no sigma. Keep an explicit,
  commented compatibility map of the legacy defaults in `render.py`, applied only where the caller
  supplied nothing, and pin each overlapping name with a test asserting the two paths are
  bit-identical. Do not change the component classes' own defaults.
- An unknown name raises `ValueError` listing the registered names, never a bare `KeyError`.

The CLI (`sensoryforge/cli.py:220-228`) and `BatchExecutor` call `render_stimulus` instead of
reaching into the pipeline. Tests: a stimulus registered only at test time (a two-line subclass)
runs end to end through `sensoryforge run`; `edge_grating` and `gabor` become runnable from a config
file. Both must fail on the Wave J merge commit.

#### K2. The four pressure-simulation stimuli

New module `sensoryforge/stimuli/tactile.py`, four `BaseStimulus` subclasses, each with
`get_param_spec()`, `from_config`, `to_dict`, `reset_state`, Google docstrings carrying shapes and
units, and registration in `register_components.py`. `forward(xx, yy)` returns `[T, H, W]`. The
formulas are transcribed from pressure-simulation and must not be "improved".

`ramp_gaussian` — from `experiments/ncn2026/_harness.py::gen_ramp_gaussian`.
`blob = exp(-(xx**2 + yy**2) / (2 * sigma_mm**2))`, multiplied by an amplitude vector that is
`torch.linspace(0, 1, ramp_ms)` over the first `ramp_ms` samples and 1.0 after.
Defaults `total_ms=1100, ramp_ms=50, sigma_mm=1.0`.

`moving_edge` — from `encoding/encode_runner.py::generate_stimulus_from_json` with `type="edge"`,
`motion="moving"`. Envelope as in K1. Centre interpolates `c = start + alpha * (end - start)` with
`alpha = 0` for `t <= ramp_up`, `alpha = 1` for `t >= ramp_up + plateau`, else
`(t - ramp_up) / plateau`. Frame is `exp(-p**2 / (2 * spread**2))` with
`p = (xx - cx) * sin(theta) + (yy - cy) * cos(theta)` and `theta = radians(orientation_deg)`.
Defaults come from the shipped payload: `start=(-7.11, 0.0)`, `end=(7.0, 0.0)`, `spread=1.0`,
`orientation_deg=50.0`, `amplitude=1.0`, `ramp_up_ms=20`, `plateau_ms=300`, `ramp_down_ms=10`,
`total_ms=330`, `dt_ms=1.0`.

`braille` — from `scripts/ebkf/_ebkf_pres_movies.py::gen_braille_H`. Dots at offsets
`[(-1.5, -1.5), (+1.5, -1.5), (+1.5, +1.5)]` in (row, col), the letter H, exposed as the default of
a `dot_offsets` parameter so other letters are expressible. The cell centre
`cy = -7.0 + v_mms * t_s` moves along the **second** axis (`yy`); dot rows are fixed. Sum over dots
of `exp(-((xx - dot_row)**2 + (yy - dot_col)**2) / (2 * sigma_dot**2))`, then a symmetric linear
ramp in and out over `ramp_n = min(ramp_ms, T // 2)` samples, then `clamp(0, n_dots)`.
Defaults `total_ms=900, ramp_ms=75, v_mms=20.0, sigma_dot=0.40`.

`drifting_grating` — from `scripts/ebkf/_ebkf_pres_movies.py::gen_drifting_grating`.
`phase = 2 * pi * spatial_freq * (xx + v_mms * t_s)`, `frame = 0.5 * (1 + cos(phase))`, the same
symmetric ramp, then `clamp(0, 1)`. Defaults `total_ms=1000, ramp_ms=100, spatial_freq=0.25`
cycles/mm, `v_mms=15.0`.

In all four, `t_s = arange(T) * dt_ms / 1000`, with `dt_ms` a constructor parameter defaulting to
1.0 (the source scripts hard-code `DT = 1.0`).

The existing `edge_grating` stimulus is a *static* stack of lobes and is not a substitute for
`drifting_grating`. Do not merge them.

#### K3. Golden test against pressure-simulation

`tests/integration/test_stimulus_parity.py`, built like the Wave E filter parity test.

- `scripts/regenerate_stimulus_golden.py` imports pressure-simulation (root from the
  `PRESSURE_SIM_ROOT` environment variable, default `~/Documents/pressure simulation`), builds
  `GridManager(grid_size=80, spacing=0.15, center=(0.0, 0.0))`, generates the four stimuli with the
  defaults above, and writes `tests/fixtures/stimulus_golden.npz`: one float32 array per stimulus,
  subsampled to every 10th time sample to keep the fixture small, plus a `meta` JSON string
  recording source file, function, parameters and the time stride. The script skips with a clear
  message when pressure-simulation is absent.
- The test loads the fixture, renders the same four stimuli through `render_stimulus` on a
  SensoryForge grid of the same geometry, applies the same stride, and asserts **exact** equality at
  zero tolerance, as `tests/integration/test_pressure_sim_parity.py` does.
- Prove the test bites: four separate mutations (swap the braille axes, drop the grating ramp, flip
  the edge orientation sign, change a sigma) each make it fail. Record all four in the report.

#### K4. Presets

New package `sensoryforge/presets/` with `__init__.py` exposing `list_presets() -> list[str]`,
`load_preset(name) -> dict` and `PRESET_DIR`, reading YAML fragments through `importlib.resources`,
never a cwd-relative path (Phase 1b). Add the directory to `package_data` and verify it survives a
wheel install, the way Wave G did for the other data files.

- `tactile_sa1_ra1.yml` — the pressure-simulation recipe: 80x80 grid at 0.15 mm, an SA population
  (regular-spiking preset, `sa` filter) and an RA population (fast-spiking preset, `ra` filter,
  k3 = 2.0), both using the `template` receptive-field builder with
  `resolvable_distance_mm: 0.40`, `dt_ms: 1.0`.
- `tactile_stochastic_control.yml` — identical except the builder is `gaussian_stochastic`, the
  named control arm (D-019).

CLI: `sensoryforge list-presets` prints each name with a one-line description;
`sensoryforge run --preset tactile_sa1_ra1` runs with no config file; `--preset X config.yml` uses
the preset as the base that the file overrides. A test loads every preset and constructs a
`SensoryForgeConfig` from it, so a broken preset fails CI.

#### K5. `examples/pressure_simulation_recipe.py`

One script, runnable from a clean checkout with no arguments, that loads the `tactile_sa1_ra1`
preset, builds the grid and both receptive-field banks from `d = 0.40` mm, renders the four K2
stimuli, runs each through `SimulationEngine`, writes one Wave J bundle per stimulus under
`examples/output/pressure_simulation_recipe/`, and prints per-population spike counts and mean
rates.

It must finish in under two minutes on CPU. Shorten durations behind a `--quick` flag rather than
shrinking the grid, so the default run is the real recipe. A test runs it with `--quick` and asserts
the four bundles exist and reload through the Wave J loader.

#### K6. Documentation

- `docs/concepts/pressure_simulation_use_case.md` — the spine end to end for this recipe: what
  pressure-simulation supplies (the resolvable distance `d`, the stimulus ensemble, the mutual
  information scoring) and what SensoryForge supplies (grid, receptive fields, filters, neurons, the
  bundle). Cross-link the receptive-field concepts page from Wave I8 and the bundle page from Wave J.
- `docs/user_guide/presets.md` — what a preset is, the two shipped ones, how to override one, how to
  add one. A preset is data, not code, so this is the cheapest extension point.
- `docs/extending/add_stimulus.md` updated for the registry dispatch: a registered stimulus is now
  runnable from a config file, with the two-line plugin from K1 as the worked example.
- All four ported stimuli appear in `sensoryforge list-components`.

#### Wave K exit

`pytest -m "not gui"` and `pytest -m gui` green under the watchdog; `black --check`, the CI flake8
subset and `mkdocs build --strict` clean; the stimulus golden test passes with its four mutation
proofs recorded; `examples/pressure_simulation_recipe.py --quick` writes four bundles that reload;
`Closes: F-052` on the K1 commit.

### Wave L — sensor channels and receptor sampling (F-010, part)

Wave L is where SensoryForge stops being a tactile-pressure simulator and becomes a general sensor
substrate. It closes the "silently wrong" half of F-010. It touches `SimulationEngine`, so it must
start from the commit where Wave N is merged, to avoid a three-way conflict in that file.

**Fact L-a (verified 2026-09-16).** `SimulationEngine.run` flattens the stimulus row-major to
`h * w` and feeds it straight to the receptive-field bank (`simulation_engine.py:412-424`).
`_stimulus_to_receptors` (`:563-586`) only adds a batch dimension; its own docstring calls itself "a
simplified implementation". So the engine assumes receptor index equals stimulus pixel index. For a
hex, Poisson, jittered or blue-noise arrangement, or for imported coordinates, the receptors are not
on the pixel lattice: the run either raises a shape error or, when the counts happen to agree,
produces a wrong answer with no warning. That is the bug.

**Fact L-b (verified 2026-09-16).** `SimulationEngine._build_grids` raises `NotImplementedError` for
`arrangement == "composite"` (`:98-105`), while `_build_populations` already has a
`CompositeReceptorGrid` branch that calls `get_all_coordinates()` and sets `use_flat = True`
(`:155-161`). The population half is written; only the grid half is missing.

**Fact L-c (verified 2026-09-16).** `CompositeReceptorGrid.add_layer_with_coords(name, coordinates,
color=None, **metadata)` registers a layer from coordinates that are already computed
(`composite_grid.py:181`), and `get_all_coordinates()` concatenates layers in insertion order
(`:339-352`). Layer order is therefore the contract for which receptor is which index; it must be
written into the bundle and into the bank provenance.

#### L1. Channels on the sensor array

`GridConfig` gains `channels: list[str] = field(default_factory=lambda: ["value"])` and
`coords_file: Optional[str] = None`. Round-trip through `to_dict`/`from_dict`/YAML, with the
single-channel default omitted from `to_dict` output so existing configs are unchanged byte for
byte. `coords_file` reads an `[M, 2]` CSV or `.pt` of receptor coordinates in mm and builds the grid
from them, using `add_layer_with_coords`.

Validation: channel names must be non-empty, unique, and valid identifiers. A duplicate or empty
name raises `ValueError` naming the grid and the offending entry.

#### L2. Stimulus tensors with a channel axis

A stimulus is `[batch, time, H, W]` today. It becomes `[batch, time, C, H, W]` when `C > 1`;
`C == 1` keeps the four-dimensional form so nothing existing changes shape. `StimulusConfig` gains
`channel: Optional[str]`, naming which plane of the target grid a stimulus drives. Several stimuli
with different `channel` values compose into one multi-channel tensor, each filling its own plane;
planes with no stimulus are zero.

`render_stimulus` from Wave K1 grows a `channels: list[str] | None` argument and returns the
channel axis when asked. Document the shape rule in one place (`docs/concepts/units_and_shapes.md`)
and reference it from the base classes rather than repeating it.

#### L3. Real receptor sampling (the F-010 fix)

Rewrite `SimulationEngine._stimulus_to_receptors` so it samples the stimulus **at receptor
coordinates** instead of assuming an index correspondence:

- Build normalised sampling coordinates from `receptor_coords [M, 2]` in mm and the grid's `xlim`,
  `ylim`, mapping to `[-1, 1]` in the order `grid_sample` expects (its last axis is `(x, y)` with x
  indexing the **width** axis, which is the second frame axis here — Fact K-a says frame `[i, j]` is
  at `(x[i], y[j])`, so the mapping is not the naive one; write the index algebra out in the
  docstring and test it against a known Gaussian).
- Call `torch.nn.functional.grid_sample(frames, coords, mode="bilinear", align_corners=True,
  padding_mode="zeros")` once for the whole `[batch, time, C, H, W]` tensor, folding time into the
  batch axis, and return `[batch, time, C, M]`.
- Keep the fast path: when the arrangement is a regular grid whose receptor count equals `H * W` and
  whose coordinates match the lattice, skip `grid_sample` and reshape, so the Wave E and Wave K
  golden parity tests stay bit-identical. Assert that equivalence in a test rather than assuming it.

Tests: a hex layout recovers a known analytic Gaussian to better than 1% RMS; a regular grid gives
results bit-identical to the reshape path; a receptor outside the stimulus bounds samples zero, not
an edge-clamped value; the golden parity fixture from Wave E is unchanged.

#### L4. Composite grids in the engine

Delete the `NotImplementedError` at `simulation_engine.py:98-105`. A `GridConfig` with
`arrangement == "composite"` builds a `CompositeReceptorGrid` whose layers come from the grid's
`layers:` list (each entry a name plus density or explicit coordinates plus an arrangement), in
declaration order. Record the layer order and per-layer receptor counts in the grid's provenance so
a bundle reader can slice `get_all_coordinates()` back into layers.

A population targeting a composite grid innervates across all layers by default, or a named subset
via `target_layers: list[str]`. Test: a two-layer composite grid runs end to end through
`SimulationEngine.run`, and a bank built on layer A only is unaffected by changing layer B's density.

#### L5. Documentation and tests

- `docs/concepts/sensor_arrays.md` — geometry and channels, the four arrangements, imported
  coordinates, composite layers, and why receptor index is not pixel index.
- `docs/extending/add_grid_arrangement.md` — the worked example is a new arrangement plugin (a
  spiral or a fovea-style radial layout), executed in CI like the other extending guides.
- `docs/concepts/units_and_shapes.md` updated with the channel axis.
- `CLAUDE.md` technical-debt entry for composite grids removed, since it is no longer true.

#### Wave L exit

`Closes: F-010` only if Wave N has already closed its half; otherwise the commit carries the partial
note and F-010 stays open until Wave M. Both suites green under the watchdog, `black --check`,
the CI flake8 subset and `mkdocs build --strict` clean, and every Wave E and Wave K golden test
unchanged.

### Wave M — multi-input populations and processing layers

Wave M is the last structural wave: a sensory neuron may read from more than one channel or grid,
and an optional transduction stage sits between the sensor and the receptive field. It depends on
Wave L.

#### M1. `PopulationInput` and the sugar that expands to it

New dataclass in `sensoryforge/config/schema.py`:

```python
@dataclass
class PopulationInput:
    grid: str                       # grid name
    channel: str = "value"          # channel within that grid
    rf: RFBuilderConfig = ...       # builder name + params (Wave I)
    gain: float = 1.0
```

`PopulationConfig` gains `inputs: list[PopulationInput]` and `combine: str = "sum"`. The existing
single-input fields stay and are **sugar**. As of Wave L that set is `target_grid`, `target_layers`,
`innervation_method`, `sigma_d_mm`, `connections_per_neuron`, `use_distance_weights`,
`resolvable_distance_mm` and `innervation_params`. Read the dataclass rather than trusting this list:
Wave L added `target_layers` to `PopulationConfig` and `channel` to `StimulusConfig`, and a field
added after this was written must not be silently dropped by the expansion.

The sugar works like this:
`from_dict` expands them into exactly one `PopulationInput`, and `to_dict` writes the short form
back when there is exactly one input whose fields fit it. Every existing config must round-trip to
byte-identical YAML. Test that explicitly over `examples/*.yml` and the Wave K presets.

Setting both the sugar fields and `inputs` in one population raises `ValueError` naming the
population, the same way the Wave I template builder rejects both parameter forms.

**Wave L's `to_dict` contract must survive.** `GridConfig.to_dict` now omits `channels` when it is
the single-channel default, and omits `coords_file` and `layers` when empty, precisely so existing
configs serialise byte for byte. Apply the same discipline to `inputs` and `combine`: a population
that has exactly one input expressible as sugar must write the sugar and nothing else. The
round-trip test over `examples/*.yml` and the Wave K presets is what proves it, and it is not
optional.

#### M2. The engine builds and combines one bank per input

`_build_populations` builds a `ReceptiveFieldBank` per `PopulationInput`, each on its own grid and
channel. `run` computes one drive per input, scales by that input's `gain`, and combines:

- `"sum"` — element-wise sum, all inputs must agree on `N`.
- `"concat"` — concatenate along the neuron axis, giving `N * len(inputs)` neurons; the population's
  neuron count and the bundle's `neuron_centers` must reflect that, with each block's provenance
  naming its input.

`_run_pop_from_drive` is untouched. Tests: two channels summed give the same answer as one channel
whose stimulus is the sum, when both banks are identical; `concat` produces the expected shape and
its blocks match the single-input runs exactly.

#### M3. Processing layers as a per-input stage

`ProcessingPipeline` (`sensoryforge/core/processing.py:129`) is wired between receptor sampling and
the receptive-field bank, per input, configured by `PopulationInput.processing: list[dict]` and
defaulting to nothing at all (not an `IdentityLayer` instance, so the default path allocates
nothing and the golden tests stay bit-identical).

Add `OnOffLayer` as the first non-trivial layer and the worked example for the docs: a
centre-surround difference of Gaussians over receptor coordinates, emitting an ON plane and an OFF
plane. It is registered in a `PROCESSING_REGISTRY` that follows the same contract as the others, and
it is what makes the vision demo in M4 meaningful.

#### M4. The vision demo

`sensoryforge/presets/vision_onoff_rgb.yml`: one grid with channels `["R", "G", "B"]`, three
stimuli each driving one channel, one population reading R and G through `OnOffLayer` and combining
with `sum`, and one reading all three with `concat`. `examples/vision_rgb_onoff.py` runs it and
writes a bundle, mirroring the pressure-simulation recipe script from Wave K5. This is the concrete
proof that the simulator is no longer tactile-only, and it is the figure for the paper's
generality claim.

#### M5. Documentation

- `docs/concepts/populations_and_inputs.md` — one neuron, several inputs; sum versus concat; where
  processing sits in the spine.
- `docs/extending/add_processing_layer.md` — `OnOffLayer` as the worked, CI-executed example.
- `docs/user_guide/configuration_schema.md` regenerated for the new fields.

#### Wave M exit

`Closes: F-010` (with Wave L and Wave N, the three halves of it are then all done). Both suites
green under the watchdog, lint and strict docs clean, every existing config round-tripping
byte-identically, and all golden parity tests unchanged.

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

## 5b. Verified interoperability with pressure-simulation (2026-09-16)

After Waves J and N were merged, the bundle contract was checked against pressure-simulation
itself, not against a re-implementation of it. The procedure, worth repeating whenever the
bundle format changes:

1. `sensoryforge run examples/canonical_config.yml --duration 50 --bundle <dir>` writes a bundle.
2. Transcribe nothing: import `encoding.grid_torch.GridManager`, `encoding.encode_runner.PopConfig`
   and `run_encoding` from `~/Documents/pressure simulation`, and run the exact sequence
   `GUIs/ebkf_viewer.py::_on_load_bundle` performs, then the `population_configs` half of `_on_run`.
3. Drive it with a ramped Gaussian and an input gain large enough to spike (70 for SA, 700 for RA,
   per that repository's own calibration).

Result: a 40x40 grid and two populations loaded, one stimulus file and one neuron-module file found
so its Run button would be enabled, both populations matched by name, and its encoder produced
339 spikes across 48 SA neurons and 1,983 across 155 RA neurons from our receptive fields.

Two failures this caught that the in-repo tests did not, both fixed in Wave J: the bundle wrote no
`neuron_modules/` directory, so the viewer could load a bundle but its Run button never enabled
(F-054); and `stimuli/stimulus.json` was an untagged dict that its stimulus generator would have
read as a default static blob without raising (F-055).

---

## 6. Phase 2 exit criteria

- Waves I to N complete with their exit checks.
- A canonical config with two channels, a `template` RF input and an `imported` RF input combined into one
  population, an analog DSL readout, and a spiking population runs from a wheel installed outside the repo
  and writes a bundle that `load_bundle` reads and pressure-simulation's viewer logic opens.
- Golden parity with pressure-simulation passes; all suites, black, flake8 and `mkdocs build --strict` pass.
- Ledger: F-010, F-011, F-013, F-050, F-051, F-052 closed.
- `sensoryforge list-components` lists every built-in component, and a stimulus registered by a
  plugin package runs from a config file (the F-052 proof).
