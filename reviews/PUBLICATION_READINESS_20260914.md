# SensoryForge — Publication-Readiness Audit (2026-09-14)

**Scope.** Release SensoryForge as the open-source simulator (Paper A, JOSS-style tool paper) that
drives the encoder used by the pressure-simulation project (Paper B, "sensory communication and
estimation" / SGA-KF). Requirement: **the engine and the conceptual framing must be the same in
both repos.** This document maps what works, what is missing, and how to remedy it. Findings are
tracked in the living ledger (`docs_root/LEDGER.md`, IDs `F-001…F-022`); this file is the narrative.

Evidence base: three read-only code sweeps (code state, engine parity vs. pressure-simulation,
packaging/docs), plus a test run in the `sensoryforge` conda env (py3.11, torch 2.5.1, PyQt5 5.15.11).

---

## 0. Verdict in one paragraph

The *encoding core* (grid → Gaussian innervation → SA/RA filters → Izhikevich/AdEx/MQIF/DSL →
spikes) is solid, vectorised, tested (560 non-GUI unit tests pass) and already shares its lineage
with pressure-simulation (SensoryForge was extracted from it in Feb 2026, commit `2b2c44e`). What
blocks a joint release is (a) **five scientific divergences** that crept in after the fork, so today
the two repos would *not* produce the same spikes for the same config; (b) the **installed package
does not run** (missing `package_data`, wrong `python_requires`, cwd-relative default config);
(c) **no CI / no citation / no changelog**; (d) a `SimulationEngine` that is the declared
canonical path yet silently mis-handles non-grid arrangements and cannot instantiate DSL neurons;
and (e) **one critical bug**: the canonical→legacy adapter squares the grid size, so the README's
own quick-start example and two integration tests allocate tens of gigabytes and get OOM-killed
(F-012). None of these is large individually. Ordered remediation is in §6.

---

## 1. Engine parity with pressure-simulation (must be identical)

Legend: A = SensoryForge, B = pressure-simulation (`~/Documents/pressure simulation`).

| Component | Status | Detail |
|---|---|---|
| Stimulus tensor & units | IDENTICAL | `[batch, T, H, W]`, mm, ms, mA. A's `filters/base.py:52-56` says `dt` in **seconds** while `sa_ra.py` uses **ms** — doc bug only. |
| Grid geometry | IDENTICAL code | 80×80 @ 0.15 mm, σ_SA 0.3 / σ_RA 0.39, K=28. B's *locked design* is a declared resolvable distance `d = 0.40 mm ⇒ σ_RF = d/π ≈ 0.127 mm` (blueprint §109) which B's own YAML does not yet apply ("document now, re-run later", 2026-08-27). A has no notion of `d` or σ/pitch. |
| Innervation builder | **DIVERGED (F-003)** | Both ship the same stochastic builder: Gaussian *selection* probability, Poisson K, **uniform-random weights independent of distance**. B has flagged this as the *control arm* (F-006 there) and specified a deterministic K-nearest analytic-weight builder (not built). A's `use_distance_weights=True` is the nearest thing and is **off by default** (`schema.py:147`). A's docstring `innervation.py:880` still says "Gaussian falloff". |
| SA filter | **DIVERGED (F-001)** | Equations and params identical (τ_r 5, τ_d 30, k1 0.05, k2 3). **A clamps I_SA ≥ 0** (`sa_ra.py:53,160`, `clip_to_positive=True`, added 2026-04-10 against subthreshold oscillations). B does not, and B's decoder recovers the *sign* of velocity from the SA channel. |
| RA filter | **DIVERGED (F-002)** | Equation identical. τ_RA = **8 ms** in B (Kandel Ch.21, commit `0ee0653`, Apr 2026); A: 30 ms class default, 15 ms in `default_config.yml:94`, 30 in `CombinedSARAFilter`. A also deleted B's physiological-basis docstring. Note: in *both* repos `core/pipeline.py:145` (A) / `pipeline_torch.py:144` (B) call `CombinedSARAFilter()` with no args, so the YAML `filters:` block is dead on that path. |
| Neuron models | IDENTICAL params; **DIVERGED (F-004)** in assignment | Izhikevich/AdEx/MQIF defaults identical. A adds `v_floor` clamps (stability). B assigns **Fast-Spiking** Izhikevich (a=0.1, d=2) to RA and RS (a=0.02, d=8) to SA; A has no FS/RS distinction. |
| input_gain | DIVERGED in rationale | A: default 50 with a "N/mm² vs mA" story. B: gains 40–200 as free per-script knobs, no theory. |
| Filter attribution | **CONFLICT (F-005)** | A's `CLAUDE.md` and `docs/user_guide/units_and_gains.md` cite "Pierzowski (1995)" (no DOI, not locatable). A's own code (`sa_ra.py`, `mechanoreceptors.py`) and all of B cite **Parvizi-Fard et al. 2021**. |
| Noise | DIVERGED (F-007) | A's forked `core/pipeline.py:239` applies receptor + membrane noise **before** the filter; A's `SimulationEngine:393-404` applies one post-gain `randn`. A fixed a real bug B still has (per-instance `torch.Generator` instead of global reseed). |
| dt / sub-stepping | DIVERGED (F-008) | B sub-steps Izhikevich at 0.05 ms inside 1 ms bins (`encode_runner.py:183-190`). A feeds `SimulationConfig.dt` (default **1.0 ms**, `schema.py:326`) straight to the neuron; only a GUI warning guards it. |
| Seeds | DIVERGED (F-006) | A's batched `torch.multinomial` consumes RNG differently from B's per-neuron loop ⇒ **same seed, different wiring**. A's innervation still reseeds the *global* RNG (`innervation.py:897`). |
| Spike format | IDENTICAL | `(v_trace, spikes)`, bool, `[batch, T+1, N]`. |

**Remedy (the "same engine" contract).** Decide each of F-001…F-004 *once*, in the ledger, and apply
the decision to both repos in the same week:
1. **SA sign.** Either make `clip_to_positive` default `False` in A (and fix the oscillation
   another way — the `v_floor` + dt=0.1 ms already cover it) or adopt rectification in B and
   update B's sign-recovery story. Recommendation: **default False in A**; B's decoder depends on
   signed SA.
2. **τ_RA = 8 ms** everywhere (class default, YAML, `CombinedSARAFilter`), restore the Kandel
   docstring. Fix the dead `CombinedSARAFilter()` call so YAML filters are honoured.
3. **Innervation.** Make `use_distance_weights=True` the default (analytic Gaussian weights),
   fix the docstring, and add the deterministic K-nearest builder as `innervation_method:
   "deterministic"` in A first (B imports it — B's blueprint wants it and has not built it).
   Keep stochastic as `"stochastic"` control arm, per B's RETIRED_FRAMINGS.
4. **FS/RS presets.** Add named Izhikevich presets (`RS`, `FS`, `IB`, …) to the neuron model
   params and set RA populations to FS by default in the GUI/schema.
5. **Citation.** Replace "Pierzowski (1995)" with Parvizi-Fard et al. 2021 (+ Kandel Ch.21 for
   τ values) in CLAUDE.md, docs, refs/. Put the citation inline in `sa_ra.py`.
6. **Parity test.** Add `tests/integration/test_pressure_sim_parity.py` that loads a fixed config
   + seed and compares A's spikes to a golden `.npz` exported from B (tolerance 0). Sub-stepping
   (F-008) and RNG order (F-006) must be aligned for this to pass — do it as part of the same
   change.

## 2. Conceptual framing (must be consistent)

B's authoritative chain: `PROJECT_SUMMARY.md` → `SCIENTIFIC_HYPOTHESIS.md` (June 2026 revision:
*engineering study, not biology*; SA1+RA1 scope; reconstruction is validation, **MI is the design
score**) → `ARCHITECTURE_BLUEPRINT.md` (data-led design, one declared `d`) → `RETIRED_FRAMINGS.md`
(20 retired ideas) → `PAPER_A_SIMULATOR.md` (this release, scoped: encoder + GUI only, JOSS-style,
timeboxed).

A embodies: dual pathway, labeled-line fixed wiring, exact geometry, "GUI is visualization only",
modality-agnostic extensibility. A **contradicts** B in one file: `docs_root/SCIENTIFIC_HYPOTHESIS.md`
is B's Oct-2025 draft, headlining the *retired* "SA/FA sufficient to reconstruct" hypothesis and the
retired four-population plan (**F-009**). It is gitignored, but CLAUDE.md skills and the Cursor rules
still name it as the grounding document.

**Remedy.** Replace A's `docs_root/SCIENTIFIC_HYPOTHESIS.md` with a short *forward-model scope*
note that points at B's `PAPER_A_SIMULATOR.md` framing: SensoryForge is the configurable,
reproducible forward model (sensor grid → sparse RF kernels → SA/RA filtering → spiking); design
scoring (MI), decoding (SGA-KF) and mismatch studies are out of scope and live in Paper B. Add the
same two-sentence scope statement to `README.md` and `docs/index.md`. Import B's symbol glossary
(`NAMING.md` § six σ's) into `docs/user_guide/units_and_gains.md`.

## 3. Code state: working / partial / broken

**Working (verified by running).** Registry system (7 registries, idempotent); `SimulationEngine.run()`
end-to-end; `_run_pop_from_drive` genuinely shared by GUI and engine; CLI `run`/`batch`/`validate`;
`BatchExecutor` sweeps, seeds, checkpoint/resume, `.pt` + HDF5 writers; `ExperimentManager`;
5-tab GUI wiring; 560/560 non-GUI unit tests green; 912 tests collect with zero import errors.

**Partial.**
- `SimulationEngine` (**F-010**): composite grids `NotImplementedError` (`:98`); `poisson/hex/
  jittered/blue_noise` arrangements are built then *ignored* — innervation uses the regular
  `GridManager` (`:107-125`, `:224-243`) ⇒ silently wrong science; DSL neurons cannot be
  instantiated (`"DSL (Custom)"` lower-cased → unknown; `"dsl"` → `NeuronModel(dt=…)` TypeError,
  `.compile()` never called, `dsl_config` never read); `_stimulus_to_receptors` is a passthrough
  (`:425-448`) so stimulus must equal `rows×cols`.
- **CRITICAL (F-012) — canonical→legacy adapter allocates `(rows·cols)²` receptors.**
  `generalized_pipeline.py:351` sets `grid_size = rows * cols`; `grid.py:32` treats an int
  `grid_size` as a *per-side* count. A 20×20 canonical config therefore builds a 400×400 lattice
  (160 000 receptors), a 40×40 one 2.56 M receptors, and the README "Basic Example" (80×80 through
  `GeneralizedTactileEncodingPipeline.from_config`) asks for **41 M receptors**. Measured: two
  integration tests (`test_gui_cli_parity::test_pipeline_accepts_canonical_config`,
  `test_regression_refactoring::test_canonical_config_loads_via_adapter`) exceed 5 GB RSS and are
  OOM-killed — this is what took down the interactive session during this audit. CLI/Batch hit the
  same adapter on every canonical run because they still build the legacy pipeline for stimulus
  generation (`batch_executor.py:98`, `cli.py:218`), papered over by bilinear resize
  (`batch_executor.py:417-425`). Fix: `grid_size = (rows, cols)` (the function already accepts a
  tuple), then delete the resize workaround.
- Batch output (**F-013**): no neuron/receptor coordinates or `dt` on the canonical path; `.pt` is
  one monolithic pickle; `spikes` is `T+1` while `drive`/`filtered` are `T`, undocumented; HDF5 drops
  list-valued stimulus params. Good for supervised (spikes → stimulus-params) datasets; weak for any
  spatial or information-theoretic analysis until coordinates are written.
- `cli validate` forces the legacy pipeline for canonical configs; `cli visualize --save` prints
  "not yet implemented"; `list-components` is a hardcoded, already-wrong print block (**F-018**).
- GUI: `neuron_modules/` is written by the spiking tab but never created by `ExperimentManager`;
  `ProjectRegistry(Path.cwd()/…)` scatters state into whatever cwd the GUI was launched from.

**Broken.**
- **SLURM export is dead (F-011):** `generate_slurm_script` emits `sensoryforge run … --stimulus-index
  --format hdf5`; neither flag exists (`cli.py:556-578`) and `run` writes `.pt` only.
- `BatchTab` progress bar never advances (signal declared, never emitted); stop uses
  `QThread.terminate()`.
- ~3500 lines of unwired GUI code: `protocol_suite_tab.py`, `protocol_backend.py`,
  `protocol_execution_controller.py`, `neuron_explorer.py` (**F-019**).

## 4. Tests

| Slice | Result |
|---|---|
| `tests/unit` minus 10 Qt files | **560 passed, 5 skipped** (h5py, torchdiffeq absent) |
| 10 Qt unit files, one session | 25 failed, 61 errors — all `AttributeError: module 'PyQt5.QtGui' has no attribute 'QColor'` |
| Same Qt files, one at a time | pass, then **segfault at interpreter exit** (`test_grid_population_ux` clean) |
| `tests/integration`, per file under a 4 GB watchdog | `engine_parity` 21 ✓ · `gui_phase2` 7 ✓ · `pipeline` 2 ✓ · `registry_integration` 8 ✓ · `simulation_engine` 23 ✓ 1 ✗ · `yaml_pipeline` 20 ✓ 1 ✗ · `phase3_pipeline` 14 ✓ 4 ✗ · **`gui_cli_parity` and `regression_refactoring` OOM-killed (>5 GB, F-012)** |

The six non-OOM integration failures: five are step-count assertions (`assert 1000 == 200`,
`assert 500 == 100`) — tests written for dt = 0.5 ms before D-005 lowered the default to 0.1 ms, or
the legacy `neurons.dt` key is no longer honoured for the stimulus time axis (check which; the
units audit's F5 "filter dt not propagated" is the same family); one is
`test_invalid_innervation_method_raises_error` — the engine no longer validates the method name.

Root causes: (**F-016**) `tests/unit/test_stimulus_tab_gui.py:86-89` installs `MagicMock` objects
into `sys.modules["PyQt5*"]` at import time and never restores them, so every Qt module collected
afterwards imports mocks. (**F-017**) `test_gui_cli_parity.py:70` passes a *file path* to
`SensoryForgeConfig.from_yaml`, which takes YAML *text*; `simulation_engine.py:16` docstring shows
the same wrong call. The segfault is the known PyQt5 + pytest teardown issue (already noted as
"pytest exit 139" in `docs_root/PROJECT_STATUS_REPORT.md`).

**Remedy.** (1) Use `monkeypatch.setitem(sys.modules, …)` in a fixture, or move the mocked tests to
their own `tests/unit/gui_mocked/` collected with `-p no:cacheprovider --forked`. (2) Add
`from_yaml_file()` / accept `Path` in `from_yaml`, fix the docstring. (3) Add `pytest-xdist`/
`pytest-forked` and run Qt tests forked, or end the session with `os._exit` in a `pytest_sessionfinish`
hook. (4) Add `pytest.ini` with `testpaths`, `filterwarnings`, and `-p no:cacheprovider`.

## 5. Packaging, docs, hygiene (release scaffolding)

- **F-014 packaging:** `python_requires=">=3.8"` but `neurons/sa.py:49` uses `X | None` (3.10+);
  no `package_data`/`MANIFEST.in` ⇒ `gui/default_params.json` and `config/default_config.yml` are
  not installed and the GUI crashes on a wheel install; `core/pipeline.py:86,392,422` open the
  default config by a **cwd-relative** string; `h5py` undeclared; `docs/getting_started/
  installation.md:34` advertises a `[full]` extra that does not exist; PyQt5 + pyqtgraph are hard
  deps (should be a `gui` extra); no classifiers / long_description / project_urls.
- **F-015 release files & hygiene:** no `pyproject.toml`, `pytest.ini`, lint config, `.github/
  workflows/`, `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`;
  `test_refactoring.py` at repo root; `devo_reports/` (raw dictated notes) tracked; `.github/
  copilot-instructions.md` tracked despite being gitignored; three author strings (`setup.py`
  "Sensory Forge Contributors" / LICENSE "benefron" / README "Efron, Ben"); dead PyPI link
  `README.md:387`. Clean of secrets, local paths and binaries.
- **F-020 docs:** entire `docs/developer_guide/` plus `units_and_gains.md`, `gui_walkthrough.md`,
  `configuration_schema.md` are absent from `mkdocs.yml` nav; 8 broken intra-doc links;
  "pip install sensoryforge" in 3 pages; `sensoryforge/config/README.md` describes 4 files that do
  not exist; `docs/api_reference/` is a `.gitkeep`.
- **F-021 stale debt lists:** `CLAUDE.md` still lists DSL-numpy-only (C-2) and `reset_states` (M-1)
  as open — both resolved (ledger R-001, D-011). `reviews/CODE_REVIEW_20260408.md` tracker says
  37/37 open; several are fixed. Decide whether `reviews/` ships publicly at all.
- **F-022 scientific validation:** one analytic test (`test_filters_vs_theory.py`); no comparison
  to TouchSim / Saal et al. 2017 or to B; the only notebook has no outputs; no benchmark suite.

## 6. Remediation plan (ordered; each step = one commit with ledger trailers)

**Tier 0 — decide (ledger `Decision:` trailers, no code):** SA sign (F-001), τ_RA (F-002),
innervation default (F-003), FS/RS (F-004), citation (F-005), whether `reviews/` and `devo_reports/`
ship, whether the decoder is *ever* in scope for A (B's `PAPER_A_SIMULATOR.md` leans no).

**Tier 1a — stop the OOM (1 hour, do first).** `generalized_pipeline.py:351` → `grid_size =
(rows, cols)`; remove the bilinear-resize workaround; add a regression test asserting receptor count
== rows·cols for a canonical config through the adapter; run the two OOM-killed integration files
under a memory cap in CI. Closes F-012 (adapter half).

**Tier 1 — make the installed package run (½ day).** `pyproject.toml` (PEP 621, setuptools),
`python_requires>=3.10`, `package_data`, `importlib.resources` for the two data files, `gui`/`hdf5`
extras, h5py declared, remove `[full]` mention, one author string. Closes F-014.

**Tier 2 — CI + test hygiene (½ day).** `pytest.ini`; fix F-016/F-017; `.github/workflows/tests.yml`
(ubuntu + macOS, py3.10/3.11, non-GUI suite + forked Qt job with `QT_QPA_PLATFORM=offscreen`);
`black --check`, `flake8` with matching line length. Closes F-015 (CI part), F-016, F-017.

**Tier 3 — engine parity (2–3 days, the scientific core).** Apply Tier-0 decisions to filters,
innervation, presets; align sub-stepping and RNG order; export golden spikes from B; add the parity
test. Fix the dead `CombinedSARAFilter()` call. Closes F-001…F-008.

**Tier 4 — SimulationEngine correctness (2 days).** Route non-grid arrangements' coordinates into
innervation; instantiate DSL neurons (`NeuronModel.from_config` + `compile`); implement
`_stimulus_to_receptors` (bilinear sample at receptor coords); either implement composite grids or
raise a clear `ValueError` at config-validation time and say so in docs; give `SimulationEngine` its
own stimulus module so CLI/Batch stop building the legacy pipeline. Closes F-010, F-012.

**Tier 5 — batch for learning/IBS (1 day).** Write `grid/receptor_xy`, `populations/<name>/
neuron_xy`, `innervation_weights` (sparse), `dt_ms`, and `time_axis` into HDF5; make HDF5 the default
for `batch`; fix SLURM flags (`--stimulus-index`, `--format`) or generate `sensoryforge batch` array
tasks; emit progress. Closes F-011, F-013.

**Tier 6 — docs & framing (1 day).** Rewrite `docs_root/SCIENTIFIC_HYPOTHESIS.md` as the forward-model
scope note; fix nav/links/install pages; `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`; refresh
CLAUDE.md debt list; regenerate `list-components` from the registries; decide `reviews/`. Closes
F-009, F-018, F-020, F-021, F-015 (files part).

**Tier 7 — validation & extension (paper time).** Executed notebook; benchmark table (grid size vs
wall-clock, CPU/MPS/CUDA); comparison figure vs TouchSim for a probe indentation; delete or
resurrect the 3500 unwired GUI lines. Closes F-019, F-022.

## 7. Extensions and conceptual gaps (beyond the release)

- **Deterministic RF builder + σ/pitch lock** (from B's blueprint): expose `resolvable_distance_mm`
  as the single design knob that derives σ, pitch, N. This is the concept B's Paper B will cite A for.
- **Receptor-space sampling**: `_stimulus_to_receptors` done properly makes stimulus resolution
  independent of receptor pitch — needed for the composite/hex/Poisson grids to mean anything.
- **Batch datasets for learning / information-bottleneck work**: sparse event export
  (`[t, neuron]` pairs), per-population coordinates, and a `torch.utils.data.Dataset` reader.
- **Per-population `dt` sub-stepping** and `input_gain`/`noise_std` per population (already in
  schema; make the engine honour them everywhere).
- **Presets library**: Kandel-grounded SA1/RA1 presets (RS/FS, τ's, σ's) as named YAML fragments,
  so "the paper config" is one line.
- **Modality claim**: README promises vision/audition; there is no example. Either ship one
  (DVS-like ON/OFF filter on an image sequence is ~50 lines with the existing RA filter) or soften
  the claim to "modality-agnostic architecture, tactile demonstrated".
- **Decoder boundary**: keep decoding out of A (B's lean). Provide instead a documented, stable
  export contract (HDF5 layout above) that B's decoder consumes — that is the interface between the
  two papers.
