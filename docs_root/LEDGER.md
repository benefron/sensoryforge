# Project Ledger — decisions, findings, retired framings

**This is the single register.** If you want to know what was decided, what is open, or what has
already been settled and must not be re-litigated, it is here. Newest first.

**Read this before proposing anything that sounds new.** A large fraction of the entries below are
questions that were already asked and answered — sometimes answered wrongly first, then corrected.
Re-opening one costs more than reading it.

This file is an instance of the **Lore** pattern (git commit messages as a structured knowledge
protocol for AI coding agents — arXiv:2603.15566). The commit trailer is the atomic unit of
institutional knowledge; this file is the pre-digested read surface over those trailers.

---

## How to use this file

**Finding things.** Grep it. Every entry is one greppable block; the terms you would naturally
search for appear in the entry bodies deliberately.

```bash
grep -A4 '^## F-0'      docs_root/LEDGER.md   # all findings
grep -A4 '· OPEN ·'     docs_root/LEDGER.md   # everything still open
grep -A4 '· retired ·'  docs_root/LEDGER.md   # do not re-propose these
grep -B1 -A4 '<topic>'  docs_root/LEDGER.md   # everything about one topic
```

**Entry format.** The header line is machine-parsed — keep it exact.

```
## <ID> · <STATUS> · <type> · <area|-> · <date>
<one or two lines of what and why>
→ <file:line evidence> · <pointer or closes-with>
```

| Field | Values |
|---|---|
| `ID` | `D-NNN` decision · `F-NNN` finding · `R-NNN` retired framing · `N-NNN` note/thought |
| `STATUS` | `OPEN` · `CLOSED` · `SUPERSEDED` · `STANDING` (retired framings are `STANDING`) |
| `type` | `decision` · `finding` · `retired` · `note` · `thought` |
| `area` | a short workstream / component tag, or `-` if it belongs to none |
| `date` | `YYYY-MM-DD`, optionally suffixed `(recorded)`, `(from <sha>)` or `(backfilled)` |

**Date honesty.** A date is only ever one of:
- **bare** — the entry was written on that date, live.
- **`(recorded)`** — transcribed from a doc that already carried that date.
- **`(from <sha>)`** — recovered via `git log -S`, i.e. the commit that introduced the claim.
- **`(backfilled)`** — reconstructed with no better evidence; treat the date as approximate.

Never write an undecorated date onto a backfilled entry. Inventing a tidy history is the exact
failure this file exists to prevent.

**Nothing is ever deleted.** Status changes; git holds the history.

---

## How entries get here

Three capture paths, all cheap:

1. **Commit trailers.** Claude writes them on every commit; `.claude/hooks/ledger-sync.sh` reads
   `git log` and appends any it has not seen. Trailer vocabulary:
   ```
   Decision: <one line>
   Finding:  <one line>
   Opens:    F-014 <one line>
   Closes:   F-007
   Retires:  <the framing being retired>
   ```
   Optional Lore trailers (`Rejected:`, `Directive:`, `Constraint:`) are recorded verbatim into the
   entry body when present — they map onto retired framings and path-scoped rules.
2. **Checkpoint proposals.** At the end of a work chunk Claude says *"recording these: …"*; you
   approve, edit or decline.
3. **"note that …"** in chat → appended immediately as a `note` or `thought`. A thought is typed as
   a thought so it is visibly not a decision.

A digest of `OPEN` items and recent decisions is injected automatically at session start and after
compaction by `.claude/hooks/digest.sh` — so a cold session already knows this, without searching.

The sync bookmark (which commit the ledger is caught up to) is **machine-local** and lives in
`.claude/.ledger-sync` (gitignored) — advancing it never dirties this file. This file changes only
when there are real new entries.

---

# Entries

<!-- newest first; ledger-sync.sh inserts directly below this marker -->
<!-- ENTRIES_START -->

## D-019 · CLOSED · decision · - · 2026-09-14
default innervation uses analytic Gaussian weights; the stochastic uniform-weight builder is the named control arm
→ commit b1f67a3

## F-036 · OPEN · finding · - · 2026-09-14
flake8 style debt after black (all default checks, 88 columns): 364 violations, mainly E501 164, F401 141, F541 18, E402 14, F841 11; CI gates only E9,F63,F7,F82 until ratcheted
→ commit 7da39f9

## F-037 · OPEN · finding · - · 2026-09-14
SensoryForge Izhikevich/AdEx/MQIF clamp voltage at v_floor (-120/-130/-120 mV, D-007) but pressure-simulation's neurons do not, so spikes can differ for strongly negative drive, which unrectified SA (F-001) now makes reachable
→ commit 7da39f9

## F-035 · OPEN · finding · - · 2026-09-14
With Python's cyclic GC enabled, pytest -m gui segfaults (3 of 3 runs) inside pyqtgraph ScatterPlotItem.renderSymbol, called from MechanoreceptorTab._add_receptor_scatter_by_weight <- _update_innervation_graphics <- _create_population_graphics <- _regenerate_selected_population_if_instantiated, via a ViewBox lambda from a previously destroyed tab. tests/conftest.py disables GC for every session (including non-GUI) to avoid it, so the harness can no longer detect this crash class; app-level impact unproven.
→ commit fb1441b

## F-033 · CLOSED · finding · - · 2026-09-14
pyproject.toml uses project.license as a TOML table plus the "License :: OSI Approved :: MIT License" classifier; setuptools 81 warns both are deprecated and builds stop being supported after 2027-02-18. Use license = "MIT", license-files = ["LICENSE"], setuptools>=77.
→ commit 5508313

## F-034 · CLOSED · finding · - · 2026-09-14
pressure-simulation encoding/encode_runner.py still defaults RA k3 to 1.0 (commented as calibrated for fast-spiking RA at input gain 40), contradicting D-018 "k3 = 2.0 everywhere"; the golden parity test (E5) cannot pass at zero tolerance until one value is used in both repos. Needs user confirmation of scope.
→ commit 5508313

## D-018 · CLOSED · decision · - · 2026-09-14
RA filter gain k3 = 2.0 everywhere (D-Q1).
→ commit e518611

## F-031 · CLOSED · finding · - · 2026-09-14
resolve_neuron_params (config/defaults.py) expands the neuron-type preset only when no a/b/c/d override is present: RA population with model_params {d: 4.0} builds a=0.1 in the GUI but a=0.02 in SimulationEngine, and GeneralizedTactileEncodingPipeline raises KeyError 'a' (generalized_pipeline.py:432,471), breaking CLI/batch runs of GUI-exported configs with one tweaked neuron parameter. test_config_defaults.py::test_explicit_d_override_suppresses_preset asserts the wrong semantics. F-026 and F-004 were closed with this present.
→ commit 6bab8cc

## F-032 · CLOSED · finding · - · 2026-09-14
RS defaults for RA remain outside the resolver: TactileEncodingPipelineTorch builds RA neurons with the RS class default (core/pipeline.py:161-162), GeneralizedTactileEncodingPipeline DEFAULT_CONFIG has RS ra_a/ra_d for hand-written legacy configs (:160-163), and CombinedSARAFilter keeps its own default dict (filters/sa_ra.py:455-456) instead of FILTER_DEFAULTS.
→ commit 6bab8cc

## D-017 · CLOSED · decision · - · 2026-09-14
SensoryForge is the general clean-slate sensory-encoding simulator (sensor channels -> receptive fields -> sensory neurons -> spiking or analog readout -> batch data); pressure-simulation is a use case that supplies the recipe (d, ensemble, MI scoring) and consumes the generated bundle
→ commit b28acda

## F-025 · CLOSED · finding · - · 2026-09-14
The canonical->legacy adapter still squares neuron counts: it writes neuron_rows*neuron_cols into neurons.sa/ra/sa2_neurons, which InnervationModule treats as per-row (generalized_pipeline.py:404,447,480 -> :640-707). A 4-per-row canonical population builds 256 neurons in the legacy pipeline vs 16 in SimulationEngine; the README 80x80 quick-start still exceeds 3 GB. F-012 fixed only the receptor-grid half.
→ commit b28acda

## F-026 · CLOSED · finding · - · 2026-09-14
Filter and neuron defaults live in several places and disagree, so GUI and CLI run different models for one config: SpikingNeuronTab uses gui/default_params.json (RA tau_RA 30, k3 100, RS a/b/c/d for all) and exports only overrides; SimulationEngine uses class defaults (tau_RA 8, k3 2.0, FS for RA). 32 vs 4 spikes on the same drive. tau_RA still 30 in generalized_pipeline.py:132,472, default_params.json:80, neuron_explorer.py:113; 15 in examples/*.yml, tests/fixtures/phase2_config.yml, README.md:128, two docs pages. F-002 was closed prematurely.
→ commit b28acda

## F-027 · CLOSED · finding · - · 2026-09-14
SensoryForgeConfig.from_yaml raises OSError (ENAMETOOLONG) on a one-line YAML/JSON string longer than 255 bytes because it probes Path.is_file() before parsing (schema.py:427).
→ commit b28acda

## F-028 · CLOSED · finding · - · 2026-09-14
Phase 0/1a behaviour is untested (Izhikevich presets and override precedence, SimulationEngine RA->FS default, from_yaml path branch and from_yaml_file, core/pipeline.py reading filters.sa/ra) and preset was inserted positionally between d and v_init, breaking positional IzhikevichNeuronTorch callers.
→ commit b28acda

## F-029 · CLOSED · finding · - · 2026-09-14
Moving reviews/ under docs/ added 9 mkdocs warnings (relative code links in REVIEW_AGENT_FINDINGS_20260211.md); mkdocs build --strict would fail.
→ commit b28acda

## F-030 · CLOSED · finding · - · 2026-09-14
RA filter gain k3 is unresolved across both repos: SensoryForge GUI 100, SensoryForge engine 2.0, pressure-simulation class/config/decoder gain 2.0 but its encode_runner uses 1.0 and notes k3=100 saturates fast-spiking RA near 1000 Hz. Needs a user decision (D-Q1 in the handover).
→ commit b28acda

## D-013 · CLOSED · decision · - · 2026-09-14
filter-calibration citation is Parvizi-Fard et al. (2021, J. Neurophysiol.) + Kandel Ch.21 for tau_RA, not "Pierzowski (1995)"
→ commit 5285378

## D-014 · CLOSED · decision · - · 2026-09-14
SAFilterTorch defaults to clip_to_positive=False (sign-preserving SA, matching pressure-simulation's decoder which recovers velocity sign from SA)
→ commit be64ab2

## D-015 · CLOSED · decision · - · 2026-09-14
RAFilterTorch.tau_RA is locked to 8 ms everywhere (Kandel Ch.21), matching pressure-simulation
→ commit be64ab2

## D-016 · CLOSED · decision · - · 2026-09-14
Izhikevich neuron presets (RS/FS/IB/CH/LTS, Izhikevich 2003) are available via preset=; RA populations default to FS in SimulationEngine (next commit)
→ commit be64ab2

## F-023 · CLOSED · finding · - · 2026-09-14
Legacy config keys neurons.sa_neurons/ra_neurons mean neurons-per-row, not a total count -- InnervationModule squares it, so a value that reads as a total (e.g. 100) silently builds a 100x100=10,000-neuron population with a dense [N,H,W] weight tensor. Confirmed via test_gui_cli_parity.py::TestConfigAdapter::test_legacy_config_still_works, which passed sa_neurons=100/ra_neurons=196 (looking like the canonical example's totals) and hit >2.7GB RSS before the test was corrected to use per-row values.
→ commit 1c93fa6

## F-024 · OPEN · finding · - · 2026-09-14
GeneralizedTactileEncodingPipeline's gaussian/step/ramp stimulus generators read config["temporal"]["dt"], a separate key from config["neurons"]["dt"] which controls neuron integration. Setting only neurons.dt (the intuitive choice) silently leaves the stimulus time axis at the temporal.dt default (0.1ms), desynchronising the two. Found via test_regression_refactoring.py::test_pipeline_forward_pass_still_works.
→ commit 1c93fa6

## D-012 · CLOSED · decision · - · 2026-09-14
SensoryForge tracks decisions/findings in docs_root/LEDGER.md via commit trailers (living-ledger, same system as pressure-simulation); docs_root/ stays gitignored except the ledger
· Directive: Before editing sa_ra.py, innervation.py, default_config.yml, schema.py or izhikevich.py read .claude/rules/engine-parity.md and mirror any scientific change in pressure-simulation
→ commit 7931c18

## F-001 · CLOSED · finding · - · 2026-09-14
SAFilterTorch rectifies I_SA (clip_to_positive=True, sa_ra.py:53,160); pressure-simulation does not and its decoder recovers velocity sign from SA. Decide once, apply to both repos
→ commit 7a188b6

## F-002 · CLOSED · finding · - · 2026-09-14
tau_RA is 30 ms (sa_ra.py:258) / 15 ms (default_config.yml:94) / 30 (CombinedSARAFilter) here; pressure-simulation locked 8 ms (Kandel Ch.21, its 0ee0653). CombinedSARAFilter() is called with no args in core/pipeline.py:145 so YAML filters are dead on that path
→ commit 7a188b6

## F-003 · CLOSED · finding · - · 2026-09-14
Default innervation is the stochastic builder (uniform-random weights; Gaussian only in selection probability, innervation.py:952) that pressure-simulation retired to a control arm; docstring innervation.py:880 says "Gaussian falloff"; use_distance_weights defaults False (schema.py:147); no deterministic K-nearest builder
→ commit 7a188b6

## F-004 · CLOSED · finding · - · 2026-09-14
No FS/RS Izhikevich split: RA populations get RS (a=0.02,d=8); pressure-simulation assigns FS (a=0.1,d=2) to RA since Apr 2026
→ commit 7a188b6

## F-005 · CLOSED · finding · - · 2026-09-14
Filter calibration attributed to "Pierzowski (1995)" in CLAUDE.md, docs/user_guide/units_and_gains.md, refs/ (unverifiable) but to Parvizi-Fard 2021 in sa_ra.py and throughout pressure-simulation
→ commit 7a188b6

## F-006 · CLOSED · finding · - · 2026-09-14
Same seed gives different wiring across repos: batched torch.multinomial (innervation.py:949) vs per-neuron loop in pressure-simulation; innervation still reseeds the global RNG (innervation.py:897)
→ commit 7a188b6

## F-007 · CLOSED · finding · - · 2026-09-14
Two noise topologies: core/pipeline.py:239 applies receptor+membrane noise before the filter; SimulationEngine:393-404 applies one post-gain randn and no receptor noise
→ commit 7a188b6

## F-008 · CLOSED · finding · - · 2026-09-14
No sub-stepping: SimulationConfig.dt=1.0 ms (schema.py:326) is fed straight to the neuron (simulation_engine.py:262); pressure-simulation sub-steps Izhikevich at 0.05 ms inside 1 ms bins
→ commit 7a188b6

## F-009 · CLOSED · finding · - · 2026-09-14
docs_root/SCIENTIFIC_HYPOTHESIS.md is pressure-simulation's Oct-2025 draft: headlines the retired "SA/FA sufficient to reconstruct" hypothesis and the retired 4-population plan; CLAUDE.md/Cursor skills still name it as grounding
→ commit 7a188b6

## F-010 · OPEN · finding · - · 2026-09-14
SimulationEngine: composite grids NotImplementedError (:98); poisson/hex/jittered/blue_noise arrangements built then ignored, innervation uses the regular GridManager (:107-125,:224-243); DSL neurons cannot be instantiated (:260-264, dsl_config never read); _stimulus_to_receptors is a passthrough (:425-448)
→ commit 7a188b6

## F-011 · OPEN · finding · - · 2026-09-14
SLURM export is dead: generate_slurm_script emits `sensoryforge run --stimulus-index --format hdf5` (batch_executor.py:729-733) but run has neither flag (cli.py:556-578) and writes .pt only; BatchTab progress never emitted
→ commit 7a188b6

## F-012 · CLOSED · finding · - · 2026-09-14
CRITICAL canonical->legacy adapter sets grid_size = rows*cols (generalized_pipeline.py:351) and grid.py:32 treats an int as per-side: 20x20 config -> 160k receptors, README 80x80 example -> 41M; test_gui_cli_parity and test_regression_refactoring exceed 5 GB and are OOM-killed; CLI/Batch hit it on every canonical run (cli.py:218, batch_executor.py:98)
→ commit 7a188b6 · closed by commit 1c93fa6 (grid_size now emits (rows, cols))

## F-013 · OPEN · finding · - · 2026-09-14
Batch export lacks neuron/receptor coordinates and dt on the canonical path; .pt is one monolithic pickle; spikes are T+1 while drive/filtered are T (undocumented); HDF5 drops list-valued stimulus params (batch_executor.py:499-592)
→ commit 7a188b6

## F-014 · CLOSED · finding · - · 2026-09-14
Installed package does not run: python_requires>=3.8 but neurons/sa.py:49 needs 3.10; no package_data so gui/default_params.json and config/default_config.yml are not installed; core/pipeline.py:86,392,422 open the default config by cwd-relative path; h5py undeclared; docs advertise a nonexistent [full] extra; PyQt5 is a hard dependency
→ commit 7a188b6

## F-015 · CLOSED · finding · - · 2026-09-14
Release scaffolding and hygiene missing: no pyproject.toml, pytest.ini, lint config, .github/workflows, CITATION.cff, CHANGELOG.md, CONTRIBUTING.md; test_refactoring.py at root; devo_reports/ raw notes tracked; .github/copilot-instructions.md tracked despite gitignore; three author strings; dead PyPI link README.md:387
→ commit 7a188b6

## F-016 · CLOSED · finding · - · 2026-09-14
Qt test suite is order-dependent: tests/unit/test_stimulus_tab_gui.py:86-89 puts MagicMocks into sys.modules["PyQt5*"] and never restores them (86x "QtGui has no attribute QColor" in later files); Qt files segfault at interpreter exit when run alone
→ commit 7a188b6

## F-017 · CLOSED · finding · - · 2026-09-14
test_gui_cli_parity.py:70 passes a file path to SensoryForgeConfig.from_yaml, which takes YAML text (schema.py:426-455); simulation_engine.py:16 docstring shows the same wrong call; five integration tests still assert dt=0.5 step counts from before D-005; test_invalid_innervation_method_raises_error no longer raises
→ commit 7a188b6

## F-018 · OPEN · finding · - · 2026-09-14
cli list-components is a hardcoded print block (cli.py:438-478) already out of sync with the registries (lists center_surround, omits fa/sa/composite/timeline/repeated_pattern); cli validate forces the legacy pipeline for canonical configs (cli.py:406)
→ commit 7a188b6

## F-019 · OPEN · finding · - · 2026-09-14
~3500 lines of unwired GUI code: gui/protocol_suite_tab.py, protocol_backend.py, protocol_execution_controller.py, neuron_explorer.py are imported by no tab, only by two tests
→ commit 7a188b6

## F-020 · OPEN · finding · - · 2026-09-14
Public docs: developer_guide/*, units_and_gains.md, gui_walkthrough.md, configuration_schema.md absent from mkdocs nav; 8 broken intra-doc links; "pip install sensoryforge" in 3 pages; sensoryforge/config/README.md describes 4 nonexistent files; docs/api_reference/ is a .gitkeep
→ commit 7a188b6

## F-021 · CLOSED · finding · - · 2026-09-14
Debt lists are stale: CLAUDE.md still lists DSL numpy-only (C-2) and reset_states (M-1) as open, both resolved (R-001, D-011); docs/development/reviews/CODE_REVIEW_20260408.md tracker says 37/37 open though several are fixed; decide whether reviews/ ships publicly
→ commit 7a188b6

## F-022 · OPEN · finding · - · 2026-09-14
No validation against reference data or pressure-simulation: one analytic filter test (test_filters_vs_theory.py), no TouchSim/Saal comparison, notebook unexecuted, no benchmark suite
→ commit 7a188b6

## N-001 · STANDING · note · release · 2026-09-14
Publication-readiness audit run (engine parity vs pressure-simulation, code state, packaging). Full
report: docs/development/reviews/PUBLICATION_READINESS_20260914.md. Open findings F-001…F-022 were opened by that
commit's trailers. Live entries below this line are dated live; everything tagged (from <sha>) was
reconstructed from git history on 2026-09-14.
→ docs/development/reviews/PUBLICATION_READINESS_20260914.md

## R-001 · STANDING · retired · neurons · 2026-02-08 (from 87f1449)
"The equation DSL is numpy-only (no CUDA / autograd)" — retired: model_dsl.py lambdifies with torch
ops and threads `device` through compile(). Remaining real limit is Euler-only integration.
→ sensoryforge/neurons/model_dsl.py:54-77,354-357 · CLAUDE.md "Known Technical Debt" C-2 is stale

## D-011 · CLOSED · decision · filters · 2026-04-10 (from 9e31c46)
SAFilterTorch / RAFilterTorch inherit BaseFilter and expose reset_state(); reset_states(batch, n,
device) is kept only as the buffer allocator.
→ sensoryforge/filters/sa_ra.py:81,271 · CLAUDE.md M-1 debt item is stale

## D-010 · CLOSED · decision · release · 2026-09-03 (from 2e12414)
README badges and install steps that point at non-existent infrastructure (CI, Pages, PyPI) are
removed until the infrastructure exists. Package is source-install only for now.
→ README.md

## D-009 · CLOSED · decision · gui · 2026-04-13 (from 4e31c82)
ExperimentManager owns the project directory (stimuli/, results/, figures/); SensoryForgeWindow
holds one instance and pushes it to every tab via set_experiment_manager().
→ sensoryforge/core/experiment_manager.py

## D-008 · CLOSED · decision · engine · 2026-04-10 (from 85b12db)
input_gain is applied inside SimulationEngine.run() (filter → gain → noise → neuron); default 50
compensates the SA/RA filter calibration units vs the mA stimulus unit.
→ sensoryforge/core/simulation_engine.py:399 · sensoryforge/config/schema.py:190 · docs/user_guide/units_and_gains.md

## D-007 · CLOSED · decision · neurons · 2026-04-10 (from 2c74b9e)
Izhikevich, AdEx and MQIF carry a v_floor clamp (−120 / −130 / −120 mV) as a numerical-stability
guard against Euler blow-up at coarse dt.
→ sensoryforge/neurons/izhikevich.py:47 · adex.py:46 · mqif.py:43

## D-006 · CLOSED · decision · filters · 2026-04-10 (from 3191080)
SAFilterTorch output is clamped to ≥ 0 (clip_to_positive=True by default) to suppress the
subthreshold oscillations reported in devo_reports/development_9_april_2026 item 8.
→ sensoryforge/filters/sa_ra.py:53,160 · contested by F-001 (pressure-simulation does not rectify SA)

## D-005 · CLOSED · decision · engine · 2026-04-10 (from e4821da)
All DEFAULT_CONFIG and GUI dt defaults lowered to 0.1 ms (from 0.5 / 1.0) for forward-Euler
stability of the neuron models.
→ sensoryforge/core/generalized_pipeline.py · sensoryforge/gui/tabs/spiking_tab.py

## D-004 · CLOSED · decision · engine · 2026-04-11 (from fa01f2d)
SimulationEngine._run_pop_from_drive() is the single shared static backend (filter → gain → noise
→ neuron); the GUI SpikingNeuronTab calls it directly so GUI and engine.run() cannot drift.
→ sensoryforge/core/simulation_engine.py:355-423

## D-003 · CLOSED · decision · engine · 2026-04-10 (from 448e5cd)
Canonical configs (grids + populations, no `pipeline` key) are routed through SimulationEngine in
the CLI and BatchExecutor; GeneralizedTactileEncodingPipeline is the legacy path (max 3 populations).
→ sensoryforge/cli.py:186-225 · sensoryforge/core/batch_executor.py:90-108

## D-002 · CLOSED · decision · config · 2026-02-20 (from 58b0139)
SensoryForgeConfig (grids / populations / stimulus / simulation dataclasses) is the canonical config
format produced by GUI and CLI; the legacy dict format stays supported for backward compatibility.
→ sensoryforge/config/schema.py

## D-001 · CLOSED · decision · scope · 2026-02-06 (from 2b2c44e)
SensoryForge is an encoding-only extraction of pressure-simulation's encoding stack (pipeline,
filters, innervation, neurons, GUI). Decoding / reconstruction / Kalman stay in pressure-simulation.
→ sensoryforge/core/pipeline.py:1-17 still carries the `encoding.pipeline_torch` header

