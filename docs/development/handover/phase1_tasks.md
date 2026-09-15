# Phase 1 handover — review of Phase 0/1a and the task list for Phase 1

Prepared 2026-09-14 for an implementation agent (Sonnet). The approved plan is
`docs/developer_guide/roadmap_v1.md`; this file turns its Phase 1 into executable tasks and adds
the repairs that the review of Phase 0/1a found necessary. Open findings are in
`docs_root/LEDGER.md` (the session-start hook injects a digest).

---

## Kickoff prompt (paste to the agent)

> You are implementing Phase 1 of `docs/developer_guide/roadmap_v1.md` in `~/sensoryforge`, on `main`
> (Wave H has been merged). Your task list is `docs/development/handover/phase1_tasks.md`. Before
> starting, run `git log --oneline -20`, read sections 1i and 2 in full, and recreate the memory
> watchdog from the appendix in your scratchpad. Then do task H6 only. Work on `main` directly, not in
> a worktree. Follow the Guardrails exactly: one commit, a single-line `Closes: F-049` trailer, run the
> "Done when" checks and paste their output, and confirm the new tests fail on `b3492c5`. Report the
> commit hash. Do not push.

---

## 1. Review of Phase 0 and Phase 1a

Verified by reading the diff `b12e034..e66f529`, running the full non-GUI suite, running every Qt
test file in isolation on both the pre-Phase-0 tree and HEAD, and reproducing each claim.

### What is correct

| Item | Evidence |
|---|---|
| F-001 SA filter no longer rectifies by default | `filters/sa_ra.py:53`; GUI and legacy pipeline inherit the class default |
| Receptor-grid half of F-012 | adapter emits `(rows, cols)`; non-square 12×20 grid round-trips through adapter and engine |
| F-012 regression test is meaningful | both tests in `tests/regression/test_f012_adapter_grid_size.py` fail on the pre-Phase-0 tree |
| Batch resize workaround replaced by a loud `ValueError` | `core/batch_executor.py` |
| Unknown innervation method now raises | engine check; every name the GUI emits is registered |
| Stale dt=0.5 test expectations fixed | diagnosis correct (D-005 moved the default to 0.1 ms) |
| `model_params` no longer mutated in place | `simulation_engine.py` copies the dict |
| Citation (F-005), scope note (F-009), reviews move (F-021) | live tree has no "Pierzowski"; index at `docs/development/reviews/README.md` |
| No regressions | non-GUI: 682 passed, 6 skipped. Qt files one at a time: 223 passed at HEAD, every file matching the pre-Phase-0 tree (whose `test_stimulus_grid_inmemory` run crashed at teardown after 4 of 5) |

### What is wrong or incomplete

| # | Severity | Problem | Ledger |
|---|---|---|---|
| 1 | Critical | The adapter still **squares neuron counts**. It writes `neuron_rows × neuron_cols` into `neurons.sa_neurons`, which `InnervationModule` treats as per-row. A canonical population of 4 per row builds 256 neurons in the legacy pipeline and 16 in `SimulationEngine`. The README 80×80 quick-start still exceeds 3 GB (killed by a watchdog). The claim that the README OOM was fixed was never reproduced. | F-025 |
| 2 | High | **GUI and CLI now run different models for the same config.** The Spiking tab builds from `gui/default_params.json` (RA τ_RA 30, k3 100, regular-spiking a/b/c/d for every population) and exports only overrides. `SimulationEngine` uses class defaults (τ_RA 8, k3 2.0, fast-spiking for RA). Same drive: 32 spikes GUI, 4 CLI. Phase 0 introduced the τ_RA and neuron divergence; k3 pre-existed. | F-026 |
| 3 | High | **F-002 closed prematurely.** τ_RA is still 30 in `core/generalized_pipeline.py:132` and `:472`, `gui/default_params.json:80`, `gui/neuron_explorer.py:113`, and 15 in `examples/example_config.yml`, `examples/batch_config.yml`, `tests/fixtures/phase2_config.yml`, `README.md:128`, `docs/user_guide/configuration_schema.md`, `docs/user_guide/yaml_configuration.md`. | F-026 |
| 4 | Medium | **`from_yaml` regression:** a one-line YAML/JSON string longer than 255 bytes raises `OSError: File name too long`, because `Path.is_file()` is probed first. | F-027 |
| 5 | Medium | **New behaviour has no tests:** presets (values, override precedence, invalid name), engine RA→FS default, `from_yaml` path branch and `from_yaml_file`, `core/pipeline.py` reading `filters.sa/ra`, neuron counts through the adapter. | F-028 |
| 6 | Low | `preset` was inserted between `d` and `v_init` in the Izhikevich signature, so a positional call `IzhikevichNeuronTorch(a, b, c, d, v_init)` now raises. No caller in this repo does that; external callers would. | F-028 |
| 7 | Low | Moving `reviews/` under `docs/` added 9 `mkdocs build` warnings (relative code links in `REVIEW_AGENT_FINDINGS_20260211.md`). `--strict` in CI would fail. | F-029 |
| 8 | Needs decision | **RA k3 is unresolved in both repos.** GUI 100, engine 2.0, pressure-simulation class/config/decoder gain 2.0, but its `encode_runner.py:155-165` uses 1.0 and notes k3=100 was tuned for regular-spiking neurons and saturates fast-spiking ones near 1000 Hz. | F-030 |
| 9 | Low | Plan step "mirror F-001/F-002 in pressure-simulation" not done: its `CombinedSARAFilter` still defaults τ_RA to 30 (`encoding/filters_torch.py:408`). | task A7 |
| 10 | Low | The plan-mandated architecture `Decision:` trailer was never recorded. Recorded by the commit that adds this file. | — |

`CLAUDE.md` "Known Technical Debt" and `.claude/rules/engine-parity.md` were corrected in the same
commit so they no longer claim these items are resolved.

## 1b. Review of Wave A (2026-09-14, commits `f9048bb..d194cd4`, pressure-simulation `fa4af7e`)

Verified by re-running every acceptance check, running each new test against the commit before its
fix, and probing the resolver with configs the tests did not cover.

| Task | Verdict | Evidence |
|---|---|---|
| A1 neuron counts | Accepted | README quick-start peaks at 332 MB (was killed above 3 GB); new tests fail on the old tree, and the cap test exhausts memory there |
| A2 `from_yaml` | Accepted | long one-line text parses; test fails on the old tree |
| A3 keyword-only preset | Accepted | test fails on the old tree |
| A4 defaults resolver | **Rejected in part** | defaults agree between GUI, engine and adapter, but a partial override breaks both (F-031, below) |
| A5 coverage tests | Accepted | both tests fail on the commits before the behaviour they cover |
| A6 docs exclusion | Accepted | 0 warnings mention `development/`; remaining 8 warnings pre-date Phase 0 |
| A7 pressure-simulation mirror | Accepted | one file changed; that repo's uncommitted ledger was left alone |
| Suites | Green | non-GUI 722 passed, 6 skipped; Qt files one at a time 226 passed |

**F-031 (A4 defect).** `resolve_neuron_params` expands the neuron-type preset only when no `a`/`b`/`c`/`d`
override is present. The GUI stores only values that differ from the resolved defaults, so a user who
changes just `d` on an RA population exports `model_params: {d: 4.0}`. Measured on that config:

| Path | a | d |
|---|---|---|
| GUI (`_gather_model_parameters`) | 0.1 | 4.0 |
| `SimulationEngine` | 0.02 | 4.0 |
| `GeneralizedTactileEncodingPipeline` adapter | raises `KeyError: 'a'` | — |

The adapter crash means CLI and batch runs of that GUI config fail, because both build the adapter
pipeline for stimulus generation. `tests/unit/test_config_defaults.py::test_explicit_d_override_suppresses_preset`
asserts the wrong semantics and must be rewritten. F-026 and F-004 were closed with this defect present.

**F-032 (not covered by A4's file list).** RA neurons in `TactileEncodingPipelineTorch`
(`core/pipeline.py:161-162`) and in hand-written legacy configs (`DEFAULT_CONFIG` `ra_a`/`ra_d`,
`core/generalized_pipeline.py:160-163`) still use regular-spiking defaults, and `CombinedSARAFilter`
keeps its own default dict (`filters/sa_ra.py:455-456`) instead of reading `FILTER_DEFAULTS`.

**Minor, no task needed yet:** the F-023 cap (2e8 elements) does not catch the smaller mistake quoted in
F-023 itself (`sa_neurons: 100` on 80×80 builds 10,000 neurons, 6.4e7 elements); `SimulationEngine`
resolves filter defaults only for the names `sa`/`ra`, not the registered aliases `safilter`/`rafilter`.

## 1c. Review of A8 and Wave B (2026-09-14, commits `e518611`, `b6701a4`)

| Task | Verdict | Evidence |
|---|---|---|
| A8 preset base on partial overrides | Accepted | new regression test fails on `d194cd4` for the two partial-override cases and the legacy RA neurons; a 30-combination probe (SA/RA × 5 neuron overrides × 3 filter overrides) found no difference between `SimulationEngine`, the GUI resolution and the legacy adapter |
| A8 k3 = 2.0 | Accepted as implemented; **decision must be confirmed by the user** | the commit states D-Q1 was decided by the user (ledger D-018). On the default trapezoidal stimulus, fast-spiking RA averaged 314 Hz at the old GUI value of 100 and 69 Hz at 2.0, with 61 of 64 neurons still firing |
| B1 `pyproject.toml` | Accepted | wheel contains `config/default_config.yml` and `gui/default_params.json`, no tests or docs; entry point works |
| B2 `importlib.resources` | Accepted | wheel installed in a scratch venv and run from `/tmp`: `sensoryforge list-components`, `create_standard_pipeline()` and both GUI parameter paths work |
| Imports | OK | every touched module imports cleanly in a fresh interpreter (no cycle from `filters/sa_ra.py` → `config.defaults` → `neurons`) |
| Suites | Green | non-GUI 728 passed, 6 skipped; Qt files one at a time 226 passed; README quick-start 338 MB |

Two loose ends, recorded in the ledger:

- **F-033:** setuptools warns that `project.license` as a table and the `License :: OSI Approved :: MIT License` classifier are deprecated; builds stop being supported after 2027-02-18. Task B3.
- **F-034:** pressure-simulation's `encoding/encode_runner.py` still defaults RA k3 to 1.0 (its comment calls that calibrated for fast-spiking RA at input gain 40), which contradicts D-018 "k3 = 2.0 everywhere" and blocks a zero-tolerance golden parity test (E5). Needs the user to confirm whether "everywhere" includes that runner.

## 1d. Review of B3 and Wave C (2026-09-14, commits `c374052`, `d9eb260`, `874b19a`, `77ef203`)

| Task | Verdict | Evidence |
|---|---|---|
| B3 SPDX license | Accepted | verbose wheel build prints 0 `SetuptoolsDeprecationWarning`; wheel metadata has `License-Expression: MIT` and `License-File: LICENSE` |
| C1 `pytest.ini` and `gui` marker | Accepted | `-m gui` collects 229, `-m "not gui"` 734, together the full 963 |
| C2 PyQt5 mock leak | Accepted | running `test_stimulus_tab_gui.py` first, then real-Qt files, in one process: 64 `QColor` errors on `874b19a`, 166 passed at HEAD; all Qt files in reverse order also pass |
| C3 Qt crashes | Accepted, with a tracked follow-up | `pytest -m gui` 228 passed, 1 skipped, exit 0, 571 MB; full `pytest` 956 passed, 7 skipped, exit 0, 1.2 GB. The `gc.disable()` is needed: with the collector on, `pytest -m gui` crashed in 3 of 3 runs and the full suite crashed too. Memory cost is small (non-GUI 1165 MB vs 1094 MB with the collector on) |

**F-035 (follow-up, not blocking CI).** The crash that `gc.disable()` avoids is in GUI code, not test
teardown. With the collector on, it segfaults inside pyqtgraph's `ScatterPlotItem` render path, called
from `MechanoreceptorTab._add_receptor_scatter_by_weight` ← `_update_innervation_graphics` ←
`_create_population_graphics` ← `_regenerate_selected_population_if_instantiated`, via a `ViewBox`
lambda left over from a previously destroyed tab's plot. The test harness now cannot detect this class
of crash, and `gc.disable()` also applies to non-GUI sessions, where it is not needed. The GUI app
keeps one long-lived tab, so user impact is plausible but unproven. Root cause belongs with the Phase 3
GUI work; task C4 narrows the workaround now.

Other observations: `pytest_unconfigure` imports PyQt5 even in non-GUI sessions (harmless, caught if
Qt libraries are missing, but unnecessary); the old per-file Qt baseline undercounted
`test_population_csv.py` (7 instead of 9) because its isolated run aborted at exit.

## 1e. Review of C4 and Wave D (2026-09-14, commits `87adf4c..9f13374`)

| Task | Verdict | Evidence |
|---|---|---|
| C4 GC scoped to GUI sessions | Accepted | `pytest -m gui` 228 passed, 1 skipped; `pytest -m "not gui"` 729 passed, 6 skipped with the collector on; full `pytest` 957 passed, 7 skipped; all exit 0, peaks 550 MB / 1.3 GB / 1.3 GB. The `trylast=True` ordering fix is correct and covered by `tests/unit/test_conftest_gc_scope.py` |
| D1 black, `BaseSolver`, `.flake8` | Accepted | black commit touches only `.py` files; after normalizing docstring whitespace no file's syntax tree changed. `black --check` and `flake8` exit 0 |
| D2 GitHub Actions | Accepted as written, not yet proven | YAML parses and every local command succeeds; the workflow has never run on GitHub. Two likely first-run problems are task D4 |
| D3 community files | Accepted with two corrections made in review | `CITATION.cff` has all required fields; no dangling references to the deleted files |

Corrections made in the review commit:

- `CHANGELOG.md` claimed default innervation weights are analytic Gaussian. That is task E1 and has not been done (`use_distance_weights` still defaults to `False`). The bullet was removed; E1 re-adds it.
- `CONTRIBUTING.md` referred to `register_components.py` without its package path.
- D1's style-debt `Finding:` was written as a wrapped multi-line trailer, which the ledger parser silently drops. It is recorded as F-036. Guardrail 8 now requires single-line trailers.

pressure-simulation: the user decided k3 = 2.0 everywhere. Commit `f7784f9` there sets the RA default in `encoding/encode_runner.py` and `GUIs/ebkf_viewer.py`, and the decoder fallbacks in `decoding/pipeline.py` and the pseudo-inverse reconstructor, to 2.0 (filter, pipeline and decoder tests: 48 passed before and after). F-034 is closed. That commit also records, in pressure-simulation's ledger, that its RA input gains were tuned at k3 = 1.0.

## 1f. Review of D4 and Wave E (2026-09-14, commits `6dbf9ac..6c02331`)

| Task | Verdict | Evidence |
|---|---|---|
| D4 CI hardening | Accepted | pins and apt libraries as specified; CI still has never run on GitHub |
| E1 analytic Gaussian weights | Accepted | defaults changed in schema, both innervation modules and the GUI; `CHANGELOG.md` updated. README example spikes change from SA 8 / RA 15 to SA 2 / RA 19 |
| E2 per-instance RNG | **Rejected in part** | global RNG no longer touched and same-seed CPU wiring is bit-identical to before, but seeded innervation now crashes on MPS (and, by the same mechanism, CUDA): `Expected a 'mps' device type for generator but found 'cpu'`. Before E2, `InnervationModule` and `SimulationEngine` ran on MPS. F-038 |
| E3 noise after filter and gain | Accepted | not in `CHANGELOG.md` (task E10) |
| E4 neuron sub-stepping | Accepted, with two regressions and two gaps | sub-stepping matches pressure-simulation (proved by E5). Regression: the canonical adapter still reads `simulation.dt`, so texture, moving, timeline, repeated-pattern and custom stimuli in CLI/batch runs use 0.1 ms whatever `dt_ms` says (F-039). Regression: `SimulationConfig(dt=...)` no longer constructs (only `from_dict` accepts the alias). Gap: a record step that is not a whole multiple of `integrate_dt_ms` silently rescales neuron time, e.g. 0.12 ms bins integrate 0.10 ms (F-042). Gap: none of the E4 behaviour changes are in `CHANGELOG.md` |
| E5 golden parity | Accepted | 159 SA and 304 RA spikes in the fixture; the fixture regenerates bit-identically from pressure-simulation; four mutations (integration step 0.1, τ_RA 9, k3 2.1, RA `d` 2.2) each make the test fail; pressure-simulation's unclamped SA voltage bottoms at −106.4 mV, so the −120 mV clamp never engages in this case |
| Suites | Green | gui 230 passed, 1 skipped; not-gui 741 passed, 6 skipped; full 971 passed, 7 skipped; all exit 0 at ~1.3 GB peak; black and flake8 exit 0 |

Two older problems became more consequential once `dt_ms` started driving filter integration and sub-steps:

- **F-040:** a CLI run of a canonical config with `dt_ms: 1.0 --duration 100` produces 10,450 bins, because trapezoid, gaussian, step and ramp stimuli use the legacy `temporal.dt` (0.1 ms, F-024) and the trapezoid ignores `--duration`. The engine then reads every bin as 1 ms, so the simulated timeline is about 100× longer than requested. The pre-Wave-E tree gives 10,451 bins, so this predates Wave E. It is on the CLI/batch data-generation path.
- **F-041:** a GUI export always writes `simulation.dt_ms: 1.0`, because `SpikingNeuronTab.get_config()` carries no time step and `gui/main.py` `_gui_to_canonical` defaults to 1.0. The GUI simulates at the stimulus step (0.1 ms by default), so a GUI-exported config runs with different filter integration and sub-steps from the CLI.

## 1g. Review of the Wave E fixes and Wave F (2026-09-15, commits `a59f385..8084d28`)

The implementation report only described Wave F; the Wave E fixes E6–E10 were also done and are reviewed here.

| Task | Verdict | Evidence |
|---|---|---|
| E6 CPU draws, then device | Accepted | the three accelerator tests run (not skipped) on this machine's MPS and pass; seeded MPS weights equal CPU weights |
| E7 one record step for CLI/batch | Accepted | `sensoryforge run` on a `dt_ms: 1.0` canonical config with `--duration 100` now gives 100 bins for both gaussian and trapezoid stimuli (was 1,000 and 10,450). Design note: the trapezoid now scales all four segments, so a 100 ms trapezoid has ~1 ms ramps instead of 10 ms, which sharply increases RA drive for short durations. `_reconcile_dt_keys` decides by comparing values with the 0.1 ms default, so a config that deliberately sets `neurons.dt: 0.1` and `temporal.dt: 0.5` resolves to 0.5 |
| E8 GUI exports the simulated step | Accepted | 4 new tests; GUI suite passes |
| E9 record-step validation | Accepted, with a GUI gap (F-044) | 0.12 and 0.01 rejected with clear messages; but the stimulus Δt spinbox still accepts 0.12 and the Spiking tab only catches `RuntimeError`, so the `ValueError` escapes a Qt slot (PyQt5 aborts by default; no exception hook installed) |
| E10 `dt` keyword alias and changelog | Accepted | changelog now covers E2–E9 |
| F1 docs navigation and strict build | Accepted | `mkdocs build --strict` exits 0 with 0 warnings |
| F2 CLI reads the registries | Accepted | `list-components` lists registered names (including aliases, e.g. `RA`, `ra`, `rafilter`); `validate` builds `SimulationEngine` for canonical configs |
| Old-code checks | Hold | the new E7, E9 and F2 test files all fail or error on `6c02331` |
| Suites | Green | gui 234 passed, 1 skipped; not-gui 780 passed, 6 skipped; full 1,014 passed, 7 skipped; exit 0; black, flake8 and golden parity pass |

**F-043 (new, blocks the Phase 1 exit criteria).** The shipped `examples/example_config.yml` and `examples/batch_config.yml` set `sa_neurons: 100`, `ra_neurons: 196`, `sa2_neurons: 25`, commented as neuron counts; legacy configs read them per row. `sensoryforge run`, `validate` and `batch --dry-run` on them fail at HEAD with the dense-weight cap error; on `4342b8b`, before the cap, `validate` was killed above 3 GB. The same values appear in `docs/user_guide/batch_processing.md`, `cli.md` and `yaml_configuration.md`, and `CLAUDE.md` tells users to run the first example. No canonical-format example exists, which the exit criteria require.

## 1h. Review of F3, F4 and Wave G, and Phase 1 exit check (2026-09-15, commits `d9e83f8..04c230b`)

| Task | Verdict | Evidence |
|---|---|---|
| F3 canonical examples | Accepted | from a wheel installed in a scratch venv and run outside the repo: `validate` and `run --duration 50` on `examples/canonical_config.yml` succeed (100 SA, 196 RA neurons), `batch --dry-run` on `canonical_batch_config.yml` succeeds |
| F4 GUI time-step safety | Accepted | spinbox snaps, `ValueError` caught, `sys.excepthook` installed; 2 new GUI tests |
| G1 `get_param_spec` everywhere | Accepted, follow-up F-045 | the five neuron models now inherit `BaseNeuron`; their `to_dict` round-trips only `dt` |
| G3 grid arrangement classes | Accepted | registry introspection only; F-010 (engine ignores arrangement for innervation) stays open |
| G2 plugin discovery | Accepted, with defects F-046 and F-048 | an external package installed with `pip install -e` registered a filter and a neuron through the entry point, and both appear in `list-components`. But the neuron registered as `DemoNeuron` fails in `SimulationEngine` ("Unknown neuron model: DemoNeuron", the engine lowercases neuron names) while `demo_neuron` works; filters are looked up exactly, so `DemoGain` works and `demogain` fails. YAML `plugins:` is honoured only by CLI config loading, not the GUI's YAML load |
| G4 contract tests | Accepted | 35 passed, 5 documented skips |
| G5 scaffold | Accepted for source checkouts, defect F-047 | from a wheel install it wrote `site-packages/sensoryforge/filters/bandpass.py`, `site-packages/tests/unit/test_bandpass_filter.py` and `site-packages/docs/...`, and tells the user to edit core `register_components.py`; it does not produce an installable entry-point plugin |
| Commit hashes in the report | Correct | all seven exist on `main` and their subjects match the tasks; ledger-sync commits are omitted from the list, which is fine. G2–G5 carry no trailers, so no sync was needed |

**Phase 1 exit criteria**

| Criterion | Status |
|---|---|
| Wheel from outside the repo runs the canonical example; GUI imports offscreen | Pass |
| `pytest -m "not gui"` and `pytest -m gui` exit 0 in one process each | Pass (867 passed / 11 skipped; 236 passed / 1 skipped). Full suite 1,103 passed, exit 0, 1.4 GB peak |
| CI commands succeed locally | Pass (black, flake8, `mkdocs build --strict` with 0 warnings). CI has still never run on GitHub |
| Parity | Pass (resolver parity tests; golden parity against pressure-simulation) |
| `mkdocs build --strict` | Pass |
| Ledger | F-043 and F-044 closed. Open beyond the allowed list: F-045, F-046, F-047, F-048 — all extensibility, handled by Wave H |

Phase 1 is therefore not closed: the extensibility promise (a third party adds a component without editing core files and uses it in a simulation) does not hold yet. Wave H closes it.

## 1i. Review of Wave H (2026-09-15, branch `worktree-wave-h`, commits `a72ff7d..b3492c5`)

Work was done in the worktree `.worktrees/wave-h` on branch `worktree-wave-h`, forked from `main` at `9a24f47`. The branch is 11 commits ahead of `main` and `main` has not moved, so it fast-forwards.

| Task | Verdict | Evidence |
|---|---|---|
| H1 case-insensitive names | Accepted | from a branch wheel in a scratch venv, a canonical config using a plugin neuron and filter runs identically as `leaky_demo`/`band_demo` and `Leaky_Demo`/`BAND_DEMO`; registering `SA` to a different class raises; re-registering the same class is idempotent |
| H2 plugin-package scaffold | Accepted | outside the repo, `new-component neuron LeakyDemo --dest .` and `new-component filter BandDemo --dest .` produced installable packages; after `pip install -e` their generated tests pass, `list-components` shows them, and the simulation above uses them; nothing was written under `site-packages`; `--in-repo` refuses outside a checkout |
| H3 neuron round trips | Accepted | tuple parameters, `seed`, `noise_std` and an FS preset with a `d` override survive `from_config(to_dict())` exactly; the preset is stored expanded |
| H4 one plugins-aware loader | Accepted | 5 tests; 4 fail on `main` |
| H5 extension docs | Accepted | `docs/examples/plugin_filter.py` executed by `tests/docs`; CLAUDE.md now documents both routes and states F-049's limit |
| Final-review fixes | Accepted | black clean; scaffolded neuron plugins now run in `SimulationEngine`; the hand edit in `b3492c5` only corrects F-049's commit pointer |
| Old-code checks | Hold | on `main`: registry 14 failed, neuron round trip 12 failed, plugin loading 4 failed, scaffold collection error |
| Suites | Green | gui 237 passed, 1 skipped; not-gui 911 passed, 11 skipped; full 1,148 passed, 12 skipped, exit 0, 1.4 GB peak; black, flake8, `mkdocs build --strict` (0 warnings), docs example, contract and golden parity tests pass |

**F-049 is a real serialization defect, not only a missing check.** `SAFilterTorch.to_dict()` and `RAFilterTorch.to_dict()` return only `{'dt': ...}`: a filter built with `tau_r=7.0, k1=0.1` is rebuilt by `from_config` with `tau_r=5.0`, and `RAFilterTorch(tau_RA=12, k3=5)` comes back as 8 and 2. The grid arrangement classes drop `density` and `EdgeGrating` drops `normalize`. Phase 2's bundle export records component parameters for reproducibility, so this must be fixed before Phase 2 (task H6).

**Merge hazard.** `main`'s ledger bookmark (`.claude/.ledger-sync`, gitignored, per checkout) is `538b91c`. In a scratch clone, fast-forwarding `main` and then running the sync hook replayed the branch's trailers and added a spurious `F-050` duplicating F-049. Any session started in `main` runs that hook automatically, so the bookmark must be advanced to the merged tip before the next session (steps below).

### Merging Wave H into `main` (user)

```bash
cd ~/sensoryforge
git status --short                       # must be clean
git merge --ff-only worktree-wave-h
git rev-parse --short HEAD > .claude/.ledger-sync   # stop the hook replaying branch trailers
CLAUDE_PROJECT_DIR=$PWD .claude/hooks/ledger-sync.sh
git status --short                       # must still be clean: no new ledger entries
git worktree remove .worktrees/wave-h && git branch -d worktree-wave-h
```

---

## 2. Guardrails (read before any task)

These come directly from what went wrong in Phase 0/1a.

1. **Memory.** Run test suites under the watchdog script in the appendix (the full suite peaks near
   1.2 GB). Since Wave C, `pytest -m "not gui"` and `pytest -m gui` each run in one process and their
   exit codes are trustworthy. Do not remove the `gc.disable()` in `tests/conftest.py` for GUI
   sessions (F-035).
2. **A default is not changed until every copy is changed.** Before editing any default, grep all
   the places listed in `.claude/rules/engine-parity.md` and change them in the same commit.
3. **Every behaviour change needs a test that fails on the old code.** Prove it: extract the parent
   commit with `git archive <sha> | tar -x -C <scratch dir>`, copy your new test in, run it there,
   and confirm it fails.
4. **Do not write "fixed" anywhere** (commit message, `CLAUDE.md`, ledger `Closes:`) until you have
   reproduced the original failing scenario and shown it now passes. For F-025 that means running
   the README quick-start under the watchdog.
5. **GUI ↔ CLI parity is a hard contract.** Any default must be resolved by one function that both
   `SpikingNeuronTab` and `SimulationEngine` call. A default applied in only one of them is a bug.
6. **New constructor parameters are keyword-only** (after `*`) and go at the end.
7. **Git.** Stage explicit paths, run `git status` after every `git add` and every `git mv` (both
   sides of a move must be staged in the same commit). No `git rebase -i`, no force-push, no amending
   commits that are already on `origin`. Commit messages follow Conventional Commits and end with the
   attribution lines from your session instructions.
8. **Ledger trailers.** `Closes: F-0NN` only with evidence. Anything new you discover →
   `Opens: F-0NN <one line>` on the commit where you found it. **Every trailer is exactly one line**
   in the final paragraph of the message; a wrapped trailer is silently dropped by the parser. Put
   the evidence in the body and a one-line summary in the trailer. Never edit `docs_root/LEDGER.md` by
   hand except to remove a stale `Directive:` line on a closed entry. After committing, run
   `CLAUDE_PROJECT_DIR=$PWD .claude/hooks/ledger-sync.sh` and commit the ledger as
   `chore: ledger sync after <task>`.
9. **Blocked decisions.** Where a task says "blocked on D-Q", do not choose a value. Skip the blocked
   part, finish the rest, and list the open question in your report.
10. **Pressure-simulation repo** (`~/Documents/pressure simulation`) is touched only by task A7. It has
    an uncommitted change to its `docs_root/LEDGER.md`; never stage it.

---

## 3. Decision needed from the user

**D-Q1 — RA filter gain k3 (F-030).** Blocks the k3 part of A4 and the golden parity test E5.

| Option | Consequence |
|---|---|
| 2.0 (recommended) | Matches both repos' `RAFilterTorch` class default, pressure-simulation's `config/pipeline_config.yml` and its decoder gain ("sync with filters.ra.k3"). pressure-simulation's `encode_runner.py` default of 1.0 would then need aligning. RA input gain likely needs retuning in presets. |
| 1.0 | Matches pressure-simulation's calibrated runner for fast-spiking RA (~120 Hz on a 20 Hz grating at gain 40). Its class, config and decoder gain would change to 1.0. |
| 100 | Keeps the current GUI behaviour. Pressure-simulation reports it saturates fast-spiking RA near 1000 Hz. Not compatible with F-004. |

---

## 4. Task list

Each task: **Goal**, **Files**, **Do**, **Done when**, **Trailers**. Line numbers predate the black
reformat in `68fc511` and are now approximate; always re-grep before editing.

### Wave A — repair Phase 0/1a (do first, in order)

#### A1. Stop the adapter squaring neuron counts
- **Goal:** a canonical config builds the same neuron count in the legacy pipeline as in `SimulationEngine`; the README quick-start runs in well under 1.5 GB.
- **Files:** `sensoryforge/core/generalized_pipeline.py` (`_canonical_to_legacy_config` near `:404`, `:447`, `:480`; `_create_innervation` near `:592-712`), `tests/regression/test_f012_adapter_grid_size.py`.
- **Do:** emit per-row values plus explicit row/column keys (for example `neurons.sa_neuron_rows` / `sa_neuron_cols`) instead of the product. In `_create_innervation` pass `neuron_rows=` / `neuron_cols=` to `InnervationModule` and `FlatInnervationModule` when present (both accept them), falling back to `neurons_per_row`. Do the same for RA and SA2. For legacy hand-written configs (F-023), keep per-row semantics but raise a `ValueError` naming the key when the dense weight tensor `N × H × W` would exceed a documented cap (suggest 2e8 elements), so a mistaken total fails fast.
- **Done when:** new tests assert SA/RA/SA2 neuron counts equal `rows × cols` for a square (4×4) and a non-square (3×5) population and fail on the parent commit; the README quick-start script from the appendix prints shapes under `memwatch.sh 1500 300`; non-GUI suite green.
- **Trailers:** `Closes: F-025`, `Closes: F-023` (only if the cap is implemented and tested).

#### A2. Fix the `from_yaml` OSError
- **Goal:** `SensoryForgeConfig.from_yaml` accepts YAML text of any length, a `str` path, or a `Path`.
- **Files:** `sensoryforge/config/schema.py` (`from_yaml` near `:427`), `tests/unit/test_config_schema.py`.
- **Do:** treat the argument as a path only if it is a `Path`, or a `str` without newlines whose `is_file()` check succeeds inside `try/except OSError`; otherwise parse as YAML.
- **Done when:** tests cover a >255-byte one-line JSON string, a multi-line YAML string, a `str` path, a `Path`, and `from_yaml_file`; the long-string test fails on the parent commit.
- **Trailers:** `Closes: F-027`.

#### A3. Make `preset` keyword-only and test the presets
- **Files:** `sensoryforge/neurons/izhikevich.py`, new `tests/unit/test_izhikevich_presets.py`.
- **Do:** move `preset` to the end of the signature after `*`.
- **Done when:** tests check every `IZHIKEVICH_PRESETS` entry matches Izhikevich (2003) Fig. 2, `preset="FS", d=4.0` gives a=0.1 and d=4.0, an unknown preset raises `ValueError`, `IzhikevichNeuronTorch(0.02, 0.2, -65.0, 8.0, -70.0).v_init == -70.0`, and default construction equals the historical RS values.
- **Trailers:** none (partial F-028; closed in A5).

#### A4. One source of truth for filter and neuron defaults (GUI ↔ CLI parity)
- **Goal:** the GUI, `SimulationEngine`, the legacy pipeline and `core/pipeline.py` resolve identical filter and neuron parameters for the same population config.
- **Files:** new `sensoryforge/config/defaults.py`; `sensoryforge/core/simulation_engine.py` (neuron and filter construction, the FS block near `:266-279`); `sensoryforge/gui/tabs/spiking_tab.py` (`_populate_model_parameter_fields` `:774`, `_collect_model_parameter_overrides` `:837`, `_collect_filter_parameter_overrides` `:853`, `_gather_model_parameters` `:886`, `_gather_filter_parameters` `:896`); `sensoryforge/gui/default_params.json`; `sensoryforge/gui/neuron_explorer.py:113`; `sensoryforge/core/generalized_pipeline.py:132` and `:472`; `sensoryforge/filters/sa_ra.py` (`CombinedSARAFilter` defaults); `examples/example_config.yml`; `examples/batch_config.yml`; `tests/fixtures/phase2_config.yml`; `README.md:128`; `docs/user_guide/configuration_schema.md`; `docs/user_guide/yaml_configuration.md`.
- **Do:**
  1. In `config/defaults.py` define `FILTER_DEFAULTS = {"sa": {...}, "ra": {...}}` (τ_RA 8.0; k3 from D-Q1), `NEURON_PRESET_BY_TYPE = {"SA": "RS", "RA": "FS"}` (SA2 → RS), and pure functions `resolve_filter_params(method, overrides) -> dict` and `resolve_neuron_params(model_name, neuron_type, overrides) -> dict`. No Qt imports.
  2. `SimulationEngine` and the Spiking tab both call these resolvers. The GUI widget defaults and the override comparison use the resolved values for the population's neuron type, not raw JSON. Remove the filter and Izhikevich a/b/c/d entries from `default_params.json` (keep GUI-only keys).
  3. The legacy `DEFAULT_CONFIG`, adapter fallbacks, `CombinedSARAFilter` and `neuron_explorer.py` read from `FILTER_DEFAULTS`.
  4. Fix every τ_RA copy in the Files list.
- **Blocked:** the k3 value only (D-Q1). If unanswered, implement everything else but leave k3 out of the resolver-owned keys: the engine keeps the class default 2.0, `default_params.json` keeps its RA `k3` of 100, and the drift test and parity test exclude k3 with a `# D-Q1 pending` comment. Say so in the commit body and the report.
- **Done when:** a Qt-free test calls both resolvers for SA and RA with empty and non-empty overrides and asserts the engine builds exactly those values; a drift test fails if `default_params.json` redefines resolver-owned keys; a Qt test (run alone) asserts the Spiking tab builds an RA Izhikevich neuron with `a == 0.1` and an RA filter with `tau_RA == 8.0`; `grep -rn "tau_RA.*30\|ra_tau_ra.*30\|tau_RA.*15\|ra_tau_ra.*15"` over `sensoryforge examples tests/fixtures README.md docs/user_guide` returns nothing; the parity script from the appendix prints identical GUI and engine parameters.
- **Trailers:** `Decision: filter and neuron defaults are resolved only by sensoryforge/config/defaults.py, used by GUI, SimulationEngine and legacy pipelines`; `Closes: F-026`; `Closes: F-004`; `Closes: F-030` only if D-Q1 was answered.

#### A5. Cover the remaining untested Phase 0/1a behaviour
- **Files:** `tests/unit/test_simulation_engine_features.py`, `tests/integration/test_pipeline.py`.
- **Done when:** tests exist and fail on the parent of the change they cover for: engine RA→FS default (and an explicit `preset`/`a` override that suppresses it); `TactileEncodingPipelineTorch` passing `filters.sa` / `filters.ra` YAML values into `CombinedSARAFilter` (use a non-default τ_RA and assert it arrives).
- **Trailers:** `Closes: F-028`.

#### A6. Keep the docs build clean after the reviews move
- **Files:** `mkdocs.yml`.
- **Do:** add `exclude_docs:` entries for `development/reviews/` and `development/handover/` (mkdocs ≥1.5) so the internal records stay in the repo but out of the site.
- **Done when:** `python -m mkdocs build -d <scratch>/site` shows no warnings mentioning `development/`.
- **Trailers:** `Closes: F-029`.

#### A7. Mirror the τ_RA fix into pressure-simulation
- **Files:** `~/Documents/pressure simulation/encoding/filters_torch.py:408` only.
- **Do:** change `CombinedSARAFilter`'s `default_ra` τ_RA from 30 to 8.0 and run that repo's `pytest tests/test_enhanced_filters.py -q`. Stage only that file. Commit there with `Decision: CombinedSARAFilter default tau_RA is 8 ms, matching RAFilterTorch and SensoryForge (SensoryForge D-015)`.
- **Done when:** its filter tests pass and `git -C "~/Documents/pressure simulation" show --stat HEAD` lists exactly one file.

#### A8. Keep the preset as the base when overriding single neuron parameters (do this next)
- **Goal:** GUI, `SimulationEngine` and both legacy pipelines build identical Izhikevich parameters for every combination of neuron type, explicit `preset`, and individual `a`/`b`/`c`/`d` overrides.
- **Files:** `sensoryforge/config/defaults.py` (`resolve_neuron_params`); `tests/unit/test_config_defaults.py` (rewrite `test_explicit_d_override_suppresses_preset`); `sensoryforge/core/generalized_pipeline.py` (`DEFAULT_CONFIG` neuron keys near `:148-175`, adapter near `:432` and `:471`, the neuron construction that reads `neuron_params`); `sensoryforge/core/pipeline.py:161-162`; `sensoryforge/filters/sa_ra.py:455-456`; `tests/regression/` (new file).
- **Do:**
  1. In `resolve_neuron_params`, always start from a preset: the explicit `preset` override if given, otherwise `NEURON_PRESET_BY_TYPE` for the neuron type. Then apply `a`/`b`/`c`/`d` overrides on top. The result for Izhikevich must always contain `a`, `b`, `c`, `d`, `threshold`. This matches `IzhikevichNeuronTorch(preset=..., d=...)` semantics.
  2. Rewrite the wrong test so `resolve_neuron_params("Izhikevich", "RA", {"d": 99.0})` gives `a == 0.1`, `d == 99.0`.
  3. Route `CombinedSARAFilter`'s defaults through `FILTER_DEFAULTS` (k3 stays at 2.0 with the D-Q1 comment).
  4. In `TactileEncodingPipelineTorch` build the RA neurons from `resolve_neuron_params("izhikevich", "RA", {})` and the SA neurons from `"SA"`. In `GeneralizedTactileEncodingPipeline`, make the RA `DEFAULT_CONFIG` a/b/c/d come from the resolver (FS) so hand-written legacy configs agree. Keep the `*_std` keys as they are.
- **Done when:** a new regression test builds an RA population with each of `{}`, `{"d": 4.0}`, `{"a": 0.05}`, `{"preset": "IB"}`, `{"preset": "IB", "d": 1.0}` and asserts identical a/b/c/d from (i) `resolve_neuron_params` as the GUI merges it, (ii) `SimulationEngine`, (iii) `GeneralizedTactileEncodingPipeline.from_config` (no `KeyError`); it fails on `d194cd4`. A second test asserts `TactileEncodingPipelineTorch` RA neurons have `a == 0.1`. The appendix parity script with `model_params={"d": 4.0}` prints identical GUI and engine values. Non-GUI suite green; `test_spiking_tab_defaults.py` still passes when run alone.
- **Trailers:** `Closes: F-031`, `Closes: F-032`.

### Wave B — packaging (plan 1b, F-014)

#### B1. `pyproject.toml` and package data
- **Files:** new `pyproject.toml`; remove `setup.py` last; `sensoryforge/__init__.py` (`__author__`); `LICENSE` (copyright name); `README.md` (remove the PyPI link near the Links section); `docs/getting_started/installation.md` (remove `[full]` and the PyPI install lines; also `docs/user_guide/cli.md`, `docs/tutorials/batch_processing_tutorial.md`).
- **Do:** PEP 621 metadata with setuptools backend; `requires-python = ">=3.10"`; core deps without Qt; extras `gui` (PyQt5, pyqtgraph), `hdf5` (h5py), `solvers` (torchdiffeq), `dsl` (sympy), `dev` (pytest, pytest-cov, black, flake8, mypy, pydocstyle), `docs` (mkdocs, mkdocs-material, mkdocstrings[python]); console script `sensoryforge = sensoryforge.cli:main`; `[tool.setuptools.package-data]` for `sensoryforge/config/*.yml` and `sensoryforge/gui/*.json`; one author string "Ben Efron" with the maintainer email from `README.md`; classifiers, readme, project URLs.
- **Done when:** installed wheel works from a directory outside the repo (appendix "Wheel check"): `sensoryforge list-components` runs and `create_standard_pipeline()` loads its default config.
- **Trailers:** none yet (B2 closes).

#### B2. Load packaged data through `importlib.resources`
- **Files:** `sensoryforge/core/pipeline.py:86`, `:402`, `:432`; `sensoryforge/gui/neuron_explorer.py:42`; `sensoryforge/gui/tabs/spiking_tab.py:54`; `examples/scripts/example_pipeline.py:8`.
- **Do:** replace the cwd-relative `"sensoryforge/config/default_config.yml"` default with `None` resolved via `importlib.resources.files("sensoryforge.config") / "default_config.yml"`; same for `default_params.json`.
- **Done when:** the wheel check passes from an unrelated working directory, including importing `sensoryforge.gui.tabs.spiking_tab` with `QT_QPA_PLATFORM=offscreen`.
- **Trailers:** `Closes: F-014`.

#### B3. SPDX license metadata (do this first in the next run)
- **Files:** `pyproject.toml`.
- **Do:** `license = "MIT"`, `license-files = ["LICENSE"]`, remove the `License :: OSI Approved :: MIT License` classifier, raise `[build-system] requires` to `setuptools>=77`.
- **Done when:** `python -m pip wheel . --no-deps --no-build-isolation -v -w <scratch>/dist 2>&1 | grep -c SetuptoolsDeprecationWarning` prints 0, and the appendix wheel check still passes.
- **Trailers:** `Closes: F-033`.

### Wave C — test hygiene (plan 1c, F-016)

#### C1. `pytest.ini` and markers
- **Do:** `testpaths = tests`; `addopts = -p no:cacheprovider`; register markers `gui` and `slow`; set `pytestmark = pytest.mark.gui` in each Qt test file (`test_expert_mode`, `test_gain_defaults`, `test_grid_population_ux`, `test_gui_agent_d`, `test_phase3_features`, `test_population_csv`, `test_stimulus_grid_inmemory`, `test_stimulus_tab_ux`, `test_unified_workflow`, `test_stimulus_tab_gui`, plus any new Qt test from A4).
- **Done when:** `pytest -m "not gui" --collect-only -q | tail -1` and `pytest -m gui --collect-only -q | tail -1` together equal the unfiltered collection count.

#### C2. Stop the PyQt5 mock leaking across test files
- **Files:** `tests/unit/test_stimulus_tab_gui.py:86-89`.
- **Do:** install the mocks inside a module-scoped fixture that saves and restores the original `sys.modules` entries (or `monkeypatch.setitem`), and import the tab under test inside that fixture.
- **Done when:** all Qt test files run in **one** process (`pytest -m gui -v`) with zero `has no attribute 'QColor'` errors.

#### C3. Qt teardown crash
- **Do:** in `tests/conftest.py`, add a `pytest_unconfigure` hook that, only when a `QApplication` instance exists, flushes stdio and calls `os._exit(<session exit status>)`. Keep the status from `pytest_sessionfinish`.
- **Done when:** `QT_QPA_PLATFORM=offscreen pytest -m gui` exits with status 0 and prints the summary line.
- **Trailers (C3 commit):** `Closes: F-016`.

#### C4. Scope the GC workaround to GUI sessions (do this first in the next run)
- **Files:** `tests/conftest.py`.
- **Do:** remove the module-level `gc.disable()`. In a `pytest_collection_modifyitems(session, config, items)` hook, call `gc.disable()` only if any collected item has the `gui` marker (`item.get_closest_marker("gui")`). In `pytest_unconfigure`, check `sys.modules.get("PyQt5.QtWidgets")` instead of importing PyQt5, and force-exit only if that module is loaded and `QApplication.instance()` is not `None`. Update the comment to point at F-035 and the crash stack in section 1d.
- **Done when:** `pytest -m gui` exits 0 in one process; full `pytest` exits 0 in one process; `pytest -m "not gui"` passes and a check in that session confirms `gc.isenabled()` is `True` (for example a tiny test in `tests/unit/test_conftest_gc_scope.py` that asserts `gc.isenabled()` and is not marked `gui`); run all three under the watchdog and report peak memory.
- **Trailers:** none (F-035 stays open for the GUI root cause).

### Wave D — CI and community files (plan 1d, F-015)

#### D1. Formatting baseline
- **Measured before this task:** `black --check` would reformat 117 of 154 files; `flake8` at 88 columns reports 3253 violations (2401 blank-line whitespace, 538 long lines, 141 unused imports, then small counts). Only 1 is in the correctness set `E9,F63,F7,F82`: `sensoryforge/neurons/model_dsl.py:321` uses `BaseSolver` in a string annotation without importing it.
- **Do:**
  1. Add `[tool.black]` (line length 88) to `pyproject.toml`.
  2. Commit `style: apply black to sensoryforge and tests` containing only `black sensoryforge tests` output. Verify `git diff --stat` touches only `.py` files and both `pytest -m "not gui"` and `pytest -m gui` still pass.
  3. Fix the `BaseSolver` annotation with `from typing import TYPE_CHECKING` and an `if TYPE_CHECKING:` import of `sensoryforge.solvers.base.BaseSolver`.
  4. Add a `.flake8` that gates CI on the correctness set only: `select = E9,F63,F7,F82`, `max-line-length = 88`, `extend-ignore = E203,W503`, `exclude = .git,build,site,.claude`. Record the remaining style debt with a `Finding:` trailer giving the post-black count by code, so it can be ratcheted later. Do not mass-delete unused imports in this wave.
  5. `pydocstyle` is not installed in the conda environment; install the dev extra with `pip install -e ".[dev]"` if you need it, and say so in the report. Do not add pydocstyle to CI yet.
- **Done when:** `black --check sensoryforge tests` exits 0; `flake8 sensoryforge tests` exits 0 with the new config; both test suites pass.

#### D2. GitHub Actions
- **Files:** `.github/workflows/tests.yml`.
- **Do:** matrix ubuntu-latest and macos-latest × Python 3.10 and 3.11; install CPU torch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) then `pip install -e ".[gui,hdf5,dsl,dev]"`; job 1 `pytest -m "not gui"`; job 2 on ubuntu with `QT_QPA_PLATFORM=offscreen` and the Qt system libraries (`libegl1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0`) running `pytest -m gui`; job 3 `black --check sensoryforge tests`, `flake8 sensoryforge tests`, and `mkdocs build` **without** `--strict` (it currently aborts on 8 broken tutorial and user-guide links that task F1 fixes; F1 switches this job to `--strict`). Add a `concurrency` group and `pip` caching.
- **Done when:** the workflow file passes `python -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml'))"` and every command in it succeeds locally. Do not push; the user pushes.

#### D3. Community and hygiene files
- **Do:** add `CITATION.cff`, `CHANGELOG.md` (an "Unreleased" section that lists every behaviour change users will notice: SA no longer rectified by default, τ_RA 8 ms, RA fast-spiking preset, RA k3 2.0 in the GUI instead of 100 (RA firing on the default ramp stimulus drops from about 314 Hz to 69 Hz), grid and neuron count fixes, defaults resolver, Python 3.10+ and the `gui` extra for PyQt5), `CONTRIBUTING.md` (absorb `DEVELOPMENT.md`, then delete it; include the extension guide pointers and the ledger trailer convention), `CODE_OF_CONDUCT.md`. Delete `test_refactoring.py` at the repo root (superseded by `tests/integration/test_regression_refactoring.py`). Untrack `.github/copilot-instructions.md` with `git rm --cached` (it is already gitignored).
- **Trailers:** `Closes: F-015`.

#### D4. CI hardening (do this first in the next run)
- **Files:** `pyproject.toml`, `.github/workflows/tests.yml`.
- **Do:** pin the formatter and linter majors in the `dev` extra (`black>=26.1,<27`, `flake8>=7,<8`) so a new black stable style cannot fail CI on its own. In the `gui` job's apt step add `libgl1 libglib2.0-0 libfontconfig1 libdbus-1-3` next to the existing libraries (PyQt5's `QtGui` links against `libGL.so.1`, which `libegl1` does not provide).
- **Done when:** the workflow still parses; `black --check sensoryforge tests` and `flake8 sensoryforge tests` still exit 0 with the pinned versions installed locally; the report tells the user to watch the first GitHub run after pushing, since CI has never executed.
- **Trailers:** none.

### Wave E — remaining engine parity (plan 1e)

#### E1. Analytic Gaussian weights by default (F-003)
- **Files:** `sensoryforge/config/schema.py:148`; the GUI `NeuronPopulation` default in `gui/tabs/mechanoreceptor_tab.py` (dataclass field and `chk_use_distance_weights.setChecked`); legacy `DEFAULT_CONFIG`; the "Gaussian falloff" docstring at `core/innervation.py:131`.
- **Do:** default `use_distance_weights=True`; keep the stochastic builder reachable and name it in docs as the control arm.
- **Done when:** a test asserts default-config innervation weights decrease monotonically with distance for one neuron; `CHANGELOG.md` "Changed" lists "Default innervation weights are analytic Gaussian; the stochastic uniform-random builder remains available as the control arm".
- **Trailers:** `Decision: default innervation uses analytic Gaussian weights; the stochastic uniform-weight builder is the named control arm`; `Closes: F-003`.

#### E2. Per-instance RNG in innervation (F-006)
- **Files:** `sensoryforge/core/innervation.py` global `torch.manual_seed` calls near `:202`, `:352`, `:454`, and the later neuron-centre and Gaussian builders.
- **Do:** follow the pattern in `sensoryforge/filters/noise.py` (a `torch.Generator` seeded per instance, passed to every random call).
- **Done when:** a test builds innervation twice with the same seed after drawing unrelated global random numbers in between and gets identical weights; the global RNG state is unchanged by construction.
- **Trailers:** `Closes: F-006`.

#### E3. One noise topology (F-007)
- **Files:** `sensoryforge/core/pipeline.py` (`apply_noise` before filtering).
- **Do:** move to filter → gain → noise, matching `SimulationEngine._run_pop_from_drive` and pressure-simulation's runner.
- **Trailers:** `Closes: F-007`.

#### E4. Neuron sub-stepping (F-008)
- **Files:** `sensoryforge/config/schema.py` (`SimulationConfig`), `sensoryforge/core/simulation_engine.py` (`_run_pop_from_drive` `:372`), GUI call site in `spiking_tab.py`.
- **Do:** add `dt_ms` (record step) and `integrate_dt_ms` (default 0.05); accept legacy `dt` as an alias of `dt_ms` in `from_dict`. Match pressure-simulation's `encoding/encode_runner.run_encoding` exactly: the filter integrates at `dt_ms`; gain then noise are applied per record bin; the neuron is constructed with `dt=integrate_dt_ms`, gets `n = max(1, round(dt_ms / integrate_dt_ms))` sub-steps per bin with the drive held constant (`repeat_interleave`), and its state carries across bins in one forward pass; drop the initial sample (`spikes[:, 1:, :]`), then return spike **counts** per record bin (pressure-simulation reports `counts > 0`) and voltages at bin ends, both shaped `[batch, T, N]`. Both `SimulationEngine` and the Spiking tab must construct neurons with `integrate_dt_ms`.
- **Done when:** with `integrate_dt_ms == dt_ms` outputs equal the current behaviour exactly; with sub-stepping the summed counts equal the raw spike count of a direct fine-step run; the GUI and engine produce identical results (extend `tests/integration/test_engine_parity.py`).
- **Trailers:** `Closes: F-008`.

#### E5. Golden parity test against pressure-simulation
- **Blocked on:** E4 only (F-034 is resolved: k3 = 2.0 in both repos).
- **Do:**
  1. Write `scripts/dev/export_pressure_sim_golden.py`. Run it from the pressure-simulation repo root with `PYTHONPATH=.` and the `sensoryforge` conda Python (pressure-simulation's tests pass in that environment). Use a small case: 8×8 grid, T = 200 bins, `dt_ms = 1.0`, one SA and one RA population with fixed weights `[N, 8, 8]` (seeded random, saved), a positive trapezoid stimulus, `input_gain` and `noise_std = 0` given explicitly, explicit `filter_params` (SA `tau_r 5, tau_d 30, k1 0.05, k2 3.0`; RA `tau_RA 8, k3 2.0`), and explicit Izhikevich `model_params` (SA regular-spiking, RA fast-spiking a/b/c/d). `run_encoding` does not return the filtered response, so compute drive and filtered response in the script with pressure-simulation's own `SAFilterTorch`/`RAFilterTorch` exactly as the runner does, and take spikes from `run_encoding`. Save stimulus, weights, parameters, drive, filtered response and spikes to `tests/fixtures/pressure_sim_golden/case_small.npz` (keep it under 1 MB).
  2. Add `tests/integration/test_pressure_sim_parity.py`: build a `SimulationEngine` for the same populations, replace each population's innervation weights with the golden weights, run the stimulus, and compare in stages so a mismatch is localized: drive (1e-6), filtered response after gain (1e-6), then spikes as `counts > 0` (exact).
  3. SensoryForge clamps Izhikevich voltage at −120 mV (`v_floor`, D-007) and pressure-simulation does not (F-037). Keep the golden stimulus positive so neither side reaches the clamp, and note the limitation in the test docstring.
- **Trailers:** `Opens:` any mismatch found, with the measured difference.

#### E6. Draw innervation randomness on CPU, then move to the device (do this first in the next run)
- **Files:** `sensoryforge/core/innervation.py` (every call that takes `generator=`).
- **Do:** keep the per-instance CPU generator, create every seeded random tensor on the CPU, then `.to(self.device)` / `.to(device)`. This keeps a given seed's wiring identical across CPU, MPS and CUDA. Where a call currently passes a device tensor (for example `torch.multinomial(prob_weights, ...)` with `prob_weights` on the device), compute that draw on a CPU copy.
- **Done when:** a test marked `skipif(not torch.backends.mps.is_available() and not torch.cuda.is_available())` builds seeded `InnervationModule` and `FlatInnervationModule` on the accelerator and asserts the weights equal the CPU build for the same seed; `SimulationEngine` with `device="mps"` (when available) runs a small config; the E2 RNG-isolation test still passes. Run the accelerator test locally (MPS is available on this machine) and paste its output, since CI cannot.
- **Trailers:** `Closes: F-038`.

#### E7. One record step end to end for CLI and batch (F-039, F-040, F-024)
- **Files:** `sensoryforge/core/generalized_pipeline.py` (`_canonical_to_legacy_config`, and the stimulus generators' `dt` lookups), `sensoryforge/cli.py` (`cmd_run` stimulus parameters), `sensoryforge/core/batch_executor.py` (canonical stimulus generation).
- **Do:** in the adapter read `dt_ms` (falling back to `dt`) and write it to both `neurons.dt` and `temporal.dt`. For legacy configs that set only `neurons.dt`, make `temporal.dt` follow it (and vice versa when only `temporal.dt` is set). Make every stimulus generator read one resolved record step. Pass `--duration` through to every stimulus type, including trapezoid, or scale the trapezoid's plateau so its total length equals the requested duration; document which.
- **Done when:** a CLI integration test runs a canonical config with `dt_ms: 1.0 --duration 100` for gaussian and trapezoid stimuli and gets 100 bins; a BatchExecutor test with `dt_ms: 0.5` gets stimulus length `duration / 0.5`; the adapter test covers `dt_ms`, legacy `dt`, and each moving/texture/timeline stimulus; each new test fails on `6c02331`.
- **Trailers:** `Closes: F-039`, `Closes: F-040`, `Closes: F-024`.

#### E8. GUI export writes the step it simulated (F-041)
- **Files:** `sensoryforge/gui/tabs/spiking_tab.py` (`get_config`, `set_config`), `sensoryforge/gui/main.py` (`_gui_to_canonical`, `_canonical_to_gui_config`).
- **Do:** export `dt_ms` from the active stimulus step and `integrate_dt_ms` from `DEFAULT_INTEGRATE_DT_MS`; read both back on load.
- **Done when:** a `gui`-marked test sets a 0.1 ms stimulus step, exports, and gets `simulation.dt_ms == 0.1` in the canonical dict; loading that dict restores 0.1 ms; the test fails on `6c02331`.
- **Trailers:** `Closes: F-041`.

#### E9. Reject record steps that are not a whole multiple of the integration step (F-042)
- **Files:** `sensoryforge/config/schema.py` (`SimulationConfig.__post_init__`), `sensoryforge/core/simulation_engine.py` (`_run_pop_from_drive`), the GUI time-step spinbox in `spiking_tab.py`.
- **Do:** raise `ValueError` naming both values when `abs(dt_ms / integrate_dt_ms - round(dt_ms / integrate_dt_ms)) > 1e-6` or `dt_ms < integrate_dt_ms`; apply the same check in `_run_pop_from_drive` for direct callers; set the GUI spinbox single step to `integrate_dt_ms` so it can only produce valid values.
- **Done when:** tests show 1.0, 0.5, 0.1 and 0.05 accepted and 0.12, 0.07 and 0.03 rejected; both suites still pass.
- **Trailers:** `Closes: F-042`.

#### E10. Keep the old constructor keyword working and document Wave E
- **Files:** `sensoryforge/config/schema.py`, `CHANGELOG.md`.
- **Do:** accept `SimulationConfig(dt=...)` as a deprecated alias for `dt_ms` (emit `DeprecationWarning`; reject passing both). Add to `CHANGELOG.md` "Changed": noise is applied after filter and gain in `TactileEncodingPipelineTorch`; `simulation.dt` is now `dt_ms` with the old key and keyword accepted; neurons integrate at `integrate_dt_ms` (0.05 ms) with the drive held per record bin, which multiplies neuron-stage runtime by `dt_ms / 0.05` (20× at 1 ms); `SimulationEngine` spikes are integer counts per bin (use `> 0` for a raster) of length T, not booleans of length T+1; innervation no longer changes the global RNG.
- **Done when:** a test constructs `SimulationConfig(dt=0.5)` and gets `dt_ms == 0.5` with a `DeprecationWarning`; the changelog lists every item above.
- **Trailers:** none.

### Wave F — docs (plan 1f, F-020, F-018)

#### F1. Navigation, links, strict build
- **Do:** add `developer_guide/*`, `user_guide/units_and_gains.md`, `user_guide/gui_walkthrough.md`, `user_guide/configuration_schema.md` to `mkdocs.yml` nav; fix the 8 broken links listed in `docs/development/reviews/PUBLICATION_READINESS_20260914.md` §5; rewrite `sensoryforge/config/README.md` to describe the files that exist; add `mkdocstrings` and an API reference page; create the docs skeleton from roadmap "Cross-cutting requirement B" with stub pages marked "coming in Phase N".
- **Done when:** `mkdocs build --strict` passes, and the docs job in `.github/workflows/tests.yml` uses `--strict`.
- **Trailers:** `Closes: F-020`.

#### F2. CLI reads the registries
- **Files:** `sensoryforge/cli.py` (`cmd_list_components`, `cmd_validate`).
- **Do:** generate `list-components` output from the registries; make `validate` use `SimulationEngine` for canonical configs.
- **Done when:** a test registers a dummy neuron and sees it in `list-components` output.
- **Trailers:** `Closes: F-018`.

#### F3. Working examples, canonical first (do this first in the next run)
- **Files:** new `examples/canonical_config.yml` and `examples/canonical_batch_config.yml`; `examples/example_config.yml`; `examples/batch_config.yml`; `examples/README.md`; `docs/user_guide/batch_processing.md`, `docs/user_guide/cli.md`, `docs/user_guide/yaml_configuration.md`; `CLAUDE.md` "Commands → CLI"; `README.md` if it references example files; new `tests/integration/test_examples_smoke.py`.
- **Do:**
  1. Write `examples/canonical_config.yml` with `SensoryForgeConfig` (one 40×40 grid, SA 10 per row and RA 14 per row, SA/RA filters, trapezoid stimulus, `dt_ms: 1.0`) and `examples/canonical_batch_config.yml` sweeping amplitude over 3 values. Generate them from dataclasses with `to_yaml()` so they are schema-valid, then add comments.
  2. In the legacy examples and the three docs pages, change the neuron counts to per-row values (`sa_neurons: 10`, `ra_neurons: 14`, `sa2_neurons: 5`) and fix the comments to say "neurons per row (population = N×N)".
  3. Point `CLAUDE.md` and `examples/README.md` at the canonical example first.
  4. `tests/integration/test_examples_smoke.py` parametrised over every `examples/*.yml`: run `sensoryforge validate`; for non-batch configs run `sensoryforge run --duration 20 --output <tmp>`; for batch configs run `sensoryforge batch --dry-run`. Assert exit code 0.
- **Done when:** the smoke test passes and fails on `8084d28` for the two legacy examples; the phase-exit wheel check below runs `sensoryforge run examples/canonical_config.yml --duration 50` from `/tmp` successfully.
- **Trailers:** `Closes: F-043`.

#### F4. Invalid time steps cannot crash the GUI (F-044)
- **Files:** `sensoryforge/gui/tabs/stimulus_tab.py` (`spin_dt`), `sensoryforge/gui/tabs/spiking_tab.py` (`_run_simulation`), `sensoryforge/gui/main.py`.
- **Do:** snap `spin_dt` to the nearest multiple of `DEFAULT_INTEGRATE_DT_MS` on `editingFinished`; catch `ValueError` alongside `RuntimeError` in `_run_simulation` and report it through the existing `errors` list; install a `sys.excepthook` in `gui/main.py` that shows unhandled exceptions in a `QMessageBox` instead of aborting.
- **Done when:** a `gui`-marked test types 0.12 into `spin_dt`, finishes editing, and reads 0.10; a second test makes `_simulate_population` raise `ValueError` and asserts `_run_simulation` returns normally with the message recorded; both fail on `8084d28`.
- **Trailers:** `Closes: F-044`.

### Wave G — extensibility baseline (plan 1g)

- **G1.** `get_param_spec()` on every base class (default `[]`); `ParamSpec` gains optional `choices`, `help`, `group`, `advanced` without breaking existing call sites.
- **G2.** Entry-point discovery in `register_all()` via `importlib.metadata.entry_points(group="sensoryforge.components")`, plus a `plugins:` list of import paths in YAML. A plugin that fails to import produces a warning, not a crash.
- **G3.** Replace the `GRID_REGISTRY.register(<name>, str)` placeholders in `register_components.py` with real arrangement classes.
- **G4.** `tests/contract/test_component_contracts.py` parametrised over every registered component: `from_config` → `to_dict` round trip, `get_param_spec` returns `ParamSpec` objects, one forward pass with the kind's canonical tensor shape.
- **G5.** `sensoryforge new-component <kind> <name>` scaffold writing the class, a unit test that passes the contract test, and a docs stub.
- **Done when (wave):** a throwaway package in the scratch directory, installed with `pip install -e`, registers a filter through an entry point, and that filter appears in `sensoryforge list-components` and passes the contract test.

---

### Wave H — Phase 1 close-out: extensibility that works for third parties

#### H1. Case-insensitive component names (F-046, do this first)
- **Files:** `sensoryforge/registry.py` (`ComponentRegistry`), `sensoryforge/core/simulation_engine.py` (remove the `.lower()` on the neuron lookup), `sensoryforge/register_components.py` (drop pure case-variant aliases such as `"Izhikevich"`/`"izhikevich"`, `"SA"`/`"sa"`; keep genuine aliases like `safilter`), `sensoryforge/cli.py` (`list-components`).
- **Do:** normalise names on `register`, `get_class`, `is_registered` and `create` (case-fold for lookup, keep the first registered spelling for display). Registering two spellings of the same name to *different* classes raises `ValueError`; registering the same class again stays idempotent. `list_registered()` returns one display name per component.
- **Done when:** a test registers a neuron `DemoNeuron` and a filter `DemoGain` from outside `register_components.py`, and `SimulationEngine` runs configs that spell them `DemoNeuron`/`demoneuron` and `DemoGain`/`demogain`; a collision test raises; `list-components` shows no case duplicates; existing configs using `Izhikevich`, `izhikevich`, `SA`, `sa`, `DSL (Custom)` still load; the plugin test fails on `04c230b`.
- **Trailers:** `Closes: F-046`.

#### H2. Scaffold produces an installable plugin package (F-047)
- **Files:** `sensoryforge/scaffold.py`, `sensoryforge/cli.py` (`new-component`), new `sensoryforge/testing/contracts.py`, `tests/contract/test_component_contracts.py`.
- **Do:**
  1. Move the contract checks from `tests/contract/test_component_contracts.py` into an importable `sensoryforge.testing.contracts` module (`check_component(kind, cls)`), and have the in-repo contract test call it.
  2. Default mode: `sensoryforge new-component <kind> <Name> [--dest DIR]` (default: current directory) writes a standalone package `DIR/sensoryforge-<name>/` with `pyproject.toml` declaring `[project.entry-points."sensoryforge.components"]`, the component module, a `register()` function, `tests/test_contract.py` that calls `sensoryforge.testing.contracts.check_component`, and a `README.md`.
  3. `--in-repo` mode keeps today's behaviour for contributors, but finds the repository root from the current directory (a `.git` directory plus `pyproject.toml` naming `sensoryforge`) instead of the installed package location, and refuses to run otherwise.
  4. Refuse to write anywhere under a `site-packages` or `dist-packages` directory.
- **Done when:** from a wheel installed in a scratch venv and a working directory outside the repo, `sensoryforge new-component filter Bandpass --dest .`, then `pip install -e ./sensoryforge-bandpass`, then its generated tests pass, `list-components` shows it, and a canonical config using it runs; nothing is written under `site-packages`. Paste the commands and output.
- **Trailers:** `Closes: F-047`.

#### H3. Neuron models round-trip all their parameters (F-045)
- **Files:** `sensoryforge/neurons/izhikevich.py`, `adex.py`, `mqif.py`, `fa.py`, `sa.py`; `sensoryforge/testing/contracts.py`.
- **Do:** give each model a `to_dict()` returning every constructor argument (Izhikevich stores resolved `a`/`b`/`c`/`d`, not `preset`), `from_config()` accepting that dict, and a `get_param_spec()` listing each parameter with units, default and a sensible range.
- **Done when:** the contract check asserts `cls.from_config(m.to_dict()).to_dict() == m.to_dict()` and that every `__init__` parameter except `self` appears in `to_dict()`; it fails on `04c230b` for all five models.
- **Trailers:** `Closes: F-045`.

#### H4. Config plugins load the same way everywhere (F-048)
- **Files:** `sensoryforge/config/yaml_utils.py` (one `load_config_file(path)` that reads YAML and honours `plugins:`), `sensoryforge/cli.py`, `sensoryforge/core/batch_executor.py`, `sensoryforge/gui/main.py` (YAML load), `sensoryforge/config/schema.py` (`from_yaml_file`).
- **Do:** route every YAML config load through that one function; log each imported plugin module at INFO level; document in the user guide that a config's `plugins:` list imports installed modules and calls the named callables.
- **Done when:** a test writes a config whose `plugins:` entry registers a component and loads it through the CLI loader, `BatchExecutor`, `SensoryForgeConfig.from_yaml_file` and the GUI loader (gui-marked), and the component is registered in each case.
- **Trailers:** `Closes: F-048`.

#### H5. Document the extension path
- **Files:** `CLAUDE.md` ("Adding a New Component" and "ParamSpec" sections), `docs/developer_guide/extensibility.md`, `add_neuron.md`, `add_filter.md`, `add_stimulus.md`, new `docs/developer_guide/plugins.md`, new `docs/examples/plugin_filter.py`, `CHANGELOG.md`, `mkdocs.yml`.
- **Do:** describe the two supported routes (entry-point plugin via `new-component`, or an in-repo contribution) and the contract checks; add one worked example under `docs/examples/` that defines, registers and simulates a filter, executed by a new test (`tests/docs/test_docs_examples.py` running each file in `docs/examples/`); add Wave F–H items to the changelog.
- **Done when:** `mkdocs build --strict` passes; `pytest tests/docs` runs the example; CLAUDE.md no longer says components "must be registered in `register_components.py`".
- **Trailers:** none.

#### H6. Every component round-trips its full configuration (F-049, closes Phase 1)
- **Files:** `sensoryforge/testing/contracts.py` (`_check_filter`, `_check_stimulus`, `_check_grid`, `_check_solver`, `_check_innervation`), `sensoryforge/filters/sa_ra.py`, `sensoryforge/filters/base.py`, `sensoryforge/core/grid_arrangements.py`, `sensoryforge/stimuli/texture.py` (`EdgeGrating`), and any other registered component the strengthened check fails.
- **Do:** call `_assert_to_dict_roundtrip_complete` from every `_check_<kind>`. Run `pytest tests/contract` and fix each component it fails by making `to_dict()` return every constructor argument and `from_config()` accept it. Keep a component in `_TO_DICT_EXCLUDE_PARAMS` only for a documented reason (for example tensors such as `receptor_coords`, which bundles store separately).
- **Done when:** `pytest tests/contract` passes with the check applied to all six kinds; a new test builds `SAFilterTorch(tau_r=7.0, tau_d=40.0, k1=0.1, k2=2.0, clip_to_positive=True)` and `RAFilterTorch(tau_RA=12.0, k3=5.0)` and asserts `from_config(to_dict())` reproduces every attribute; both fail on `b3492c5`; all suites, black, flake8 and `mkdocs build --strict` still pass.
- **Trailers:** `Closes: F-049`.

## 5. Phase 1 exit criteria (from the roadmap)

- Wheel installed in a clean environment runs `sensoryforge run examples/canonical_config.yml --duration 50` from `/tmp` and the GUI imports offscreen (appendix wheel check).
- `pytest -m "not gui"` and `pytest -m gui` both exit 0 in one process each.
- CI workflow commands all succeed locally.
- Parity: GUI and engine resolve identical parameters (A4); golden parity test green (E5) or its blocker reported.
- `mkdocs build --strict` passes.
- Ledger: F-045 to F-049 closed; F-010, F-011, F-013, F-019, F-022, F-035, F-036 and F-037 may stay open for Phase 2 and later.
- A third-party plugin created with `sensoryforge new-component` installs from outside the repo and runs in a simulation (H2 acceptance).
- CI has run green at least once on GitHub (user pushes).

---

## Appendix — scripts and commands

### Memory watchdog

Save as `<scratch>/memwatch.sh` and `chmod +x`. Usage:
`memwatch.sh <limit_mb> <timeout_s> <log> -- <command...>`

```bash
#!/bin/bash
LIM=$1; TO=$2; LOG=$3; shift 4
"$@" > "$LOG" 2>&1 &
PID=$!; PEAK=0; T=0
while kill -0 $PID 2>/dev/null; do
  RSS=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' '); RSS=$(( ${RSS:-0} / 1024 ))
  for c in $(pgrep -P $PID); do r=$(ps -o rss= -p $c 2>/dev/null | tr -d ' '); RSS=$((RSS + ${r:-0}/1024)); done
  [ $RSS -gt $PEAK ] && PEAK=$RSS
  if [ $RSS -gt $LIM ]; then pkill -P $PID; kill -9 $PID; echo "KILLED_BY_WATCHDOG rss=${RSS}MB" >> "$LOG"; break; fi
  if [ $T -ge $TO ]; then pkill -P $PID; kill -9 $PID; echo "TIMEOUT" >> "$LOG"; break; fi
  sleep 1; T=$((T+1))
done
wait $PID 2>/dev/null
echo "peak_rss_mb=$PEAK"
```

### Test commands

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 QT_QPA_PLATFORM=offscreen
PY=/opt/miniconda3/envs/sensoryforge/bin/python

# Since Wave C (pytest.ini + gui marker + conftest hooks), each suite runs in one process.
<scratch>/memwatch.sh 3000 600 <scratch>/nongui.log -- $PY -m pytest -m "not gui" -q
<scratch>/memwatch.sh 3000 600 <scratch>/gui.log    -- $PY -m pytest -m gui -q
<scratch>/memwatch.sh 4000 900 <scratch>/full.log   -- $PY -m pytest -q
# Baseline at 77ef203: not gui 728 passed, 6 skipped (~1.2 GB); gui 228 passed, 1 skipped (~0.6 GB);
# full 956 passed, 7 skipped (~1.2 GB); all exit 0
```

### Old-code check for a new test

```bash
git archive <parent-sha> | tar -x -C <scratch>/base
cp <your new test file> <scratch>/base/tests/<same path>
cd <scratch>/base && $PY -m pytest <that test> -q -p no:cacheprovider   # must FAIL
```

### README quick-start (A1 acceptance)

```python
from sensoryforge.config.schema import SensoryForgeConfig, GridConfig, PopulationConfig, StimulusConfig, SimulationConfig
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
config = SensoryForgeConfig(
    grids=[GridConfig(name="Main Grid", arrangement="grid", rows=80, cols=80, spacing=0.15)],
    populations=[
        PopulationConfig(name="SA Population", neuron_type="SA", neuron_model="izhikevich", filter_method="sa", innervation_method="gaussian", neurons_per_row=10),
        PopulationConfig(name="RA Population", neuron_type="RA", neuron_model="izhikevich", filter_method="ra", innervation_method="gaussian", neurons_per_row=14),
    ],
    stimulus=StimulusConfig(type="gaussian", amplitude=30.0, sigma=0.5),
    simulation=SimulationConfig(device="cpu", dt=0.5),
)
p = GeneralizedTactileEncodingPipeline.from_config(config.to_dict())
assert p.sa_innervation.num_neurons == 100 and p.ra_innervation.num_neurons == 196
r = p.forward(stimulus_type="gaussian", amplitude=30.0, sigma=0.5)
print(tuple(r["sa_spikes"].shape), tuple(r["ra_spikes"].shape))
```

### GUI ↔ engine parameter parity (A4 acceptance)

Build an RA population config with empty `model_params` and `filter_params`, construct
`SimulationEngine`, and print `populations[0]["neuron"].a/.d` and `populations[0]["filter"].tau_RA/.k3`.
Then compute what the Spiking tab builds by calling the same resolver functions it uses. Both lines
must be identical. Before A4 this printed `0.1 2.0 | 8.0 2.0` (engine) versus `0.02 8.0 | 30 100.0` (GUI).

### Wheel check (B1/B2 acceptance)

```bash
cd ~/sensoryforge && $PY -m pip wheel . --no-deps -w <scratch>/dist
$PY -m venv --system-site-packages <scratch>/venv
<scratch>/venv/bin/pip install --no-deps --force-reinstall <scratch>/dist/sensoryforge-*.whl
cd /tmp && <scratch>/venv/bin/sensoryforge list-components \
  && <scratch>/venv/bin/python -c "from sensoryforge.core.pipeline import create_standard_pipeline; create_standard_pipeline(); print('default config loaded')"
```

The venv reuses the conda environment's torch through `--system-site-packages`, so nothing large is
downloaded. Confirm `python -c "import sensoryforge; print(sensoryforge.__file__)"` points into the
venv, not the repo.
