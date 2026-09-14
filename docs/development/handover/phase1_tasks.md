# Phase 1 handover — review of Phase 0/1a and the task list for Phase 1

Prepared 2026-09-14 for an implementation agent (Sonnet). The approved plan is
`docs/developer_guide/roadmap_v1.md`; this file turns its Phase 1 into executable tasks and adds
the repairs that the review of Phase 0/1a found necessary. Open findings are in
`docs_root/LEDGER.md` (the session-start hook injects a digest).

---

## Kickoff prompt (paste to the agent)

> You are implementing Phase 1 of `docs/developer_guide/roadmap_v1.md` in `~/sensoryforge`. Your task
> list is `docs/development/handover/phase1_tasks.md`. Waves A and B are done and reviewed (sections 1b
> and 1c). Start with task B3, then do Wave C (C1, C2, C3). Read the "Guardrails" section first and follow it exactly. One task = one commit
> with the ledger trailers the task names. Before you mark a task done, run its "Done when" checks
> and paste their output into your final summary. If a task says it is blocked on a user decision,
> skip it and continue with the next unblocked task. Stop and report when Wave C is finished.

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

---

## 2. Guardrails (read before any task)

These come directly from what went wrong in Phase 0/1a.

1. **Memory.** Never run the whole test suite in one process, and never run anything that builds a
   canonical config through `GeneralizedTactileEncodingPipeline` without a watchdog until A1 lands.
   Use the watchdog script in the appendix. Run Qt test files one file at a time with `-v`; they
   crash at interpreter exit (F-016), so read the streamed `PASSED`/`FAILED` lines, not the exit code.
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
   `Opens: F-0NN <one line>` on the commit where you found it. Never edit `docs_root/LEDGER.md` by
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

Each task: **Goal**, **Files**, **Do**, **Done when**, **Trailers**. Line numbers are as of commit
`e66f529` plus this handover commit; re-grep before editing.

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

### Wave D — CI and community files (plan 1d, F-015)

#### D1. Formatting baseline
- **Do:** add `[tool.black]` (line length 88) and a `.flake8` with `max-line-length = 88` and `extend-ignore = E203,W503`. Run `black sensoryforge tests` once as a formatting-only commit (`style: apply black`), with no logic changes, and verify the non-GUI suite is still green.

#### D2. GitHub Actions
- **Files:** `.github/workflows/tests.yml`.
- **Do:** matrix ubuntu-latest and macos-latest × Python 3.10 and 3.11; install CPU torch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) then `pip install -e ".[gui,hdf5,dsl,dev]"`; job 1 `pytest -m "not gui"`; job 2 on ubuntu with `QT_QPA_PLATFORM=offscreen` and the Qt system libraries (`libegl1 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0`) running `pytest -m gui`; job 3 `black --check`, `flake8`, and `mkdocs build --strict`.
- **Done when:** the workflow file passes `python -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml'))"` and every command in it succeeds locally. Do not push; the user pushes.

#### D3. Community and hygiene files
- **Do:** add `CITATION.cff`, `CHANGELOG.md` (an "Unreleased" section that lists every behaviour change users will notice: SA no longer rectified by default, τ_RA 8 ms, RA fast-spiking preset, RA k3 2.0 in the GUI instead of 100 (RA firing on the default ramp stimulus drops from about 314 Hz to 69 Hz), grid and neuron count fixes, defaults resolver, Python 3.10+ and the `gui` extra for PyQt5), `CONTRIBUTING.md` (absorb `DEVELOPMENT.md`, then delete it; include the extension guide pointers and the ledger trailer convention), `CODE_OF_CONDUCT.md`. Delete `test_refactoring.py` at the repo root (superseded by `tests/integration/test_regression_refactoring.py`). Untrack `.github/copilot-instructions.md` with `git rm --cached` (it is already gitignored).
- **Trailers:** `Closes: F-015`.

### Wave E — remaining engine parity (plan 1e)

#### E1. Analytic Gaussian weights by default (F-003)
- **Files:** `sensoryforge/config/schema.py:148`; the GUI `NeuronPopulation` default in `gui/tabs/mechanoreceptor_tab.py` (dataclass field and `chk_use_distance_weights.setChecked`); legacy `DEFAULT_CONFIG`; the "Gaussian falloff" docstring at `core/innervation.py:131`.
- **Do:** default `use_distance_weights=True`; keep the stochastic builder reachable and name it in docs as the control arm.
- **Done when:** a test asserts default-config innervation weights decrease monotonically with distance for one neuron; `CHANGELOG.md` lists the change.
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
- **Do:** add `dt_ms` (record step) and `integrate_dt_ms` (default 0.05); accept legacy `dt` as an alias of `dt_ms` in `from_dict`; integrate the neuron at `integrate_dt_ms` holding the drive constant within a record bin; return spike **counts** per record bin and voltages at bin ends.
- **Done when:** with `integrate_dt_ms == dt_ms` outputs equal the current behaviour exactly; with sub-stepping the summed counts equal the raw spike count of a direct fine-step run; the GUI and engine produce identical results (extend `tests/integration/test_engine_parity.py`).
- **Trailers:** `Closes: F-008`.

#### E5. Golden parity test against pressure-simulation
- **Blocked on:** E4, and the user's answer to F-034 (whether pressure-simulation's `encode_runner.py` k3 default moves from 1.0 to 2.0). Do not edit pressure-simulation for this without that answer.
- **Do:** write `scripts/dev/export_pressure_sim_golden.py`, run manually in pressure-simulation's environment, that calls its `encoding/encode_runner.run_encoding` with a fixed stimulus and fixed innervation weights and saves `weights`, `stimulus`, per-population filtered response and spikes to `tests/fixtures/pressure_sim_golden/*.npz`. Add `tests/integration/test_pressure_sim_parity.py`, which feeds the same weights and stimulus through `SimulationEngine` and compares (filtered response to 1e-6, spikes exactly).
- **Trailers:** `Opens:` any mismatch found, with the measured difference.

### Wave F — docs (plan 1f, F-020, F-018)

#### F1. Navigation, links, strict build
- **Do:** add `developer_guide/*`, `user_guide/units_and_gains.md`, `user_guide/gui_walkthrough.md`, `user_guide/configuration_schema.md` to `mkdocs.yml` nav; fix the 8 broken links listed in `docs/development/reviews/PUBLICATION_READINESS_20260914.md` §5; rewrite `sensoryforge/config/README.md` to describe the files that exist; add `mkdocstrings` and an API reference page; create the docs skeleton from roadmap "Cross-cutting requirement B" with stub pages marked "coming in Phase N".
- **Done when:** `mkdocs build --strict` passes.
- **Trailers:** `Closes: F-020`.

#### F2. CLI reads the registries
- **Files:** `sensoryforge/cli.py` (`cmd_list_components`, `cmd_validate`).
- **Do:** generate `list-components` output from the registries; make `validate` use `SimulationEngine` for canonical configs.
- **Done when:** a test registers a dummy neuron and sees it in `list-components` output.
- **Trailers:** `Closes: F-018`.

### Wave G — extensibility baseline (plan 1g)

- **G1.** `get_param_spec()` on every base class (default `[]`); `ParamSpec` gains optional `choices`, `help`, `group`, `advanced` without breaking existing call sites.
- **G2.** Entry-point discovery in `register_all()` via `importlib.metadata.entry_points(group="sensoryforge.components")`, plus a `plugins:` list of import paths in YAML. A plugin that fails to import produces a warning, not a crash.
- **G3.** Replace the `GRID_REGISTRY.register(<name>, str)` placeholders in `register_components.py` with real arrangement classes.
- **G4.** `tests/contract/test_component_contracts.py` parametrised over every registered component: `from_config` → `to_dict` round trip, `get_param_spec` returns `ParamSpec` objects, one forward pass with the kind's canonical tensor shape.
- **G5.** `sensoryforge new-component <kind> <name>` scaffold writing the class, a unit test that passes the contract test, and a docs stub.
- **Done when (wave):** a throwaway package in the scratch directory, installed with `pip install -e`, registers a filter through an entry point, and that filter appears in `sensoryforge list-components` and passes the contract test.

---

## 5. Phase 1 exit criteria (from the roadmap)

- Wheel installed in a clean environment runs `sensoryforge run` on a canonical example and the GUI imports offscreen.
- `pytest -m "not gui"` and `pytest -m gui` both exit 0 in one process each.
- CI workflow commands all succeed locally.
- Parity: GUI and engine resolve identical parameters (A4); golden parity test green (E5) or its blocker reported.
- `mkdocs build --strict` passes.
- Ledger: F-003, F-006, F-007, F-008, F-014, F-015, F-016, F-018, F-020, F-033, F-034 closed or explicitly reported as blocked (F-014, F-023, F-025–F-032 closed in Waves A and B).

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

# Non-GUI suite (until C1 adds markers)
$PY -m pytest tests/unit tests/integration tests/regression -q -p no:cacheprovider \
  --deselect tests/unit/test_expert_mode.py --deselect tests/unit/test_gain_defaults.py \
  --deselect tests/unit/test_grid_population_ux.py --deselect tests/unit/test_gui_agent_d.py \
  --deselect tests/unit/test_phase3_features.py --deselect tests/unit/test_population_csv.py \
  --deselect tests/unit/test_stimulus_grid_inmemory.py --deselect tests/unit/test_stimulus_tab_gui.py \
  --deselect tests/unit/test_stimulus_tab_ux.py --deselect tests/unit/test_unified_workflow.py
# Baseline at e66f529: 682 passed, 6 skipped

# Qt files, one at a time, streamed (until C2/C3 land)
for f in test_expert_mode test_gain_defaults test_grid_population_ux test_gui_agent_d \
         test_phase3_features test_population_csv test_stimulus_grid_inmemory \
         test_stimulus_tab_ux test_unified_workflow test_stimulus_tab_gui; do
  <scratch>/memwatch.sh 2500 180 <scratch>/$f.log -- $PY -m pytest tests/unit/$f.py -v -p no:cacheprovider --tb=line
  echo "$f pass=$(grep -c ' PASSED' <scratch>/$f.log) fail=$(grep -c ' FAILED' <scratch>/$f.log) err=$(grep -c ' ERROR' <scratch>/$f.log)"
done
# Baseline at e66f529: 8, 5, 12, 3 (+1 skip), 28, 7, 5, 17, 36, 102 passed; 0 failed
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
