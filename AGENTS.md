# AGENTS.md

SensoryForge simulates sensory encoding: a stimulus is sampled by a sensor array, passed
through receptive fields, temporal filters and neuron models, and comes out as population
activity. Worlds and data sets generate stimuli at scale, and `sensoryforge/evaluate`
audits what was generated. Touch is the first use, not the only one. Python and PyTorch.

This is the front door for any coding agent. It is short on purpose: each fact lives in
one place and this file points to it. Claude Code reads `CLAUDE.md`, which imports this
file.

## Set up, test, lint

```bash
conda env create -f environment.yml && conda activate sensoryforge
pip install -e ".[dev]"                    # never from a git worktree: use PYTHONPATH (F-053)
pytest tests/unit -q                       # fast tests; the full suite takes about 8 minutes
pytest tests/unit/test_filters.py -v       # one module
black sensoryforge tests && flake8 sensoryforge tests   # what CI checks
mkdocs build --strict                      # the docs, as CI builds them
sensoryforge --help                        # every command; `sensoryforge <command> --help`
python sensoryforge/gui/main.py            # the GUI
```

GUI tests need `QT_QPA_PLATFORM=offscreen` without a display; set `OMP_NUM_THREADS=1` for
reproducible timings. More commands: `CONTRIBUTING.md`, `docs/user_guide/cli.md`.

To *use* SensoryForge through an agent rather than develop it, the skills ship in
`sensoryforge/agents/`; start with `sensoryforge agents setup`
(`docs/user_guide/agent_setup.md`).

## Where things live

Terms are defined in `docs/agents/glossary.md`; a task in the user's words, with its files,
tests and rules, is in `docs/agents/task_index.md`.

| To work on | Code | Read first |
|---|---|---|
| Running a config (canonical engine, CLI, batch) | `sensoryforge/core/simulation_engine.py`, `sensoryforge/cli.py`, `sensoryforge/core/batch_executor.py` | `docs/developer_guide/architecture_notes.md` |
| Config schema and defaults | `sensoryforge/config/schema.py`, `sensoryforge/config/defaults.py` | `docs/user_guide/configuration_schema.md` |
| A new neuron, filter, stimulus, grid, solver, receptive-field builder or processing layer | `sensoryforge/registry.py`, `sensoryforge/register_components.py`, `sensoryforge new-component` | `docs/developer_guide/extensibility.md`, `docs/developer_guide/plugins.md` |
| Receptive fields | `sensoryforge/core/rf_bank.py`, `sensoryforge/core/rf_builders/` | `docs/user_guide/receptive_fields.md` |
| Event converters | `sensoryforge/neurons/event_encoders.py` | `docs/reference/converter_contract.md` |
| Worlds, sampling, data sets | `sensoryforge/world/` | `docs/user_guide/worlds.md`, `docs/reference/world_contract.md` |
| Data bundles (the files a run writes) | `sensoryforge/io/bundle.py` | `docs/user_guide/bundles.md` |
| Evaluating a data set (audit, compare, checks, server, export) | `sensoryforge/evaluate/` | `docs/user_guide/evaluate_pipeline.md`, `docs/reference/report_contract.md` |
| The GUI | `sensoryforge/gui/app.py`, `sensoryforge/gui/screens/`, `sensoryforge/gui/widgets/` | `docs/developer_guide/architecture_notes.md` |
| Shipped recipes and their gains | `sensoryforge/presets/`, `scripts/calibrate_recipe_gains.py` | `docs/user_guide/units_and_gains.md` |
| Tests and fixtures | `tests/unit/`, `tests/integration/`, `tests/contract/`, `tests/gui_v2/`, `tests/scale/`, `tests/conftest.py` | `CONTRIBUTING.md` |
| Performance | `benchmarks/` | `docs/reference/benchmarks.md` |
| Agent skills, setup and this file's siblings | `sensoryforge/agents/` | `docs/developer_guide/agent_files.md` |
| Decisions and open findings | `docs_root/LEDGER.md` | the ledger entries' ids (`F-`, `D-`) |

## Dangerous areas

- **Contracts with pressure-simulation**, which pins this code by git sha. A change that
  alters a world draw, frame, seed, entry id, bundle field or converter step needs a
  format-tag bump and its contract page updated: `docs/reference/world_contract.md`,
  `docs/reference/converter_contract.md`, `docs/reference/report_contract.md`. Read
  `.claude/rules/world-engine.md` and `.claude/rules/engine-parity.md` first.
- **Defaults come from one resolver** (`sensoryforge/config/defaults.py`). Never add a
  hard-coded copy of a filter, neuron or gain default.
- **Golden fixtures**: compare with `sensoryforge.testing.golden.assert_matches_golden`,
  not `torch.equal` (F-071).
- **Worktrees**: never `pip install -e .` from one (F-053); use `PYTHONPATH=<worktree>`.
- **The blind audit** (`sensoryforge/evaluate/audit`) never imports `sensoryforge.world`;
  a test enforces it.
- **GUI**: preferences only through `sensoryforge.gui.settings.gui_settings()` (F-072);
  connect Qt signals to bound methods, never a lambda or `functools.partial` over `self`
  (F-085); plots only through `sensoryforge/gui/widgets/plot_factory.py`.
- **Noise keys**: `sensor_noise_std` and `membrane_noise_std` are separate; `noise_std` is
  a deprecated alias (C-130).
- Peak memory from the watchdog is noise below about 2x (F-056); use `benchmarks/`.
- The rest, with reasons: `docs/developer_guide/architecture_notes.md` (Known Technical
  Debt).

## Conventions

- Units: ms, mm and mA at user-facing APIs; batch first (`[batch, time, ...]`);
  coordinates `(x, y)` in mm. Vectorise; no loops over neurons or spatial dimensions.
- Google-style docstrings with tensor shapes and units. `raise ValueError`, never
  `assert`, for input checks. Catch specific exceptions, never a bare `except Exception`.
- A module of 40 lines or more ends its docstring with tag lines (`Topic:`, `Tests:`,
  `Entry points:`, `Pinned by:`, `Do not change without:`); search them with `grep -rn
  "^Topic:" sensoryforge/`. `docs/developer_guide/file_headers.md`.
- Extend by registry or plugin; every component passes
  `sensoryforge.testing.contracts.check_component`, and every world shape, pattern,
  modulation, distribution and class kind passes
  `sensoryforge.testing.world_contracts.check_world_component`.
- Modality-neutral words in the core (sensor, channel, frame, population, draw, entry);
  tactile words only in tactile-specific code.
- Do not invent domain facts. Physics, models, parameters and units come from the user, a
  cited source the user approves, or reasoning the user approves.
- Every change ships its docs in `docs/` (mkdocs).

## Commits and the ledger

Conventional Commits (`feat:`, `fix:`, `docs:`, ...). Every commit carries a ledger
trailer in its last paragraph: `Decision:`, `Finding:`, `Opens:`, `Fixed:`, `Action:`
(with `Due:`), `Retires:`, `Closes:`/`Supersedes:`/`Refs: <id>`, or
`Ledger: none — <reason>`. `.githooks/commit-msg` enforces it; never pass `--no-verify`.
`docs_root/LEDGER.md` is the record of decisions and open findings: check it before
re-proposing something, and record a new open problem as `Opens:`.

## Documentation layout

- `docs/`: the published site (`mkdocs.yml`); `docs/development/` holds engineering
  records that are not published.
- `docs_root/`: internal notes and plans, gitignored except `docs_root/LEDGER.md` and
  `docs_root/DECISIONS.md`.
- `llms.txt`: an index of the published docs for tools that read it, generated by
  `scripts/build_llms_txt.py`.
