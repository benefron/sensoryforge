# Custom Layouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A layout written once as a grid plugin can be named in a config for sensors, for
neurons, or both.

**Architecture:** One module, `sensoryforge/core/layouts.py`, turns a registered layout name,
an area and its settings into checked `[N, 2]` points. The engine uses it for both uses and
the CLI uses the same checks. Built-in names keep their present code paths.

**Tech Stack:** Python, PyTorch, pytest, `GRID_REGISTRY`, `BaseGrid`.

**Spec:** `docs/development/specs/2026-11-27-custom-layouts-design.md`

## Global Constraints

- Built-in sensor and neuron layouts stay bit-identical; golden files are not touched.
- No change to a world draw, frame, seed, entry id or converter step. If one is needed, stop.
- New config keys: `grids[].layout_params`, `populations[].neuron_layout_params`; empty by
  default and left out of `to_dict()` when empty.
- A name is matched exactly as registered. Built-in names: `grid`, `poisson`, `hex`,
  `jittered_grid`, `blue_noise` (sensors also `composite`).
- Points are never moved or dropped; bad points raise `ValueError` naming the layout.
- `raise ValueError`, never `assert`, for input checks; Google docstrings with shapes and mm.
- Run with `PYTHONPATH=$PWD OMP_NUM_THREADS=1`; every commit has a ledger trailer.

## Review Focus

- A custom sensor layout with a non-square area or `rows != cols`: x and y must not swap.
- `batch` with two sensor grids, one custom: it must run (the older pipeline is also built).
- `evaluate` checks on a data set that uses a custom layout: skipped or passing, never `error`.
- A config naming a layout whose plugin is not installed: the error names it; nothing falls back.
- A layout that returns float64 or points exactly on the area's edge: accepted.

---

## Stage 1: core

### Task 1: The shared layout builder

**Files:** Create `sensoryforge/core/layouts.py`; Test `tests/unit/test_custom_layouts.py`;
shared test layouts in `tests/unit/layout_plugins.py`.

**Interfaces (produces):**
- `BUILTIN_LAYOUTS: Tuple[str, ...]` (the five names)
- `custom_layout_names() -> List[str]`: registered names whose constructor accepts `xlim`,
  `ylim` and `device`
- `layout_problem(name: str, params: Mapping[str, Any]) -> Optional[str]`: unknown name,
  constructor without the area, unknown or missing setting, a reserved setting
  (`xlim`, `ylim`, `device`, `seed`)
- `build_layout_points(name, xlim, ylim, *, params=None, seed=None, require_seed_support=False,
  device="cpu") -> torch.Tensor` `[N, 2]` float32 mm

- [ ] Failing tests: returns the plugin's points as float32; passes `seed` only to a layout
  that takes it; `require_seed_support=True` with a seed and no `seed` parameter raises;
  each `layout_problem` case raises with the layout's name; shape `[N, 3]`, empty, NaN and
  outside-area points raise; points on the edge and float64 output are accepted; a built-in
  or unknown name raises listing built-ins and custom names.
- [ ] Run, watch them fail; implement; run, pass.
- [ ] Commit `feat(layouts): build and check the points of a registered layout`.

### Task 2: Config keys

**Files:** Modify `sensoryforge/config/schema.py` (`GridConfig`, `PopulationConfig`),
regenerated `schemas/`; Test `tests/unit/test_custom_layouts.py`.

**Interfaces (produces):** `GridConfig.layout_params: Dict[str, Any]`,
`PopulationConfig.neuron_layout_params: Dict[str, Any]`, both `field(default_factory=dict)`.

- [ ] Failing tests: both round-trip through `to_dict()`/`from_dict()`; neither key appears
  in `to_dict()` when empty; settings with a built-in name raise `ValueError` at construction.
- [ ] Implement, regenerate with `scripts/build_schemas.py`, pass, commit.

### Task 3: Custom neuron layouts in the engine

**Files:** Modify `sensoryforge/core/simulation_engine.py` (`_build_input_bank`); Test
`tests/unit/test_custom_layouts.py`.

**Interfaces (consumes):** `build_layout_points`, `layout_problem`.

- [ ] Failing tests: the bank's `neuron_centers` equal the plugin's points and the neuron
  count is the plugin's, with `gaussian`; spikes have that width after `run`; the area given
  is the grid bounds shrunk by `edge_offset`; `UserWarning` naming each of `neurons_per_row`/
  `neuron_rows`/`neuron_cols`/`neuron_jitter_factor` the user set, none otherwise; with a
  builder that derives centres a warning containing "derives its own neuron lattice"; two
  inputs share one set of centres; an unknown name raises naming it.
- [ ] Implement: built-in names keep the `create_neuron_centers` call unchanged.
- [ ] Pass, commit.

### Task 4: Custom sensor layouts in the engine

**Files:** Modify `sensoryforge/core/simulation_engine.py` (`build_grid`, `_build_grids`);
Test `tests/unit/test_custom_layouts.py`.

- [ ] Failing tests: receptor coordinates equal the plugin's points; bounds equal the
  rectangle from `rows`, `cols`, `spacing`, centre (rows along x); **with `rows=12`,
  `cols=8` and a Gaussian wider in x than y, a plugin returning the exact lattice points
  gives the same filtered output as arrangement `grid`** (`allclose`, `atol=1e-5`);
  `density`, `coords_file`, a zero-width area and `GridConfig.seed` on a layout without
  `seed` each raise.
- [ ] Implement: a `CompositeReceptorGrid` with one layer, bounds = declared area, and
  `provenance={"layers": [...], "source": "layout", "layout": name}`; no `GridManager`.
- [ ] Pass, commit.

### Task 5: CLI, older pipeline and batch

**Files:** Modify `sensoryforge/cli.py` (`validate_config`),
`sensoryforge/core/generalized_pipeline.py` (`_canonical_to_legacy_config`),
`sensoryforge/core/batch_executor.py` (`__init__`); Test `tests/unit/test_custom_layouts.py`.

- [ ] Failing tests: `validate_config` accepts a registered layout for neurons and sensors,
  and rejects an unknown name and a bad setting with `layout_problem`'s message; the older
  pipeline built from a canonical config with a custom name raises naming it; `BatchExecutor`
  with two grids, one custom, writes its bundles, and its frames equal those of the same
  config with `grid` layouts.
- [ ] Implement: the executor gives the older pipeline (stimulus generation only) the first
  grid alone with the layout keys removed; the engine still gets the full config.
- [ ] Pass, commit.

### Task 6: Contract check, scaffold, example and docs

**Files:** Modify `sensoryforge/testing/contracts.py` (`_check_grid`),
`sensoryforge/scaffold.py` (grid templates), `docs/examples/grid_arrangement_plugin.py`,
`docs/how-to/extend/add_grid_arrangement.md`, `docs/reference/configuration_schema.md`,
`CHANGELOG.md`, generated reference pages and `llms.txt`; Tests
`tests/unit/test_contract_checks.py`, `tests/unit/test_scaffold.py`,
`tests/docs/test_docs_examples.py`.

- [ ] Failing tests: `check_component("grid", ...)` fails a layout that takes the area but
  puts a point outside it, or ignores its seed; a grid plugin without `xlim` still passes;
  the scaffolded grid builds through `build_layout_points`.
- [ ] Implement; the spiral example takes `xlim`/`ylim` and runs as sensor and neuron
  layout with no `coords_file`; the how-to covers both uses and the `coords_file` route.
- [ ] Pass, commit.

### Task 7: End to end

**Files:** Test `tests/unit/test_custom_layouts.py`.

- [ ] Failing-or-pinning tests: a YAML config runs through `sensoryforge run` with the spiral
  as sensor layout, neuron layout and both; the saved bundle's `config` rebuilds an engine
  with identical spikes; `evaluate` engine checks on such a data set report no `error`.
- [ ] Run the full suite, `black`, `flake8`, `mkdocs build --strict`. Commit.

## Stage 2: the app

### Task 8: Lists, settings and missing plugins

**Files:** Modify `sensoryforge/gui/screens/populations_cards.py`,
`sensoryforge/gui/screens/sensors.py`, `sensoryforge/gui/validation.py`,
`sensoryforge/config/schema.py` (`grid_config_param_specs`), `docs/how-to/gui_walkthrough.md`;
Tests under `tests/gui_v2/`.

- [ ] Failing tests: both lists hold built-ins then `custom_layout_names()`; choosing a custom
  layout shows a form from its `get_param_spec()` writing into the settings key; switching to
  a built-in clears those settings; rows, columns and per-row fields of neurons are disabled
  for a custom layout; sensor density is disabled; a config naming a missing layout keeps the
  name in the list and validation reports it.
- [ ] Implement, pass, full suite, commit.

## Not in this plan

Merging and pushing wait for the user. Sweeping layout settings, composite layers and the
other non-goals in the spec stay out.
