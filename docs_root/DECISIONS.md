# Decisions record — the reasoning behind the ledger

**Level 2 of three.** `docs_root/LEDGER.md` holds the *fact* of each decision (one line, generated from
the commit trailer). This file holds the *reasoning*: why, on what evidence, against what
alternative. Level 3 — a deck script, a paper, the README's claims — is whatever audience surface
this repo has, if it has one; `.claude/ledger.conf` names it under `AUDIENCE_SURFACE=`, and it is
updated on request, never automatically.

A decision recorded here but not in the ledger is invisible to future sessions. A decision in the
ledger but not here is a verdict with no argument behind it. Write both.

**This file is append-only, and it keeps its history.** Every section is dated. A decision that is
later superseded is **never deleted and never edited**: it gains a `**Superseded by:**` line, and
the section that replaces it opens with `**Supersedes:**`. The corrections are the record — tidying
them away is the exact failure this system exists to prevent.

**How to add one.** `ledger-sync.sh` has already appended the dated fact to the **log** at the
bottom for you. Write the reasoning as a section here, newest first, using this template:

```markdown
## D-0NN · <short title> · <YYYY-MM-DD>

**Supersedes:** D-0MM · <date>          <!-- omit unless it replaces an earlier decision -->

**What was decided.** <quote the ledger entry verbatim — level 1 and level 2 must agree>

**Why.** <the evidence: numbers, measurements, the `F-`/`D-` ids that forced it. Not "it seemed
cleaner" — what was observed.>

**What was rejected.** <the alternative that was seriously considered, and the specific reason it
lost. This is the half that stops the question being re-opened in three months.>

**Where it lives.** <file:line, config key, module — where a reader verifies the decision is real>

**Ledger id + sha.** D-0NN · `<short sha>`

**Validation pending.** <what would still falsify or confirm this, or "none — settled">
```

And on the section it replaces, add one line — nothing else changes:

```markdown
**Superseded by:** D-0NN · <date>
```

---

# Sections

<!-- newest first; written by hand -->
<!-- SECTIONS_START -->

## D-ea0f017 · Tactile recipes get per-population gains, calibrated against P5 · 2026-09-24

**What was decided.** SensoryForge's tactile recipes give SA and RA their own input gains, calibrated on the responsive-set rate against the P5 bands over the four benchmark stimuli, with the drive scale reconciled with pressure-simulation's design-time model (its C-032)

**Why.** At the shared gain of 50, the Izhikevich SA baseline reached 8.75 Hz on the recipe's one genuine hold, against P5's 20-100 Hz (F-093). AdEx RA fired nothing on `drifting_grating`, whose onset drive peaks at 3.08 mA against a 7.57 mA rheobase (F-092). A gain sweep (`scripts/calibrate_recipe_gains.py`, 10% steps from 30 to about 400) showed that no single gain meets both populations' targets. Two rules were fixed before choosing, so the gains follow from P5 rather than from a preference:
- **SA:** the gain that puts the geometric mean of the four stimuli's responsive-set rates at the band's log centre, 44.7 Hz.
- **RA:** the geometric centre of the gains at which every onset burst reaches 150-400 Hz per afferent and a held stimulus is silent.

Results: `tactile_sa1_ra1` SA 220 / RA 61; `tactile_sa1_ra1_adex` SA 55 / RA 86. `tactile_stochastic_control` shares `tactile_sa1_ra1`'s gains, so the control arm still differs only in its receptive fields. Two methodological corrections came out of the sweep:
1. **The static hold was scored from the instant the ramp ended.** That counted RA spikes 1-6 ms later, the tail of the ramp response through the 8 ms RA filter, as hold firing. P5 allows spikes for the first ~30 ms of a static stimulus, so the hold is now scored from 30 ms after the ramp.
2. **F-092's objection to raising RA sensitivity does not hold.** It treated `moving_edge`'s steady interval as a hold, but the edge never stops moving there (F-089).

The drive-scale reconciliation needed no change on this side. Pressure-simulation's 150x (its C-032) was its decoder multiplying `input_gain` into SensoryForge's `filtered` array a second time. That array already includes the gain, and pressure-simulation fixed its decoder in `d448fd8`.

**What was rejected.** Leaving input gain entirely to pressure-simulation's design directories. The recipes must run sensibly without a design, and one shared gain cannot put SA and RA in their bands at once. Also rejected: tuning AdEx's membrane resistance R instead. R and `input_gain` scale the same current, and the gain is the knob every neuron model shares.

**Where it lives.** `sensoryforge/presets/tactile_sa1_ra1*.yml`, `tactile_stochastic_control.yml`; `scripts/calibrate_recipe_gains.py`; `benchmarks/results/recipe_calibration/`; `tests/integration/test_recipe_calibration.py`, which fails if either recipe leaves P5 or comes within 20 mV of `v_floor` (F-037).

**Ledger id + sha.** D-ea0f017 · `b4a853b` (decision), `63d37c3` (calibration)

**Validation pending.** P5 is the calibration target, so meeting it validates nothing by itself. The TouchSim comparison (`benchmarks/results/touchsim_comparison/`, `a2e8c0b`) is the independent check. It agrees on the Izhikevich SA rate-intensity curve and on RA's silent hold, but finds RA far less sensitive relative to SA than TouchSim's RA. That points at the RA gain, or at the RA filter, as the next thing to revisit.

## D-030 · The canonical stimulus block is what `sensoryforge run` renders · 2026-09-21

**What was decided.** the canonical stimulus: block is what sensoryforge run renders (via stimuli.render.render_for_config, shared with the GUI); a legacy stimuli: list still wins with a deprecation notice

**Why.** `sensoryforge run` chose its stimulus from the legacy top-level `stimuli:` list and never read `SensoryForgeConfig.stimulus`, so a canonical YAML exported from the GUI ran a default trapezoid instead of the stimulus it declared (the finding on `e79ff48`). Rendering through the same `render_for_config` the GUI uses makes GUI and CLI runs of one config identical, which `tests/integration/test_gui_engine_equality.py` pins.

**What was rejected.** Dropping the legacy `stimuli:` list outright: existing legacy configs would change behaviour silently. It keeps precedence when present, with a deprecation notice, until the legacy path is removed.

**Where it lives.** `sensoryforge/cli.py` (the stimulus selection, around the deprecation comment at line 321); `sensoryforge/stimuli/render.py::render_for_config`.

**Ledger id + sha.** D-030 · `e79ff48`

**Validation pending.** none — settled.

## D-029 · One receptor/receptive-field preview widget · 2026-09-21

**What was decided.** the receptor/receptive-field preview is one reusable widget (sensoryforge/gui/widgets/grid_preview.py) built from GridConfig via core.simulation_engine.build_grid, the same code path SimulationEngine uses

**Why.** The GUI audit (`docs_root/gui_audit/10_plan.md`) found the old GUI drew grids through private code that differed from the engine: the Spiking tab sampled receptors with a flat reshape the engine does not use (F-076), and `poisson`/`hex` grids crashed config paths. A preview built by `build_grid` shows exactly the receptors a run uses, for every arrangement.

**What was rejected.** Keeping the preview inside the Mechanoreceptor tab's drawing code: that tab was deleted in Phase 3, and its preview was bound to the tab's private dataclasses rather than `GridConfig`. The drawing code was lifted, not rewritten.

**Where it lives.** `sensoryforge/gui/widgets/grid_preview.py`; `sensoryforge/core/simulation_engine.py::build_grid`.

**Ledger id + sha.** D-029 · `c4d6317`

**Validation pending.** none — settled.

## D-027 · Every pyqtgraph widget is built by plot_factory · 2026-09-17

**What was decided.** every pyqtgraph widget in GUI v2 is built by sensoryforge/gui/widgets/plot_factory.py; pyqtgraph signals are connected only through plot_factory.connect (functools.partial, no bound methods or widget-closing lambdas) and released by teardown()

**Why.** F-035: with the cyclic GC on, the old GUI segfaulted (3 of 3 runs) in `ScatterPlotItem.renderSymbol` through a ViewBox lambda left by a destroyed tab, and the test suite had to turn the GC off, hiding the whole crash class. Signal wiring that holds no reference back to a widget removes that failure mode; putting all construction in one module gives it one place to be enforced. A 50-times build/destroy test under `gc.collect()` proves it (`tests/gui_v2/test_plot_factory.py`), and the suite now runs with the GC on.

**What was rejected.** Fixing each offending lambda in place: the audit also found three separate plotting paths (F-063), so the hazard would have come back with the next plot.

**Where it lives.** `sensoryforge/gui/widgets/plot_factory.py` (`connect`, `teardown`).

**Ledger id + sha.** D-027 · `0bfc3ea`

**Validation pending.** F-085 remains open: collecting pyqtgraph objects left in cycles by a destroyed window can still destroy a live ViewBox, mitigated by the test harness but not explained.

## D-026 · Forms are generated from get_param_spec() · 2026-09-17

**What was decided.** GUI v2 forms are generated from get_param_spec() by sensoryforge/gui/widgets/param_form.py and bind to the Session by dotted path; no per-component GUI code

**Why.** The audit found three large form tabs with private dataclasses that File > Load/Save barely reached, and parameters editable in two places. `get_param_spec()` is required on every component since G1, so a form generated from it covers built-ins and plugins alike, and a new plugin appears in the GUI with no GUI code. Binding by dotted path (`populations.1.filter_params.tau_r`) writes straight into the one `SensoryForgeConfig`, so there is nothing to translate on save.

**What was rejected.** Hand-written forms per component: each plugin would need GUI code, and each form would drift from the component's real parameters. The Circuit tab's `build_param_form` was the model and was lifted.

**Where it lives.** `sensoryforge/gui/widgets/param_form.py`.

**Ledger id + sha.** D-026 · `8657bea`

**Validation pending.** none — settled.

## D-025 · Stimuli are rendered on a regular canvas and sampled at receptors · 2026-09-17

**What was decided.** stimuli are rendered on a regular canvas of the array's extent (sensoryforge/stimuli/canvas.py) for every arrangement and sampled at receptor coordinates by SimulationEngine; the "grid" canvas is bit-identical to ReceptorGrid.get_coordinates()

**Why.** `poisson` and `hex` arrangements crashed every config-driven path (CLI, BatchExecutor, Circuit tab) at stimulus render, because they built a `ReceptorGrid` only to call `get_coordinates()`, which raises when an arrangement has no lattice (`04ab68a`). The engine already samples frames at each receptor's real position (`grid_sample`, Wave L), so a regular canvas over the array's extent serves every arrangement. Keeping the `grid` canvas bit-identical preserves the golden fixtures.

**What was rejected.** Giving each irregular arrangement a lattice of its own: `poisson` and `hex` have none that matches their receptors, and any invented one would be sampled anyway.

**Where it lives.** `sensoryforge/stimuli/canvas.py`; `SimulationEngine`'s receptor sampling.

**Ledger id + sha.** D-025 · `04ab68a`

**Validation pending.** none — settled.

## D-022 · One light theme · 2026-09-17

**What was decided.** GUI v2 uses one light theme (sensoryforge/gui/theme.py): app #F4F5F7, panels #FFFFFF, accent #2563EB, population colours SA #2563EB / RA #E8630A; no dark mode

**Why.** The audit's visual findings included a dark DockArea beside white heatmaps. One palette removes that mismatch. Light was chosen because the existing light tab was the best-received one, and figures exported from a light GUI match print. SA and RA get fixed, distinguishable colours so a population keeps its colour across every screen.

**What was rejected.** A dark theme, or both themes: listed as an explicit non-goal in the GUI v2 plan, since two themes double the styling that has to be checked.

**Where it lives.** `sensoryforge/gui/theme.py`.

**Ledger id + sha.** D-022 · `5369bc9`

**Validation pending.** none — settled.

---

# Log — every recorded decision, in order

Appended automatically by `.claude/hooks/ledger-sync.sh` from `Decision:` and `Retires:` trailers,
so level 2 always holds at least the dated fact even before anyone writes the reasoning above.
Do not hand-edit between the markers.

| Date | Id | One line | Commit |
|---|---|---|---|
<!-- DECISIONS_LOG_START -->
| 2026-09-22 | D-039 | upgrade the living-ledger template from v1 to v3 — enforced commit trailers, automatic post-commit ledger sync, Refs: backlinks, the level-2 decisions record, stale-rule detection, automatic cross-repo index push | `36d2e3f` |
| 2026-09-24 | D-0437899 | every stimulus defaults to peak amplitude 1.0, pressure-simulation's convention: the named gaussian, texture and moving types and their layered presets change from 30 to 1.0, and no stimulus renders negative values by default (gabor, texture) | `b4a853b` |
| 2026-09-24 | D-ea0f017 | SensoryForge's tactile recipes give SA and RA their own input gains, calibrated on the responsive-set rate against the P5 bands over the four benchmark stimuli, with the drive scale reconciled with pressure-simulation's design-time model (its C-032) | `b4a853b` |
| 2026-09-24 | D-4aafcdc | the quantitative afferent comparison uses touchsim output generated once in a throwaway environment and committed as fixture data; touchsim never becomes a dependency | `b4a853b` |
| 2026-09-24 | D-88b4b41 | GridConfig.density sets the receptor count of poisson, hex and blue_noise layouts (density times the rows x cols x spacing extent); setting it on grid or jittered_grid, where spacing fixes the count, is an error | `b4a853b` |
<!-- DECISIONS_LOG_END -->
