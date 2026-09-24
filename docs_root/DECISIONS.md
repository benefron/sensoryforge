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

## D-f4d0967 · SA matches TouchSim's SA1 · 2026-09-24

**What was decided.** SensoryForge's SA matches TouchSim's SA1 in its ramp (dynamic) response and in a graded rise of rate with indentation, for both the Izhikevich and the AdEx recipe

**Why.** The TouchSim comparison (`a2e8c0b`, F-20e111e) found two problems. SA's ramp response was 2-3x weaker than SA1's (hold/onset about 0.55 against 0.22). AdEx SA had a hard threshold: silent up to the 0.7 mm level, then 40 Hz. `scripts/validation/fit_afferents.py` searched grids scored by the RMS log error against SA1's onset and hold rates over 0.1-1.25 mm. The amplitude per mm was refitted at every point (the model is linear up to the neuron, so this absorbs SA's gain). The search found two levers:
- **The ramp response comes from the SA filter's `k2`,** the gain on the input's rate of change, the only velocity-sensitive term. Parvizi-Fard et al. (2021) used 3.0. Izhikevich's best is 10 and AdEx's is 5. **8.0** costs each about one spike at one depth, and it keeps one SA filter for both recipes and for pressure-simulation: its `design/drive.py` builds filters from the same resolver defaults, and its design directories cannot carry filter parameters (F-094).
- **The graded rise comes from spike-frequency adaptation.** Izhikevich `d` goes 8 -> 15, set in the recipe, because RS is Izhikevich's published preset. AdEx `SA1_tonic` gets b 0 -> 28, tau_w 200 -> 110 ms and v_reset -58 -> -70 mV.

Result (`benchmarks/results/touchsim_comparison/`): both recipes agree with SA1 on the hold rate, the onset rate and hold/onset at every compared depth. Izhikevich reaches 40/60/100/160 Hz onset against 40/60/100/180, and 8.6/17/26/40 Hz hold against 8.6/11/26/43.

Two consequences followed:
- **SA's gain rule changed.** With SA1-like dynamics, SA answers motion at several times its hold rate. On the AdEx recipe the static and moving benchmark rates (13 against 61-74 Hz) spread wider than P5's 20-100 Hz band, so the old rule could not be met: it wanted all four stimuli in the band. SA's gain now puts the one held stimulus (`ramp_gaussian`'s hold) at 44.7 Hz, which is how P5 states its SA band. The moving stimuli's rates are reported (Izhikevich 83-86 Hz, AdEx 167-196 Hz); TouchSim's SA1 itself reaches 180 Hz during a ramp.
- **The release current reaches the voltage floor.** The larger `k2` makes the negative SA current on a trailing edge larger. Izhikevich SA reaches its -120 mV floor there. Its spikes are identical with and without the clamp, which is what F-037 is about, and the recipe test now asserts exactly that.

**What was rejected.**
- **Tuning `k2` per recipe.** pressure-simulation runs AdEx on the default filters, so a recipe-only value would not reach its designs.
- **Adaptation alone, without `k2`.** It flattened the rate-intensity curve, but left the ramp response at about half of SA1's, because the drive is still rising during the ramp.
- **The first AdEx fit (b = 48-56).** It looked best, but in this AdEx form `w` enters dv/dt in mV, so it drove the voltage into the -130 mV clamp after every spike, and the clamp shaped the dynamics. Fits now reject any point whose hold voltage comes within 20 mV of the floor.

**Where it lives.** `sensoryforge/config/defaults.py::FILTER_DEFAULTS` (and `SAFilterTorch`'s defaults); `sensoryforge/neurons/adex.py::ADEX_PRESETS`; `sensoryforge/presets/tactile_sa1_ra1.yml` and `tactile_stochastic_control.yml` (`model_params.d`); `scripts/validation/fit_afferents.py`, `benchmarks/results/afferent_fit/`; `scripts/calibrate_recipe_gains.py`.

**Ledger id + sha.** D-f4d0967 · `ce166e0` (decision)

**Validation pending.** TouchSim is one model of SA1, not recordings. Real afferent data would be the stronger test. At its calibrated gains the AdEx recipe leaves the physiological voltage range (open entry), so AdEx SA's fit holds on the probe's range, not on the benchmark stimuli.

## D-d9bd411 · RA matches TouchSim's RA sensitivity · 2026-09-24

**What was decided.** SensoryForge's RA matches TouchSim's RA in sensitivity relative to SA, firing at the small indentations where TouchSim's RA fires, so small movements are detected; TouchSim's RA, not P5 alone, sets RA's calibration

**Why.** At intensities matched on SA's hold rate, RA stayed silent at 0.2-0.7 mm, where TouchSim's RA fires 20-60 Hz at onset (F-47cc697). RA's job is to report small movements. Because the amplitude per mm absorbs SA's gain, TouchSim constrains RA's gain relative to SA's.
- **RA's gain:** `scripts/calibrate_recipe_gains.py` sweeps it on a 15% grid at the calibrated SA gain, and takes the geometric centre of the gains whose onset rates best match TouchSim's RA at 0.1-1.25 mm. The 0.1 mm depth, where TouchSim's RA is silent, penalises an RA that is too sensitive.
- **RA's adaptation:** it makes RA's rate grow gradually with ramp speed instead of saturating a few spikes above threshold. Izhikevich `d` goes 2 -> 24; AdEx `RA1_phasic` gets b 20 -> 40 and v_reset -58 -> -70 mV.

Result: RA's onset rates match TouchSim's at every depth: Izhikevich 0/20/40/60/100 Hz, identical to TouchSim; AdEx 0/20/40/60/120 Hz. Gains: Izhikevich 410 against SA 380; AdEx 660 against SA 500.

**What was rejected.** Keeping P5's RA rule, the centre of the gains whose bursts stay within 150-400 Hz, which set RA's sensitivity from the benchmark stimuli alone and left RA blind to small indentations. P5's RA criteria are now reported as a check. The silent hold still holds. The burst band does not: on the fast `moving_edge`, RA reaches 600 Hz (Izhikevich) and 1200 Hz (AdEx).

**Where it lives.** `scripts/calibrate_recipe_gains.py::choose_ra_gain_touchsim`; `scripts/validation/compare_with_touchsim.py::ra_onset_error`; the three tactile presets.

**Ledger id + sha.** D-d9bd411 · `ce166e0` (decision)

**Validation pending.** Two points stay open:
- **RA's release.** It is as strong as its onset, because the RA filter responds to the absolute rate of change. TouchSim's RA releases more weakly and is silent at 0.2 mm (open entry).
- **Peak rates on fast stimuli** exceed what afferents reach, AdEx's especially, which has no refractory period (open entry).

## D-ea0f017 · Tactile recipes get per-population gains, calibrated against P5 · 2026-09-24

**Superseded in part by:** D-d9bd411 · 2026-09-24 (RA's gain rule; SA's is refined in D-f4d0967's section)

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
| 2026-09-24 | D-d9bd411 | SensoryForge's RA matches TouchSim's RA in sensitivity relative to SA, firing at the small indentations where TouchSim's RA fires, so small movements are detected; TouchSim's RA, not P5 alone, sets RA's calibration | `ce166e0` |
| 2026-09-24 | D-f4d0967 | SensoryForge's SA matches TouchSim's SA1 in its ramp (dynamic) response and in a graded rise of rate with indentation, for both the Izhikevich and the AdEx recipe | `ce166e0` |
<!-- DECISIONS_LOG_END -->
