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

## D-5cdc524 · Sensor noise and membrane noise are separate; a design's declared noise is simulated · 2026-09-30

**What was decided.** Sensor (receptor-current) noise and membrane noise are separate SensoryForge parameters; a design directory's declared noise_std is simulated as sensor noise and membrane_noise_std as membrane noise

**Why.** pressure-simulation now declares its sensor noise as a specification (its DECISIONS §110):
each population record in `design.json` carries `noise_std` (mA, the declared design-unit floor
converted to the injected current at the design's gains; 3.59 mA SA and 37.9 mA RA on the default
40x40 design), `noise_seed`, and a separate `membrane_noise_std` (default 0). Two things stopped
SensoryForge from running that (pressure-simulation's C-130): `load_design` ignored the noise keys, so
every `run --design` was noise-free; and `PopulationConfig.noise_std` was the only noise key, used
both as the post-gain current noise (`_run_pop_from_drive`) and as the neuron's Langevin noise
(`build_neuron`), so "sensor noise on, membrane noise off" could not be expressed. The two are
different physical things: one is the receptor's noise, which the design's information numbers
assume and which shows up in the recorded `filtered` current; the other belongs to the spiking
stage and never appears in `filtered`.

**What was rejected.** (1) Keeping one key and accepting the coupling for noisy runs: a declared
membrane noise of 0 would then be silently replaced by 3.59 mA-equivalent Langevin noise on SA.
(2) Renaming `noise_std` outright: every existing config, bundle and preset override would change
meaning or break. It stays as a deprecated alias that sets both (a `FutureWarning` when non-zero;
an explicit new key overrides it), and `tests/integration/test_noise_split_golden.py` pins every
preset and two alias configs to spikes recorded with the pre-change code.

**Where it lives.** `sensoryforge/config/schema.py` (`PopulationConfig.sensor_noise_std`,
`membrane_noise_std`, `effective_sensor_noise_std()`, `effective_membrane_noise_std()`);
`sensoryforge/core/simulation_engine.py` (`build_neuron`, `run`); `sensoryforge/io/design.py`
(`load_design`); `docs/user_guide/bundles.md` § "The design's declared noise is simulated";
`tests/unit/test_sensor_noise_split.py`.

**Ledger id + sha.** D-5cdc524 · `7f7c6a3`

**Validation pending.** pressure-simulation's own sensor-noise parity test
(`tests/design/test_sensor_noise.py`) pins the old coupling and must be updated when its pin moves
to this commit; its `design.export.sensoryforge_config` workaround becomes redundant.

## D-673e0ed · The non-adapting AdEx RA also gets the low-threshold gain · 2026-09-28

**What was decided.** the AdEx recipe's RA gain is set like its SA gain -- RA fires from about 10% of the benchmark stimulus's rate of pressure change (10x the lowest gain at which the held benchmark's ramp makes RA fire) -- instead of matching TouchSim's RA sensitivity, so RA's rate stays proportional to the rate of change over the benchmark range; the Izhikevich recipe keeps the TouchSim-matched RA gain

**Why.** SA's low-threshold gain (D-8dde454) made SA sensitive, so the amplitude per mm fitted against TouchSim shrank to 0.21. Matching TouchSim's RA sensitivity relative to SA then pushed AdEx RA's gain to 740. There its onset rate over the benchmark ramp speeds (5-40 units/s) is compressed to 149-320 Hz near the refractory ceiling (R^2 0.81 against speed). At gains 100-200 it rises proportionally (R^2 0.94). pressure-simulation needs RA proportional to the rate of pressure change (D-6bae4df), and firing from 10% of the benchmark's rate still detects small movements.

**What was rejected.** The TouchSim-matched gain (740), for the compression above.

**Where it lives.** `scripts/calibrate_recipe_gains.py` (`RA_RULES`, `choose_ra_gain_low_threshold`); `sensoryforge/presets/tactile_sa1_ra1_adex.yml`.

**Ledger id + sha.** D-673e0ed · `df3320e` (decision)

**Validation pending.** How well pressure-simulation's Kalman filter decodes with these gains.

## D-8dde454 · Izhikevich keeps its adaptation; the non-adapting AdEx SA gets a low threshold · 2026-09-28

**What was decided.** the Izhikevich recipe keeps its TouchSim-fitted spike-frequency adaptation (d 15 for SA, 24 for RA), because without it SA is too steep to be graded; the AdEx recipe, pressure-simulation's model, stays non-adapting, with SA's gain set so SA fires from about 10% of the benchmark pressure (10x the lowest gain at which a held benchmark stimulus fires), accepting rates above P5's band for a held stimulus

**Why.** Measured with a 1 mm probe after removing adaptation. AdEx at the P5 gain was silent below about 40% of the benchmark pressure (0, 0, 0, 18, 35, 55 ... 107 Hz from 0.1 to 2.0). At about 3-4x the gain it fires from about 10% and rises roughly linearly. Izhikevich without adaptation is too steep: at its P5 gain it is silent below about 60%, and lowering the threshold pushes it to 470-800 Hz. Both choices are yours (2026-09-28). The drive is linear in both gain and pressure, so a held benchmark at 10% of its amplitude with gain g drives the neuron exactly as the full benchmark does with gain g/10. The rule therefore finds the lowest firing gain g0 by bisection and takes 10 g0.

**What was rejected.** Keeping P5's 45 Hz held-stimulus gain for AdEx (a dead zone below 40% of the benchmark pressure). A non-adapting Izhikevich recipe (no gain gives it both a low threshold and physiological rates).

**Where it lives.** `scripts/calibrate_recipe_gains.py` (`SA_RULES`, `choose_sa_gain_low_threshold`); `sensoryforge/presets/tactile_sa1_ra1.yml` (`d`), `tactile_sa1_ra1_adex.yml` (gains); `tests/integration/test_recipe_calibration.py`.

**Ledger id + sha.** D-8dde454 · `67c45b5` (decision)

**Validation pending.** A held benchmark stimulus now drives AdEx SA above P5's 20-100 Hz band (about 125 Hz). The Kalman filter's behaviour with these rates is untested.

## D-ce22df3 · Simple neurons by default; adaptation and the clamp opt-in · 2026-09-28

**What was decided.** the tactile recipes' neurons have no spike-frequency adaptation and no voltage clamp by default -- AdEx SA1_tonic and RA1_phasic get a = b = 0, the Izhikevich recipe sets d = 0, and v_floor defaults to none on Izhikevich and AdEx; adaptation (the TouchSim-fitted values, kept as AdEx presets SA1_adapting and RA1_adapting and as documented Izhikevich d values) and the clamp are opt-in options

**Superseded in part by:** D-8dde454 · 2026-09-28 (the Izhikevich recipe keeps d 15 / 24)

**Why.** pressure-simulation is the priority (D-6bae4df). Its models should stay simple enough for hardware, and adaptation adds a history-dependent term between rate and pressure that its Kalman-filter inference would have to model. The clamp was a guard against Euler blow-up at coarse steps (D-007). The neurons now integrate at 0.05 ms by default (F-008), and the input floor (D-43dc520) keeps negative drive out of tactile neurons, so the recipes' voltage stays near rest without it. With the clamp off, SensoryForge's Izhikevich also matches pressure-simulation's unclamped neurons (F-037).

**What was rejected.** Keeping the TouchSim-fitted adaptation by default (the Kalman concern above). Keeping the clamp by default: it was never reached in a way that changed spikes, and it hid the AdEx adaptation problem (F-f59aa11) rather than solving it.

**Where it lives.** `sensoryforge/neurons/adex.py` (`ADEX_PRESETS`: `SA1_tonic`/`RA1_phasic` non-adapting, `SA1_adapting`/`RA1_adapting` opt-in; `v_floor` default `None`); `sensoryforge/neurons/izhikevich.py` (`v_floor` default `None`). MQIF keeps its -120 mV clamp; you named only Izhikevich and AdEx.

**Ledger id + sha.** D-ce22df3 · `5251e45` (decision)

**Validation pending.** A user integrating strongly negative input at a coarse step should set `v_floor`; nothing warns them.

## D-4e669b4 · RA answers release as strongly as indentation, by design · 2026-09-28

**What was decided.** RA's response to release as strong as to indentation (the RA filter is symmetric in the rate of change) is kept by design and documented as a known difference from TouchSim's RA

**Why.** It suits the pressure-simulation design: RA reports the magnitude of the rate of pressure change in either direction. TouchSim's RA releases at 0.67-0.8 of its onset and is silent at 0.2 mm (F-bad9126).

**Where it lives.** `sensoryforge/filters/sa_ra.py::RAFilterTorch` (eq. 8, `|dI/dt|`); documented in `docs/user_guide/units_and_gains.md`.

**Ledger id + sha.** D-4e669b4 · `5251e45` (decision)

**Validation pending.** None -- settled.

## D-407c639 · A larger gabor default · 2026-09-28

**What was decided.** the named gabor stimulus's default sigma and wavelength become 1.0 mm (those of the layered gabor shape), so a default gabor spans many receptors of a 0.15 mm grid

**Why.** At sigma 0.3 mm and wavelength 0.5 mm, a default gabor covered about two receptors of the recipes' 0.15 mm grid. It drove tactile_sa1_ra1 to 54 SA and 0 RA spikes in 500 ms, against 1214 and 82 for a default gaussian (F-d33d335). At 1.0 mm, 73 receptors of a 41 x 41 grid sit above half its peak.

**What was rejected.** Changing `texture`'s defaults too. It uses the same class but its own render-level defaults (2.0 mm), and it was not the problem.

**Where it lives.** `sensoryforge/stimuli/texture.py::GaborTexture` (constructor and ParamSpec defaults).

**Ledger id + sha.** D-407c639 · `5251e45` (decision), `63ec27a` (change)

**Validation pending.** None -- settled.

## D-43dc520 · A tactile afferent's neuron input is floored at zero · 2026-09-24

**What was decided.** the current a tactile afferent population's neuron receives is floored at 0 mA (PopulationConfig.input_floor, which resolves to 0 for SA, RA and SA2 populations and to no floor otherwise); the recorded filtered drive stays signed, so bundles and pressure-simulation's decoder see the same signal

**Why.** With SA fitted to SA1's ramp response (k2 = 8, D-f4d0967) and a gain that puts a held stimulus at 45 Hz, the SA current on a moving stimulus's trailing edge reached about -100 to -136 mA (F-96ff772). Izhikevich SA hit its -120 mV floor there. AdEx, with only a linear leak, would have fallen to about -1000 mV without its clamp. A mechanoreceptor's transduction current cannot reverse, so negative mechanical drive should silence the afferent, not hyperpolarize it.

The floor sits where the current enters the neuron (`SimulationEngine._run_pop_from_drive`, after gain and noise). The recorded `filtered` keeps its sign, and pressure-simulation's decoder reads velocity sign from it (F-001). It resolves by population: 0 mA for SA, RA and SA2 on a built-in neuron model. A DSL model, which may be an analog readout of a signed signal, gets none, and so does any other type. The vision preset sets `-.inf` explicitly, because it reuses the SA type label for non-tactile populations.

**What was rejected.** Rectifying the SA filter's output (`clip_to_positive`). The recorded, decoded signal must stay signed (F-001).

**Where it lives.** `sensoryforge/config/schema.py::PopulationConfig.input_floor`; `sensoryforge/config/defaults.py::resolve_input_floor`; `sensoryforge/core/simulation_engine.py::_run_pop_from_drive`; `tests/unit/test_input_floor.py`.

**Ledger id + sha.** D-43dc520 · `9e2b70a` (decision)

**Validation pending.** The legacy pipelines (`GeneralizedTactileEncodingPipeline`, `TactileEncodingPipelineTorch`) do not apply the floor. The GUI does not show the field yet.

## D-f5853a4 · AdEx gets an absolute refractory period · 2026-09-24

**What was decided.** AdEx neurons get an absolute refractory period t_ref (default 0 ms, so existing configs are unchanged); the SA1_tonic and RA1_phasic presets use 2 ms, capping their rates near 500 Hz

**Why.** At the TouchSim-fitted sensitivity, AdEx RA reached 1200 Hz per afferent on `moving_edge` (six spikes in 5 ms), because AdEx has no refractory period. After a spike, `t_ref` holds the voltage at v_reset while the adaptation current keeps evolving. With 2 ms, a constant 400 mA drive gives 395 Hz instead of 880 Hz, and `moving_edge` peaks at 400 Hz. With `t_ref` = 0 the update is bit-identical to the code before the change.

**What was rejected.** Capping rates in the analysis (a scoring ceiling), which would have left the simulated spike trains unphysiological.

**Where it lives.** `sensoryforge/neurons/adex.py` (`t_ref`, `ADEX_PRESETS`); `tests/unit/test_adex_refractory.py`.

**Ledger id + sha.** D-f5853a4 · `9e2b70a` (decision)

**Validation pending.** Izhikevich has no refractory period and reaches 600-800 Hz on `moving_edge`. Its f-I curve is steep, and no decision covers it yet.

## D-f4d0967 · SA matches TouchSim's SA1 · 2026-09-24

**Update 2026-09-28 (D-ce22df3, D-8dde454):** the fitted adaptation stays in the Izhikevich recipe. For AdEx it moved to the opt-in presets `SA1_adapting`/`RA1_adapting`, and the AdEx recipe is non-adapting. The SA filter's `k2` = 8.0 is unchanged.

**Update 2026-09-24 (D-43dc520, D-f5853a4):** with the neuron-input floor and AdEx's refractory period, the AdEx recipe's gains became SA 370 / RA 490. AdEx now agrees with SA1 at every compared depth except 0.2 mm, where its hold is 2.9 Hz against 8.6 (one spike against three). Its hold ISI CV is about 0.5-0.6, against P5's 0.5. Izhikevich is unchanged.

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

**Update 2026-09-28 (D-673e0ed):** the Izhikevich recipe keeps the TouchSim-matched RA gain. The AdEx recipe's RA gain now follows the low-threshold rule instead.

**Update 2026-09-24 (D-43dc520, D-f5853a4):** RA's peak on `moving_edge` is now 400 Hz for AdEx (refractory period) and 600 Hz for Izhikevich. AdEx's RA gain is 490 against SA 370.

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

**Update 2026-09-28 (D-8dde454, D-673e0ed):** the AdEx recipe's SA and RA gains follow the low-threshold rule, not P5. The Izhikevich recipe keeps P5 for SA and TouchSim for RA.

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
| 2026-09-24 | D-7e71f68 | SA's calibrated gain puts the one held benchmark stimulus (ramp_gaussian's static hold) at 44.7 Hz, the centre of P5's band, with ISI CV below 0.5; the moving stimuli's SA rates are reported, not required to lie in the band, because SA now answers motion as TouchSim's SA1 does | `3b30da8` |
| 2026-09-24 | D-43dc520 | the current a tactile afferent population's neuron receives is floored at 0 mA (PopulationConfig.input_floor, which resolves to 0 for SA, RA and SA2 populations and to no floor otherwise); the recorded filtered drive stays signed, so bundles and pressure-simulation's decoder see the same signal | `9e2b70a` |
| 2026-09-24 | D-f5853a4 | AdEx neurons get an absolute refractory period t_ref (default 0 ms, so existing configs are unchanged); the SA1_tonic and RA1_phasic presets use 2 ms, capping their rates near 500 Hz | `9e2b70a` |
| 2026-09-24 | D-6bae4df | for now SensoryForge's tactile models serve the pressure-simulation project and stay simple enough to implement in hardware (Izhikevich or AdEx, no new model terms); the requirement is that SA and RA both fire, SA's rate proportional to pressure and RA's to its rate of change; fixing AdEx's adaptation voltage range and other SensoryForge-specific extensions are deferred | `c368ac7` |
| 2026-09-28 | D-ce22df3 | the tactile recipes' neurons have no spike-frequency adaptation and no voltage clamp by default -- AdEx SA1_tonic and RA1_phasic get a = b = 0, the Izhikevich recipe sets d = 0, and v_floor defaults to none on Izhikevich and AdEx; adaptation (the TouchSim-fitted values, kept as AdEx presets SA1_adapting and RA1_adapting and as documented Izhikevich d values) and the clamp are opt-in options | `5251e45` |
| 2026-09-28 | D-4e669b4 | RA's response to release as strong as to indentation (the RA filter is symmetric in the rate of change) is kept by design and documented as a known difference from TouchSim's RA | `5251e45` |
| 2026-09-28 | D-407c639 | the named gabor stimulus's default sigma and wavelength become 1.0 mm (those of the layered gabor shape), so a default gabor spans many receptors of a 0.15 mm grid | `5251e45` |
| 2026-09-28 | D-8dde454 | the Izhikevich recipe keeps its TouchSim-fitted spike-frequency adaptation (d 15 for SA, 24 for RA), because without it SA is too steep to be graded; the AdEx recipe, pressure-simulation's model, stays non-adapting, with SA's gain set so SA fires from about 10% of the benchmark pressure (10x the lowest gain at which a held benchmark stimulus fires), accepting rates above P5's band for a held stimulus | `67c45b5` |
| 2026-09-28 | D-673e0ed | the AdEx recipe's RA gain is set like its SA gain -- RA fires from about 10% of the benchmark stimulus's rate of pressure change (10x the lowest gain at which the held benchmark's ramp makes RA fire) -- instead of matching TouchSim's RA sensitivity, so RA's rate stays proportional to the rate of change over the benchmark range; the Izhikevich recipe keeps the TouchSim-matched RA gain | `df3320e` |
| 2026-09-30 | D-5cdc524 | Sensor (receptor-current) noise and membrane noise are separate SensoryForge parameters; a design directory's declared noise_std is simulated as sensor noise and membrane_noise_std as membrane noise | `7f7c6a3` |
<!-- DECISIONS_LOG_END -->
