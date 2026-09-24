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

**Three levels of a decision.** A decision is captured at up to three levels, each for a different
reader, and the levels must agree:

| Level | Where | Reader | What it holds |
|---|---|---|---|
| 1. Fact | **this file**, via the `Decision:` trailer of the commit that enacted it | future sessions, the injected digest | one sentence, status, evidence pointer |
| 2. Reasoning | **`docs_root/DECISIONS.md`** — append-only, every section dated | you, collaborators | what was decided (quoting level 1), why — the numbers and the `F-`/`D-` ids, what was rejected and why, where it lives in code, what validation is still pending |
| 3. Audience | *only if this repo has an audience surface* — a deck script, a paper, the README's claims. `.claude/ledger.conf` names it under `AUDIENCE_SURFACE=` | that audience | the current state. Updated **on request**; nothing rebuilds it automatically, and it never rewrites history |

The order is fixed: the trailer goes on the commit → the post-commit hook writes level 1 → `ledger-sync.sh`
appends the dated fact to level 2's log → a human writes the reasoning above it. Level 3 moves only when
you ask for it.

A decision that exists at level 1 only is a verdict with no argument behind it. A decision that exists at
level 2 only is invisible to every future session. Both are incomplete.

**Level 2 is append-only and keeps its history.** A superseded decision is never deleted and never
edited: it gains a `**Superseded by:** <section / ledger id> · <date>` line, and the section that
replaces it opens with `**Supersedes:** …`. The corrections *are* the record.

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

## D-0437899 · CLOSED · decision · - · 2026-09-24
every stimulus defaults to peak amplitude 1.0, pressure-simulation's convention: the named gaussian, texture and moving types and their layered presets change from 30 to 1.0, and no stimulus renders negative values by default (gabor, texture)
· Rejected: keeping each type's own amplitude with per-type gain guidance | a user switching type would still have to know each type's scale, and 30 matches nothing pressure-simulation calibrates against
→ commit b4a853b

## D-ea0f017 · CLOSED · decision · - · 2026-09-24
SensoryForge's tactile recipes give SA and RA their own input gains, calibrated on the responsive-set rate against the P5 bands over the four benchmark stimuli, with the drive scale reconciled with pressure-simulation's design-time model (its C-032)
· Rejected: leaving input gain entirely to pressure-simulation's design directories | the recipe must run sensibly without a design, and one shared gain cannot put SA and RA in their bands at once
→ commit b4a853b

## D-4aafcdc · CLOSED · decision · - · 2026-09-24
the quantitative afferent comparison uses touchsim output generated once in a throwaway environment and committed as fixture data; touchsim never becomes a dependency
→ commit b4a853b

## D-88b4b41 · CLOSED · decision · - · 2026-09-24
GridConfig.density sets the receptor count of poisson, hex and blue_noise layouts (density times the rows x cols x spacing extent); setting it on grid or jittered_grid, where spacing fixes the count, is an error
· Rejected: removing GridConfig.density | irregular receptor layouts are naturally specified in receptors per mm2, the unit afferent densities are published in
→ commit b4a853b

## D-039 · CLOSED · decision · - · 2026-09-22
upgrade the living-ledger template from v1 to v3 — enforced commit trailers, automatic post-commit ledger sync, Refs: backlinks, the level-2 decisions record, stale-rule detection, automatic cross-repo index push
→ commit 36d2e3f

## F-094 · STANDING · finding · - · 2026-09-22
a design directory's filter_params/model_params must be empty -- the two repos do not share filter parameter names, so any value there makes the hand-off unloadable rather than merely redundant.
→ commit a05fb8b
· tidied 2026-09-24: a settled result, not an open problem

## D-038 · CLOSED · decision · - · 2026-09-22
SensoryForge accepts an externally designed encoder as a design directory (design.json + per-population npz) via `sensoryforge run --design`, and stamps the design manifest into the bundle.
→ commit 7eba10d

## F-095 · CLOSED · finding · - · 2026-09-22
RampGaussianStimulus did not clamp its ramp to the requested duration, so any run shorter than the 50 ms default ramp raised a tensor-size error.
→ commit 7eba10d
✓ closed by 4d93f5c chore(ledger): tidy

## D-035 · CLOSED · decision · - · 2026-09-22
AdEx SA/RA populations resolve to the SA1_tonic and RA1_phasic presets by neuron_type through the same resolve_neuron_params mechanism as Izhikevich's F-004 RS/FS split, rather than a second parallel resolver.
→ commit 326ef6f

## D-036 · CLOSED · decision · - · 2026-09-22
the AdEx recipe ships as a separate preset file (tactile_sa1_ra1_adex) rather than a neuron_model switch on tactile_sa1_ra1, so the Izhikevich golden parity fixture stays untouched.
→ commit 326ef6f

## D-037 · CLOSED · decision · - · 2026-09-22
the AdEx presets' operating point is set through R, the membrane resistance, tuned against the tactile recipe's measured responsive-neuron drive (SA rheobase 3.07 mA against a hold drive of p10/p50/p90 = 3.82/5.25/7.30 mA; RA rheobase 7.57 mA, below the 5.67-13.87 mA onset transients and above the ~0.15 mA hold drive); input_gain stays 50.0 and no preset YAML is touched.
→ commit 326ef6f

## F-086 · STANDING · finding · - · 2026-09-22
the four pressure-simulation benchmark stimuli already render at a common peak amplitude at the recipe defaults (ramp_gaussian 0.9944, moving_edge 1.0000, braille 0.9825, drifting_grating 1.0000, within 2% of each other) and none of them routes through render.py's _LEGACY_DEFAULTS, the source of F-083's 30x split, so no change is needed for these four (partial F-083 resolution, scoped to them only).
→ commit 326ef6f
· tidied 2026-09-24: a settled result, not an open problem

## F-087 · STANDING · finding · - · 2026-09-22
AdEx presets tuned against a flat constant-current bench step are silent in the tactile recipe -- their rheobase lay above the drive the recipe actually delivers -- so a bench probe is not a sufficient tuning target for a population that sees filtered tactile drive; tune against the measured drive instead.
→ commit 326ef6f
· tidied 2026-09-24: a settled result, not an open problem

## F-088 · STANDING · finding · - · 2026-09-22
the P5 SA rate criterion is only meaningful on a drive-derived responsive set -- on a spatially localized stimulus a whole-population mean cannot reach 20-100 Hz for any neuron model, Izhikevich included.
→ commit 326ef6f
· tidied 2026-09-24: a settled result, not an open problem

## F-089 · STANDING · finding · - · 2026-09-22
RA's silence during a hold comes mainly from the RA filter differentiating a static drive to ~0, not from AdEx adaptation alone -- moving_edge's steady-drive interval, where the edge never stops moving, still yields 594 RA spikes.
→ commit 326ef6f
· tidied 2026-09-24: a settled result, not an open problem

## F-090 · STANDING · finding · - · 2026-09-22
SA1_tonic's R = 6.0 was selected by scanning R for the smallest value whose responsive-set ISI CV cleared 0.5, so the reported CV of 0.470 is a fitted outcome rather than an independent check; the purely principled placement (rheobase at the measured hold drive's p10) gives R ~= 4.8, within about 25%.
→ commit 326ef6f
· tidied 2026-09-24: a settled result, not an open problem

## F-091 · STANDING · finding · - · 2026-09-22
the RA onset "burst" that passes P5 is a single spike per responsive afferent synchronized across the set, not a multi-spike burst within one afferent -- the 200 Hz peak is exactly the 5 ms bin's cap for one spike.
→ commit 326ef6f
· tidied 2026-09-24: a settled result, not an open problem

## F-092 · OPEN · finding · - · 2026-09-22
RA1_phasic fires nothing on drifting_grating (peak per-afferent rate 0 Hz against P5's ~300 Hz) because that stimulus's RA onset drive peaks at 3.08 mA, below the 7.57 mA rheobase; reaching it needs R >~ 20, which would leave too little margin over moving_edge's 6.06 mA RA hold drive.
→ commit 326ef6f
↔ b4a853b decide: stimulus amplitude, recipe gains, touchsim reference, grid density

## F-093 · OPEN · finding · - · 2026-09-22
under the corrected responsive-set metric the Izhikevich SA baseline reaches only 8.75 Hz on the recipe's one genuine hold, an order of magnitude below P5's 20-100 Hz band; whether that is the recipe's input_gain or the RS preset is not settled.
→ commit 326ef6f
↔ b4a853b decide: stimulus amplitude, recipe gains, touchsim reference, grid density

## D-034 · CLOSED · decision · - · 2026-09-22
stimuli are designed as layered stimuli (stimuli.layered): a stack of layers combined by sum or max, each a primitive shape (gaussian, disc, bar, grating, gabor) placed by a pattern (single, grid+mask, list, random, braille), moved (none, linear, circular, path) and timed explicitly (onset, ramp up, hold, ramp down); the named stimulus types stay registered and exact and are also offered as layered presets
→ commit 774e9c4

## D-033 · CLOSED · decision · - · 2026-09-22
every rendered stimulus changes over time: a still stimulus ramps up and down over one eighth of the run each unless ramp_up_ms/plateau_ms/ramp_down_ms are set, and stimuli with their own total_ms span the run unless it is set (stimuli.render.default_envelope, _clock_to_render_step)
→ commit b4afe01

## F-084 · CLOSED · finding · - · 2026-09-21
render_for_config ignored the stimulus's own clock, so with a run dt_ms other than 1 ms the tactile stimuli (moving_edge, braille, drifting_grating, ramp_gaussian) played at the wrong speed, and a timeline stimulus never advanced past its first sub-stimulus; fixed here, but results made with dt_ms != 1 ms from those stimuli, or with any timeline, were wrong
→ commit 6956f81

## F-085 · OPEN · finding · - · 2026-09-21
with the cyclic GC enabled, collecting pyqtgraph objects left in reference cycles by a destroyed window can destroy a live ViewBox of another window (measured in the test suite: RuntimeError "wrapped C/C++ object of type ViewBox has been deleted" in GridPreview.set_grids); the suite collects at test boundaries, and the app builds each plot once per window lifetime, but any future screen that discards and rebuilds pyqtgraph widgets at runtime would be exposed
→ commit 32646fb

## D-032 · CLOSED · decision · - · 2026-09-21
StimulusConfig.params (dict) stores stimulus-type parameters that have no named field and is forwarded to the constructor by render_for_config; sensoryforge.stimuli.render.effective_defaults(type) is the only source for the default a form displays, so the displayed default is the value that runs
→ commit 301af19

## F-081 · OPEN · finding · - · 2026-09-21
GridConfig.density is accepted and round-tripped but never read by core.simulation_engine.build_grid: every arrangement (grid, hex, poisson, jittered_grid, blue_noise) is sized from rows x cols x spacing, so a YAML density value silently does nothing (composite layers' own density is separate and is used)
→ commit 8674a6a
↔ b4a853b decide: stimulus amplitude, recipe gains, touchsim reference, grid density

## F-082 · CLOSED · finding · - · 2026-09-21
sensoryforge run ignored the config's simulation.duration_ms and always ran --duration (default 1000 ms); fixed here, but any earlier result produced from a config that set duration_ms without --duration was 1000 ms long
→ commit c65f3af

## D-029 · CLOSED · decision · - · 2026-09-21
the receptor/receptive-field preview is one reusable widget (sensoryforge/gui/widgets/grid_preview.py) built from GridConfig via core.simulation_engine.build_grid, the same code path SimulationEngine uses
→ commit c4d6317

## F-077 · CLOSED · finding · - · 2026-09-21
sensoryforge run chose its stimulus from the legacy top-level stimuli: list and never read SensoryForgeConfig.stimulus, so a canonical YAML exported from the GUI ran a default trapezoid instead of the stimulus it declared
→ commit e79ff48

## D-030 · CLOSED · decision · - · 2026-09-21
the canonical stimulus: block is what sensoryforge run renders (via stimuli.render.render_for_config, shared with the GUI); a legacy stimuli: list still wins with a deprecation notice
→ commit e79ff48

## F-078 · CLOSED · finding · - · 2026-09-21
render_for_config forwarded every StimulusConfig schema default to the stimulus constructor, overriding the type's own defaults, so a bare 'type: moving_edge' block rendered a static edge (start == end == [0, 0]) with no error
→ commit a123772

## D-031 · CLOSED · decision · - · 2026-09-21
every GUI v2 run goes through execution.run_controller.RunController, a QThread worker that renders with execution.render.render_for_config and calls SimulationEngine.run(progress_cb=...) on a config snapshot; cancel is cooperative; results land in Session.last_results and in a bundle under the project's runs/ directory
→ commit 461b4b7

## F-080 · CLOSED · finding · - · 2026-09-21
forwarding only stimulus fields that differ from the schema default discarded a value deliberately set equal to it (a Gaussian sigma of 2.0 ran as the type default 1.0), and the CLI's trapezoid fallback for a default-looking stimulus block made the same YAML run different stimuli in the GUI and on the CLI
→ commit ca78f62

## D-026 · CLOSED · decision · - · 2026-09-17
GUI v2 forms are generated from get_param_spec() by sensoryforge/gui/widgets/param_form.py and bind to the Session by dotted path; no per-component GUI code
→ commit 8657bea

## D-027 · CLOSED · decision · - · 2026-09-17
every pyqtgraph widget in GUI v2 is built by sensoryforge/gui/widgets/plot_factory.py; pyqtgraph signals are connected only through plot_factory.connect (functools.partial, no bound methods or widget-closing lambdas) and released by teardown()
→ commit 0bfc3ea

## D-028 · CLOSED · decision · - · 2026-09-17
GUI v2 is one window with a left stage navigation (Sensors, Stimulus, Populations, Run & Results, Batch), a pipeline strip showing sensor array → receptive field → filter → neuron → readout per population, and a bottom run bar; there are no tabs and no node graph
→ commit 24caa84

## D-022 · CLOSED · decision · - · 2026-09-17
GUI v2 uses one light theme (sensoryforge/gui/theme.py): app #F4F5F7, panels #FFFFFF, accent #2563EB, population colours SA #2563EB / RA #E8630A; no dark mode
→ commit 5369bc9

## F-074 · CLOSED · finding · engine · 2026-09-17
PopulationConfig.noise_seed was read only by the legacy adapter (core/generalized_pipeline.py) and ignored by SimulationEngine, and neither the CLI nor the GUI seeded the RNG, so noise_std > 0 runs were reproducible only through BatchExecutor
→ commit cb04bf0

## D-023 · CLOSED · decision · - · 2026-09-17
SimulationConfig.seed seeds torch/numpy/random at the start of SimulationEngine.run(); PopulationConfig.noise_seed drives a per-population torch.Generator for membrane noise; run() takes an optional progress_cb(index, n, name) called once per population
→ commit cb04bf0

## D-024 · CLOSED · decision · - · 2026-09-17
GUI v2 holds exactly one SensoryForgeConfig in a Session(QObject) that emits configChanged(dotted path); every screen binds to it; a project is a directory with config.yml and runs/<bundle dirs>, no other GUI persistence format
→ commit d774217

## F-075 · CLOSED · finding · stimuli · 2026-09-17
poisson and hex receptor arrangements crashed every config-driven path (CLI, BatchExecutor, Circuit tab) at stimulus render because they built a ReceptorGrid only to call get_coordinates(), which raises when the arrangement has no lattice
→ commit 04ab68a

## D-025 · CLOSED · decision · - · 2026-09-17
stimuli are rendered on a regular canvas of the array's extent (sensoryforge/stimuli/canvas.py) for every arrangement and sampled at receptor coordinates by SimulationEngine; the "grid" canvas is bit-identical to ReceptorGrid.get_coordinates()
→ commit 04ab68a

## F-076 · CLOSED · finding · gui · 2026-09-17
the GUI Spiking Neurons tab renders stimuli with its own third renderer (spiking_tab.py:2027) and samples receptors with an unconditional flat reshape (spiking_tab.py:2596), so for any arrangement other than a resolution-matched regular grid its neuron drive differs from what SimulationEngine.run() computes for the same config; tests/unit/test_gui_rf_banks.py:164 encodes the same flat reshape as its expected value
→ commit a3b1607

## F-073 · CLOSED · finding · - · 2026-09-17
(cited as F-072 in commit 4836b19 and the code comments, an id that was already taken) the GUI tests read and wrote the real per-user Qt preferences of whoever ran them, so a test passed where a collapsible section had once been saved expanded and failed on a fresh CI runner, and a test run could overwrite a developer's saved GUI state
→ commit 4836b19

## F-072 · CLOSED · finding · - · 2026-09-17
sensoryforge list-components never listed PROCESSING_REGISTRY, so a processing-layer plugin (or any processing layer) was invisible to that command even once correctly registered
→ commit 92d8ad0

## D-021 · CLOSED · decision · - · 2026-09-17
SensoryForge and pressure-simulation engines are held in parity by SensoryForge tests/integration/test_pressure_sim_parity.py (drive and filtered response to 1e-6, spikes exact, 8x8 grid, one SA and one RA population, non-negative stimulus) and test_stimulus_parity.py (four benchmark stimuli, zero tolerance); known exceptions are the -120 mV voltage clamp under strongly negative drive (SensoryForge F-037) and that parity has only been verified on macOS arm64 (SensoryForge F-071)
→ commit 52642b7

## F-068 · CLOSED · finding · - · 2026-09-17
F-068 the reproducibility test compared spike counts recorded on macOS arm64 exactly against runs on the Linux CI matrix, with no record of the reference's platform, and its docstring justified exact counts by the very cross-platform rounding differences that can flip a threshold-edge spike
→ commit 2eeac0e

## F-069 · CLOSED · finding · - · 2026-09-17
F-069 SimulationEngine warned that neurons_per_row, neuron_rows and neuron_cols were ignored whenever a lattice-deriving receptive-field builder was used, even when the user had set none of them, so every run of the shipped tactile_sa1_ra1 preset printed a warning about a value nobody chose
→ commit 2ce266f

## F-070 · OPEN · finding · - · 2026-09-17
F-070 the comparison against published afferent data is qualitative only -- SA sustains and RA adapts during a hold, checked against cited literature -- because touchsim cannot be installed here and no digitised Saal et al. (2017) or Izhikevich (2003) figure data was available, so no quantitative comparison with a published afferent model exists
→ commit 992e6ca
↔ b4a853b decide: stimulus amplitude, recipe gains, touchsim reference, grid density

## F-071 · CLOSED · finding · - · 2026-09-17
F-071 three zero-tolerance golden tests -- tests/integration/test_pressure_sim_parity.py, tests/integration/test_stimulus_parity.py and the receptive-field golden weights in tests/fixtures/rf_engine_golden_weights.pt -- compare against fixtures generated on macOS arm64 and have never run on another platform, so a failure on the Linux CI runner may be floating-point rounding rather than a regression and should be diagnosed before either loosening the test or changing code
→ commit 992e6ca

## F-067 · CLOSED · finding · - · 2026-09-17
F-067 the benchmark CI guard compared raw milliseconds from a baseline measured on an Apple M3 Pro against runs on a GitHub Linux runner, so a hardware difference alone could fail it; pinning the run to this machine's efficiency cores made the engine 4.56x slower in raw time, which fails the 3.0x limit with no code changed
→ commit 98a7593

## F-066 · CLOSED · finding · - · 2026-09-17
F-066 the Circuit screenshot test ran the generator against the committed docs/assets/gui directory and checked that same directory, so every GUI suite run rewrote tracked images and the test could not fail, because previously committed images satisfied its size check even when the script wrote nothing
→ commit 84dd282

## F-065 · CLOSED · finding · - · 2026-09-16
F-065 Wave O specified saving Circuit node positions to a sibling <config>.layout.json via Flowchart.saveState() but no part of it was implemented, so a graph's arrangement was lost on every reload and grep for saveState across sensoryforge/ returned nothing
→ commit d9e93e7

## F-064 · CLOSED · finding · - · 2026-09-16
StimulusDesignerTab.set_config() raises TypeError on any full get_config()-shaped dict (QSpinBox.setValue(float) at spin_edge_count, stimulus_tab.py _set_spin); pre-existing on d763bc9, uncaught because no test called set_config() directly.
→ commit fcf0c4a

## F-063 · CLOSED · finding · - · 2026-09-16
F-063 the Circuit inspector draws its sensor-array, stimulus and receptive-field previews with its own small widgets rather than reusing the Mechanoreceptor and Stimulus Designer plot widgets, because those share one plot item driving mouse-based population placement inside two 3,500-line tabs and could not be extracted safely in Wave P; the two drawing paths can now drift apart without any test noticing
→ commit 2f399c9

## F-061 · CLOSED · finding · - · 2026-09-16
F-061 the Circuit tab discovered which stimulus parameters a class accepts by retrying and deleting whatever the constructor rejected, so a field the user had deliberately set was dropped with no message and the graph then described a run that did not happen
→ commit 7a5ec87

## F-062 · CLOSED · finding · - · 2026-09-16
F-062 the Circuit tab's GraphValidationError covers dangling terminals, a readout with no filter and a drive with neither receptive-field bank nor combine, but not a combine whose inputs disagree on neuron count, because that needs building receptor coordinates from the registries to know the counts; a sum-combined graph with mismatched inputs therefore fails at run time rather than at validation
→ commit e356451

## F-060 · CLOSED · finding · - · 2026-09-16
F-060 sensoryforge run raised KeyError 'spikes' for any config with an analog population because the CLI summary loop read pop_results["spikes"] unconditionally, after the run had succeeded and the bundle had been written; neither Wave J nor Wave N could test the other's half of that seam
→ commit 577f2dd

## F-058 · CLOSED · finding · - · 2026-09-16
F-058 the processing-layer kind is not wired into sensoryforge.testing.contracts.check_component, so OnOffLayer and any third-party processing plugin are checked only by hand-written tests while every other component kind goes through the shared contract harness a plugin author is told to run
→ commit 8a6c816

## F-059 · CLOSED · finding · - · 2026-09-16
F-059 docs/user_guide/configuration_schema.md documents the Wave M config fields but not the Wave L ones -- GridConfig.channels, coords_file and layers, PopulationConfig.target_layers and readout, StimulusConfig.channel and dsl_config are all absent, so the published schema reference understates what a config may contain
→ commit 8a6c816

## F-057 · CLOSED · finding · - · 2026-09-16
render_stimulus never advances a stateful registered stimulus's .step(), so "moving" (and any stepped stimulus) renders as a static repeated frame instead of animating
→ commit 1d4d210

## F-056 · STANDING · finding · - · 2026-09-16
F-056 the memory watchdog's peak RSS varies from about 800 MB to 1500 MB run to run for identical code (83b735d measured 873 MB and 1517 MB on two runs), so it cannot detect a regression below roughly a factor of two and its numbers must never be compared across runs or across machines
→ commit 0c13d47
· tidied 2026-09-24: a settled result, not an open problem

## F-055 · CLOSED · finding · - · 2026-09-16
the bundle wrote stimuli/stimulus.json as an untagged caller dict, or {} when none was given, and pressure-simulation's generate_stimulus_from_json defaults every field, so a bundle could be read there as a static Gaussian blob at the origin and encoded and plotted with no error anywhere; fixed by tagging every payload with schema_version and kind and emitting pressure-simulation's schema only for the types proven to regenerate exactly
→ commit e722c0a

## F-054 · CLOSED · finding · - · 2026-09-16
the bundle lacked neuron_modules/, so pressure-simulation's viewer could load it but never run it; fixed by writing neuron_modules/sensoryforge.json with one population_configs entry per population, matched by raw population name. (Recorded as F-052 on branch wave-j, which forked before F-052 was taken; renumbered to F-054 at the merge.)
→ commit 6382319

## F-053 · CLOSED · finding · - · 2026-09-16
F-053 docs examples run as subprocesses resolve `import sensoryforge` through the environment's editable install rather than the checkout under test, so with git worktrees a `pip install -e .` from one worktree makes every other checkout's docs-example tests silently exercise that worktree's code while still reporting green; fixed by setting cwd and PYTHONPATH in tests/docs/test_docs_examples.py
→ commit ad7162d

## F-052 · CLOSED · finding · - · 2026-09-16
GeneralizedTactileEncodingPipeline.generate_stimulus dispatches stimulus names through a hard-coded if/elif chain (generalized_pipeline.py:1030-1073) that the CLI calls even for canonical configs (cli.py:222), so registered stimuli composite/edge_grating/gabor/static cannot be run from a config file and a third-party stimulus plugin can be registered but never executed
→ commit 09a14e7

## D-020 · CLOSED · decision · - · 2026-09-15
the template receptive-field builder derives sigma = d/pi and pitch = d from one resolvable distance d, truncates to the k nearest receptors with analytic Gaussian weights and unit-L2 rows by default
→ commit d53c018

## F-050 · CLOSED · finding · - · 2026-09-15
ReceptorGrid and CompositeReceptorGrid take no seed; jittered_grid, blue_noise and poisson draw from the global RNG, so identical builds differ, building changes global RNG state, and from_config(to_dict()) does not reproduce a Poisson grid
→ commit d244faa

## F-051 · CLOSED · finding · - · 2026-09-15
On non-composite grids SimulationEngine builds InnervationModule without innervation_method, so gaussian, uniform, one_to_one and distance_weighted yield bit-identical weights in the engine, CLI and batch paths; only the flat path honours the method
→ commit d244faa

## F-049 · CLOSED · finding · - · 2026-09-15
_assert_to_dict_roundtrip_complete (H3) is only called from _check_neuron, not _check_filter/_check_stimulus/_check_grid/_check_solver/_check_innervation; applied to today's built-ins it would fail SAFilterTorch (missing tau_r, tau_d, k1, k2, clip_to_positive), RAFilterTorch (missing tau_RA, k3), the grid arrangement classes (missing density), and EdgeGrating (missing normalize)
→ commit 5b2cc4e

## F-046 · CLOSED · finding · - · 2026-09-15
SimulationEngine lowercases neuron_model before a case-sensitive NEURON_REGISTRY lookup (filters are looked up exactly), so a plugin neuron registered as DemoNeuron is listed by list-components but fails with "Unknown neuron model"; built-ins only work because both spellings are registered
→ commit 2abdab2

## F-047 · CLOSED · finding · - · 2026-09-15
sensoryforge new-component writes relative to the installed package (parents[1] of cli.py): from a wheel install it creates files under site-packages/sensoryforge, site-packages/tests and site-packages/docs, and instructs editing core register_components.py instead of producing an entry-point plugin package
→ commit 2abdab2

## F-048 · CLOSED · finding · - · 2026-09-15
YAML plugins: lists are honoured only by cli.load_config_file; the GUI YAML load (gui/main.py), BatchExecutor given a path, and SensoryForgeConfig.from_yaml_file ignore them, so a config using plugin components loads in the CLI but not elsewhere
→ commit 2abdab2

## F-045 · CLOSED · finding · - · 2026-09-15
concrete neuron models only round-trip dt via to_dict/from_config, not a/b/c/d/tau_m etc.
→ commit 51eb198

## F-043 · CLOSED · finding · - · 2026-09-15
examples/example_config.yml and batch_config.yml (and docs batch_processing.md, cli.md, yaml_configuration.md) set sa_neurons 100 / ra_neurons 196 as totals; legacy configs read them per row (10,000 and 38,416 neurons), so run/validate/batch --dry-run fail with the dense-weight cap (killed above 3 GB before the cap); no canonical example exists
→ commit 42f5758

## F-044 · CLOSED · finding · - · 2026-09-15
Stimulus dt spinbox accepts values such as 0.12 ms; validate_dt_ms then raises ValueError inside SpikingNeuronTab._run_simulation, which only catches RuntimeError, and no sys.excepthook is installed, so the exception escapes a Qt slot (PyQt5 aborts by default)
→ commit 42f5758

## F-038 · CLOSED · finding · - · 2026-09-14
Seeded innervation fails on MPS/CUDA since fed09be: per-instance CPU torch.Generator used with device tensors raises "Expected a 'mps' device type for generator but found 'cpu'"; SimulationEngine(device="mps") with a seeded population crashes; CI is CPU-only
→ commit bd13a0b

## F-039 · CLOSED · finding · - · 2026-09-14
_canonical_to_legacy_config reads simulation "dt", which SimulationConfig.to_dict no longer writes (dt_ms since fd73a0e), so texture/moving/timeline/repeated_pattern/custom stimuli in CLI and batch canonical runs use 0.1 ms regardless of dt_ms
→ commit bd13a0b

## F-040 · CLOSED · finding · - · 2026-09-14
CLI run of a canonical config with dt_ms 1.0 and --duration 100 yields 10450 stimulus bins (trapezoid/gaussian/step/ramp use legacy temporal.dt 0.1 ms; trapezoid ignores --duration), which the engine reads as 1 ms each; pre-existing, on the CLI/batch data-generation path
→ commit bd13a0b

## F-041 · CLOSED · finding · - · 2026-09-14
GUI export always writes simulation.dt_ms 1.0: SpikingNeuronTab.get_config carries no time step and gui/main.py _gui_to_canonical defaults to 1.0, while the GUI simulates at the stimulus step (0.1 ms default)
→ commit bd13a0b

## F-042 · CLOSED · finding · - · 2026-09-14
dt_ms that is not a whole multiple of integrate_dt_ms silently rescales neuron time in _run_pop_from_drive (0.12 ms bins integrate 0.10 ms, 0.07 ms bins integrate 0.05 ms); no validation in SimulationConfig or the GUI
→ commit bd13a0b

## D-019 · CLOSED · decision · - · 2026-09-14
default innervation uses analytic Gaussian weights; the stochastic uniform-weight builder is the named control arm
→ commit b1f67a3

## F-036 · OPEN · finding · - · 2026-09-14
flake8 style debt after black (all default checks, 88 columns): 364 violations, mainly E501 164, F401 141, F541 18, E402 14, F841 11; CI gates only E9,F63,F7,F82 until ratcheted
→ commit 7da39f9

## F-037 · STANDING · finding · - · 2026-09-14
SensoryForge Izhikevich/AdEx/MQIF clamp voltage at v_floor (-120/-130/-120 mV, D-007) but pressure-simulation's neurons do not, so spikes can differ for strongly negative drive, which unrectified SA (F-001) now makes reachable
→ commit 7da39f9
· settled 2026-09-24: never reached by the tactile recipes -- the lowest voltage on the four benchmark stimuli is -94.1 mV at the calibrated gains (Izhikevich SA, moving_edge), 26 mV above the floor; tests/integration/test_recipe_calibration.py fails if a recipe comes within 20 mV of it

## F-035 · CLOSED · finding · - · 2026-09-14
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

## F-024 · CLOSED · finding · - · 2026-09-14
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

## F-010 · CLOSED · finding · - · 2026-09-14
SimulationEngine: composite grids NotImplementedError (:98); poisson/hex/jittered/blue_noise arrangements built then ignored, innervation uses the regular GridManager (:107-125,:224-243); DSL neurons cannot be instantiated (:260-264, dsl_config never read); _stimulus_to_receptors is a passthrough (:425-448)
→ commit 7a188b6

## F-011 · CLOSED · finding · - · 2026-09-14
SLURM export is dead: generate_slurm_script emits `sensoryforge run --stimulus-index --format hdf5` (batch_executor.py:729-733) but run has neither flag (cli.py:556-578) and writes .pt only; BatchTab progress never emitted
→ commit 7a188b6

## F-012 · CLOSED · finding · - · 2026-09-14
CRITICAL canonical->legacy adapter sets grid_size = rows*cols (generalized_pipeline.py:351) and grid.py:32 treats an int as per-side: 20x20 config -> 160k receptors, README 80x80 example -> 41M; test_gui_cli_parity and test_regression_refactoring exceed 5 GB and are OOM-killed; CLI/Batch hit it on every canonical run (cli.py:218, batch_executor.py:98)
→ commit 7a188b6 · closed by commit 1c93fa6 (grid_size now emits (rows, cols))

## F-013 · CLOSED · finding · - · 2026-09-14
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

## F-018 · CLOSED · finding · - · 2026-09-14
cli list-components is a hardcoded print block (cli.py:438-478) already out of sync with the registries (lists center_surround, omits fa/sa/composite/timeline/repeated_pattern); cli validate forces the legacy pipeline for canonical configs (cli.py:406)
→ commit 7a188b6

## F-019 · CLOSED · finding · - · 2026-09-14
~3500 lines of unwired GUI code: gui/protocol_suite_tab.py, protocol_backend.py, protocol_execution_controller.py, neuron_explorer.py are imported by no tab, only by two tests
→ commit 7a188b6

## F-020 · CLOSED · finding · - · 2026-09-14
Public docs: developer_guide/*, units_and_gains.md, gui_walkthrough.md, configuration_schema.md absent from mkdocs nav; 8 broken intra-doc links; "pip install sensoryforge" in 3 pages; sensoryforge/config/README.md describes 4 nonexistent files; docs/api_reference/ is a .gitkeep
→ commit 7a188b6

## F-021 · CLOSED · finding · - · 2026-09-14
Debt lists are stale: CLAUDE.md still lists DSL numpy-only (C-2) and reset_states (M-1) as open, both resolved (R-001, D-011); docs/development/reviews/CODE_REVIEW_20260408.md tracker says 37/37 open though several are fixed; decide whether reviews/ ships publicly
→ commit 7a188b6

## F-022 · CLOSED · finding · - · 2026-09-14
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

## F-083 · OPEN · finding · - · 2026-09-21
stimulus types disagree on amplitude scale by about 30x at their defaults (gaussian, texture and moving peak about 30 mA; braille, gratings, moving_edge and ramp_gaussian about 1; gabor 0.55), while input_gain 50 is calibrated for about 30, so with tactile_sa1_ra1 on a 40x40 grid for 500 ms a default gabor gives 0 SA and 1 RA spike against 9731 and 3288 for a default gaussian; a user switching type in the GUI sees a silent population with no warning
→ measured 2026-09-21 on gui-v2 0b06c3b; decision needed (common default amplitude, or per-type gain guidance), not a code fix
→ root cause, measured 2026-09-22 (tactile_sa1_ra1 on a 40x40 grid, 800 ms, default ramps): (1) two amplitude conventions -- stimuli/render.py _LEGACY_DEFAULTS gives gaussian, texture and moving amplitude 30 (the pre-pressure-simulation "mA" generator), while the ported pressure-simulation stimuli and the texture-module constructors use 1.0, the scale pressure-simulation calibrates its filters and gains against (config/pipeline_config.yml: every stimulus amplitude 1.0; its gains 40-200); at gain 50 the legacy-30 types fire at 50-110 Hz mean, the unit types at 1-10 Hz; (2) repeated_pattern sums six overlapping amplitude-30 copies (peak 115.6); (3) gabor's own defaults (sigma 0.3 mm, wavelength 0.5 mm) on a 0.15 mm grid are a 2-receptor blob with a 0.55 sampled peak and equal negative lobes, which the receptive fields sum away (SA drive 0.86 vs 1.25 for the positive part alone): 0 spikes; (4) gabor and texture are signed (texture min -26.5), i.e. negative pressure driving negative current
↔ b4a853b decide: stimulus amplitude, recipe gains, touchsim reference, grid density

