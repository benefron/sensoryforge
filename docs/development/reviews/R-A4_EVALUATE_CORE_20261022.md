# R-A4 · Review of the evaluation core (audit, declared side, comparison)

Date 2026-10-22 · reviewer Claude Opus 5.5 · branch `phase/a-evaluate` at `48d8665` · scope
roadmap rows A2a–A2d, A3 and A4 (`sensoryforge/evaluate/`, `schemas/evaluate/`, the report
contract, the world commit `8cf4e60`).

**Verdict:** the structure is sound and the blind wall holds. The numbers are not yet ready
to build on: one core metric reports noise as a direction, and three declared/measured
pairings give large raw differences that the declaration itself predicts, so a reader would
take them for data faults. All five need a change before A5 shows them. Each is small, and
row **A4f** (Sonnet, the start of S4) makes them, as listed under "Required changes".

## How this was checked

- The diff `084d604..48d8665` (source, schemas, contract sections 2–10, the stage pages).
- A fresh 50-entry data set from `tests/fixtures/worlds/tactile_small.yml` on the 8×8 test
  design (seed 5, 40 train, the stratified test split, one session), run through
  `run_audit(packs=core,tactile)`, `run_declared` and `run_compare`. Every pair with
  `|diff| > 1.01` (one frame step) was listed and explained; the results below come from it.
- The full suite, once, in the background (result in the R-A4 session log).

## What holds

- **The blind wall.** The reader projects `config.json`, `data.h5`, the manifest and the index
  onto `allowlist.py` at read time; `dataset.json`, `stimuli/` and `neuron_modules/` are never
  opened. The import walk follows every import under `sensoryforge.evaluate` (lazy ones too)
  and import-time imports elsewhere; a planted chain proves it fires; a `python -I`
  subprocess confirms no `sensoryforge.world` module loads. `registry.py` names the
  tactile pack but not the declared adapter, which registers itself from `declared/`.
- **Raw numbers before verdicts.** `compare.json` holds measured, declared, `diff`,
  `rel_diff` and a `pairing` that only says whether a difference could be computed. There is
  no tolerance anywhere (Q8). The table prints with `repr`, so nothing is rounded away.
- **No tactile word in the audit core** (lint, mutation-checked). The world commit `8cf4e60`
  is additive: a `units` block is written only for a world that names a unit other than
  mm/ms, so tactile bundles are byte-for-byte unchanged.
- **Agreement where the definitions meet.** On still, unmodulated draws, onset, offset,
  active time, bouts, centroid and the three episode phases agree to within one frame step.

## Findings

| # | Severity | Where | Finding |
|---|---|---|---|
| 1 | bug | `audit/frame_metrics.py` `_MOVE_FLOOR` | `core.frames.direction` gives a random angle for a shape that does not move: 20 of 28 still entries (dots, edges, braille, vibes) carry one. The centroid of a still shape jitters by 1e-11 to 1e-9 mm during its ramps (float32 rounding of the scaled frame), and the 1e-9 × extent floor sits inside that noise. Moving steps on the same data are at least 1e-4 mm. |
| 2 | definition | `tactile/metrics.py` speed | `tactile.motion.speed` reads 24–72 % low on all 9 sliders. Two causes. First, the plateau contains the still hold, so its pooled steps include zeros (the point A3 left open). Second, for a moving shape the envelope `max |x|` changes as the shape crosses the lattice and the canvas edge, so the 1 % plateau covers only part of the slide (5–36 frames measured, against 28–60 declared). |
| 3 | adapter | `tactile/declared.py` episode phases | Rise, plateau and fall are declared for moving shapes, where the envelope is not flat (cause 2 of finding 2): rise +8 to +45 ms, plateau −5 to −47 ms and fall +3 to +21 ms on every slider and orbit entry. These differences come from the metric's definition, not from the data. |
| 4 | adapter | `tactile/declared.py` timeline metrics | A pulse train with `depth` 1 (the default) renders exactly 0 between taps (`stimuli/episode.py`: `1 − depth·(1 − pulse)`), but the timeline basis declares one bout and the whole contact time. On all 8 taps entries, bouts read +3 to +5, active time −12 to −31 ms and offset −2 to −14 ms. The world declares this, so every pulsed data set would show it. This difference is where the data set's `bouts` 1.57 vs 1.14 came from (A4's log). |
| 5 | comparison | `compare/run.py` `pair_summaries` | The stat pairs of a group compare the audit's summary of *all* its entries with the declared summary of only the entries the declaration determines (A4's finding: centroid `n` 13 vs 7). A difference in a mean or quantile then mixes two populations. |
| 6 | expected | `core.frames.peak` | Measured peak is below the declared one in all 26 pairs (−2e-5 to −0.069). The sensors sample the kernel at lattice points, so the sampled maximum reaches `amplitude` only when the centre falls on a sensor. This is the `shape peak` basis as defined; the contract should say so. No code change. |
| 7 | open | centroid, direction, speed near the canvas edge | Direction differs by 1–14° on sliders. Even with the speed rule of D2, which isolates the slide (22–40 moving steps against `slide_ms` 20–40), speed still reads 30–55 % low. The 8×8 canvas is 1.05 mm wide, and a σ 0.3 mm Gaussian on it is cut off at about ±1.75 σ. The weighted centroid of a cut-off shape moves at about 0.6–0.7 of its true speed and is pulled towards the middle. The code is right, but the declared side does not know the sensor extent, so it declares centroid, direction and speed even for shapes the canvas cuts off. The ledger records this as an open point: a later change could give the adapter the canvas and leave such entries out. |
| 8 | gap | `tests/contract/test_audit_import_wall.py` | The tactile-word lint covers `audit/**` and `evaluate/*.py` but not `compare/**` or `declared/**`. Both are neutral by the contract, and both are clean today apart from `declared/__init__.py`, which imports the tactile adapter. |
| 9 | minor | `audit/reader.py` `read_index` | A corrupt index line raises `JSONDecodeError` and stops the whole audit, where a corrupt bundle fails only its own entry. |

## Decisions

- **D1 (finding 1).** The motion floor is `1e-6` of the layout's largest extent, used by
  `core.frames.direction` and by the speed rule in D2. Float32 centroid noise is at most
  about 1e-7 of the extent, and the slowest motion worth reporting (1e-4 mm/ms on a 6 mm
  canvas at 1 ms) is about 2e-5 of it. Contract section 6.5 changes from 1e-9 to 1e-6.
- **D2 (finding 2, the question A3 left to R-A4).** Change the metric, not the declared
  value. `tactile.motion.speed` becomes the pooled length of the centroid steps that move
  (a step between consecutive active frames longer than the D1 floor), divided by their
  number times `dt`, over the entry's bouts. The plateau plays no part. `motion_tolerance`
  stays and applies to the entry's total moving path. The declared value stays the slide
  speed, so the raw difference measures how well the slide is rendered and sampled. The
  id, unit and pack version stay: the branch is unmerged and nothing has read version 1.
- **D3 (finding 3).** The adapter leaves the three episode phases out when the shape moves
  (reason `motion`, the same test the centroid uses).
- **D4 (finding 4).** When the modulation reaches 0 (`depth` ≥ 1, for pulses and for sine),
  the adapter leaves out `offset_time`, `active_time`, `active_fraction`, `bouts`,
  `gap_mean` and `gap_max`, with the reason `modulation reaches 0`. Onset stays, since the
  first tap starts the contact. This keeps Q6, because the timeline basis cannot fix these
  values when the frames go to 0 inside a contact. Declaring them from the modulation
  instead (counting taps) is a possible later addition.
- **D5 (finding 5, the question A4 left to R-A4).** For each group, the compare stage pairs
  the declared summary with a summary of the *measured values of the same (entry, key)
  set*: the entries whose declared row carries the key. It uses the audit's summariser
  (`audit/summary.py`) with the audit's `histogram_bins`. The measured `n` then equals the
  declared `n` minus that group's `missing_measured`. `audit.json` keeps the full-set
  summaries. The tag stays `sensoryforge-compare/1` because the format is unreleased.
- **Approved as written:** the implementation choices in contract sections 6.5 (except the
  floor, D1), 7.1 (except speed, D2), 8.4 (deviations 1–4 from A3's log) and 9.1 (the
  pairing rule order; the `distributions` block).

## Required changes (row A4f)

1. `frame_metrics._MOVE_FLOOR = 1e-6`. Add a test: a float32 gaussian with ramps that
   stays still gives `direction` null, and the same gaussian moving at 1e-3 mm per frame
   gives the right angle.
2. `tactile/metrics.py`: speed per D2 (`measure_channel` no longer pools plateau steps).
   Add tests: still hold then slide gives the slide speed; a still shape gives null; a
   path below `motion_tolerance` gives null. Update contract section 7 (the definition
   row) and 7.1, and the pack's page in `audit_metrics.md`.
3. `tactile/declared.py`: D3 (`motion` for the three phases) and D4 (`modulation reaches 0`
   for the six timeline metrics). Add tests on the adapter. Update contract 8.3's table
   ("left out when") and 8.4, and `declared_adapter.md`.
4. `compare/run.py`: D5. Keep the measured value for every (entry, key) whose declared
   value is present, and summarise per group with `summarise_values`. Add a test where the
   audit has more entries than the declared side, which must give equal `n` and equal
   means for identical values. Update contract 9 and 9.1 and `compare_stage.md`.
5. The wall lint covers `compare/**` and `declared/**`, with `declared/__init__.py` exempt
   (pinned the same way as `registry.py`).
6. `read_index`: a line that is not JSON is skipped and counted, so the entry fails with
   stage `batch` or `read` and the audit goes on. A test.
7. Contract 8.4: one sentence on the sampled peak (finding 6).
8. Re-run the 50-entry check from this review (the script is in the R-A4 session log).
   Expected: no direction on still entries; no pair beyond one frame step on still,
   unmodulated draws; no episode phases on moving draws and no timeline counts on taps;
   slider speeds still low by the cut-off of finding 7. Add one unit test with a shape
   much smaller than a wide canvas moving at a known speed, which must read within 5 %.
