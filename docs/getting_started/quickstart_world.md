# Quick Start: World to Data Set to Report

In five minutes you describe a small world of stimuli, draw a data set from it, simulate
every draw and get a report that sets what was measured in the generated data beside what
the world declared. You need SensoryForge installed ([Installation](installation.md)).

## 1. Describe a world

A world is a YAML file of classes of stimuli and the ranges their values are drawn from.
Save this as `world.yml`:

```yaml
world:
  name: first
  modality: tactile
  defaults:
    delay_ms:   {range: [0, 20]}
    touch_ms:   {range: [5, 15]}
    hold_ms:    {range: [20, 60]}
    release_ms: {range: [5, 15]}
    amplitude:  {range: [0.5, 1.0]}
    x_mm: {range: [-0.3, 0.3]}
    y_mm: {range: [-0.3, 0.3]}
  classes:
    dots:
      weight: 0.5
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}
    edges:
      weight: 0.4
      layer: {shape: {kind: bar, profile: gaussian, length_mm: 0}}
      axes:
        width_mm: {range: [0.05, 0.15]}
        orientation_deg: {range: [0, 180], circular: true}
    quiet:
      kind: quiet
      weight: 0.1
      axes: {quiet_ms: {range: [20, 60]}}
```

Check it and look at two draws:

```bash
sensoryforge world validate world.yml
sensoryforge world sample world.yml -n 2 --seed 7
```

```text
w-b820f0863eba  first  channels=['value']
  class dots (layered) weight=0.5: amplitude, delay_ms, hold_ms, release_ms, sigma_mm, touch_ms, x_mm, y_mm
  class edges (layered) weight=0.4: amplitude, delay_ms, hold_ms, orientation_deg, release_ms, touch_ms, width_mm, x_mm, y_mm
  class quiet (quiet) weight=0.1: quiet_ms
{"class": "dots", "draw_seed": ..., "index": 0, ...}
{"class": "dots", "draw_seed": ..., "index": 1, ...}
```

The first word is the world's id; it changes whenever a value in the file changes.

## 2. Build, simulate and evaluate in one command

```bash
sensoryforge evaluate run world.yml --out runs/first --preset tactile_sa1_ra1 \
    --draws 40 --duration-ms 150 --seed 7 --packs core,tactile
```

```text
[batch] done in ... s
audited 40 entries (0 failed) -> runs/first/evaluate/audit.json
...
report folder: runs/first/evaluate
  build    ok
  batch    ok
  audit    ok
  declared ok
  compare  ok
  checks   ok
checks: 13 passed, 0 failed, 0 skipped, 0 errors
```

The sensor is the `tactile_sa1_ra1` preset. Forty draws of 150 ms take about half a minute
on a laptop CPU. `runs/first/` now holds the data set (its manifest and one
[bundle](../user_guide/bundles.md) per draw), and `runs/first/evaluate/` the report:

| file | what it holds |
|---|---|
| `audit.json` | statistics measured from the generated frames and spikes alone, never from the world file |
| `declared.json` | what the world file says each draw should show |
| `compare.json` | measured, declared and their raw difference, per metric |
| `checks.json` | the reference-free checks |
| `pipeline.json` | the run: inputs, each stage's status and time |

## 3. Look at it

As a table in the terminal:

```bash
python -m sensoryforge.evaluate.compare runs/first/evaluate runs/first/evaluate
```

```text
metric                    stat  measured  declared  diff  ...
core.frames.duration      mean  150.0     150.0     0.0   ...
core.frames.onset_time    mean  ...
```

In the browser, with a draw picker, a movie of each draw, linked traces and the
dashboards:

```bash
sensoryforge evaluate serve runs/first     # then open http://127.0.0.1:8765/
```

The server listens on your own machine only; stop it with Ctrl+C. To send the result to
someone, write a snapshot that opens offline, with no server:

```bash
sensoryforge evaluate export runs/first snapshot --n-draws 4    # open snapshot/index.html
```

## Next

- Every class kind, shape, motion and modulation a world can use:
  [Worlds and Data Sets](../user_guide/worlds.md).
- Splits, held-out classes and sessions: a data-set file instead of `--draws`
  ([Evaluating a World](../user_guide/evaluate_pipeline.md)).
- The browser: [Evaluation Browser](../user_guide/evaluation_browser.md); the snapshot:
  [Sharing an Evaluation](../user_guide/evaluation_export.md).
- What each metric measures: the [report contract](../reference/report_contract.md).
