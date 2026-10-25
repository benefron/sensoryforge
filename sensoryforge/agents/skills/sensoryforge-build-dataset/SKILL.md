---
name: sensoryforge-build-dataset
description: Turn a SensoryForge world into a data set, choosing the splits, the seed and the sizes with the user, and build its manifest without simulating anything. Use when someone wants training, validation or test examples, held-out classes, probes, long sessions or named draws from a world, or wants to change or rebuild a data set, in whatever words they put it.
---

# Build a data set

A data set fixes a world, a seed, a duration and how many of what to draw in each split.
Building it writes a manifest, one row per entry, and simulates nothing, so it is cheap
and safe to repeat. You end with a data-set file, a built manifest and a short report.

## 0. Read the working style

Run `sensoryforge agents brief` and follow it. If the command is not found, SensoryForge
is not installed here: say so and stop (it installs from source; it is not on PyPI yet).

## 1. Check the inputs

You need a world file that validates (`sensoryforge validate --json --kind world
FILE`). If there is none, or it does not validate, use `sensoryforge-design-world` first.
Ask what the data are for, in the user's words: fitting something, checking something,
testing how far it generalises, comparing sensors, checking the pipeline itself.

## 2. Choose the splits

Map the purpose to the kinds of split by meaning. `sensoryforge describe --schema dataset`
is the file's JSON Schema, `sensoryforge describe --json --section world` lists the split
kinds that exist, and the page that teaches them is
https://benefron.github.io/sensoryforge/user_guide/worlds/#data-sets. In plain words:

- Training and validation splits: draws, optionally repeated with fresh noise.
- A test split: balanced across classes and across each axis's range, so every part of a
  range is tested equally.
- Probes: draws just outside each axis's range, to see what happens beyond it.
- Held-out classes: kinds the world marks as held out, kept out of every other split.
- Sessions: long recordings that join several draws with gaps.
- Fixed draws: named draws with explicit values.

Sizes are the user's choice. If they have none, offer a small trial data set labelled as a
trial, to check the pipeline, and say that its numbers say nothing about the final one.

## 3. Seed and duration

The data set has one seed. Use the user's; if they have none, pick one, write it in the
file, say which and that it can be changed. A draw never depends on how many are drawn, so
adding draws later keeps the earlier ones. The duration (ms) is how long each entry is
simulated: it must cover the longest draw. Check it: `sensoryforge world sample WORLD -n
200 --seed 0` prints `"end_ms"` for each draw; tell the user if a draw ends after the
duration, because the end is then cut off.

## 4. Write, validate, build

Write the data-set file (`world:` is a path relative to the data-set file), then:

```
sensoryforge validate --json --kind dataset dataset.yml
sensoryforge dataset build dataset.yml --out data/NAME
```

The build prints the entry count and the data set's id and writes `dataset.json` (the
spec, the ids, and the probes it had to skip) and `manifest.jsonl`. It fails if any seed
appears twice. Fix the cause (a problem names its path) and build again; never edit the
manifest by hand.

## 5. Count and report

Count the entries per split and class from the manifest, and read `dataset.json` for
probes that were skipped. Report: the data-set file and folder, world id, data-set id,
seed, duration, the counts, anything skipped, and the cost to come: entries times the time
and disk one entry takes, which `sensoryforge-run-simulation` measures. Do not guess it.
Stop and ask whether to simulate.

## Verify

Validation is clean, the build printed the expected total, the per-split counts match what
the user asked for, and nothing in `dataset.json` is skipped without being reported.

## Worked example

A world with two classes, a small data set from it, and the counts:

```yaml
# file: world.yml
world:
  name: demo
  defaults:
    delay_ms:   {range: [0, 10]}
    touch_ms:   {range: [5, 15]}
    hold_ms:    {range: [20, 40]}
    release_ms: {range: [5, 15]}
    amplitude:  {range: [0.5, 1.0]}
    x_mm: {range: [-0.3, 0.3]}
    y_mm: {range: [-0.3, 0.3]}
  classes:
    dots:
      weight: 0.6
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}
    edges:
      weight: 0.4
      layer: {shape: {kind: bar, profile: gaussian, length_mm: 0}}
      axes:
        width_mm: {range: [0.05, 0.15]}
        orientation_deg: {range: [0, 180], circular: true}
```

```yaml
# file: dataset.yml
dataset:
  name: demo
  world: world.yml
  seed: 7
  duration_ms: 80
  splits:
    train: {n: 4}
    test: {stratified: {bins: 2, per_bin: 1}}
```

```python
# file: count.py
"""Entries per split and class in a built data set."""
import collections
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1] + "/manifest.jsonl")]
counts = collections.Counter((r["split"], r["class"]) for r in rows)
for (split, cls), n in sorted(counts.items()):
    print(f"{split:8s} {cls:8s} {n}")
print(f"{len(rows)} entries")
```

```bash
sensoryforge validate --json --kind dataset dataset.yml
# prints: "valid": true
sensoryforge dataset build dataset.yml --out data/demo
# prints: 8 entries
python count.py data/demo
# prints: 8 entries
```
