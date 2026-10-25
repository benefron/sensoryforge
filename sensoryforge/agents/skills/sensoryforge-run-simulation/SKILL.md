---
name: sensoryforge-run-simulation
description: Simulate a SensoryForge data set (or one config) through a sensor and write one data bundle per entry, timed, resumable and splittable over jobs. Use when someone wants to run, simulate, encode or batch a data set or a stimulus through a recipe, design or config, to resume an interrupted run, or to spread a run over several jobs, in whatever words they put it.
---

# Run a simulation

A run simulates every entry of a data set through a sensor (receptive fields, filters,
neurons) and writes one data bundle per entry. It can take hours and gigabytes, so you
measure one entry first, say what the whole run costs and ask before starting it.

## 0. Read the working style

Run `sensoryforge agents brief` and follow it. If the command is not found, SensoryForge
is not installed here: say so and stop (it installs from source; it is not on PyPI yet).

## 1. Check the inputs

- **The data set file** and a built manifest (`sensoryforge-build-dataset`).
- **The sensor** is the user's choice: a shipped recipe (`sensoryforge list-presets`;
  `sensoryforge describe --section recipes` shows each one's grid and populations), a
  design folder, or a config file they wrote. Do not pick one for them, and never edit a
  recipe's gains or parameters: if none matches, go back to `sensoryforge-design-and-plan`.
- **Where to write** (a new folder; the whole run is about as large as the entries times
  one bundle), and the device (`--device cpu|cuda|mps`).

For a config file, check it first: `sensoryforge validate CONFIG`.

## 2. Time one entry, then say what the run costs

```
sensoryforge batch --dataset dataset.yml --preset NAME --output runs/NAME --entries 0:1
```

(`--design DIR` or a config file replaces `--preset NAME`.) Read the time on its line and
the size of its bundle in the output folder, multiply by the number of entries and say
so, with "measured on one entry" next to it. Check that the disk has the room. Do not use
`--dry-run` with `--dataset`: it does not stop the run, it simulates every entry. To see
what a run would do without doing it, use `--print-tasks` (step 4).

## 3. Run

```
sensoryforge batch --dataset dataset.yml --preset NAME --output runs/NAME --resume
```

`--resume` skips every entry whose bundle is complete (the entry from step 2 included).
A bundle is written to a temporary folder and moved into place, so an interrupted run
leaves no half bundle and the same command continues it. Each entry's noise is seeded
from the entry, so a rerun gives the same spikes. The command prints one line per entry,
a total, and exits non-zero if any entry failed.

## 4. Scale

- `--splits a,b` runs only those splits; `--entries 100:200` runs a slice of the manifest.
- `--tasks N --print-tasks` prints N commands, one per task, for a job array or several
  terminals, and runs nothing; `--tasks N --task-index K` runs the K-th task of N (a
  scheduler's array index goes there). Every task writes its own file under `index/`, so
  tasks never share a file; they may share the output folder.
- `--resume` is also how a failed task is retried.

## 5. When entries fail

Read the failed rows: `python -c "from sensoryforge.world import read_batch_index as r;
print([x for x in r('runs/NAME') if x['status'] != 'ok'])"` gives each row's entry id,
status and error text. Say what the error is in words, fix its cause (a config value, not
the data), and run the same command with `--resume`. If you cannot tell the cause, show the
user the error and ask; do not loop.

## 6. Verify and report

The output folder has one folder per entry, `batch.json` (data set, config and
SensoryForge's version), and `index/`. Check that the number of ok rows equals the number of
manifest rows. Report: ok, failed and skipped counts, total time, size on disk, the folder,
the sensor you used, and the seed. Stop and ask whether to evaluate
(`sensoryforge-evaluate`).

One simulation without a data set: `sensoryforge run --preset NAME --duration MS`
(`sensoryforge run --help` lists the stimulus and output options).

## Worked example

A tiny data set on a shipped recipe: time one entry, run the rest, and count.

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
      weight: 1.0
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}
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
```

```python
# file: count.py
"""Entries by status in a run's index."""
import collections
import sys

from sensoryforge.world import read_batch_index

rows = read_batch_index(sys.argv[1])
print(dict(collections.Counter(r["status"] for r in rows)), "of", len(rows), "entries")
```

```bash
sensoryforge dataset build dataset.yml --out data/demo
# prints: 4 entries
sensoryforge batch --dataset dataset.yml --preset tactile_sa1_ra1_adex --output runs/demo --entries 0:1
# prints: 1 ok, 0 failed
sensoryforge batch --dataset dataset.yml --preset tactile_sa1_ra1_adex --output runs/demo --resume
# prints: 3 ok, 0 failed, 1 skipped
python count.py runs/demo
# prints: {'ok': 4} of 4 entries
sensoryforge batch --dataset dataset.yml --preset tactile_sa1_ra1_adex --output runs/demo --tasks 2 --print-tasks
# prints: --task-index 1
```
