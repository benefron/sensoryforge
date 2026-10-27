---
name: sensoryforge-evaluate
description: Evaluate a SensoryForge world or data set (audit the generated data, set it against what the world declared, run the checks) and read the report back in words, raw numbers first. Use when someone wants to check, audit, evaluate, validate or compare what a data set or a simulation produced, to look at the draws in the browser, or to share a snapshot, in whatever words they put it.
---

# Evaluate

One command takes a world or a data set to a report folder: it builds the data set,
simulates it, audits the generated data (it never sees the world file), extracts what the
world declares for each entry, compares the two and runs reference-free checks. Your job
is to run it, then say what the report says: the numbers first, then what they mean, and
nothing the numbers do not support.

## 0. Read the working style

Run `sensoryforge agents brief` and follow it. If the command is not found, SensoryForge
is not installed here: say so and stop (it installs from source; it is not on PyPI yet).

## 1. Check the inputs

- **A world file** (then also the number of draws, the duration in ms and a seed) or a
  **data-set file** (it declares its own splits and duration; do not pass `--draws` or
  `--duration-ms` with it). Bundles already simulated into the output folder are kept with
  `--resume`, so a repeat does not simulate them again. That holds only for the same data
  set (same file, or same world, draws, duration and seed) on the same sensor: `--resume`
  refuses a folder written for another data set or sensor, and the stage fails with the
  values that differ. For a different data set or sensor, give a new `--out`.
- **The sensor**, the user's choice, exactly as for `sensoryforge-run-simulation`: `--preset`,
  `--design` or `--config`.
- **The metric packs**, named by the user, never guessed from the world:
  `sensoryforge describe --section evaluation` lists them. The default is the general
  core pack; a pack for one modality is named with `--packs core,<pack>`. If the world
  declares an adapter whose pack you did not run, the report says so in its notes: pass
  that on to the user.
- **Size.** For a first look use a small run. Say what a large run costs from a measured
  entry (`sensoryforge-run-simulation`), not from a guess.

## 2. Run it

```
sensoryforge evaluate run FILE --out runs/NAME --preset NAME --draws N --duration-ms MS \
    --seed S --packs core
```

It prints a line per stage and the stage times, then the check counts. Exit 0 means every
stage finished and no check failed; exit 1 means a stage failed or a check did. A stage
that fails stops the rest, they are recorded as skipped and the finished stages stay in
`runs/NAME/evaluate/`. If entries failed to simulate, fix the cause and run it again with
`--resume`.

## 3. Read the report back

Run the summary script from the worked example below, or read the files yourself. They are
all in `runs/NAME/evaluate/`; the report contract at
https://benefron.github.io/sensoryforge/reference/report_contract/ defines each field.
Say, in this order:

1. **Did it finish?** `pipeline.json`: status, the error if any, the counts (entries,
   audited, compared pairs, checks passed, failed, skipped, errored) and every note.
2. **What failed.** `checks.json`: for each failed or errored check, its title, what it
   tests, how many it tested and how many failed, its numbers, and the first failing
   entries. For each skipped check, say that it was not run and why (its note).
3. **What the data show against what the world declares.** The comparison table, with the
   three numbers always together: measured (from the generated data alone), declared (from
   the world file) and the raw difference (measured minus declared); the resolution of the
   measurement and its unit next to them. Print it with `python -m
   sensoryforge.evaluate.compare runs/NAME/evaluate runs/NAME/evaluate`.
4. **What the report did not cover.** Metrics only one side has, pairs whose units could
   not be compared, populations or splits that were not run.

Do not call a difference good or bad. The report states no tolerance, and neither do you:
if the user asks "is this good enough?", give the numbers and the resolution and ask what
difference they accept, or for a source that says. Do not hide a failed or skipped check
in a summary. When the user asks what a number means or what is typical, use
`sensoryforge-interpret-report`; when a number or a check is not what they expected, use
`sensoryforge-diagnose`.

## 4. Look and share

`sensoryforge evaluate serve runs/NAME` serves the data set to a browser on this machine
only (it prints the address); start it in the background, give the user the address and
stop it when they are done. `sensoryforge evaluate export runs/NAME OUT` writes a
snapshot that opens offline with no server (`--help` shows what it records).

## 5. Stop and report

Give the report folder, the counts, what failed and the comparison in the form above.
A failure is the user's to decide on, whatever the pace: ask what they want to do, and
offer to trace it with `sensoryforge-diagnose`. With no failure, offer the next step at the
pace the brief sets (ask first, or say what comes next and carry on): explaining a metric
with `sensoryforge-interpret-report`, or tracing a number that surprises them.

## Worked example

A tiny world through a shipped recipe, and a summary of the report.

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

```python
# file: summarize.py
"""Say what an evaluation report folder says: status, counts, notes, failed checks."""
import json
import sys

folder = sys.argv[1] + "/evaluate"
pipe = json.load(open(folder + "/pipeline.json"))
print("status:", pipe["status"], "| error:", pipe["error"])
print("counts:", json.dumps(pipe["counts"]))
for note in pipe["notes"]:
    print("note:", note)
for check in json.load(open(folder + "/checks.json"))["checks"]:
    if check["status"] != "pass":
        print(check["status"], check["id"], "-", check["title"], "|", check["note"])
        print("  tests:", check["what"])
        print("  tested", check["tested"], "failed", check["failed"], check["numbers"])
compare = json.load(open(folder + "/compare.json"))
print("pairing:", json.dumps(compare["counts"]), "| difference is", compare["difference"])
```

The user names the pack for a touch world (tactile); the example does the same.

```bash
sensoryforge evaluate run world.yml --out runs/demo --preset tactile_sa1_ra1_adex --draws 6 --duration-ms 80 --seed 7 --packs core,tactile
# prints: 0 failed
python summarize.py runs/demo
# prints: status: ok
python -m sensoryforge.evaluate.compare runs/demo/evaluate runs/demo/evaluate
# prints: core.frames.duration
```
