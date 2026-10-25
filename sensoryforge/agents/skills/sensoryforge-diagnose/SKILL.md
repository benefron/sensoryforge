---
name: sensoryforge-diagnose
description: Trace a surprising or bad SensoryForge result back to the draws, parameters and code behind it. Use when a report, check, plot or run gives something unexpected (a failed or skipped check, a silent or saturated population, a measured value far from the declared one, an entry that did not run, two runs that disagree), in whatever words they put it.
---

# Diagnose

A bad number is a symptom. Your job is to find which entries it comes from, open their
draws, and say which of four places holds the cause, with the evidence for it and what is
not yet shown. You change nothing until the user has chosen what to change.

## 0. Read the working style

Run `sensoryforge agents brief` and follow it. If the command is not found, SensoryForge
is not installed here: say so and stop (it installs from source; it is not on PyPI yet).

## 1. Pin down the symptom

Get it as numbers from a file, not as an impression: which report folder, which metric
or check id, which value, and what the user expected instead (and where that expectation
comes from). If the user has no report yet, run `sensoryforge-evaluate` first; if they
want to know what a metric means before judging it, use `sensoryforge-interpret-report`.
Write the symptom down in one line before you look further.

## 2. Find the entries behind it

The report folder is the one given to `sensoryforge evaluate run --out`. The script in the
worked example reads it:

- `python diagnose.py DIR odd` lists every entry that did not run, has no active
  stimulus frame, or has a value the comparison could not pair (and why). It exits 1 when
  it finds any.
- `python diagnose.py DIR largest ID 5` lists the entries where one metric differs most
  between measured and declared, with the resolution beside each.
- For a failed check, the entries are in its listed failures (`checks.json`, and
  "entries_with_failures"); for a population that is silent or saturated in many entries,
  the anomaly gallery ranks them (`sensoryforge evaluate serve DIR`, then the Gallery
  page; see `sensoryforge-evaluate` step 4).

A few entries behind a symptom is a draw problem. Every entry of a class, or of one
population, is a sensor, data-set or code problem. Say which you see.

## 3. Open one entry

`python diagnose.py DIR entry ENTRY` prints what the data set holds about it: its class,
split, the duration it was cut to and whether the draw was cut off by it, the seeds, the
draw's declared values and timeline, the batch status and error, and each compared metric
with measured, declared, difference and pairing. This reads the world's declaration. That
is allowed here because the number is already in the report; the audit itself stays blind.

For the data behind it, `sensoryforge world sample WORLD --n N --seed S` prints draws as
JSON lines, and the bundle's own `config.json` shows the sensor the entry ran on.

## 4. Sort the cause into one of four places

Name the place, the evidence and how sure you are (shown by a test, likely, or unknown).

1. **The data set asks for it.** The draw's values or timeline put the stimulus where the
   sensors cannot see it: it starts after the duration, is cut off by it, lies off the
   canvas, or is a probe outside an axis's range. Evidence: the entry's timeline against
   its duration. Owner: the world or data-set file (`sensoryforge-design-world`,
   `sensoryforge-build-dataset`).
2. **The sensor does it.** A recipe's or config's receptive fields, filter, gain, input
   floor or neuron model produce silence, saturation or a rate the user did not expect.
   Evidence: the same draws through another sensor the user names, or the bundle's config.
   Owner: the user. Never edit a shipped recipe; propose a config change as a file the
   user approves (`sensoryforge-run-simulation`).
3. **The measurement cannot see it.** The metric's definition, or its resolution (one
   frame step for time values), explains the number: activity shorter than a frame, a value
   that is null by definition, a shape the canvas cuts off. Evidence: the raw difference
   against the resolution, and the definition in the report contract,
   https://benefron.github.io/sensoryforge/reference/report_contract/ (each metric's
   "how the audit applies these definitions" notes). Say what the metric measures, then
   what it cannot.
4. **The code does it.** A check under the run area fails (determinism, a re-run that does
   not reproduce the stored spikes), an entry fails with an error, or two runs of one data
   set disagree: `python -m sensoryforge.evaluate.diff BASELINE NEW` names the first
   step where they differ. Find the code by searching the installed package for the
   check's or metric's id; read it before you say what it does. If that code sits behind a
   contract (the glossary's word), do not change it to make a number look right: report it.

When two places fit, say so and say which test would tell them apart.

## 5. Test the cause, one change at a time

Change one thing in a copy (a new output folder, a copied config), re-run only the entries
that showed the symptom (`sensoryforge batch ... --entries I:J --output runs/try`, or
`sensoryforge evaluate run ... --out runs/try`), and set the new numbers beside the old
ones. A value you changed is the user's to approve first; a value you only read is yours.
If the test does not move the symptom, the cause was not that: say so and go back to
step 4. Stop after three tries and show the user what you have.

## 6. Stop and report

Give, in this order: the symptom as numbers; the entries and their ids; the cause (place,
evidence, how sure); what you tested and what moved; the options, with who decides each
(the choices are the user's); and what you did not check. Do not call a difference good or
bad (the report states no tolerance). If the cause is a missing component, go to the
"When a component is missing" part of `sensoryforge-design-and-plan`.

## Worked example

A world whose draws start 30 to 60 ms in, in a data set cut to 40 ms. The report is built,
then traced; the user then chooses 130 ms, the longest a draw can take (60 + 15 + 40 + 15,
from the world's ranges), and the same check passes.

```yaml
# file: world.yml
world:
  name: late
  defaults:
    delay_ms:   {range: [30, 60]}
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
  name: late
  world: world.yml
  seed: 7
  duration_ms: 40
  splits:
    train: {n: 6}
```

```yaml
# file: dataset_long.yml
dataset:
  name: late
  world: world.yml
  seed: 7
  duration_ms: 130
  splits:
    train: {n: 6}
```

```python
# file: diagnose.py
"""Trace a surprising number in an evaluation report back to the draws behind it.

    python diagnose.py DIR odd             entries that did not run, are silent or unpaired
    python diagnose.py DIR largest ID [N]  the N entries where metric ID differs most
    python diagnose.py DIR entry ENTRY     what the data set holds about one entry

DIR is the folder given to `sensoryforge evaluate run --out`. It reads the world's
declaration (the manifest) on purpose: the audit stays blind, a diagnosis does not.
"""
import json
import sys
from pathlib import Path

from sensoryforge.world import read_batch_index

root, mode, rest = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]


def rows(name):
    return [json.loads(line) for line in open(root / name)]


def short(value):
    return round(value, 4) if isinstance(value, float) else value


pairs = {r["entry"]: r["pairs"] for r in rows("evaluate/compare_entries.jsonl")}
index = {r["entry"]: r for r in read_batch_index(root)}

if mode == "odd":
    found = 0
    for entry, row in index.items():
        if row["status"] != "ok":
            print(entry, "did not run:", row["status"], row["error"])
            found += 1
    for row in rows("evaluate/audit_entries.jsonl"):
        bouts = [v["value"] for v in row["values"] if v["id"] == "core.frames.bouts"]
        if bouts and not any(bouts):
            print(row["entry"], "silent: no active stimulus frame")
            found += 1
    for entry, entry_pairs in pairs.items():
        for kind in sorted({p["pairing"] for p in entry_pairs} - {"ok", "unit_unchecked"}):
            ids = [p["id"] for p in entry_pairs if p["pairing"] == kind]
            print(entry, kind, "->", ", ".join(ids))
            found += 1
    sys.exit(1 if found else 0)
elif mode == "largest":
    metric, count = rest[0], int(rest[1]) if len(rest) > 1 else 5
    found = [
        (entry, p)
        for entry, entry_pairs in pairs.items()
        for p in entry_pairs
        if p["id"] == metric and p["diff"] is not None
    ]
    for entry, p in sorted(found, key=lambda item: -abs(item[1]["diff"]))[:count]:
        print(entry, "measured", short(p["measured"]), "declared", short(p["declared"]),
              "diff", short(p["diff"]), "resolution", p["resolution"])
elif mode == "entry":
    entry = rest[0]
    row = next(r for r in rows("manifest.jsonl") if r["entry"] == entry)
    print("entry", entry, "| class", row["class"], "| split", row["split"])
    print("duration_ms", row["duration_ms"], "| cut off by it:", row["truncated"],
          "| seeds", row["seeds"])
    print("draw values:", json.dumps({k: short(v) for k, v in row["draw"]["values"].items()}))
    print("timeline:", [(n, short(a), short(b)) for n, a, b in row["draw"]["timeline"]])
    print("batch:", index[entry]["status"], index[entry]["error"] or "")
    for p in pairs[entry]:
        print(" ", p["id"], "measured", short(p["measured"]), "declared", short(p["declared"]),
              "diff", short(p["diff"]), p["pairing"])
else:
    sys.exit("modes: odd | largest ID [N] | entry ENTRY")
```

```bash
sensoryforge evaluate run dataset.yml --out runs/late --preset tactile_sa1_ra1_adex --packs core,tactile --no-checks
# prints: report folder
python diagnose.py runs/late odd
# prints: silent: no active stimulus frame
# exit: 1
python diagnose.py runs/late entry train/r0/00002
# prints: cut off by it: True
python diagnose.py runs/late largest core.frames.active_time 2
# prints: resolution 1.0
sensoryforge evaluate run dataset_long.yml --out runs/long --preset tactile_sa1_ra1_adex --packs core,tactile --no-checks
# prints: report folder
python diagnose.py runs/long odd
# exit: 0
```
