---
name: sensoryforge-interpret-report
description: Explain in plain words what a SensoryForge report's metrics, checks and comparison columns mean and what is typical for the data in front of you. Use when someone asks what a number, metric, check, column or file in a report means, whether a value is normal or typical, or how to read a report, in whatever words they put it.
---

# Interpret a report

A report is numbers with definitions. Your job is to say what a number measures, in what
unit and at what resolution, and what is typical for this data set, from the report and
the contract only. You never judge a value good or bad: the report states no tolerance.

## 0. Read the working style

Run `sensoryforge agents brief` and follow it. If the command is not found, SensoryForge
is not installed here: say so and stop (it installs from source; it is not on PyPI yet).

## 1. Find what is being asked about

The user names a metric, a check, a column, a file, or the whole report. Map their words
to what exists by meaning, not by string: `python explain.py REPORT --list` prints every
metric id in the report (the folder given to `--out` of `sensoryforge evaluate run`), and
`checks.json` lists the check ids. When two ids could fit, say which two and ask. If the
report does not hold it, say so; do not describe a metric the report did not measure.

## 2. Say what it measures

Read its definition in the report contract before you explain it:
https://benefron.github.io/sensoryforge/reference/report_contract/. The core metrics are
in section 6, a modality's own pack follows it (section 7 is the tactile pack), the
comparison is section 9, the checks section 13, the run-to-run diff section 14 and
`pipeline.json` section 15. If you cannot open the page, say so and give only what the
report folder itself shows (the unit, the resolution and the scope below); do not explain
a definition from memory.

Then say, in plain words and in this order:

1. **What it measures**, in one sentence, and over what: which frames or units, per entry
   or for the whole data set, per channel or per population.
2. **Its unit** and where the unit came from (the data, the user, a metric pack, derived
   from other units, or unknown). A metric with no stated unit is still a number; say the
   unit is unknown rather than inventing one.
3. **Its resolution**: for a time value measured on the frame grid, one frame step is the
   smallest difference the data can show, and a difference below it is not one the data can
   see. The report lists it for the frame times and the response latency. Where it lists none
   (a metric that is not a time, or bundles whose steps differ), say so; do not supply one.
4. **When it is empty** (null): the contract says when. Say it in words, for example that
   a response latency is empty when the stimulus never became active.

`python explain.py REPORT ID` prints the report's own row for an id: its unit with the
unit's origin, its resolution, its scope and whether it is per entry or for the data set.
It does not know the meaning; the contract does.

## 3. Say what is typical, and only from these sources

"Normal" has three sources. Name which one you use, and use no other.

- **This data set's own spread.** `python explain.py REPORT ID` prints, for the whole data
  set and for each class and split, how many entries have a value, how many are empty, and
  the minimum, the 5th, 50th and 95th percentile, the maximum, mean and standard
  deviation. Say "in this data set, half the entries are below X", with the numbers.
- **A statement in the contract or the docs.** Quote it with its section.
- **A source the user approves** (a paper, a benchmark, a measurement of theirs). Ask for
  it, or offer a short search and report for them to approve first. Never fill in "typical"
  values for a neuron, a sensor or a stimulus from memory: those are domain facts, and they
  come from the user or from a source the user approves.

If the user asks "is this good enough?", give the numbers and the resolution and ask what
difference they accept.

## 4. Read a comparison, a check, a file

- **A comparison row** has three numbers that always go together: measured (from the
  generated data alone), declared (from the world file) and the difference (measured minus
  declared), with the resolution beside them. The relative difference is empty when the
  declared value is 0. The "pairing" column says whether a difference could be computed:
  read its word from the contract's section 9 and say it in plain words.
- **A check** is pass, fail, skipped or error. A skipped check was not run (its note says
  why) and is not a failure; an error is a read or engine error. Its numbers are the raw
  values behind the verdict, and a failure lists the first entries it failed on.
- **The files:** the pipeline file says what was run, not what was found; the audit file is
  measured from the data alone; the declared file is what the world says; the comparison
  sets them side by side; the checks file tests the simulator on its own; the diff file
  compares two runs of one data set.

## 5. Stop and offer the next step

Answer what was asked and stop. If a number looks surprising to the user, offer
`sensoryforge-diagnose`; if they want more of the report read back, `sensoryforge-evaluate`.

## Worked example

A small report, then one metric explained from it: its row and this data set's spread.

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
# file: explain.py
"""What a report holds about a metric: its row and this data set's own spread.

    python explain.py REPORT --list     every metric id in the report
    python explain.py REPORT ID         its unit, resolution and scope, then the spread

REPORT is the folder given to `sensoryforge evaluate run --out`. The meaning of a metric is
in the report contract, not here.
"""
import json
import sys

folder, what = sys.argv[1] + "/evaluate", sys.argv[2]
audit = json.load(open(folder + "/audit.json"))
metrics = {m["id"]: m for m in audit["metrics"]}

if what == "--list":
    for metric_id, m in metrics.items():
        print(metric_id, "|", (m["unit"] or {}).get("symbol"), "|", m["scope"])
    sys.exit()

m = metrics[what]
unit = m["unit"] or {}
print(what, "| unit:", unit.get("symbol"), "| origin:", unit.get("origin"))
print("resolution", m["resolution"], "| scope", m["scope"], "| level", m["level"])
groups = {"data set": audit["dataset"]}
for kind, labels in audit["groupings"].items():
    groups.update({kind + " " + label: group for label, group in labels.items()})
columns = ("n", "n_null", "min", "q05", "q50", "q95", "max", "mean", "std")
for name, group in groups.items():
    for row in group["summaries"]:
        if row["id"] == what:
            where = row.get("population") or row.get("channel") or ""
            numbers = [f"{c} {row[c]:.4g}" if row[c] is not None else f"{c} -" for c in columns]
            print(name, where, "|", ", ".join(numbers))
```

```bash
sensoryforge evaluate run world.yml --out runs/demo --preset tactile_sa1_ra1_adex --draws 6 --duration-ms 80 --seed 7 --packs core --no-checks
# prints: report folder
python explain.py runs/demo --list
# prints: core.response.latency
python explain.py runs/demo core.response.latency
# prints: unit: ms
# prints: resolution 1.0
```
