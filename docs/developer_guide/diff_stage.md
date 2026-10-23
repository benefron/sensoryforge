# Run-to-run diff

You change the simulator (a refactor, a different neuron, a new torch) and want to know
exactly what moved. The diff stage compares two runs of the **same** data set, a baseline and
a new one, and reports: a verdict, per-draw divergence, the first frame at which each draw's
arrays differ, and per-metric deltas. A change that alters nothing reports zero differences.
The format is in the [report contract](../reference/report_contract.md), section 14.

## Run it

```bash
python -m sensoryforge.evaluate.diff BASELINE_DIR NEW_DIR --out OUT_DIR
```

`BASELINE_DIR` and `NEW_DIR` are the batch outputs of the two runs (manifest, index, one
bundle per entry). The command writes `diff.json` and `diff_entries.jsonl`, plus the two
audits it makes for the metric deltas (`audit_baseline/`, `audit_new/`), and prints the
report. It exits with status 0 when there is no difference (identical, or equal within the
tolerance), 1 when the runs differ, 2 on a usage or data error.

| option | meaning |
|---|---|
| `--atol`, `--rtol` | the tolerance (default: the golden fixtures', `atol` 2e-7, `rtol` 0) |
| `--no-metrics` | skip the per-metric deltas (no audits are run) |
| `--packs` | metric packs of the two audits (default `core`) |
| `--audit-baseline`, `--audit-new` | reuse a folder that already holds that side's audit |

From Python:

```python
from sensoryforge.evaluate.diff import run_diff

doc = run_diff("runs/before", "runs/after", "out/diff")
print(doc["verdict"], doc["counts"])
print(doc["first_divergence"])     # {'entry': ..., 'step': 40, 'ms': 40.0}
```

## What it compares

For every entry both runs hold, the arrays the audit may read: the stimulus frames, the frame
times and each population's readout (spikes, events or state), drive and filtered current.
Each array is compared along its time axis, value by value, and ends up **identical** (bit for
bit), **equivalent** (a difference within the tolerance) or **different**.

The tolerance is `sensoryforge.testing.golden.FLOAT32_GOLDEN_ATOL`, the one the golden
fixtures use. It exists because another processor rounds float32 differently in the last step
(measured up to 6e-8, roughly 3 times below `atol`); a diff that flagged that would report a
change on every machine. Spike and event counts are whole numbers, so any change in one is a
difference, never rounding. Comparisons *within* one process stay bit-exact by convention, so
if you compare two runs from the same machine, pass `--atol 0` to see every last step.

An entry's **first divergence** is the earliest frame, over all its arrays, at which a value is
beyond the tolerance. That is where to look first: a change that starts at frame 0 touches the
stimulus or the wiring; one that starts mid-run is the dynamics.

The **metric deltas** run the [audit](audit_metrics.md) on both sides and pair the per-entry
values: for each metric (per channel and population where it has them), how many entries
changed, the mean on each side and the largest change with its entry. A change in the spike
count of one entry shows as one changed pair of `core.response.count`.

## Reading the result

- `verdict` is `identical`, `equivalent` (only last-step differences) or `different`.
- `counts` are entries by status, including entries on one side only (`only_a`, `only_b`),
  entries one side could not run (`unreadable`) and entries that failed the same way on both
  sides (`both_failed`, which is no change).
- `by_array` says which arrays moved (`sa.spikes`, `frames`) and in how many entries;
  `most_divergent` lists the worst draws, to open in the [evaluation
  server](evaluation_server.md) (`/#entry=ID`).
- `notes` flags a pair of data sets with different `dataset_id`s (entries are still paired by
  id).

## Limits

- Both runs must be the same data set (the same manifest). A diff of two *different* data sets
  pairs only the entries they share and says so.
- Entries are compared whole; a run that differs only in one sample of a long entry shows its
  array in `arrays` with the count of differing values and the first frame, not a plot.
- The diff reads bundles through the blind reader, so it does not compare `voltage_probe`
  (the audit never opens it) or the bundle's recorded configuration. Compare the voltage with
  the checks stage (`core.run.rerun_matches`).
- No page in the evaluation server yet: the report is the command's output and `diff.json`.

## Where the code is

`sensoryforge/evaluate/diff/`: `arrays.py` (one array), `entries.py` (one entry), `metrics.py`
(the deltas), `run.py` (the stage) and `__main__.py`. The schema is
`schemas/evaluate/diff.schema.json`. The tests are `tests/unit/test_eval_diff.py`, including
that two engine runs of one data set report no difference.
