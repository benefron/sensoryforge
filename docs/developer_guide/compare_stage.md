# The raw comparison

The audit says what a data set *shows* ([the audit's metrics](audit_metrics.md)) and the
declared stage says what its declaration says it *should* show ([the declared
side](declared_adapter.md)). The compare stage puts the two side by side: for every value
the declared side carries, the measured value, the declared value and the raw difference
`measured - declared`. It gives **no verdict**. A pass or fail rule is a later stage with
its own file, which reads `compare.json` and never changes it. The formats are in the
[report contract](../reference/report_contract.md), section 9.

The stage reads `audit.json` and `declared.json` and the two `_entries.jsonl` files beside
them. It reads no bundle, no world file and no manifest, and it converts no unit.

## Run it

```bash
python -m sensoryforge.evaluate.compare AUDIT DECLARED --out OUT_DIR
```

`AUDIT` and `DECLARED` are the two files or the folders that hold them; `--out` defaults to
the audit's folder. It writes `compare.json` and `compare_entries.jsonl`, then prints the
data set's table. From Python:

```python
from sensoryforge.evaluate.compare import run_compare, format_stat_table

compare = run_compare("out/audit", "out/declared", "out/compare")
print(format_stat_table(compare, "dataset", stats=["mean", "q50"]))
```

Both inputs must come from the same data set: the stage checks each entry file against the
hash its `.json` records and refuses a pair whose entry ids differ. Two runs over the same
inputs are byte-identical apart from `created_at`. The input paths are recorded relative to
the output folder, with their hashes. The command exits with status 2 and a message on
stderr for a missing file, a hash or format mismatch, or a table group that does not exist.

## How a value is paired

An entry's row holds one pair for every value its declared row carries: the metric id (and
the channel or population when it has one), `measured`, `declared`, `diff`, `rel_diff`,
`resolution` and `pairing`. `diff = measured - declared`, exact for two integers.
`rel_diff = diff / |declared|`, empty when `declared` is 0. `resolution` is the audit's
quantisation step for the metric, usually `dt` for a time: a difference smaller than about
one `dt` is the frame grid, not an error.

`pairing` says whether a difference could be computed. The first rule that applies decides.

| pairing | when | `diff` |
|---|---|---|
| `missing_measured` | the measured value is absent or `null` (a silent entry, a failed entry, a metric the audit did not run) | empty |
| `missing_declared` | the declared value is `null` | empty |
| `not_numeric` | either value is not a number (a digest) | empty |
| `unit_mismatch` | both units are known and they differ | empty |
| `unit_unchecked` | both are numbers and one unit is unknown (the frame values carry no unit) | computed |
| `ok` | both are numbers and the units are known and equal | computed |

A `unit_mismatch` keeps both values so you can see them; nothing is converted.

## The files

`compare.json` lists, once each: the metrics both sides carry with their unit, resolution
and `n_pairs` (the entries with a computed difference); the ids only the audit measured
(`unpaired.measured_only`) and the ids only the declaration carries
(`unpaired.declared_only`, which pair as `missing_measured` in every entry); and `counts`,
the entry pairs by pairing.

`dataset` and `groupings` pair the two sides' **summaries** statistic by statistic (`n`,
`mean`, `std`, `min`, `q05` to `q95`, `max`) for the data set, each class and each split.
Read the `n` rows first. The two summaries cover different entries when the declaration
leaves a value out (a modulated draw has no declared peak, a still draw has no speed), so
their means differ for that reason alone. The statistics of the *same entries* are in
`distributions`.

`distributions` (optional field, the tag is unchanged) has, for each metric and each group,
over the entries whose difference was computed:

- `n_pairs`;
- `diff_mean` and `diff_abs_mean`, the mean and the mean absolute value of the entry
  differences;
- `ks`, the Kolmogorov-Smirnov distance between the measured and the declared values
  (unitless, 0 to 1);
- `wasserstein`, the first Wasserstein distance (the mean absolute difference of the sorted
  values, in the metric's unit; both samples have the same size, so it is exact);
- `histogram`, the two samples' counts on shared equal-width bins (`--histogram-bins`).

All four are `null` where `n_pairs` is 0. They are numbers to read, not thresholds.

## Printing

Every number is printed with `repr`, the shortest text that reads back as the same number,
so a printed difference is the stored one. A missing value is `-`.

```bash
python -m sensoryforge.evaluate.compare out out --table class:dots --stats mean,q50
python -m sensoryforge.evaluate.compare out out --table entry:test/dots/0042
python -m sensoryforge.evaluate.compare out out --distributions
python -m sensoryforge.evaluate.compare out out --histogram core.frames.onset_time
```

`--table` takes `dataset` (default), `class:LABEL`, `split:LABEL` or `entry:ID`;
`--metrics` and `--stats` narrow the rows. `--quiet` writes the files only.

## Why there is no tolerance file here

An earlier note on this task asked for verdicts in an optional file of tolerances. The
contract (choice Q8, approved) keeps the comparison free of them: the reference-free and
tolerance checks are their own stage (roadmap A8), so the table cannot change when a
tolerance does.
