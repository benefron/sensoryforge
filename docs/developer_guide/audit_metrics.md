# The audit's metrics

The blind audit measures a data set from its generated data alone (see
[the audit reader](audit_reader.md) for how it reads). This page covers what it
measures, how to run it, and how to add a metric pack. The formats and every metric
definition are in the [report contract](../reference/report_contract.md), sections 4 to 6;
this page does not repeat them.

## Run it

From a data-set folder alone (the batch output: `manifest.jsonl`, `index/`, one bundle per
entry):

```bash
python -m sensoryforge.evaluate.audit DATA_DIR --out OUT_DIR
python -m sensoryforge.evaluate.audit DATA_DIR --units frame=kPa --param core.zero_tolerance=0.01
```

`--out` defaults to `DATA_DIR/evaluate`. The command writes two files:

| File | Holds |
|---|---|
| `audit.json` | the inventory (layout, populations, resolved units), every metric's definition, the summaries over the data set, per class and per split, the failures and the data-set values |
| `audit_entries.jsonl` | one row per entry, sorted by entry id: its labels and every value |

The same thing from Python:

```python
from sensoryforge.evaluate.audit.run import run_audit

audit = run_audit("data", "out", packs=["core"], params={"core": {"zero_tolerance": 0.0}})
```

`audit.json` validates against `schemas/evaluate/audit.schema.json` and each row against
`#/$defs/entry_row`. Two runs over the same data with the same parameters are byte-identical
apart from `created_at`. The command exits with status 2 and a message on stderr when the
folder has no manifest, a pack is unknown or a unit contradicts what the data names.

An entry that cannot be measured does not stop the run. It gets a row with `values: []` and
a record in `failures` with a stage: `batch` (the index says it failed), `read` (the bundle
is missing or differs from the first one) or `measure` (a pack raised on it).

## What the core pack measures

| Section | Scope | Reads |
|---|---|---|
| `core.frames.*` | per channel | the stimulus frames: timeline (onset, offset, bouts, gaps), amplitude (peak, mean, rms), spatial extent (footprint, centroid, spread), motion (path length, speed, direction), the dominant frequency of the longest bout and a digest |
| `core.input.*`, `core.response.*` | per population | the drive and filtered current; spike counts and rates, latency, ISI CV, active-unit fraction (spikes); signed event counts (events); the state's statistics (analog) |
| `core.run.seconds` | per entry | the batch index |
| `core.dataset.*` | once per data set | how many entries are silent and how many repeat an earlier entry's frames |

All of it is computed from arrays, so none of it names a modality. A frame metric has no
unit of its own: its unit is the one the data names for the quantity (`time`, `space`,
`frame`, ...) or `null`; a ratio such as a speed is composed from them (`px/ms`) and is
`null` if either part is. Where the contract leaves a detail open, section 6.5 says what
the audit does.

Every metric is kept as a distribution, not only a mean: each summary carries `n`,
`n_null`, mean, standard deviation, min, the 5, 25, 50, 75 and 95 percent quantiles, max and
a histogram (`histogram_bins`, default 32).

## Parameters

| Pack | Parameter | Default | Meaning |
|---|---|---|---|
| `core` | `zero_tolerance` | 0 | a frame value is active when its magnitude exceeds this (frame unit) |
| `core` | `histogram_bins` | 32 | bins of every summary's histogram |

Pass them with `--param core.zero_tolerance=0.01` or `params={"core": {...}}`. They are
recorded under `params` in `audit.json`.

## Write a metric pack

A pack is a class with a name, a version, its parameters and two methods. It follows the
component pattern, so it round-trips through `to_dict()`/`from_config()` and lists its
parameters in `get_param_spec()`.

```python
from sensoryforge.evaluate.pack import MetricPack, metric_definition
from sensoryforge.evaluate.registry import METRIC_PACK_REGISTRY


class PeakCount(MetricPack):
    name = "peaks"
    version = "1"

    def metrics(self, layout, units, dt=None):
        return [metric_definition("peaks.frames.n_sensors_over_half", None, scope="entry",
                                  value_type="integer")]

    def measure(self, view):
        # view.frames [T, C, M], view.populations, view.layout, view.units, view.dt
        n = int((view.frames.max(axis=(0, 1)) > 0.5).sum())
        return [{"id": "peaks.frames.n_sensors_over_half", "value": n}]


METRIC_PACK_REGISTRY.register("peaks", PeakCount)
```

Run it with `--packs core,peaks`. A third-party pack registers itself from the callable of a
`sensoryforge.components` entry point, like any other component. Two things to know:

- `measure()` sees only the entry view built by the reader. It cannot reach the world file
  or a draw's record, and a test fails if any module reachable from the audit imports
  `sensoryforge.world`.
- A pack that needs data-set level values lists the per-entry metric ids it needs in
  `dataset_inputs` and implements `measure_dataset(rows)`; the audit keeps those values for
  every entry.

The pack and registry modules are `sensoryforge/evaluate/pack.py` and
`sensoryforge/evaluate/registry.py`; the core pack is
`sensoryforge/evaluate/audit/core_pack.py`, with its arithmetic in `frame_metrics.py`
and `response_metrics.py` and the summariser in `summary.py`.
