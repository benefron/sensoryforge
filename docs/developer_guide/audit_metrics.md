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

## The tactile pack

`--packs core,tactile` adds four per-channel metrics that read a touch episode back from the
frames: how long it takes to reach its peak, how long it stays there and how long it takes to
release, and how fast its centre moves while it holds. They exist so the declared side, which
knows the episode's touch, hold and release times, has something measured to be set against.

| Metric | Unit | Reads |
|---|---|---|
| `tactile.episode.rise_time` | time | from the start of a bout to the first frame on its plateau |
| `tactile.episode.plateau_time` | time | the plateau: first to last frame near the bout's peak |
| `tactile.episode.fall_time` | time | from the end of the plateau to the end of the bout |
| `tactile.motion.speed` | space/time | the mean speed of the centroid steps that move, in the entry's bouts |

Speed is the pooled length of the centroid steps between consecutive frames that are
longer than the *motion floor* (`1e-6` of the layout's largest extent, so float32 rounding of
a still shape is not motion), over their number times `dt`; the plateau plays no part. It is
null when nothing moves.

The three phases are measured per bout on the **envelope**, the largest absolute value over the sensors
in each frame, and averaged over the entry's bouts. A bout that only rises or only falls has
a fall or rise of 0; a bout with a flat top is all plateau. The frames lie at the data's own
step, so a phase is a whole number of frames times `dt` and carries `dt` as its resolution.
For an unmodulated stimulus the three phases add up to `core.frames.bout_mean` when the entry
has one bout; under a modulation the envelope oscillates and the numbers still come out, but
they no longer mean touch, hold and release (contract, section 7).

The pack also states the unit of `drive` and `filtered` as `mA`, with the units guide as its
source, so `core.input.*` carries `mA` when it runs. It states no unit for the frame values:
a world's amplitude is unitless, so a frame unit comes from `--units frame=...` or stays
`null`.

Section 7.1 of the contract lists the choices the implementation makes where the definition
leaves a detail open. The code is `sensoryforge/evaluate/tactile/metrics.py`; it sits inside
the import wall and is free to use tactile words, where the core is not.

## Parameters

| Pack | Parameter | Default | Meaning |
|---|---|---|---|
| `core` | `zero_tolerance` | 0 | a frame value is active when its magnitude exceeds this (frame unit) |
| `core` | `histogram_bins` | 32 | bins of every summary's histogram |
| `tactile` | `plateau_tolerance` | 0.01 | a frame is on the plateau when its envelope is within this share of the bout's peak |
| `tactile` | `motion_tolerance` | 0 | the entry counts as moving when its moving centroid steps total more than this (space unit) |

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
and `response_metrics.py` and the summariser in `summary.py`. A modality's pack is a module
of its own (`sensoryforge/evaluate/tactile/metrics.py`); `registry.py` is the one core file
that names it, where the built-in packs are registered.
