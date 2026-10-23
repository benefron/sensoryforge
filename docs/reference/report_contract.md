# Evaluation report contract

!!! note "Status: accepted (roadmap task A1, reviewed and approved 2026-10-22)"
    This page fixes the formats the evaluation stages exchange before any of them is written.
    The audit's reader layer (A2a, see the [audit reader](../developer_guide/audit_reader.md))
    implements section 2; the core metrics, the summariser and `audit.json` (A2b–A2c, see
    [the audit's metrics](../developer_guide/audit_metrics.md)) implement sections 4–6, the
    tactile pack (A2d) section 7, the declared side (A3, see [the declared
    side](../developer_guide/declared_adapter.md)) section 8, the comparison (A4, see [the raw
    comparison](../developer_guide/compare_stage.md)) section 9. The JSON Schemas are in
    `schemas/evaluate/`. Section 11 lists the choices this page makes, all approved as
    written.

Evaluating a data set takes three stages that never share code. Each stage writes one file
the next one reads:

```
batch output (bundles, index) ──► audit ──────────► audit.json     (+ audit_entries.jsonl)
data set + world file ─────────► declared adapter ─► declared.json  (+ declared_entries.jsonl)
audit.json + declared.json ────► compare ─────────► compare.json   (+ compare_entries.jsonl)
```

- **audit**: everything that can be measured from the generated data alone. It never reads
  the world file or any draw's record (the **blind rule**, section 2).
- **declared**: what the world file and the data set's manifest imply each entry should
  show, expressed with the same metric ids and units as the audit.
- **compare**: for every metric both sides carry, the measured value, the declared value and
  the raw difference `measured − declared`. It gives no verdict. Pass/fail checks are a later
  stage with its own file (A8), and they read `compare.json`. They never change it.

The core is **modality-neutral**. Its words are *sensor*, *channel*, *frame*, *population*,
*draw* and *entry*, and it assumes no unit and no square grid. A modality brings a **metric
pack**,
a **frame renderer** and a **declared adapter**, registered like any other component (section
10). Tactile is the first of each. Tactile words (contact, pressure, indentation, receptor)
appear only in the tactile pack's metrics (section 7) and adapter (section 8.3).

## 1. Words

| Word | Meaning here |
|---|---|
| data set | The output of `sensoryforge dataset build` (`dataset.json`, `manifest.jsonl`) and of the batch run over it (one bundle per entry, `index/task_*.jsonl`) |
| entry | One row of the manifest and one bundle; its id is the manifest's `entry` (`test/dots/0042`) |
| draw | The world's sampled values behind an entry. Several entries can share a draw (`noise_repeats`); only the declared side reads it |
| frame | The stimulus at one time step: `[C, M]` values, one per channel and sensor |
| sensor | One point at which frames are given, with coordinates `[D]` (D = 1, 2 or 3) in the data's space unit |
| channel | One plane of a multi-plane frame (a grid's `channels`); one channel when the grid names none |
| population | A group of `N` units with one readout: `spikes`, `events` (signed counts) or `state` (analog) |
| class, split | The manifest's grouping labels for an entry |
| active | A frame value with `|x| > zero_tolerance` (default 0: exactly non-zero). A frame is active when any of its values is |
| bout | A maximal run of consecutive active frames; a gap is a run of inactive frames between two bouts |

Time is measured from the entry's start: frame `k` is at `t_k = k · dt`. Every array the core
reads is batch-free: frames `[T, C, M]`, a population's arrays `[T, N]`.

## 2. What the audit may read: the blind rule

The audit sees the data through one reader, which projects everything it opens onto the
allow-list below **at read time**: a field not listed never reaches a metric pack. A test
(A2a) builds the reader's view of a bundle and fails if any other key is present, and a
second test fails if any module reachable from the audit or from a registered metric pack
imports `sensoryforge.world` (D-c52e668).

| Source | Allowed | Never read |
|---|---|---|
| `data.h5` | attrs `dt_ms`, `integrate_dt_ms`, `sensoryforge_version`, `sensoryforge_sha`; `/time_ms`; `/stimulus/frames`; `/populations/<name>/{drive, filtered, spikes, events, state}` and the `events` attrs `signed`, `polarity`, `theta` | `/meta` (its `config_yaml` holds the stimulus section); `/populations/<name>/voltage_probe` (stored membrane voltage, schema 2.4.0: a bundle may hold it, no metric reads it) |
| `config.json` | `schema_version`; `grid` (`rows`, `cols`, `spacing_mm`, `center_mm`); `populations[*].{name, readout, event_encoding}`; `config.grids` (the sensor layout and channel names); `config.simulation.dt_ms`; `units` (when present); `world.{world_id, dataset_id, entry}` as opaque identifiers | `config.stimulus`, `design`, everything else |
| `stimuli/` | nothing | `stimulus.json` (it carries the manifest row and the draw as a layer) |
| `neuron_modules/` | nothing | all |
| `index/task_*.jsonl` | every column (`entry, bundle, status, error, seconds, finished_at, design_id, sensoryforge_sha, task`) | — |
| `manifest.jsonl` | the labels `entry`, `split`, `class`, `repeat`, `noise_repeat` | `draw`, `bins`, `probe`, `seeds`, `duration_ms`, `truncated` and every other column |
| `dataset.json`, the world file | nothing | all |

**Sensor coordinates.** For a world entry the frames are the stimulus rendered on grid 0's
canvas, so the sensors are that canvas's points. The reader computes their coordinates from
`config.json["grid"]` with SensoryForge's centred layout (dim 0 = x, `indexing="ij"`, sensor
`k = i · cols + j`). It does this without importing the world package. A2a's test checks
the result against `Canvas.from_grid_config`; the test may import the world package, the
reader may not.

**Class labels are grouping keys, not declarations.** The audit groups entries by `class`
and `split` so it can summarise per class. It never reads what a class declares. `bins` and
`probe` are excluded because they encode axis values.

## 3. Units

The core assumes no unit. Each quantity's unit is resolved once per data set and recorded
with its **origin**:

| Origin | Meaning |
|---|---|
| `data` | The data names it: a `units` block in `config.json` (written when a world names a unit other than `mm`/`ms`; it outranks a key's suffix), else a key's suffix (`dt_ms` gives time in `ms`, `spacing_mm` gives space in `mm`) |
| `user` | Passed to the audit (`--units frame=kPa`); recorded as given |
| `pack:<name>` | A metric pack states it, with a `source` (a doc page or citation) |
| `derived` | Composed by the core from other units (`mm/ms`, `1/ms`); `null` if any part is `null` |
| `null` | Nobody states it. The value is still reported, with the whole `unit` set to `null` (not a unit with a null symbol) |

Resolution order: a `data` unit is final. A `user` or pack unit for a quantity the data already
names must equal it, or the audit fails and names both. For a quantity the data does not name,
the `user` unit applies first, then a pack's, then `null`. The core composes units as
symbols (`"mm/ms"`, `"1/ms"`) and never converts between them. A declared adapter must
express each value in the unit the audit reports for that metric; when the two differ, the
compare stage computes no difference (section 9).

Quantities: `time`, `space`, `frame` (frame values), `drive` and `filtered` (a population's
input before and after its filter), `state` (an analog readout), plus the fixed `count`, `1`
(dimensionless) and `deg` (angles, counter-clockwise from the +x sensor axis, in [0, 360)).

What a schema-2.3.0 bundle gives today: `time` = `ms` and `space` = `mm` (origin `data`), unless
its `units` block names another (the [toy vision fixture](../developer_guide/toy_vision_fixture.md)
gives `space` = `px`; see the world contract, 10.14). Without such a block it gives no unit for
`frame`, `drive`, `filtered` or `state`. The tactile pack states `drive`
and `filtered` in `mA` (source: `docs/user_guide/units_and_gains.md`). It states no frame unit:
a world's `amplitude` is unitless in SensoryForge, so a tactile frame unit comes from the user
or stays `null`.

## 4. Common shapes

**Files and format tags.** Each stage writes `<stage>.json` plus `<stage>_entries.jsonl` next
to it, with one row per entry sorted by entry id. Rows go in a separate file because a data
set can hold 10^5 entries with tens of metrics each, and a reader should not have to load them
all to see the summaries. The tags are `sensoryforge-audit/1`, `sensoryforge-declared/1` and
`sensoryforge-compare/1`. A new optional field keeps the tag; removing a field or changing its
meaning bumps it. The schemas list each object's fields exactly, so adding an optional field
means updating the schema in the same change.

**Determinism.** Every field except `created_at` is a function of the inputs and parameters
alone. Entries are sorted by id, and metric lists follow the catalogue's order (packs in the
order given, metrics in each pack's declared order). Floats are written in Python's shortest
round-trip form. NaN and infinity are never written: an undefined value is `null`. Two audits
of the same data with the same parameters are byte-identical apart from `created_at`, which is
what the run-to-run diff (A10) relies on. The `_entries.jsonl` file's sha256 is recorded in its
`.json`.

**Metric ids** are `<pack>.<quantity>.<name>` (`core.frames.onset_time`). A metric's
definition is listed once per file, in `metrics`:

```json
{"id": "core.frames.onset_time", "pack": "core", "level": "entry",
 "value_type": "number", "unit": {"symbol": "ms", "origin": "derived"},
 "resolution": 1.0, "scope": "channel", "summarised": true}
```

`scope` says what one value is per: `entry` (one per entry), `channel` (one per channel) or
`population` (one per population; `applies_to` lists the readouts it applies to).
A metric whose `level` is `dataset` has one value for the whole data set and carries scope `entry`, meaning no channel or population index. `resolution` is the quantisation step a frame-based measurement carries (dt for times, null
when none applies). `level` is `entry` or `dataset` (section 6.4).

**A value** is `{"id", "value"}`, plus `"channel"` (the channel's 0-based index; only when the data has more than one
channel, the names are in `layout.channel_names`) or `"population"` (its name), matching the metric's scope. `value` is a number, an integer, a
string (digests) or `null` (undefined for this entry, e.g. the onset of a silent entry).

**A summary** describes one metric over a set of entries: `{id, channel?, population?, n,
n_null, mean, std, min, q05, q25, q50, q75, q95, max, histogram: {edges, counts}}`. `std` uses
ddof 0, quantiles use linear interpolation (numpy's default), and the histogram has
`histogram_bins` equal-width bins from `min` to `max` (one bin when they are equal). Only
non-null numeric values enter it: `n` counts them and `n_null` counts the entries of the group whose value is
`null` (failed entries count in neither). When `n` is 0, every statistic and `histogram` are `null`. String
metrics are not summarised. A summary carries `channel` or `population` as its metric's scope requires.

**Groupings.** Summaries are given for the whole data set (`dataset`), per `class` and per
`split`. An entry without a class (a session) appears under its split and in the data set,
not under `class`. Each group is `{n_entries, n_failed, summaries: [...]}`.

## 5. `audit.json`

```json
{
  "format": "sensoryforge-audit/1",
  "created_at": "2026-10-22T14:03:11Z",
  "producer": {"sensoryforge_version": "1.3.0", "sensoryforge_sha": "…"},
  "inputs": {"batch_dir": "…", "manifest": "…", "data_sensoryforge_sha": ["…"],
             "bundle_schema_versions": ["2.3.0"]},
  "packs": [{"name": "core", "version": "1"}, {"name": "tactile", "version": "1"}],
  "params": {"core": {"zero_tolerance": 0.0, "histogram_bins": 32},
             "tactile": {"plateau_tolerance": 0.01, "motion_tolerance": 0.0}},
  "layout": {"n_sensors": 1600, "dims": 2, "n_channels": 1, "channel_names": null,
             "bounds": [[-2.925, 2.925], [-2.925, 2.925]],
             "regular_grid": {"rows": 40, "cols": 40, "spacing": 0.15}},
  "populations": [{"name": "SA", "readout": "spikes", "n_units": 196}],
  "units": {"time": {"symbol": "ms", "origin": "data", "source": "data.h5:dt_ms"}, "…": "…"},
  "metrics": ["… one definition per metric, section 4 …"],
  "entries": {"file": "audit_entries.jsonl", "n": 1200, "n_ok": 1199, "n_failed": 1,
              "sha256": "…"},
  "failures": [{"entry": "test/dots/0042", "stage": "batch", "error": "…"}],
  "dataset": {"n_entries": 1200, "n_failed": 1, "summaries": ["…"]},
  "groupings": {"class": {"dots": {"…": "…"}}, "split": {"test": {"…": "…"}}},
  "dataset_values": [{"id": "core.dataset.duplicate_frames", "value": 0}]
}
```

`layout.regular_grid` is `null` when the sensors do not form a regular grid. `units` lists a
quantity with `null` when nobody states its unit (section 3).

A row of `audit_entries.jsonl`:

```json
{"entry": "test/dots/0042", "split": "test", "class": "dots", "repeat": 0,
 "noise_repeat": 0, "bundle": "test/dots/0042",
 "values": [{"id": "core.frames.onset_time", "value": 12.0}, "…"]}
```

Every entry in the data set has a row. A failed entry's row has `values: []` and appears in
`failures`, whose `stage` is `batch` (the index says it failed), `read` (the bundle is missing
or unreadable) or `measure` (a pack raised). Every bundle must share one sensor layout and one
set of populations. A bundle that differs fails at `read` with `layout_mismatch`.

## 6. Core metrics (pack `core`)

Parameters: `zero_tolerance` (frame unit, default 0), `histogram_bins` (default 32).

### 6.1 Frames (scope: channel; per entry)

With `x[k, m]` the channel's value at frame `k` and sensor `m`, `A` the set of active frames
and `w = |x|`:

| id | unit | definition |
|---|---|---|
| `core.frames.n_frames` | count | `T` |
| `core.frames.dt` | time | the frame step |
| `core.frames.duration` | time | `T · dt` |
| `core.frames.peak` | frame | `max x` |
| `core.frames.trough` | frame | `min x` |
| `core.frames.mean` | frame | mean of `x` over all frames and sensors |
| `core.frames.rms` | frame | root mean square of `x` over all frames and sensors |
| `core.frames.negative_fraction` | 1 | share of values `< −zero_tolerance` |
| `core.frames.onset_time` | time | `t` of the first active frame; null if none |
| `core.frames.offset_time` | time | `t + dt` of the last active frame (the end of activity); null if none |
| `core.frames.active_time` | time | `|A| · dt` |
| `core.frames.active_fraction` | 1 | `|A| / T` |
| `core.frames.bouts` | count | number of bouts |
| `core.frames.bout_mean` | time | mean bout length; null if no bout |
| `core.frames.gap_mean`, `core.frames.gap_max` | time | mean and longest gap between bouts; null with fewer than two bouts |
| `core.frames.footprint_mean`, `core.frames.footprint_max` | 1 | mean and maximum over `A` of the share of sensors that are active |
| `core.frames.centroid_x`, `_y`, `_z` | space | mean over `A` of each frame's `w`-weighted sensor position, one per dimension the layout has |
| `core.frames.spread` | space | mean over `A` of each frame's `w`-weighted RMS distance from its centroid |
| `core.frames.path_length` | space | sum of centroid steps between consecutive active frames within a bout |
| `core.frames.centroid_speed` | space/time | `path_length` over the time those steps span; null with no step |
| `core.frames.direction` | deg | direction of the net centroid displacement from the first to the last active frame (dims 0 and 1); null when it is 0 or `D < 2` |
| `core.frames.dominant_frequency` | 1/time | the frequency with the most power (the zero frequency excluded) in the real FFT of the sensor-mean signal over the longest bout, after removing its mean; null if that bout is shorter than 4 frames or carries no power. Its resolution is 1 / (that bout's length) |
| `core.frames.digest` | — | sha256 of the channel's frames as little-endian float32 in C order (a string; not summarised) |

All time metrics have resolution `dt`.

### 6.2 Population input (scope: population; per entry)

| id | unit | definition |
|---|---|---|
| `core.input.drive_peak`, `core.input.drive_mean` | drive | maximum and mean of `drive` over time and units |
| `core.input.filtered_max`, `core.input.filtered_min` | filtered | extremes of `filtered` (it is signed) |

### 6.3 Population response (scope: population; per entry)

The windows "active" and "quiet" are the stimulus's active and inactive frames (any channel),
as measured above. `N` is the population's unit count.

| id | applies to | unit | definition |
|---|---|---|---|
| `core.response.n_units` | all | count | `N` |
| `core.response.count` | spikes | count | total spikes |
| `core.response.rate_mean` | spikes | 1/time | `count / (N · duration)` |
| `core.response.rate_active`, `core.response.rate_quiet` | spikes | 1/time | spikes in the window over `N ·` the window's time; null for an empty window |
| `core.response.active_unit_fraction` | spikes, events | 1 | share of units with at least one spike or event |
| `core.response.latency` | spikes, events | time | first spike or event at or after the onset, minus the onset; null if none or no onset |
| `core.response.isi_cv` | spikes | 1 | mean over units with at least 3 spikes of their inter-spike-interval CV; null if no such unit |
| `core.response.events_on`, `core.response.events_off` | events | count | total events of each sign (`polarity` attr) |
| `core.response.state_mean`, `_std`, `_min`, `_max` | state | state | over time and units |
| `core.response.digest` | all | — | sha256 of the readout array as stored |

Per-entry runtime: `core.run.seconds` (time in `s`, origin `data`: the index's `seconds`).

### 6.4 Data-set metrics (level: dataset)

| id | unit | definition |
|---|---|---|
| `core.dataset.silent_entries` | count | entries with no active frame |
| `core.dataset.duplicate_frames` | count | entries with `noise_repeat` 0 and at least one active frame whose `core.frames.digest` equals that of an earlier such entry (by id). Noise repeats share their draw's frames by design and are left out |

### 6.5 How the audit applies these definitions

Where a definition above leaves a detail open, the audit (A2b, A2c) does this. These are
implementation choices, listed for the review at R-A4; none changes a metric's name, unit or
shape.

- **Activity per channel and across channels.** `core.frames.*` use the channel's own active
  frames. `core.response.*` windows and the onset use the *union* of the channels' active
  frames ("any channel", 6.3). `core.dataset.silent_entries` counts entries whose every channel
  has `core.frames.bouts` 0. `core.dataset.duplicate_frames` compares the tuple of an entry's
  channel digests.
- **Times.** Frame `k` is at `k · dt`, not read from `/time_ms`. `core.response.latency` is the
  index of the first spike or event at or after the onset frame, minus the onset index, times
  `dt`.
- **Centroid and weights.** `w = |x|` for every value, including values at or below
  `zero_tolerance`; a frame is a centroid candidate only when it is active, so its weight sum
  is positive.
- **`core.frames.direction`.** The net displacement counts as 0 when it is at most `1e-6`
  of the layout's largest extent (the *motion floor*), so a static blob does not get a
  direction from float32 rounding of its scaled frames, which moves its centroid by up to
  about `1e-7` of the extent. The slowest motion worth reporting is about `2e-5` of it.
  The angle is in [0, 360).
- **`core.frames.dominant_frequency`.** A bout whose mean-removed signal swings by at most
  `1e-9` of the signal's own largest magnitude carries no power (the value is `null`). Ties
  go to the lowest frequency. The metric definition's `resolution` is `null`, because the
  true resolution, `1 / (bout length · dt)`, differs per entry.
- **Rates.** `core.response.rate_*` are per time unit (1/ms for ms data), not per second.
- **Inter-spike intervals.** `core.response.isi_cv` uses the population standard deviation
  (ddof 0) of a unit's intervals, in frames; a unit needs at least 3 spikes.
- **Events.** `core.response.events_on` sums the positive counts, `events_off` the magnitude of
  the negative counts. An event readout that is not signed has `events_off` 0.
- **Digests.** `core.response.digest` hashes the readout array's bytes in C order with
  little-endian byte order, in the dtype it is stored in.
- **Run time.** `core.run.seconds` is the index's `seconds` for the entry, `null` when the
  index has no row.
- **Metric definitions need the layout and the frame step.** `centroid_z` exists only when
  the layout has three dimensions, and the time metrics carry `dt` as their resolution (`null`
  if the bundles' steps differ), so a pack's `metrics()` receives the layout, the resolved
  units and `dt` (10).

## 7. Tactile pack metrics (pack `tactile`)

Parameters: `plateau_tolerance` (share of a bout's peak, default 0.01) and `motion_tolerance`
(space unit, default 0). Measured per bout on the **envelope** `e(k) = max over sensors of
|x[k, m]|`, then averaged over bouts (scope: channel; per entry). A bout covers frames `[a, b)`
and peaks at `p = max e`. Its **plateau** is the frames with `e ≥ (1 − plateau_tolerance) · p`,
first `p0` and last `p1`.

| id | unit | definition |
|---|---|---|
| `tactile.episode.rise_time` | time | `(p0 − a) · dt` |
| `tactile.episode.plateau_time` | time | `(p1 − p0 + 1) · dt` |
| `tactile.episode.fall_time` | time | `(b − 1 − p1) · dt` |
| `tactile.motion.speed` | space/time | mean speed of the centroid steps that move (longer than the motion floor, section 6.5), within the bouts, if their total length exceeds `motion_tolerance`; null if none |

These are the world's episode phases (touch, hold and slide, release), read back from the
frames. They mean what their names say only for an unmodulated envelope. Under a modulation
they still give numbers, and the declared side leaves them out (section 8.3). The pack also
states the units `drive` = `filtered` = `mA` (section 3).

### 7.1 How the audit applies these definitions

Where section 7 leaves a detail open, the tactile pack (A2d) does this. None changes a metric's
name, unit or shape; they are listed for the review at R-A4.

- **Bouts.** The pack finds bouts itself, on the envelope, with a frame active when its
  envelope is exactly above 0. It has no `zero_tolerance` parameter (section 7 lists two), so
  running the core with a non-zero `zero_tolerance` changes the core's bouts and not this
  pack's.
- **Plateau.** The threshold is `(1 − plateau_tolerance) · p` in 64-bit floating point, and
  the plateau spans the first to the last frame at or above it, including any dip between
  them. With a tolerance of 1 the whole bout is plateau.
- **Averaging.** Rise, plateau and fall are counted in frames per bout, averaged over the
  entry's bouts, then multiplied by `dt`.
- **Speed** (changed at R-A4, D2; it first pooled the plateau's steps). A *moving step* is
  the centroid step between two consecutive frames of a bout (the `core.frames.centroid_*`
  weighting) that is longer than the motion floor of section 6.5 (`1e-6` of the layout's
  largest extent). The value pools the moving steps of all the entry's bouts: their total
  length over their number times `dt`. The plateau plays no part, so a still hold before a
  slide does not dilute it, and the changing envelope of a shape crossing the lattice does
  not shorten it. It is null when the total length of the moving steps is not longer than
  `motion_tolerance` (with the default 0: when nothing moves), and its `resolution` is null.
- **Units.** `drive` and `filtered` are stated as `mA` with the source
  `docs/user_guide/units_and_gains.md`; a data set that names another unit for either fails
  the audit with a unit conflict (section 3).

## 8. `declared.json`

A declared adapter reads the data set (`dataset.json`, `manifest.jsonl`) and its world file,
and writes for each entry the value its declaration implies for a metric id the audit also
uses. **It writes a value only where the declaration determines it.** Where it does not, the
adapter leaves the value out and counts the reason.

```json
{
  "format": "sensoryforge-declared/1",
  "created_at": "…", "producer": {"…": "…"},
  "adapter": {"name": "tactile", "version": "1", "params": {}},
  "inputs": {"dataset_dir": "…", "dataset_id": "…", "world_id": "w-…",
             "world_format": "sensoryforge-world/1"},
  "units": {"time": {"symbol": "ms", "origin": "pack:tactile", "source": "world_contract.md §2"}},
  "metrics": [{"id": "core.frames.onset_time", "unit": {"symbol": "ms", "origin": "pack:tactile"},
               "basis": "timeline", "source": "world_contract.md §2"}],
  "entries": {"file": "declared_entries.jsonl", "n": 1200, "sha256": "…"},
  "omitted": {"core.frames.peak": {"modulated": 310, "several shapes": 0}},
  "bounds": [{"grouping": "class", "label": "dots", "id": "core.frames.centroid_x",
              "lo": -1.0, "hi": 1.0, "source": "axis x_mm"}],
  "dataset": {"…": "…"}, "groupings": {"…": "…"}
}
```

A row of `declared_entries.jsonl` has the shape of an audit row (`entry`, `split`, `class`,
`repeat`, `noise_repeat`, `values`). Entries that share a draw get the same values. The
`dataset` and `groupings` summaries are computed from these rows **by the audit's own
summariser**, so the two sides' summaries can be compared stat by stat. `bounds` (the
range an axis allows, with `grouping` `class` or `split`) are for display and are not compared. The
adapter knows nothing of the batch, so its groups carry `n_failed` 0.

### 8.1 What a metric's `basis` says

Each declared metric names its basis, the part of the declaration it reads. For example,
`timeline` means the contact intervals and ignores shape, amplitude and modulation. A raw
difference is then read against that basis: a declared contact whose shape lies off the
sensors shows as a measured onset of `null` (`missing_measured`). That is a finding, not an
adapter error.

### 8.2 Units

The adapter states the unit of each declared value. The comparison pairs it with the audit's
unit for the same id and converts nothing.

### 8.3 The tactile adapter

Source of each rule: the [world engine contract](world_contract.md). The contact intervals
are the draw's `timeline` phases in its contact set (touch, hold, slide, release), clipped to
the entry's duration and merged where they touch. For a session, the same rules are applied to
each item at its start time.

| id | declared value | basis | left out when |
|---|---|---|---|
| `core.frames.duration` | the row's `duration_ms` | manifest | never |
| `core.frames.onset_time` | start of the first contact interval | timeline | no contact within the duration |
| `core.frames.offset_time` | end of the last contact interval | timeline | no contact; modulation reaches 0 |
| `core.frames.active_time`, `active_fraction` | total length of the intervals (over the duration) | timeline | modulation reaches 0 |
| `core.frames.bouts`, `gap_mean`, `gap_max` | count of the merged intervals, and the gaps between them | timeline | modulation reaches 0; `gap_*`: fewer than two |
| `core.frames.peak` | `amplitude` (+ `background`) | shape peak (§9) | modulated; several shapes; `clamp_min` above 0 |
| `core.frames.centroid_x`, `_y` | `x_mm`, `y_mm` | position | an unbounded shape; motion |
| `core.frames.direction`, `tactile.motion.speed` | `direction_deg`, `speed_mm_per_ms` | motion | no motion or `slide_ms` 0; an unbounded shape |
| `core.frames.dominant_frequency` | `frequency_hz / 1000` (`sine`), `rate_hz / 1000` (`pulses`) in 1/ms | modulation | no modulation |
| `tactile.episode.rise_time`, `plateau_time`, `fall_time` | `touch_ms`, `hold_ms + slide_ms`, `release_ms` | timeline | modulated; motion |

Data-set level: each group's `n_entries` is the manifest's count (it shows entries the batch
lost). The world declares nothing about responses, so no `core.response.*` or `core.input.*`
value is declared.

### 8.4 How the tactile adapter applies these definitions

Where section 8.3 leaves a detail open, the tactile adapter (A3) does this. These are
implementation choices, listed for the review at R-A4; none changes a metric's name, unit or
shape. The reasons below are the strings counted under `omitted`.

- **Contact intervals** come from the draw's `timeline` (`touch`, `hold`, `slide`,
  `release`), in the unrounded ms the world gives, clipped to `duration_ms` and merged when
  one ends where the next starts. A draw with no contact within the duration has no onset
  or offset (`no contact`) and declares 0 for `active_time` and `bouts`.
- **A modulation that reaches 0** (R-A4, D4). A `sine` or `pulses` modulation with `depth`
  1 or more (the default is 1) renders exactly 0 between taps, which cuts the contact into
  bouts the timeline cannot see. The adapter then leaves out `offset_time`, `active_time`,
  `active_fraction`, `bouts`, `gap_mean` and `gap_max` (`modulation reaches 0`); `onset_time`
  stays, since the first tap starts the contact. Declaring them by counting taps is a
  possible later addition.
- **Peak.** Declared as `amplitude` (times the pattern element's weight, plus `background`)
  for the shapes whose kernel peaks at 1: gaussian, disc, bar and grating. Any other shape
  is left out (`shape not known`, or `indenter` for a depth-driven shape). Other left-out
  cases: `modulated`, `several shapes` (a pattern of more than one element, except for an
  unbounded kernel, which is drawn once), `peak not reached` (the duration ends before the
  first touch does), `session`. The `clamp_min above 0` case of 8.3 cannot occur: the world
  refuses a floor above 0 when it loads.
- **Centroid.** Declared as the pattern element's position in mm (the pattern's `x_mm`,
  `y_mm` plus the element's offset) for the shapes that are symmetric about their centre and
  finite: gaussian, disc and a bar with a length. Left out: `unbounded shape` (a grating, a
  bar of length 0), `shape not known`, `motion` (a slide longer than 0 starts within the
  duration), `several shapes`, `background` (a constant level pulls the weighted centroid
  towards the middle of the sensors), `session`. A modulation does not move the centroid, so
  it does not leave it out.
- **Direction and speed.** Declared for a linear slide that starts within the duration. A
  class without its own motion slides at the draw's `speed_mm_per_ms` towards its
  `direction_deg` (taken modulo 360); a class that declares a linear motion slides along it,
  with the travel spread over every contact's slide. Left out: `no motion`, `motion not
  linear`, `unbounded shape`, `shape not known`, `background`, `session`. The declared
  speed is the speed *during* the slide, and `tactile.motion.speed` pools the steps that
  move (section 7.1), so the raw difference measures how well the slide is rendered and
  sampled. Where the canvas cuts the shape off (a σ 0.3 mm Gaussian on a 1.05 mm canvas is
  cut at about ±1.75 σ) the weighted centroid moves at about 0.6–0.7 of the true speed and
  is pulled towards the middle, so speed, direction and centroid read low there; the
  adapter does not know the sensor extent and still declares them (open, see the review R-A4,
  finding 7).
- **Frequency.** `frequency_hz / 1000` (sine) or `rate_hz / 1000` (pulses), in 1/ms; a
  modulation kind the adapter does not know is left out (`modulation not known`).
- **Episode phases.** `touch_ms`, `hold_ms + slide_ms`, `release_ms`, per contact. Left out
  when `modulated`, when the shape moves (`motion`: the envelope `max |x|` of a moving shape
  changes as it crosses the lattice and the canvas edge, so it has no flat plateau), when the
  draw is `truncated` by the duration, when contacts `merge` into fewer bouts than `contacts`
  (a zero pause), and for a `session`.
- **The sampled peak** (R-A4, finding 6). `core.frames.peak` is the declared shape peak. The
  sensors sample the kernel at lattice points, so the measured peak reaches `amplitude` only
  when the centre falls on a sensor; the measured value is below the declared one in every
  pair, by up to the kernel's change over half a sensor spacing. This is the definition of the
  `shape peak` basis, not a fault.
- **Channels.** A world with more than one channel gets one value per channel, each from
  the draws that drive that channel; a channel nothing drives declares `active_time` 0 and
  `bouts` 0 and leaves the rest out (`no contact`).
- **Units.** `time` and `space` are the symbols the world names (`world.units`), with origin
  `pack:tactile` and the source `world.units.<quantity>`; the others are `null`. A metric's
  unit is composed the way the audit composes it.

## 9. `compare.json`

```json
{
  "format": "sensoryforge-compare/1",
  "created_at": "…", "producer": {"…": "…"},
  "inputs": {"audit": {"path": "audit.json", "sha256": "…"},
             "declared": {"path": "declared.json", "sha256": "…"}},
  "difference": "measured - declared",
  "metrics": [{"id": "core.frames.onset_time", "unit": {"symbol": "ms"}, "resolution": 1.0,
               "n_pairs": 1199}],
  "unpaired": {"measured_only": ["core.frames.rms", "…"], "declared_only": []},
  "counts": {"ok": 9000, "unit_unchecked": 0, "unit_mismatch": 0, "missing_measured": 3,
             "missing_declared": 0, "not_numeric": 0},
  "entries": {"file": "compare_entries.jsonl", "n": 1200, "sha256": "…"},
  "dataset": [{"id": "core.frames.onset_time", "stat": "q50", "measured": 41.0,
               "declared": 40.2, "diff": 0.8, "rel_diff": 0.0199, "pairing": "ok"}],
  "groupings": {"class": {"dots": ["…"]}, "split": {"test": ["…"]}}
}
```

`metrics[*].unit` is the audit's unit for the metric (symbol only; `null` when the audit states none), so
a `unit_mismatch` is read from the pair's `pairing`, not from the metric. Group and data-set rows
carry `channel` or `population` as the summary they pair does; entry pairs carry them as their value does.

A row of `compare_entries.jsonl`:

```json
{"entry": "test/dots/0042", "split": "test", "class": "dots",
 "pairs": [{"id": "core.frames.onset_time", "measured": 13.0, "declared": 12.4,
            "diff": 0.6, "rel_diff": 0.0484, "resolution": 1.0, "pairing": "ok"}]}
```

- An entry's `pairs` cover every value its declared row carries. Measured-only metrics are
  listed once, in `unpaired`, not per entry.
- Group and data-set rows pair the two summaries stat by stat (`n`, `mean`, `std`, `min`,
  `q05` … `q95`, `max`). Histograms are not paired.
- `diff = measured − declared`. `rel_diff = diff / |declared|`, null when `declared` is 0.
- `pairing` says whether a difference could be computed. It is not a verdict:

| pairing | when | diff |
|---|---|---|
| `ok` | both numbers, units equal and known | computed |
| `unit_unchecked` | both numbers, at least one unit `null` | computed |
| `unit_mismatch` | both units known and different | null |
| `missing_measured` | declared, but measured is `null` or absent (silent frames, failed entry) | null |
| `missing_declared` | a summary stat the declared side lacks | null |
| `not_numeric` | either value not a number | null |

### 9.1 How the compare stage applies these definitions

Choices made while building A4 (none changes a metric id, unit or an existing field):

- **Order of the pairing rules.** The first that applies decides: measured `null` or absent
  is `missing_measured` (also when the declared value is `null` too), declared `null` is
  `missing_declared`, a non-number is `not_numeric`, two known different units are
  `unit_mismatch`, an unknown unit is `unit_unchecked`, otherwise `ok`. A `bool` is not a
  number.
- **What the rows follow.** The entry rows and the group rows follow the declared side. A
  group the audit lacks pairs as `missing_measured`. The two entry files must name the same
  entries in the same order, or the stage refuses; each entry file must match the hash in its
  `.json`.
- **`metrics`** lists the ids both catalogues carry, in the audit's order; `n_pairs` counts
  the entry pairs whose difference was computed. A declared id the audit does not carry is
  listed in `unpaired.declared_only` and still pairs, as `missing_measured`, in each entry.
- **Input paths** are written relative to the output folder.
- **Summaries cover the same entries** (R-A4, D5). For each key and group, the stage
  summarises the *measured* values of the entries whose declared row carries that key, with
  the audit's summariser (`audit/summary.py`) and `histogram_bins`, and pairs that with the
  declared summary. The measured `n` is then the declared `n` minus the group's
  `missing_measured`, and a difference in a mean or quantile is not a difference of
  populations. `audit.json` keeps the summaries of all the entries.
- **`distributions` (optional, added in A4, the tag stays).** For each declared value key
  and each group (`dataset`, then `class` and `split` labels): over the entries whose
  difference was computed, `n_pairs`, `diff_mean`, `diff_abs_mean`, `ks`, `wasserstein` and
  a paired `histogram` (`edges`, `measured`, `declared`) on shared equal-width bins
  (`histogram_bins`, default 32). All are `null` for `n_pairs` 0. The distances are not
  paired summaries, so this is the only place the two histograms appear.

## 10. The three seams

Each seam is a `ComponentRegistry` (`sensoryforge.registry`) in
`sensoryforge/evaluate/registry.py`. Its classes follow the component pattern: they are
registered by name, configured by `from_config`/`to_dict`, and declare their parameters with
`get_param_spec()`. They are discovered from plugins through the `sensoryforge.components`
entry-point group. Contract checks for these kinds are added in C2d.

| Registry | Built-in | Interface |
|---|---|---|
| `METRIC_PACK_REGISTRY` | `core`, `tactile` | `name`, `version`; `metrics(layout, units, dt) → [definition]` (section 4); `units() → {quantity: (symbol, source)}`; `measure(view) → [value]` for one entry; `measure_dataset(rows) → [value]` (optional, from the entry values named in the pack's `dataset_inputs`) |
| `FRAME_RENDERER_REGISTRY` | `points` (any layout), `grid` (a regular grid), `tactile` | `accepts(layout) → bool`; `render(frames [T, C, M], layout, channel) → images [T, H, W]` (float32, with an `extent` in space units) |
| `DECLARED_ADAPTER_REGISTRY` | `tactile` | `name`, `version`; `accepts(dataset_meta) → bool` (`tactile`: world format `sensoryforge-world/1`); `load(dataset_meta) → world`; `describe(dataset_meta, world)`; `channels(world)`; `units(world)`; `metrics(units) → [{id, unit, basis, source}]`; `declare(manifest_row, world) → Declaration(values, omitted)` (the values, and a reason for each one left out, which the stage counts under `omitted`); `bounds(world) → [bound]` |

A metric pack's `measure` receives an **entry view** built by the reader (section 2):

- `frames [T, C, M]` float32, `time [T]`, `dt`
- `layout`: sensor coordinates `[M, D]`, channel names, and the regular grid when there is one
- `populations`: `{name: {readout, data [T, N], drive [T, N], filtered [T, N], attrs}}`
- `units`, `seconds` (the index's run time), plus the labels `entry`, `split`, `class`, `repeat`, `noise_repeat`

A view carries nothing else, so a pack cannot reach the declaration. Packs are chosen
explicitly (`--packs core,tactile`) and recorded in `audit.json`. The audit never picks a pack
by looking at the data set's world. The default is `core` alone. The one-command pipeline
(A11a) passes `tactile` for tactile data sets. Modules a metric pack imports are inside the
import wall. A declared adapter lives in its own module, outside the wall, and may import the
world package: `sensoryforge/evaluate/tactile/metrics.py` (inside) and
`sensoryforge/evaluate/tactile/declared.py` (outside).

A second modality that is not touch, for tests, is the [toy vision fixture](../developer_guide/toy_vision_fixture.md): three colour channels on a 16 × 16 array.

## 11. Choices to review

Each of these was reviewed and approved as written on 2026-10-22. Changing one later changes
only the sections named, and needs a new decision.

| # | Choice | Sections |
|---|---|---|
| Q1 | The audit may read `class` and `split` as grouping labels (not `bins` or `probe`), so it can summarise per class without reading any declaration | 2, 4 |
| Q2 | "Active" means exactly non-zero by default (`zero_tolerance` 0), relying on the world's guarantee that quiet stretches render exactly 0 | 1, 6.1 |
| Q3 | Per-entry rows go to a `_entries.jsonl` file beside each `.json` (scale: 10^5 entries) | 4, 5, 8, 9 |
| Q4 | Units are never converted. The adapter writes declared values in the audit's unit, and a mismatch gives no difference | 3, 9 |
| Q5 | Bundles state units for time and space only. The tactile pack states `drive`/`filtered` = `mA` from the units guide. Frame values stay unitless unless the user gives a unit | 3 |
| Q6 | The declared side writes a value only where the declaration determines it, and the tactile adapter's basis is the contact timeline, ignoring shape and amplitude (section 8.3's table) | 8 |
| Q7 | Episode phases (rise, plateau, fall) and plateau speed are tactile metrics; everything else is core | 6, 7 |
| Q8 | No verdicts in `compare.json`; checks (A8) write their own file | 9 |
| Q9 | Packs are chosen explicitly, never detected from the world; default `core` | 10 |

## 12. Schemas

| File | Validates |
|---|---|
| `schemas/evaluate/common.schema.json` | units, metric definitions, values, summaries, groups (shared `$defs`) |
| `schemas/evaluate/audit.schema.json` | `audit.json`; `#/$defs/entry_row` validates a row of `audit_entries.jsonl` |
| `schemas/evaluate/declared.schema.json` | `declared.json`; `#/$defs/entry_row` a row of `declared_entries.jsonl` |
| `schemas/evaluate/compare.schema.json` | `compare.json`; `#/$defs/entry_row` a row of `compare_entries.jsonl` |

They are JSON Schema 2020-12 and refer to each other by relative `$ref`. A2a, A3 and A4
validate everything they write against them in their tests.
