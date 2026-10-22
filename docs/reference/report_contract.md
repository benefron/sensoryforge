# Evaluation report contract

!!! note "Status: accepted (roadmap task A1, reviewed and approved 2026-10-22)"
    This page fixes the formats the evaluation stages exchange before any of them is written.
    No code implements it yet: the audit is built in A2a–A2c, the declared side in A3, the
    comparison in A4. The JSON Schemas are in `schemas/evaluate/`. Section 11 lists the choices
    this page makes, all approved as written.

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
| `data.h5` | attrs `dt_ms`, `integrate_dt_ms`, `sensoryforge_version`, `sensoryforge_sha`; `/time_ms`; `/stimulus/frames`; `/populations/<name>/{drive, filtered, spikes, events, state}` and the `events` attrs `signed`, `polarity`, `theta` | `/meta` (its `config_yaml` holds the stimulus section) |
| `config.json` | `schema_version`; `grid` (`rows`, `cols`, `spacing_mm`, `center_mm`); `populations[*].{name, readout, event_encoding}`; `config.grids` (the sensor layout and channel names); `config.simulation.dt_ms`; `world.{world_id, dataset_id, entry}` as opaque identifiers | `config.stimulus`, `design`, everything else |
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
| `data` | The data names it: a key's suffix (`dt_ms` gives time in `ms`, `spacing_mm` gives space in `mm`) or a `units` block in the data's metadata |
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

What a schema-2.3.0 bundle gives today: `time` = `ms` and `space` = `mm` (origin `data`). It
gives no unit for `frame`, `drive`, `filtered` or `state`. The tactile pack states `drive`
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
| `tactile.motion.speed` | space/time | mean speed of the centroid steps within plateaus whose length exceeds `motion_tolerance`; null if none |

These are the world's episode phases (touch, hold and slide, release), read back from the
frames. They mean what their names say only for an unmodulated envelope. Under a modulation
they still give numbers, and the declared side leaves them out (section 8.3). The pack also
states the units `drive` = `filtered` = `mA` (section 3).

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
| `core.frames.offset_time` | end of the last contact interval | timeline | no contact |
| `core.frames.active_time`, `active_fraction` | total length of the intervals (over the duration) | timeline | never |
| `core.frames.bouts`, `gap_mean`, `gap_max` | count of the merged intervals, and the gaps between them | timeline | `gap_*`: fewer than two |
| `core.frames.peak` | `amplitude` (+ `background`) | shape peak (§9) | modulated; several shapes; `clamp_min` above 0 |
| `core.frames.centroid_x`, `_y` | `x_mm`, `y_mm` | position | an unbounded shape; motion |
| `core.frames.direction`, `tactile.motion.speed` | `direction_deg`, `speed_mm_per_ms` | motion | no motion or `slide_ms` 0; an unbounded shape |
| `core.frames.dominant_frequency` | `frequency_hz / 1000` (`sine`), `rate_hz / 1000` (`pulses`) in 1/ms | modulation | no modulation |
| `tactile.episode.rise_time`, `plateau_time`, `fall_time` | `touch_ms`, `hold_ms + slide_ms`, `release_ms` | timeline | modulated |

Data-set level: each group's `n_entries` is the manifest's count (it shows entries the batch
lost). The world declares nothing about responses, so no `core.response.*` or `core.input.*`
value is declared.

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

## 10. The three seams

Each seam is a `ComponentRegistry` (`sensoryforge.registry`) in
`sensoryforge/evaluate/registry.py`. Its classes follow the component pattern: they are
registered by name, configured by `from_config`/`to_dict`, and declare their parameters with
`get_param_spec()`. They are discovered from plugins through the `sensoryforge.components`
entry-point group. Contract checks for these kinds are added in C2d.

| Registry | Built-in | Interface |
|---|---|---|
| `METRIC_PACK_REGISTRY` | `core`, `tactile` | `name`, `version`; `metrics() → [definition]` (section 4); `units() → {quantity: unit}` with origin and source; `measure(view) → [value]` for one entry; `measure_dataset(rows) → [value]` (optional) |
| `FRAME_RENDERER_REGISTRY` | `points` (any layout), `grid` (a regular grid), `tactile` | `accepts(layout) → bool`; `render(frames [T, C, M], layout, channel) → images [T, H, W]` (float32, with an `extent` in space units) |
| `DECLARED_ADAPTER_REGISTRY` | `tactile` | `name`, `version`; `accepts(dataset_meta) → bool` (`tactile`: world format `sensoryforge-world/1`); `declare(manifest_row, world) → [value]`; `units()`; `bounds(world) → [bound]` |

A metric pack's `measure` receives an **entry view** built by the reader (section 2):

- `frames [T, C, M]` float32, `time [T]`, `dt`
- `layout`: sensor coordinates `[M, D]`, channel names, and the regular grid when there is one
- `populations`: `{name: {readout, data [T, N], drive [T, N], filtered [T, N], attrs}}`
- `units`, plus the labels `entry`, `split`, `class`, `repeat`, `noise_repeat`

A view carries nothing else, so a pack cannot reach the declaration. Packs are chosen
explicitly (`--packs core,tactile`) and recorded in `audit.json`. The audit never picks a pack
by looking at the data set's world. The default is `core` alone. The one-command pipeline
(A11a) passes `tactile` for tactile data sets. Modules a metric pack imports are inside the
import wall. A declared adapter lives in its own module, outside the wall, and may import the
world package: `sensoryforge/evaluate/tactile/metrics.py` (inside) and
`sensoryforge/evaluate/tactile/declared.py` (outside).

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
