# The declared side

The audit says what a data set *shows* ([the audit's metrics](audit_metrics.md)). The
declared stage says what the data set's own declaration says each entry *should* show, in
the same metric ids and units, so the comparison stage can put the two side by side. The
formats are in the [report contract](../reference/report_contract.md), section 8; this page
covers running the stage, what the tactile adapter declares, and how to write an adapter for
another modality.

The stage reads `dataset.json` and `manifest.jsonl` only. It opens no bundle, runs no
simulation and never touches the audit's code, so it can run before the batch has finished.

## Run it

```bash
python -m sensoryforge.evaluate.declared DATA_DIR --out OUT_DIR
python -m sensoryforge.evaluate.declared DATA_DIR --adapter tactile
```

`--out` defaults to `DATA_DIR/evaluate`. It writes two files:

| File | Holds |
|---|---|
| `declared.json` | the adapter and its version, the units it states, one line per metric it can declare (unit, basis, source), how many values were left out and why, the ranges a class's axes allow, and the summaries over the data set, per class and per split |
| `declared_entries.jsonl` | one row per manifest entry, sorted by entry id: its labels and every value its declaration determines |

From Python:

```python
from sensoryforge.evaluate.declared import run_declared

declared = run_declared("data", "out")
```

Without `--adapter` the stage uses the first registered adapter that accepts the data set,
and fails with a message if none does. Both files validate against
`schemas/evaluate/declared.schema.json` (each row against `#/$defs/entry_row`). Two runs
over the same data set are byte-identical apart from `created_at`. The summaries come from
the audit's own summariser, so the two sides' statistics line up one for one. The command
exits with status 2 and a message on stderr for a missing file, an unknown adapter or a
declaration that does not match its data set.

## What the tactile adapter declares

The adapter rebuilds the world embedded in `dataset.json` (and checks that it still hashes
to the recorded `world_id`) and reads each row's draw. Entries that share a draw (noise
repeats) get the same values.

A value is written only where the declaration determines it. Every other case is left out,
and `declared.json` counts the cases by reason under `omitted`. A missing value is not an
error: it means the world says nothing definite there.

| Metric | Declared value | Left out when |
|---|---|---|
| `core.frames.duration` | the manifest's `duration_ms` | never |
| `core.frames.onset_time`, `offset_time` | start of the first, end of the last contact interval | no contact |
| `core.frames.active_time`, `active_fraction`, `bouts` | total length of the intervals, its share of the duration, their count | never |
| `core.frames.gap_mean`, `gap_max` | the gaps between the intervals | fewer than two bouts |
| `core.frames.peak` | `amplitude` of a shape that peaks at 1 (gaussian, disc, bar, grating), plus `background` | modulated, several shapes, another shape, the peak is cut off by the duration, no contact |
| `core.frames.centroid_x`, `centroid_y` | the pattern's placement | an unbounded shape (grating, a bar with no length), any motion during the entry, several shapes, a `background`, another shape |
| `core.frames.direction`, `tactile.motion.speed` | the direction and speed of a linear slide | no slide within the duration, a motion that is not linear, an unbounded shape, a `background` |
| `core.frames.offset_time`, `active_time`, `active_fraction`, `bouts`, `gap_mean`, `gap_max` | the merged contact intervals | a modulation that reaches 0 (a `sine` or `pulses` at `depth` 1, the default): the frames are 0 between taps, which the timeline cannot see. `onset_time` stays |
| `core.frames.dominant_frequency` | `frequency_hz / 1000` (sine) or `rate_hz / 1000` (pulses), in 1/ms | no modulation |
| `tactile.episode.rise_time`, `plateau_time`, `fall_time` | `touch_ms`, `hold_ms + slide_ms`, `release_ms` | modulated, the shape moves (the envelope of a moving shape is not flat), the draw is cut off by the duration, contacts that merge into one bout |

The **contact intervals** are the draw's timeline phases touch, hold, slide and release,
clipped to the entry's duration and merged where they touch; a pause or quiet stretch
separates two bouts. For a session the same rule is applied to each item at its start time,
and everything that needs one draw's shape (peak, centroid, motion, frequency, phases) is
left out with the reason `session`.

A declared slide speed is the speed *during* the slide; the measured one is the speed of the
steps that move, so the raw difference shows how well the slide is rendered and sampled. A
shape the canvas cuts off (a σ 0.3 mm Gaussian on a 1.05 mm canvas) reads slower and pulled
towards the middle, and the adapter does not know the canvas, so it still declares centroid,
direction and speed there (open: R-A4, finding 7). The measured peak is below the declared
one by design: the sensors sample the kernel at lattice points.

Times are the unrounded milliseconds of the world's timeline. The audit samples frames at
`k * dt`, so a raw difference of up to about `dt` between a declared and a measured time is
expected and does not mean anything is wrong. A declared contact whose shape lies off the
sensors shows on the comparison as a measured onset of `null`: that is a finding about the
world, not an adapter error (contract, section 8.1).

A world with more than one channel gets one value per channel (`channel` is the 0-based
index): a channel nothing touches is declared as 0 active time and 0 bouts.

The adapter's `bounds` are the ranges a class's axes allow for the onset, the peak and the
centroid. They are for display and are never compared.

!!! note "The world cannot declare a floor above 0"
    The contract's table lists `clamp_min` above 0 as a case that leaves the peak out. The
    world engine refuses such a floor when it loads the world (its domain is `<= 0`), so the
    adapter has no case for it.

## Write an adapter for another modality

An adapter is a component like any other: a class registered by name, configured by
`from_config`/`to_dict`, with `get_param_spec()`. It lives outside the audit's import wall, so
it may import whatever reads its declaration; the audit never imports an adapter.

```python
from sensoryforge.evaluate.declared import Declaration, DeclaredAdapter
from sensoryforge.evaluate.registry import DECLARED_ADAPTER_REGISTRY


class MyAdapter(DeclaredAdapter):
    name = "my_modality"

    @classmethod
    def accepts(cls, dataset_meta): ...          # can it read this dataset.json?
    def load(self, dataset_meta): ...            # the declaration, as your own object
    def describe(self, dataset_meta, world): ... # dataset_id, world_id, world_format
    def units(self, world): ...                  # {quantity: {symbol, origin, source} | None}
    def metrics(self, units): ...                # [{id, unit, basis, source}], catalogue order
    def declare(self, row, world): ...           # a Declaration(values, omitted) per row
    def bounds(self, world): ...                 # optional, display only


DECLARED_ADAPTER_REGISTRY.register("my_modality", MyAdapter)
```

Rules that keep the two sides comparable:

- Use the metric ids the audit uses, and state each value in the unit the audit reports for
  that id (`sensoryforge.evaluate.audit.units.unit_of` builds the same unit objects). The
  comparison converts nothing: a unit that differs gives no difference.
- Write a value only where the declaration determines it. Give every value you leave out a
  reason in `Declaration.omitted`; the stage counts them.
- Return values in catalogue order, with `channel` set when the data has more than one
  channel.
- Register the adapter from a plugin through the `sensoryforge.components` entry-point group,
  or import its module from `sensoryforge/evaluate/declared/__init__.py` as the built-in one
  is. Never register it from `sensoryforge/evaluate/registry.py`: that module is inside the
  audit's wall.

`sensoryforge/evaluate/tactile/declared.py` is the worked example; its tests
(`tests/unit/test_declared.py`) re-derive the timeline values from the manifest alone and
check the shape values against the world's own renderer.
