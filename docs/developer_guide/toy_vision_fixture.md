# The toy vision fixture

A small data set that is not about touch. It exists so the evaluation code (audit,
comparison, server; see the [report contract](../reference/report_contract.md)) always has a
second modality to run on, and so the core cannot quietly come to depend on contact,
pressure or receptors. It is a test fixture, not a video feature.

Nothing is stored. `tests/fixtures/toy_vision.py` declares the world and the data set in
code and builds them through the same path a tactile data set takes
(`build_dataset`, `write_dataset`, then `run_dataset`, the engine behind
`sensoryforge batch --dataset`). The output has the usual layout: `dataset.json`,
`manifest.jsonl`, `batch.json`, `index/task_0000.jsonl` and one bundle per entry. It builds
in about two seconds.

```python
from tests.fixtures.toy_vision import build_toy_dataset

toy = build_toy_dataset(tmp_path)        # tmp_path / "data" holds the data set
toy.data, toy.entries, toy.spec          # directory, built entries, declared spec
```

## What it holds

| | |
|---|---|
| Sensor | one grid `retina`, 16 × 16 at spacing 1, with three channels `red`, `green`, `blue` |
| Stimulus | a soft Gaussian blob in one channel, held still for 30 ms, then drifting for 60 ms, then gone (frames are exactly zero from 90 ms) |
| Classes | `red_blob`, `green_blob`, `blue_blob` (one channel each), and `dark` (all zero) |
| Populations | `luma`: 16 units reading all three channels, summed. `chroma`: 32 units reading `red` and `blue` side by side (concat) |
| Size | 8 train entries, plus one test entry per class (12 in all), 100 ms each |

The draws come from these ranges, so a measurement from the frames can be compared with the
draw's record:

| Field | Range |
|---|---|
| blob width (sigma) | 1.5 to 2.5 pixels |
| speed | 0.02 to 0.06 pixels/ms, direction 0 to 360 degrees (counter-clockwise from the +x axis) |
| start | within 3 pixels of the array centre |
| peak level | 0.5 to 1.0 |

`tests/unit/test_toy_fixture.py` measures the start position, speed, direction, peak and
active interval from the frames and checks each against the record, so the fixture's
"known statistics" are tested, not assumed.

## Limits you will meet

- **The engine's only space unit is mm**, and the world's fields end in `_mm`
  (`sigma_mm`, `x_mm`). The toy reads 1 mm as 1 pixel. The audit will therefore report
  space in `mm` for this data too; a user unit (`--units space=px`) must equal it or be
  refused, as the report contract says.
- The world still uses its touch-era timing fields (`hold_ms`, `slide_ms`, `speed_mm_per_ms`)
  for the blob's timeline. They are field names of the layered class kind, not a tactile
  assumption in the core.
- A class has one channel. A blob in two colours at once would need a class kind that takes
  several layers, which the toy does not add.

## Using it in a test

Build it once per module (`scope="module"`) and read the bundles with
`sensoryforge.io.bundle.load_bundle` or `h5py`. Do not read its world or draws in code that is
meant to be blind; use them only on the declared side.
