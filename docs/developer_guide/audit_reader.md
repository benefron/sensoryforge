# The audit reader

The blind audit measures a data set from its generated data alone. This
page describes its first layer, `sensoryforge.evaluate.audit`: one reader that opens a
data-set folder under the **blind rule** of the [report contract](../reference/report_contract.md)
(section 2), and an inventory of counts and metadata. The metrics that sit on top of it
arrive in the next sessions.

## Reading a data set

```python
from pathlib import Path
from sensoryforge.evaluate.audit import DatasetReader, inventory

reader = DatasetReader(Path("data"))          # manifest.jsonl, index/, one bundle per entry
inv = inventory(reader)                       # shapes and attributes only; no array is loaded
view = reader.read_entry(reader.rows[0].entry)
view.frames          # [T, C, M] float32, read-only
view.layout          # sensor coordinates [M, D], channel names, regular grid
view.populations     # {name: PopulationView(readout, data, drive, filtered, attrs)}
view.units           # resolved units by quantity
```

The folder is the batch output: `manifest.jsonl`, `index/task_*.jsonl` and one bundle per
entry. Nothing else is needed, and the reader needs only numpy and h5py.

## The wall

The reader projects everything it opens onto the allow-list in
`sensoryforge/evaluate/audit/allowlist.py`, which mirrors the contract's table:

- from `data.h5`: the attributes `dt_ms`, `integrate_dt_ms`, `sensoryforge_version`,
  `sensoryforge_sha`; `/time_ms`, `/stimulus/frames` and each population's `drive`,
  `filtered` and readout. It never opens `/meta`.
- from `config.json`: the grid block, population names and readouts, `config.grids` (layout and
  channel names), the step, the `units` block and opaque world identifiers. Not the
  stimulus section and not `design`.
- from `manifest.jsonl`: the labels `entry`, `split`, `class`, `repeat`, `noise_repeat`. Each
  line is reduced to those at once, so a draw never stays in memory.
- from `index/task_*.jsonl`: every column.
- never opened: `dataset.json`, `batch.json`, `stimuli/stimulus.json`, `neuron_modules/`, `*.pt`.

Three tests hold the wall. `tests/unit/test_audit_reader.py` plants a marker in every
excluded place and requires that it appears in nothing the reader returns, and records
every file and HDF5 key the reader touches. `tests/contract/test_audit_import_wall.py` walks
the import graph from the audit (and from every metric pack module) and fails if
`sensoryforge.world` is reachable, runs a clean `python -I` process that reads a data set
and checks `sys.modules`, and fails if the core audit source uses a tactile word (tactile,
contact, pressure, indentation, receptor, touch, skin, afferent). Those words belong to the
tactile pack.

## Sensor coordinates

For a world entry the frames are the stimulus on grid 0's canvas. The reader rebuilds the
coordinates from `config.json`'s grid block with SensoryForge's centred layout
(`indexing="ij"`, sensor `k = i * cols + j`) and never imports the world package; a test
compares them with `Canvas.from_grid_config`. A bundle whose sensors cannot be located
(frames that do not match the grid, or a coordinates file) is refused with the code
`layout_unknown` instead of being given invented positions.

## Failures

The inventory never raises for one bad entry; it records it in `failures` with a stage:

| stage | when |
|---|---|
| `batch` | the index says the entry failed |
| `read` | the bundle is missing (`missing_bundle`), unreadable (`unreadable`, `bad_shape`), cannot be located (`layout_unknown`) or differs from the first bundle that opened in sensor layout, populations or units (`layout_mismatch`) |

A failed entry still counts in the manifest's totals. A unit given by the user that
contradicts one the data names stops the audit (`UnitConflict`) and names both.

## What the inventory gives

`inv.counts()` (entries by split, class and both, repeat totals), `inv.lengths()` (frame steps
and the number of entries of each length), `inv.layout`, `inv.populations`, `inv.units`,
`inv.failures`, and `inv.header(packs, params)`, which is the front of `audit.json`; its sections
validate against `schemas/evaluate/audit.schema.json`. It does not count draws: how many
distinct stimuli a data set holds is a measurement (`core.frames.digest`), not something to
read from the declaration.
