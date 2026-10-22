"""What the audit may read (``docs/reference/report_contract.md``, section 2).

The reader projects every file it opens onto these names at read time, so a field that
is not listed never reaches a metric pack. Anything that needs a new name here changes
the contract (and its page) first.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: ``data.h5`` attributes.
H5_ATTRS: Tuple[str, ...] = (
    "dt_ms",
    "integrate_dt_ms",
    "sensoryforge_version",
    "sensoryforge_sha",
)

#: ``data.h5`` datasets (``<pop>`` is a population name from ``config.json``).
H5_DATASETS: Tuple[str, ...] = (
    "/time_ms",
    "/stimulus/frames",
    "/populations/<pop>/drive",
    "/populations/<pop>/filtered",
    "/populations/<pop>/spikes",
    "/populations/<pop>/events",
    "/populations/<pop>/state",
)

#: Attributes read from ``/populations/<pop>/events``.
H5_EVENT_ATTRS: Tuple[str, ...] = ("signed", "polarity", "theta")

#: ``config.json`` paths (dotted; ``[*]`` is every list item). ``world.*`` are opaque
#: ids.
CONFIG_PATHS: Tuple[str, ...] = (
    "schema_version",
    "grid.rows",
    "grid.cols",
    "grid.spacing_mm",
    "grid.center_mm",
    "populations[*].name",
    "populations[*].readout",
    "populations[*].event_encoding",
    "config.grids",
    "config.simulation.dt_ms",
    "units",
    "world.world_id",
    "world.dataset_id",
    "world.entry",
)

#: Keys kept from each ``config.grids[*]`` item.
GRID_KEYS: Tuple[str, ...] = (
    "arrangement",
    "rows",
    "cols",
    "spacing",
    "center_x",
    "center_y",
    "channels",
    "coords_file",
)

#: Columns kept from ``index/task_*.jsonl`` (every column).
INDEX_COLUMNS: Tuple[str, ...] = (
    "entry",
    "bundle",
    "status",
    "error",
    "seconds",
    "finished_at",
    "design_id",
    "sensoryforge_sha",
    "task",
)

#: Columns kept from ``manifest.jsonl``: grouping labels, not declarations (Q1).
MANIFEST_LABELS: Tuple[str, ...] = (
    "entry",
    "split",
    "class",
    "repeat",
    "noise_repeat",
)

#: Files in a data-set folder the audit never opens.
NEVER_OPENED: Dict[str, str] = {
    "dataset.json": "the data set's declaration",
    "batch.json": "carries the run's config and stimulus section",
    "stimuli/stimulus.json": "the manifest row and the draw as a layer",
    "neuron_modules/": "model code",
    "*.pt": "weights; not an input of any metric",
}
