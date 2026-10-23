"""A toy non-tactile data set, built through the real engine (roadmap task A1b).

Three colour channels (``red``, ``green``, ``blue``) on a 16x16 sensor array; the
stimulus is a soft blob drifting across one channel; two populations read it. Nothing is
committed as data: :func:`build_toy_dataset` declares a world and a data set in code,
builds the manifest and runs ``run_dataset`` (the engine path behind ``sensoryforge
batch --dataset``), so the result has exactly the layout a tactile data set has.

It is a second modality for the evaluation code (audit, comparison, server) to run on,
so the core cannot quietly depend on touch. It is a test fixture, not a video feature.

The world declares its space unit as ``px`` and writes its fields without the unit
suffix (``sigma``, ``x``, ``speed``); the loader maps them onto the canonical names
(``sigma_mm``, ...), which still appear in a draw's record, where the ``_mm`` only means
"the world's space unit". The bundle carries ``units: {"space": "px"}``.

Known statistics, per class (a blob in one channel):

====== =========================================================
field  range
====== =========================================================
sigma  1.5 to 2.5 px
speed  0.02 to 0.06 px/ms, in a direction drawn from 0 to 360 deg
start  within 3 px of the array centre
level  0.5 to 1.0
====== =========================================================

The draw's record carries these values (the declared side); the bundle carries only the
frames and the responses (the measured side).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    PopulationInput,
    RFBuilderConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.world.dataset import (
    DatasetSpec,
    Entry,
    build_dataset,
    load_dataset,
    write_dataset,
)
from sensoryforge.world.runner import run_dataset

CHANNELS = ["red", "green", "blue"]
ROWS = COLS = 16
SPACING = 1.0
DT_MS = 1.0
DURATION_MS = 100

SIGMA_RANGE = (1.5, 2.5)
SPEED_RANGE = (0.02, 0.06)
LEVEL_RANGE = (0.5, 1.0)
START_RADIUS = 3.0

_BLOB = {"shape": {"kind": "gaussian"}}
_BLOB_AXES = {"sigma": {"range": list(SIGMA_RANGE)}}

TOY_WORLD: Dict[str, Any] = {
    "world": {
        "name": "toy_vision",
        "description": "soft blobs drifting across one colour channel of a 16x16 array",
        "modality": "toy_vision",
        "channels": CHANNELS,
        "units": {"space": "px", "time": "ms"},
        "defaults": {
            "delay_ms": {"value": 0},
            "touch_ms": {"value": 0},
            "release_ms": {"value": 0},
            "hold_ms": {"value": 30},
            "slide_ms": {"value": 60},
            "speed": {"range": list(SPEED_RANGE)},
            "direction_deg": {"range": [0, 360], "circular": True},
            "amplitude": {"range": list(LEVEL_RANGE)},
            "x": {"range": [-START_RADIUS, START_RADIUS]},
            "y": {"range": [-START_RADIUS, START_RADIUS]},
        },
        "classes": {
            "red_blob": {
                "weight": 1.0,
                "channel": "red",
                "layer": _BLOB,
                "axes": _BLOB_AXES,
            },
            "green_blob": {
                "weight": 1.0,
                "channel": "green",
                "layer": _BLOB,
                "axes": _BLOB_AXES,
            },
            "blue_blob": {
                "weight": 1.0,
                "channel": "blue",
                "layer": _BLOB,
                "axes": _BLOB_AXES,
            },
            "dark": {
                "kind": "quiet",
                "weight": 0.5,
                "axes": {"quiet_ms": {"value": 100}},
            },
        },
    }
}

TOY_DATASET: Dict[str, Any] = {
    "dataset": {
        "name": "toy_vision",
        "seed": 20261022,
        "duration_ms": DURATION_MS,
        "splits": {
            "train": {"n": 8, "repeats": 1},
            "test": {"stratified": {"bins": 1, "per_bin": 1}},
        },
    }
}


def toy_config(device: str = "cpu") -> SensoryForgeConfig:
    """The toy sensor: one 3-channel 16x16 grid and two populations.

    * ``luma``: 16 units, each reading all three channels with a summed drive.
    * ``chroma``: 32 units, the ``red`` and ``blue`` inputs side by side (concat).

    Args:
        device: Torch device for the engine.

    Returns:
        A config the engine runs as it stands; no tactile preset is used.
    """
    rf = RFBuilderConfig(method="gaussian")
    luma = PopulationConfig(
        name="luma",
        neuron_model="Izhikevich",
        neuron_type="SA",
        neurons_per_row=4,
        filter_method="none",
        input_gain=40.0,
        inputs=[PopulationInput(grid="retina", channel=c, rf=rf) for c in CHANNELS],
        combine="sum",
    )
    chroma = PopulationConfig(
        name="chroma",
        neuron_model="Izhikevich",
        neuron_type="RA",
        neurons_per_row=4,
        filter_method="ra",
        input_gain=120.0,
        inputs=[
            PopulationInput(grid="retina", channel=c, rf=rf) for c in ("red", "blue")
        ],
        combine="concat",
    )
    grid = GridConfig(
        name="retina",
        arrangement="grid",
        rows=ROWS,
        cols=COLS,
        spacing=SPACING,
        channels=list(CHANNELS),
    )
    return SensoryForgeConfig(
        grids=[grid],
        populations=[luma, chroma],
        simulation=SimulationConfig(device=device, dt_ms=DT_MS),
    )


@dataclass
class ToyDataset:
    """What :func:`build_toy_dataset` made.

    Attributes:
        root: The directory it was built in.
        data: The data-set directory (``dataset.json``, ``manifest.jsonl``, ``index/``,
            one bundle per entry).
        spec: The data-set spec (the declared side).
        entries: The built entries, in manifest order.
        summary: ``run_dataset``'s ``{"ok", "failed", "skipped"}``.
    """

    root: Path
    data: Path
    spec: DatasetSpec
    entries: List[Entry]
    summary: Dict[str, int]


def toy_spec(
    n_train: int = 8, duration_ms: float = DURATION_MS, seed: Optional[int] = None
) -> DatasetSpec:
    """The toy data-set spec (world inline, nothing read from disk).

    Args:
        n_train: Entries in the ``train`` split.
        duration_ms: Entry length, ms.
        seed: Data-set seed (default: the fixture's).

    Returns:
        A loaded :class:`~sensoryforge.world.dataset.DatasetSpec`.
    """
    raw = copy.deepcopy(TOY_DATASET)
    ds = raw["dataset"]
    ds["world"] = copy.deepcopy(TOY_WORLD)
    ds["duration_ms"] = duration_ms
    ds["splits"]["train"]["n"] = n_train
    if seed is not None:
        ds["seed"] = seed
    return load_dataset(raw, base_dir=Path.cwd())


def build_toy_dataset(
    root: Path,
    *,
    n_train: int = 8,
    duration_ms: float = DURATION_MS,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> ToyDataset:
    """Build the toy data set: manifest, then one bundle per entry.

    Args:
        root: An empty directory; the data set goes in ``root / "data"``.
        n_train: Entries in the ``train`` split (the test split adds one per class).
        duration_ms: Entry length, ms.
        seed: Data-set seed (default: the fixture's).
        device: Torch device for the engine.

    Returns:
        The :class:`ToyDataset`.

    Raises:
        RuntimeError: If any entry failed to simulate.
    """
    spec = toy_spec(n_train, duration_ms, seed)
    entries = build_dataset(spec)
    data = Path(root) / "data"
    write_dataset(spec, entries, data)
    summary = run_dataset(toy_config(device), spec, data)
    if summary["failed"]:
        raise RuntimeError(
            f"toy data set: {summary['failed']} entries failed: {summary}"
        )
    return ToyDataset(Path(root), data, spec, entries, summary)
