"""The audit's one reader of a data set (report contract, section 2).

``DatasetReader`` opens a data-set folder (``manifest.jsonl``, ``index/task_*.jsonl``
and one bundle per entry) and hands out :class:`EntryView` objects. Everything it opens
is projected onto ``allowlist`` at read time, so a field the blind rule excludes (a
draw, a class's declaration, the stimulus section of a config) never reaches a metric
pack. It uses only numpy and h5py, and never imports ``sensoryforge.world``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

from sensoryforge.evaluate.audit import allowlist
from sensoryforge.evaluate.audit.layout import Layout, regular_layout
from sensoryforge.evaluate.audit.units import UnitTable, data_units, resolve_units

_READOUTS = ("spikes", "events", "state")


class ReadError(Exception):
    """An entry's bundle could not be read, or differs from the data set's first one.

    Attributes:
        entry: The entry id.
        code: A short machine-readable reason (``missing_bundle``, ``unreadable``,
            ``layout_mismatch``, ``layout_unknown``, ``bad_shape``), or ``None``.
    """

    def __init__(self, entry: str, message: str, code: Optional[str] = None) -> None:
        super().__init__(f"{entry}: {message}")
        self.entry = entry
        self.code = code
        self.message = message


@dataclass(frozen=True)
class EntryRow:
    """One entry: its grouping labels and what the batch index says about it.

    Attributes:
        entry: Entry id (``test/dots/0042``).
        split: Split label.
        cls: Class label, or ``None`` for an entry without one (a session).
        repeat: Repeat number.
        noise_repeat: Noise-repeat number.
        bundle: Bundle folder relative to the data-set folder, or ``None`` when the
        index has no row for the entry.
        status: The index's status (``"ok"``, ``"failed"``), or ``None`` without a row.
        error: The index's error text, or ``None``.
        seconds: The index's run time in seconds, or ``None``.
    """

    entry: str
    split: str
    cls: Optional[str]
    repeat: int
    noise_repeat: int
    bundle: Optional[str]
    status: Optional[str]
    error: Optional[str]
    seconds: Optional[float]


@dataclass(frozen=True)
class PopulationView:
    """One population of an entry.

    Attributes:
        name: Population name.
        readout: ``"spikes"``, ``"events"`` or ``"state"``.
        data: The readout array ``[T, N]`` as stored.
        drive: The population's drive ``[T, N]`` before its filter.
        filtered: The drive after its filter ``[T, N]``.
        attrs: Allow-listed attributes of the readout (``signed``, ``polarity``,
            ``theta`` for events), plus ``event_encoding`` from ``config.json``.
    """

    name: str
    readout: str
    data: np.ndarray
    drive: np.ndarray
    filtered: np.ndarray
    attrs: Dict[str, Any]


@dataclass(frozen=True)
class EntryView:
    """Everything a metric pack may see of one entry, and nothing else.

    Attributes:
        entry: Entry id.
        split: Split label.
        cls: Class label or ``None``.
        repeat: Repeat number.
        noise_repeat: Noise-repeat number.
        frames: Stimulus frames ``[T, C, M]`` (float32, read-only).
        time: Frame times ``[T]`` in the time unit (float64).
        dt: The frame step.
        layout: Sensor coordinates, channel names and the regular grid, if any.
        populations: Populations by name, in the bundle's order.
        units: Resolved units by quantity (``None`` where nobody states one).
        seconds: The batch index's run time for the entry in seconds, or ``None``.
    """

    entry: str
    split: str
    cls: Optional[str]
    repeat: int
    noise_repeat: int
    frames: np.ndarray
    time: np.ndarray
    dt: float
    layout: Layout
    populations: Dict[str, PopulationView]
    units: Dict[str, Optional[Dict[str, str]]]
    seconds: Optional[float] = None


@dataclass(frozen=True)
class BundleInfo:
    """What one bundle says about its own shape; read without loading any array.

    Attributes:
        entry: Entry id.
        schema_version: The bundle's ``schema_version``.
        sensoryforge_sha: The sha in ``data.h5``.
        dt: Frame step.
        n_frames: Number of frames ``T``.
        layout: Sensor layout.
        populations: ``(name, readout, n_units)`` in the bundle's order.
        data_units: Units the bundle names.
        world_ids: Opaque ``world`` identifiers from ``config.json`` (may be empty).
    """

    entry: str
    schema_version: str
    sensoryforge_sha: str
    dt: float
    n_frames: int
    layout: Layout
    populations: Tuple[Tuple[str, str, int], ...]
    data_units: UnitTable
    world_ids: Dict[str, str] = field(default_factory=dict)


def _label(value: Any, default: Any = None) -> Any:
    return default if value is None else value


def read_manifest_labels(path: Path) -> Dict[str, EntryRow]:
    """Read ``manifest.jsonl`` keeping only the grouping labels (report contract Q1).

    Each line is parsed and immediately reduced to ``allowlist.MANIFEST_LABELS``; the
    draw, bins, probe and seeds columns are dropped and never stored.

    Args:
        path: The manifest file.

    Returns:
        ``{entry: EntryRow}`` in file order; the index fields are still empty.

    Raises:
        ValueError: If a line is not JSON, lacks ``entry`` or repeats an entry id.
    """
    rows: Dict[str, EntryRow] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: not JSON ({exc})") from exc
            labels = {k: raw.get(k) for k in allowlist.MANIFEST_LABELS}
            del raw
            entry = labels["entry"]
            if not isinstance(entry, str) or not entry:
                raise ValueError(f"{path}:{n}: no 'entry' id")
            if entry in rows:
                raise ValueError(f"{path}:{n}: entry {entry!r} appears twice")
            rows[entry] = EntryRow(
                entry=entry,
                split=str(_label(labels["split"], "")),
                cls=None if labels["class"] is None else str(labels["class"]),
                repeat=int(_label(labels["repeat"], 0)),
                noise_repeat=int(_label(labels["noise_repeat"], 0)),
                bundle=None,
                status=None,
                error=None,
                seconds=None,
            )
    return rows


def read_index(
    folder: Path, skipped: Optional[List[Tuple[str, int]]] = None
) -> Dict[str, Dict[str, Any]]:
    """Read ``index/task_*.jsonl``, the last row per entry winning.

    A line that is not a JSON object is skipped, so one corrupt line costs only the
    entry it described (which then falls back to its bundle) and the audit goes on.

    Args:
        folder: The data-set folder.
        skipped: If given, ``(file name, line number)`` of each skipped line is
            appended to it.

    Returns:
        ``{entry: {column: value}}`` restricted to ``allowlist.INDEX_COLUMNS``. A later
        ``finished_at`` replaces an earlier row (a resumed run).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted((folder / "index").glob("task_*.jsonl")):
        with open(path, "r", encoding="utf-8") as fh:
            for number, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except ValueError:  # JSONDecodeError is a ValueError
                    raw = None
                if not isinstance(raw, dict):
                    if skipped is not None:
                        skipped.append((path.name, number))
                    continue
                row = {k: raw.get(k) for k in allowlist.INDEX_COLUMNS}
                entry = row["entry"]
                if not isinstance(entry, str):
                    continue
                old = out.get(entry)
                if old is None or (row["finished_at"] or "") >= (
                    old["finished_at"] or ""
                ):
                    out[entry] = row
    return out


@dataclass(frozen=True)
class _ConfigView:
    """``config.json`` projected onto the allow-list."""

    schema_version: str
    rows: int
    cols: int
    spacing: Optional[float]
    center: Tuple[float, float]
    populations: Tuple[Tuple[str, Optional[str], Any], ...]
    grid0: Dict[str, Any]
    dt_ms: Optional[float]
    units: Dict[str, str]
    world_ids: Dict[str, str]


def _project_config(entry: str, raw: Mapping[str, Any]) -> _ConfigView:
    """Reduce a parsed ``config.json`` to the allow-listed keys."""
    try:
        grid = raw["grid"]
        rows, cols = int(grid["rows"]), int(grid["cols"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReadError(
            entry, "config.json has no grid rows and cols", "unreadable"
        ) from exc
    center = grid.get("center_mm") or (0.0, 0.0)
    spacing = grid.get("spacing_mm")
    inner = raw.get("config") or {}
    grids = inner.get("grids") or []
    grid0 = (
        {k: grids[0].get(k) for k in allowlist.GRID_KEYS}
        if grids and isinstance(grids[0], Mapping)
        else {}
    )
    populations = []
    for pop in raw.get("populations") or []:
        populations.append(
            (str(pop["name"]), pop.get("readout"), pop.get("event_encoding"))
        )
    world = raw.get("world") or {}
    return _ConfigView(
        schema_version=str(raw.get("schema_version", "")),
        rows=rows,
        cols=cols,
        spacing=None if spacing is None else float(spacing),
        center=(float(center[0]), float(center[1])),
        populations=tuple(populations),
        grid0=grid0,
        dt_ms=(inner.get("simulation") or {}).get("dt_ms"),
        units={k: v for k, v in (raw.get("units") or {}).items() if isinstance(v, str)},
        world_ids={
            k: str(world[k])
            for k in ("world_id", "dataset_id", "entry")
            if world.get(k) is not None
        },
    )


def _attr(value: Any) -> Any:
    """A python scalar or string from an h5py attribute."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


class DatasetReader:
    """Reads a data-set folder under the blind rule.

    The first entry that opens successfully fixes the data set's sensor layout,
    population set and units; any later bundle that differs raises ``ReadError`` with
    code
    ``layout_mismatch``.

    Args:
        folder: The data-set folder (the batch output: ``manifest.jsonl``, ``index/``,
            bundles).
        manifest: The manifest file; default ``folder / "manifest.jsonl"``.
        user_units: Units given to the audit, by quantity (report contract, section 3).
        pack_units: ``(pack name, {quantity: (symbol, source)})`` per metric pack.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If the manifest is malformed.
    """

    def __init__(
        self,
        folder: Path,
        manifest: Optional[Path] = None,
        user_units: Optional[Mapping[str, str]] = None,
        pack_units: Sequence[Tuple[str, Mapping[str, Tuple[str, str]]]] = (),
    ) -> None:
        self.folder = Path(folder)
        self.manifest = Path(manifest) if manifest else self.folder / "manifest.jsonl"
        if not self.manifest.is_file():
            raise FileNotFoundError(f"no manifest at {self.manifest}")
        self._user_units = dict(user_units or {})
        self._pack_units = tuple(pack_units)
        labels = read_manifest_labels(self.manifest)
        #: ``(index file, line number)`` of each index line that was not JSON.
        self.index_skipped: List[Tuple[str, int]] = []
        index = read_index(self.folder, self.index_skipped)
        self.index_only: List[str] = sorted(set(index) - set(labels))
        self._rows: Dict[str, EntryRow] = {}
        for entry in sorted(labels):
            row, idx = labels[entry], index.get(entry)
            if idx is not None:
                row = EntryRow(
                    row.entry,
                    row.split,
                    row.cls,
                    row.repeat,
                    row.noise_repeat,
                    idx["bundle"] or entry,
                    idx["status"],
                    idx["error"],
                    None if idx["seconds"] is None else float(idx["seconds"]),
                )
            self._rows[entry] = row
        self._reference: Optional[BundleInfo] = None
        self._units: Optional[Dict[str, Optional[Dict[str, str]]]] = None

    @property
    def rows(self) -> List[EntryRow]:
        """Every manifest entry, sorted by entry id."""
        return list(self._rows.values())

    @property
    def units(self) -> Dict[str, Optional[Dict[str, str]]]:
        """Resolved units; available once one bundle has been opened."""
        if self._units is None:
            raise RuntimeError("units are resolved when the first bundle is opened")
        return self._units

    @property
    def reference(self) -> Optional[BundleInfo]:
        """The first bundle opened (it fixes layout, populations and units), or None."""
        return self._reference

    def _bundle_dir(self, row: EntryRow) -> Path:
        base = self.folder.resolve()
        path = (base / (row.bundle or row.entry)).resolve()
        if base != path and base not in path.parents:
            raise ReadError(
                row.entry,
                f"bundle path {row.bundle!r} leaves the data set",
                "unreadable",
            )
        if not (path / "data.h5").is_file() or not (path / "config.json").is_file():
            raise ReadError(row.entry, f"no bundle at {path}", "missing_bundle")
        return path

    def _open(self, row: EntryRow, load: bool) -> Tuple[BundleInfo, Dict[str, Any]]:
        """Open a bundle; ``load`` reads the arrays, else only shapes and attributes."""
        bdir = self._bundle_dir(row)
        try:
            with open(bdir / "config.json", "r", encoding="utf-8") as fh:
                cfg = _project_config(row.entry, json.load(fh))
            with h5py.File(bdir / "data.h5", "r") as f:
                return self._read_h5(row, cfg, f, load)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise ReadError(
                row.entry, f"unreadable bundle ({exc})", "unreadable"
            ) from exc

    def _read_h5(
        self, row: EntryRow, cfg: _ConfigView, f: h5py.File, load: bool
    ) -> Tuple[BundleInfo, Dict[str, Any]]:
        entry = row.entry
        attrs = {k: _attr(f.attrs[k]) for k in allowlist.H5_ATTRS if k in f.attrs}
        if "dt_ms" not in attrs:
            raise ReadError(entry, "data.h5 has no dt_ms attribute", "unreadable")
        frames_ds = f["stimulus/frames"]
        shape = tuple(int(s) for s in frames_ds.shape)
        if len(shape) == 3:
            n_ch, hw = 1, shape[1:]
        elif len(shape) == 4:
            n_ch, hw = shape[1], shape[2:]
        else:
            raise ReadError(
                entry,
                f"frames must be [T,H,W] or [T,C,H,W], got {list(shape)}",
                "bad_shape",
            )
        if hw != (cfg.rows, cfg.cols):
            raise ReadError(
                entry,
                f"frames are {hw[0]}x{hw[1]} but config.json's grid is "
                f"{cfg.rows}x{cfg.cols}; "
                "the sensor coordinates cannot be located",
                "layout_unknown",
            )
        if cfg.grid0.get("coords_file") or cfg.spacing is None:
            raise ReadError(
                entry,
                "sensor coordinates come from a file or are missing",
                "layout_unknown",
            )
        names = cfg.grid0.get("channels")
        names = tuple(str(n) for n in names) if names else None
        if names is not None and len(names) != n_ch:
            raise ReadError(
                entry,
                f"frames have {n_ch} channels but the grid names {len(names)}",
                "layout_mismatch",
            )
        layout = regular_layout(
            cfg.rows, cfg.cols, cfg.spacing, cfg.center, names, n_ch
        )
        n_frames = shape[0]
        if f["time_ms"].shape != (n_frames,):
            raise ReadError(entry, "time_ms and frames differ in length", "bad_shape")

        pops: List[Tuple[str, str, int]] = []
        loaded: Dict[str, PopulationView] = {}
        for name, readout, encoding in cfg.populations:
            path = f"populations/{name}"
            if path not in f:
                raise ReadError(
                    entry,
                    f"population {name!r} is in config.json but not data.h5",
                    "layout_mismatch",
                )
            grp = f[path]
            present = [k for k in _READOUTS if k in grp]
            if readout is None and len(present) == 1:
                readout = present[0]
            if readout not in present:
                raise ReadError(
                    entry,
                    f"population {name!r}: readout {readout!r} not in "
                    f"data.h5 {present}",
                    "layout_mismatch",
                )
            ds = grp[readout]
            if len(ds.shape) != 2 or ds.shape[0] != n_frames:
                raise ReadError(
                    entry,
                    f"population {name!r} readout has shape {list(ds.shape)}",
                    "bad_shape",
                )
            pops.append((name, readout, int(ds.shape[1])))
            if load:
                pattrs: Dict[str, Any] = {}
                if readout == "events":
                    pattrs = {
                        k: _attr(ds.attrs[k])
                        for k in allowlist.H5_EVENT_ATTRS
                        if k in ds.attrs
                    }
                if encoding is not None:
                    pattrs["event_encoding"] = encoding
                loaded[name] = PopulationView(
                    name,
                    readout,
                    _ro(ds[...]),
                    _ro(grp["drive"][...]),
                    _ro(grp["filtered"][...]),
                    pattrs,
                )
        info = BundleInfo(
            entry=entry,
            schema_version=cfg.schema_version,
            sensoryforge_sha=str(attrs.get("sensoryforge_sha", "")),
            dt=float(attrs["dt_ms"]),
            n_frames=n_frames,
            layout=layout,
            populations=tuple(pops),
            data_units=data_units(cfg.units, cfg.spacing is not None, True),
            world_ids=cfg.world_ids,
        )
        arrays: Dict[str, Any] = {}
        if load:
            frames = np.asarray(frames_ds[...], dtype=np.float32).reshape(
                n_frames, n_ch, -1
            )
            arrays = {
                "frames": _ro(frames),
                "time": _ro(np.asarray(f["time_ms"][...], dtype=np.float64)),
                "populations": loaded,
            }
        return info, arrays

    def _check_reference(self, info: BundleInfo) -> None:
        """Fix the data set's shape on the first bundle; compare later ones to it."""
        ref = self._reference
        if ref is None:
            self._units = resolve_units(
                info.data_units, self._user_units, self._pack_units
            )
            self._reference = info
            return
        if not info.layout.same_as(ref.layout):
            raise ReadError(
                info.entry,
                f"sensor layout differs from {ref.entry}'s",
                "layout_mismatch",
            )
        if info.populations != ref.populations:
            raise ReadError(
                info.entry, f"populations differ from {ref.entry}'s", "layout_mismatch"
            )
        if info.data_units != ref.data_units:
            raise ReadError(
                info.entry, f"units differ from {ref.entry}'s", "layout_mismatch"
            )

    def peek(self, entry: str) -> BundleInfo:
        """Read a bundle's shape and attributes without loading its arrays.

        Args:
            entry: Entry id.

        Returns:
            The bundle's :class:`BundleInfo`.

        Raises:
            KeyError: If the entry is not in the manifest.
            ReadError: If the bundle is missing, unreadable or differs from the first.
        """
        info, _ = self._open(self._rows[entry], load=False)
        self._check_reference(info)
        return info

    def read_entry(self, entry: str) -> EntryView:
        """Read one entry's allow-listed data.

        Args:
            entry: Entry id.

        Returns:
            The :class:`EntryView` a metric pack measures.

        Raises:
            KeyError: If the entry is not in the manifest.
            ReadError: If the bundle is missing, unreadable or differs from the first.
        """
        row = self._rows[entry]
        info, arrays = self._open(row, load=True)
        self._check_reference(info)
        return EntryView(
            entry=entry,
            split=row.split,
            cls=row.cls,
            repeat=row.repeat,
            noise_repeat=row.noise_repeat,
            frames=arrays["frames"],
            time=arrays["time"],
            dt=info.dt,
            layout=info.layout,
            populations=arrays["populations"],
            units=self.units,
            seconds=row.seconds,
        )


def _ro(array: np.ndarray) -> np.ndarray:
    """The array with writing switched off, so a metric cannot alter the view."""
    array.setflags(write=False)
    return array
