"""The data behind the evaluation server (roadmap A5a).

:class:`DatasetService` answers every question the browser asks about one data-set
folder, with no HTTP in it: a summary, a filtered list of draws, one draw's record and
the traces of its stimulus and populations. It is a viewer, not the blind audit: it
reads the whole manifest (the declared record of each draw is what the viewer shows),
and it reads bundles through the audit's reader, so a bundle is checked the same way
and a failed one is reported, not crashed on.

The index is built once from ``manifest.jsonl`` (labels, the draw seed and the byte
offset of the line) and ``index/task_*.jsonl`` (status); no bundle is opened to list
draws, so a data set of 10,000 draws lists in the time it takes to parse its manifest.
A draw's full record is read from its line on demand.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

from sensoryforge.evaluate.audit.reader import (
    BundleInfo,
    DatasetReader,
    ReadError,
    read_index,
)

#: The stage files the server shows, in pipeline order.
STAGES: Tuple[str, ...] = ("audit", "declared", "compare")

#: ``source`` values of a trace; ``readout`` is whatever the population stores
#: (``spikes``, ``events`` or ``state``).
SOURCES: Tuple[str, ...] = ("stimulus", "drive", "filtered", "readout")

#: Most series one trace request returns, and most points per series.
MAX_SERIES = 64
MAX_POINTS = 20000
DEFAULT_POINTS = 2000

#: Frames per chunk by default, and the most bytes one chunk carries.
DEFAULT_CHUNK_FRAMES = 32
MAX_CHUNK_BYTES = 8 << 20

#: Default size of the decompressed-slab cache, bytes (``--frame-cache-mb`` sets it).
DEFAULT_FRAME_CACHE_BYTES = 256 << 20

#: Failed bundle reads the summary tries before it gives up on a reference draw.
SUMMARY_MAX_PEEKS = 50

#: Most draws one list request returns.
MAX_LIMIT = 1000
DEFAULT_LIMIT = 100

#: The filter token for "no class" in a ``class`` filter.
NO_CLASS = "~"


class ServiceError(Exception):
    """A request the service cannot answer.

    Attributes:
        status: The HTTP status the message belongs to (400, 404 or 422).
        message: What went wrong, for the person reading it.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class DrawRow:
    """One manifest entry as the draw list shows it.

    Attributes:
        entry: Entry id.
        split: Split label.
        cls: Class label or ``None``.
        repeat: Repeat number.
        noise_repeat: Noise-repeat number.
        seed: The draw's seed as a decimal string (seeds exceed what a browser's
            numbers hold exactly), or ``None``.
        duration_ms: The entry's duration, or ``None``.
        status: ``"ok"``, ``"failed"`` or ``None`` when the batch index has no row.
        seconds: Run time of the entry in the batch, or ``None``.
        offset: Byte offset of the entry's line in ``manifest.jsonl``.
    """

    entry: str
    split: str
    cls: Optional[str]
    repeat: int
    noise_repeat: int
    seed: Optional[str]
    duration_ms: Optional[float]
    status: Optional[str]
    seconds: Optional[float]
    offset: int

    def to_dict(self) -> Dict[str, Any]:
        """The row as the list endpoint returns it."""
        return {
            "entry": self.entry,
            "split": self.split,
            "class": self.cls,
            "repeat": self.repeat,
            "noise_repeat": self.noise_repeat,
            "seed": self.seed,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "seconds": self.seconds,
        }


@dataclass(frozen=True)
class FrameChunk:
    """A run of frames ready to send.

    Attributes:
        body: ``count * height * width`` little-endian float32 values, frame by frame.
        start: Index of the first frame in the draw.
        count: Frames in the chunk.
        height: Rows of a frame.
        width: Columns of a frame.
        total: Frames in the whole draw.
        dt_ms: Frame step, ms.
        low: Smallest finite value in the chunk.
        high: Largest finite value in the chunk.
    """

    body: bytes
    start: int
    count: int
    height: int
    width: int
    total: int
    dt_ms: float
    low: float
    high: float


def _scan_manifest(path: Path) -> List[Tuple[Dict[str, Any], int]]:
    """Parse ``manifest.jsonl`` once, keeping each line's fields and byte offset."""
    out: List[Tuple[Dict[str, Any], int]] = []
    with open(path, "rb") as fh:
        while True:
            offset = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except ValueError as exc:
                raise ValueError(
                    f"{path}: line at byte {offset} is not JSON ({exc})"
                ) from exc
            entry = raw.get("entry")
            if not isinstance(entry, str) or not entry:
                raise ValueError(f"{path}: line at byte {offset} has no 'entry' id")
            seed = (raw.get("seeds") or {}).get("draw")
            out.append(
                (
                    {
                        "entry": entry,
                        "split": str(raw.get("split") or ""),
                        "class": (
                            None if raw.get("class") is None else str(raw["class"])
                        ),
                        "repeat": int(raw.get("repeat") or 0),
                        "noise_repeat": int(raw.get("noise_repeat") or 0),
                        "seed": None if seed is None else str(seed),
                        "duration_ms": raw.get("duration_ms"),
                    },
                    offset,
                )
            )
    return out


def _csv(value: Optional[str]) -> Optional[List[str]]:
    """A comma-separated query value as a list, or ``None`` when absent or empty."""
    if value is None or value == "":
        return None
    return [v for v in value.split(",")]


def _int(name: str, value: Optional[str], default: int, lo: int, hi: int) -> int:
    if value is None or value == "":
        return default
    try:
        number = int(value)
    except ValueError as exc:
        raise ServiceError(400, f"{name} must be an integer, got {value!r}") from exc
    if not lo <= number <= hi:
        raise ServiceError(400, f"{name} must be in [{lo}, {hi}], got {number}")
    return number


def decimate(values: np.ndarray, max_points: int) -> np.ndarray:
    """The frames to keep of one series so that it has at most ``max_points`` samples.

    The frames are cut into buckets of equal size; each bucket keeps the frame of its
    smallest and the frame of its largest value, so a one-frame spike survives.

    Args:
        values: One series ``[T]``.
        max_points: Most frames to keep (at least 2).

    Returns:
        Increasing frame indices ``[P]``; all of them when ``T <= max_points``.
    """
    n_t = values.shape[0]
    if n_t <= max_points:
        return np.arange(n_t)
    n_buckets = max(1, max_points // 2)
    size = -(-n_t // n_buckets)
    blocks = np.pad(values, (0, n_buckets * size - n_t), mode="edge").reshape(
        n_buckets, size
    )
    base = np.arange(n_buckets) * size
    frames = np.concatenate(
        [blocks.argmin(axis=1) + base, blocks.argmax(axis=1) + base]
    )
    return np.unique(np.minimum(frames, n_t - 1))


class DatasetService:
    """Questions about one data-set folder, answered without HTTP.

    Args:
        folder: The data-set folder (``manifest.jsonl``, ``index/``, one bundle per
            entry, ``dataset.json``).
        stages: Where ``audit.json``, ``declared.json`` and ``compare.json`` live;
            default ``folder / "evaluate"``.
        frame_cache_bytes: Most bytes of decompressed stimulus frames kept between
            requests (default 256 MiB; 0 keeps nothing). See :meth:`frames`.

    Raises:
        FileNotFoundError: If the folder has no manifest.
        ValueError: If the manifest is malformed.
    """

    def __init__(
        self,
        folder: Path,
        stages: Optional[Path] = None,
        frame_cache_bytes: int = DEFAULT_FRAME_CACHE_BYTES,
    ) -> None:
        if frame_cache_bytes < 0:
            raise ValueError(f"frame_cache_bytes must be >= 0, got {frame_cache_bytes}")
        self.folder = Path(folder)
        self.stages = Path(stages) if stages else self.folder / "evaluate"
        manifest = self.folder / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"no manifest at {manifest}")
        started = time.perf_counter()
        scanned = _scan_manifest(manifest)
        status = read_index(self.folder)
        self.rows: List[DrawRow] = []
        for fields, offset in scanned:
            idx = status.get(fields["entry"]) or {}
            seconds = idx.get("seconds")
            self.rows.append(
                DrawRow(
                    fields["entry"],
                    fields["split"],
                    fields["class"],
                    fields["repeat"],
                    fields["noise_repeat"],
                    fields["seed"],
                    fields["duration_ms"],
                    idx.get("status"),
                    None if seconds is None else float(seconds),
                    offset,
                )
            )
        self.by_entry: Dict[str, DrawRow] = {r.entry: r for r in self.rows}
        if len(self.by_entry) != len(self.rows):
            raise ValueError(f"{manifest}: an entry id appears twice")
        self.index_seconds = time.perf_counter() - started
        self._manifest = manifest
        self._lock = threading.RLock()
        self._reader: Optional[DatasetReader] = None
        self._infos: "OrderedDict[str, BundleInfo]" = OrderedDict()
        self._reference: Optional[str] = None
        # Decompressed time slabs of ``stimulus/frames``, least recently used first,
        # keyed (entry, channel, first frame of the slab). One lock covers the cache
        # and the read that fills it, so two requests for a slab decompress it once.
        self.frame_cache_bytes = int(frame_cache_bytes)
        self._slabs: "OrderedDict[Tuple[str, int, int], np.ndarray]" = OrderedDict()
        self._slab_lock = threading.Lock()
        self._slab_bytes = 0
        self.slab_hits = 0
        self.slab_misses = 0

    # ------------------------------------------------------------------ helpers

    def _row(self, entry: Optional[str]) -> DrawRow:
        if not entry:
            raise ServiceError(400, "give the draw as ?entry=<entry id>")
        row = self.by_entry.get(entry)
        if row is None:
            raise ServiceError(404, f"no draw {entry!r} in this data set")
        return row

    def reader(self) -> DatasetReader:
        """The audit's reader over the folder (made on first use, then shared)."""
        with self._lock:
            if self._reader is None:
                self._reader = DatasetReader(self.folder)
            return self._reader

    def warm(self) -> None:
        """Make the reader now, so the first draw opened does not pay for it.

        The reader parses the manifest once more (about 0.3 s at 10,000 draws). A
        manifest the reader refuses is reported again when a draw is opened.
        """
        try:
            self.reader()
        except (OSError, ValueError):
            pass

    def info(self, entry: str) -> BundleInfo:
        """A bundle's shape and attributes, cached for the last 256 draws.

        Raises:
            ServiceError: 404 for an unknown entry, 422 for a bundle that cannot be
                read (the message says why).
        """
        self._row(entry)
        with self._lock:
            cached = self._infos.get(entry)
            if cached is not None:
                self._infos.move_to_end(entry)
                return cached
            try:
                info = self.reader().peek(entry)
            except ReadError as exc:
                text = f"{exc.code}: {exc.message}" if exc.code else exc.message
                raise ServiceError(422, f"draw {entry!r} cannot be read ({text})")
            self._infos[entry] = info
            if len(self._infos) > 256:
                self._infos.popitem(last=False)
            return info

    def _units(self) -> Dict[str, Optional[str]]:
        try:
            units = self.reader().units
        except RuntimeError:
            return {}
        return {k: (None if v is None else v["symbol"]) for k, v in units.items()}

    def record(self, entry: str) -> Dict[str, Any]:
        """The draw's full manifest record (its declaration), read from its line."""
        row = self._row(entry)
        with open(self._manifest, "rb") as fh:
            fh.seek(row.offset)
            return json.loads(fh.readline())

    # ------------------------------------------------------------------ dataset

    def summary(self) -> Dict[str, Any]:
        """The data set at a glance.

        Returns:
            Names and ids from ``dataset.json`` when it is present, the entry counts by
            split, class and status, the sensor layout, populations, frame step and
            units of the first readable bundle, and the state of the stage files.
        """
        meta: Dict[str, Any] = {}
        path = self.folder / "dataset.json"
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                raw = {}
            for key in (
                "name",
                "dataset_id",
                "world_id",
                "format",
                "duration_ms",
                "seed",
                "n_entries",
                "splits",
                "world_description",
            ):
                if key in raw:
                    meta[key] = raw[key]
        out: Dict[str, Any] = {
            "folder": str(self.folder),
            **meta,
            "n_draws": len(self.rows),
            "counts": {
                "split": dict(sorted(Counter(r.split for r in self.rows).items())),
                "class": dict(
                    sorted(Counter(r.cls or NO_CLASS for r in self.rows).items())
                ),
                "status": dict(
                    sorted(Counter(r.status or "unknown" for r in self.rows).items())
                ),
            },
            "no_class_token": NO_CLASS,
            "index_seconds": round(self.index_seconds, 4),
            "sensors": None,
            "populations": [],
            "dt_ms": None,
            "n_frames": None,
            "units": {},
            "stages": self.stage_status(),
        }
        info: Optional[BundleInfo] = None
        reference = self._reference
        if reference is not None:
            try:
                info = self.info(reference)
            except ServiceError:
                self._reference = reference = None
        failed = 0
        if info is None:
            for row in self.rows:
                if row.status not in (None, "ok"):
                    continue
                try:
                    info = self.info(row.entry)
                except ServiceError:
                    failed += 1
                    if failed >= SUMMARY_MAX_PEEKS:
                        break
                    continue
                self._reference = reference = row.entry
                break
        if info is not None:
            out["sensors"] = info.layout.summary()
            out["populations"] = [
                {"name": n, "readout": r, "n_units": u} for n, r, u in info.populations
            ]
            out["dt_ms"] = info.dt
            out["n_frames"] = info.n_frames
            out["units"] = self._units()
            out["reference_draw"] = reference
        else:
            out["reference_draw"] = None
            if failed >= SUMMARY_MAX_PEEKS:
                out["reference_note"] = (
                    f"no bundle read in the first {failed} draws tried, so the "
                    "layout, populations and units are not shown"
                )
        return out

    # ------------------------------------------------------------------- stages

    def stage_path(self, name: str) -> Path:
        """Where a stage's JSON file is (it may not exist).

        Raises:
            ServiceError: 404 for a name that is not a stage.
        """
        if name not in STAGES:
            raise ServiceError(404, f"no stage {name!r}; the stages are {list(STAGES)}")
        return self.stages / f"{name}.json"

    def stage_status(self) -> Dict[str, Any]:
        """Per stage: whether its file exists and its header."""
        out: Dict[str, Any] = {}
        for name in STAGES:
            path = self.stage_path(name)
            if not path.is_file():
                out[name] = {"present": False, "path": str(path)}
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                out[name] = {"present": True, "path": str(path), "readable": False}
                continue
            entries = doc.get("entries") or {}
            out[name] = {
                "present": True,
                "readable": True,
                "path": str(path),
                "format": doc.get("format"),
                "created_at": doc.get("created_at"),
                "n_entries": entries.get("n"),
                "bytes": path.stat().st_size,
            }
        return out

    def stage_bytes(self, name: str) -> Optional[bytes]:
        """A stage file's content as stored, or ``None`` when it does not exist."""
        path = self.stage_path(name)
        return path.read_bytes() if path.is_file() else None

    # -------------------------------------------------------------------- draws

    def filtered(self, query: Mapping[str, str]) -> List[DrawRow]:
        """The rows matching the query's filters, in manifest order.

        Filters: ``split``, ``class`` (``~`` for no class), ``status`` (``unknown`` for
        entries the index has no row for), each a comma-separated list; ``q`` (a
        substring of the entry id); ``seed`` (a decimal string, exact); ``repeat``.
        """
        splits = _csv(query.get("split"))
        classes = _csv(query.get("class"))
        statuses = _csv(query.get("status"))
        needle = (query.get("q") or "").lower()
        seed = query.get("seed") or None
        repeat = query.get("repeat")
        repeat_n = None if repeat in (None, "") else _int("repeat", repeat, 0, 0, 10**9)
        out = []
        for r in self.rows:
            if splits is not None and r.split not in splits:
                continue
            if classes is not None and (r.cls or NO_CLASS) not in classes:
                continue
            if statuses is not None and (r.status or "unknown") not in statuses:
                continue
            if needle and needle not in r.entry.lower():
                continue
            if seed is not None and r.seed != seed:
                continue
            if repeat_n is not None and r.repeat != repeat_n:
                continue
            out.append(r)
        return out

    def list_draws(self, query: Mapping[str, str]) -> Dict[str, Any]:
        """A page of the draw list.

        Args:
            query: The filters of :meth:`filtered`, plus ``sort`` (``entry``,
                ``seconds`` or manifest order by default), ``order`` (``asc`` or
                ``desc``), ``offset`` and ``limit`` (default 100, at most 1000).

        Returns:
            ``{total, offset, limit, items}``; ``total`` counts the matches.
        """
        rows = self.filtered(query)
        sort = query.get("sort") or ""
        if sort == "entry":
            rows = sorted(rows, key=lambda r: r.entry)
        elif sort == "seconds":
            rows = sorted(rows, key=lambda r: (r.seconds is None, r.seconds or 0.0))
        elif sort:
            raise ServiceError(400, f"sort must be 'entry' or 'seconds', got {sort!r}")
        order = query.get("order") or "asc"
        if order not in ("asc", "desc"):
            raise ServiceError(400, f"order must be 'asc' or 'desc', got {order!r}")
        if order == "desc":
            rows = rows[::-1]
        offset = _int("offset", query.get("offset"), 0, 0, 10**12)
        limit = _int("limit", query.get("limit"), DEFAULT_LIMIT, 1, MAX_LIMIT)
        return {
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "items": [r.to_dict() for r in rows[offset : offset + limit]],
        }

    def random_draw(self, query: Mapping[str, str]) -> Dict[str, Any]:
        """One draw picked at random from the filtered ones.

        Args:
            query: The filters of :meth:`filtered`, plus ``pick_seed`` (an integer; the
                same seed and filters always pick the same draw).

        Returns:
            ``{pick_seed, total, draw}`` where ``draw`` is the chosen list row.

        Raises:
            ServiceError: 404 when no draw matches.
        """
        rows = self.filtered(query)
        if not rows:
            raise ServiceError(404, "no draw matches the filters")
        pick = query.get("pick_seed")
        seed = (
            random.SystemRandom().randrange(2**31)
            if pick in (None, "")
            else _int("pick_seed", pick, 0, 0, 2**63)
        )
        chosen = rows[random.Random(seed).randrange(len(rows))]
        return {"pick_seed": seed, "total": len(rows), "draw": chosen.to_dict()}

    def draw(self, entry: Optional[str]) -> Dict[str, Any]:
        """One draw: its list row, declared record and what its bundle holds.

        Args:
            entry: Entry id.

        Returns:
            ``row``, ``record`` (the manifest line as stored), ``seeds`` (as strings)
            and, when the bundle reads, ``bundle`` (``dt_ms``, ``n_frames``,
            ``duration_ms``, ``layout`` with the sensor coordinates, ``populations``,
            ``units``); otherwise ``bundle`` is ``None`` and ``error`` says why.

        Raises:
            ServiceError: 400 without an entry, 404 for an unknown one.
        """
        row = self._row(entry)
        record = self.record(row.entry)
        out: Dict[str, Any] = {
            "row": row.to_dict(),
            "record": record,
            "seeds": {k: str(v) for k, v in (record.get("seeds") or {}).items()},
            "bundle": None,
            "error": None,
        }
        try:
            info = self.info(row.entry)
        except ServiceError as exc:
            if exc.status != 422:
                raise
            out["error"] = exc.message
            return out
        layout = info.layout.summary()
        layout["coords"] = np.asarray(info.layout.coords, dtype=float).tolist()
        out["bundle"] = {
            "dt_ms": info.dt,
            "n_frames": info.n_frames,
            "duration_ms": info.n_frames * info.dt,
            "layout": layout,
            "populations": [
                {"name": n, "readout": r, "n_units": u} for n, r, u in info.populations
            ],
            "units": self._units(),
            "schema_version": info.schema_version,
            "sensoryforge_sha": info.sensoryforge_sha,
        }
        return out

    # ------------------------------------------------------------------- frames

    def frames(self, entry: Optional[str], query: Mapping[str, str]) -> "FrameChunk":
        """A run of stimulus frames of one draw as raw little-endian float32.

        Args:
            entry: Entry id.
            query: ``start`` (first frame, default 0), ``count`` (frames wanted, default
                32; fewer come back at the end of the draw or when the chunk would pass
                8 MiB) and ``channel`` (default 0).

        Returns:
            The chunk: ``count`` frames of one channel, ``[count, H, W]`` row-major.

        Raises:
            ServiceError: 400 for a bad query, 404 for an unknown draw, 422 for an
                unreadable bundle.
        """
        row = self._row(entry)
        info = self.info(row.entry)
        n_total = info.n_frames
        n_ch = info.layout.n_channels
        start = _int("start", query.get("start"), 0, 0, max(n_total - 1, 0))
        want = _int("count", query.get("count"), DEFAULT_CHUNK_FRAMES, 1, 10**6)
        channel = _int("channel", query.get("channel"), 0, 0, max(n_ch - 1, 0))
        path = self._h5_path(row.entry)
        try:
            with h5py.File(path, "r") as f:
                ds = f["stimulus/frames"]
                height, width = int(ds.shape[-2]), int(ds.shape[-1])
                count = max(
                    1,
                    min(want, n_total - start, MAX_CHUNK_BYTES // (4 * height * width)),
                )
                block = self._frame_run(row.entry, ds, channel, start, count)
        except (OSError, KeyError, ValueError) as exc:
            raise self._unreadable(entry, exc) from exc
        finite = block[np.isfinite(block)]
        return FrameChunk(
            body=block.tobytes(),
            start=start,
            count=count,
            height=height,
            width=width,
            total=n_total,
            dt_ms=float(info.dt),
            low=float(finite.min()) if finite.size else 0.0,
            high=float(finite.max()) if finite.size else 0.0,
        )

    def _h5_path(self, entry: str) -> Path:
        try:
            return self.reader().bundle_path(entry) / "data.h5"
        except ReadError as exc:
            raise ServiceError(422, f"draw {entry!r} cannot be read ({exc.message})")

    @staticmethod
    def _unreadable(entry: Optional[str], exc: Exception) -> ServiceError:
        """The 422 for an h5 read that failed after the bundle's peek succeeded."""
        return ServiceError(422, f"draw {entry!r} cannot be read ({exc})")

    def _frame_run(
        self, entry: str, ds: h5py.Dataset, channel: int, start: int, count: int
    ) -> np.ndarray:
        """Frames ``[start, start + count)`` of one channel, ``[count, H, W]`` float32.

        HDF5 decompresses a whole stored chunk to give any frame in it, and a chunk
        spans hundreds of frames, so reading a 32-frame run straight from the file
        repeats that work for every run. The frames are read instead as time slabs
        aligned to the stored chunk (``ds.chunks[0]`` frames) and the decompressed
        slabs are kept, least recently used first, up to ``frame_cache_bytes``. A
        slab larger than the whole cache is read and used without being kept, and a
        dataset that is not chunked is read for the run alone (a contiguous read
        decompresses nothing).
        """
        n_total = int(ds.shape[0])
        slab_len = ds.chunks[0] if ds.chunks else 0

        def read(lo: int, hi: int) -> np.ndarray:
            raw = ds[lo:hi] if ds.ndim == 3 else ds[lo:hi, channel]
            return np.ascontiguousarray(raw, dtype="<f4")

        if slab_len <= 0:
            return read(start, start + count)
        runs = []
        stop = start + count
        with self._slab_lock:
            for lo in range((start // slab_len) * slab_len, stop, slab_len):
                key = (entry, channel, lo)
                slab = self._slabs.get(key)
                if slab is None:
                    self.slab_misses += 1
                    slab = read(lo, min(lo + slab_len, n_total))
                    self._keep(key, slab)
                else:
                    self.slab_hits += 1
                    self._slabs.move_to_end(key)
                runs.append(slab[max(start - lo, 0) : stop - lo])
        return runs[0] if len(runs) == 1 else np.concatenate(runs)

    def _keep(self, key: Tuple[str, int, int], slab: np.ndarray) -> None:
        """Add a slab to the cache and evict the oldest until the total fits."""
        if slab.nbytes > self.frame_cache_bytes:
            return
        self._slabs[key] = slab
        self._slab_bytes += slab.nbytes
        while self._slab_bytes > self.frame_cache_bytes:
            _, old = self._slabs.popitem(last=False)
            self._slab_bytes -= old.nbytes

    def cache_stats(self) -> Dict[str, int]:
        """Slab cache counters: ``hits``, ``misses``, ``slabs``, ``bytes``, ``limit``.

        Returns:
            The five counters as integers.
        """
        with self._slab_lock:
            return {
                "hits": self.slab_hits,
                "misses": self.slab_misses,
                "slabs": len(self._slabs),
                "bytes": self._slab_bytes,
                "limit": self.frame_cache_bytes,
            }

    # ------------------------------------------------------------------- traces

    def trace(self, entry: Optional[str], query: Mapping[str, str]) -> Dict[str, Any]:
        """Series of one draw as JSON.

        Args:
            entry: Entry id.
            query: ``source`` (``stimulus``, ``drive``, ``filtered`` or ``readout``),
                ``population`` (for the three population sources), ``index`` (a
                comma-separated list of sensor ids, for ``stimulus``, or unit ids; at
                most 64), ``channel`` (stimulus channel, default 0) and ``max_points``
                (default 2000, at most 20000; longer series are decimated keeping each
                bucket's extremes).

        Returns:
            ``{entry, source, population, readout, dt_ms, n_frames, decimated,
            series, units}``; ``series`` is ``[{index, time_ms, values}]``, each with
            its own kept frames when it was decimated.

        Raises:
            ServiceError: 400 for a bad query, 404 for an unknown draw or population,
                422 for an unreadable bundle.
        """
        row = self._row(entry)
        info = self.info(row.entry)
        source = query.get("source") or ""
        if source not in SOURCES:
            raise ServiceError(400, f"source must be one of {list(SOURCES)}")
        ids = [
            _int("index", v, 0, 0, 10**9) for v in (_csv(query.get("index")) or ["0"])
        ]
        if len(ids) > MAX_SERIES:
            raise ServiceError(400, f"at most {MAX_SERIES} series per request")
        ids = sorted(set(ids))
        points = _int(
            "max_points", query.get("max_points"), DEFAULT_POINTS, 2, MAX_POINTS
        )
        population: Optional[str] = None
        readout: Optional[str] = None
        path = self._h5_path(row.entry)
        try:
            with h5py.File(path, "r") as f:
                time_ms = np.asarray(f["time_ms"][...], dtype=np.float64)
                if source == "stimulus":
                    data = self._stimulus_series(f["stimulus/frames"], info, ids, query)
                else:
                    population = query.get("population")
                    known = {n: r for n, r, _u in info.populations}
                    if population not in known:
                        raise ServiceError(
                            404,
                            f"population must be one of {sorted(known)}, "
                            f"got {population!r}",
                        )
                    readout = known[population]
                    group = f[f"populations/{population}"]
                    ds = group[readout if source == "readout" else source]
                    if ids[-1] >= ds.shape[1]:
                        raise ServiceError(
                            400, f"unit {ids[-1]} is out of range: {ds.shape[1]} units"
                        )
                    data = np.asarray(ds[:, ids], dtype=np.float64)
        except (OSError, KeyError, ValueError) as exc:
            raise self._unreadable(entry, exc) from exc
        series = []
        decimated = False
        for k, unit in enumerate(ids):
            frames = decimate(data[:, k], points)
            decimated = decimated or frames.size < info.n_frames
            col = data[frames, k]
            if np.isfinite(col).all():
                values: List[Any] = col.tolist()
            else:
                values = [float(v) if np.isfinite(v) else None for v in col]
            series.append(
                {"index": unit, "time_ms": time_ms[frames].tolist(), "values": values}
            )
        return {
            "entry": row.entry,
            "source": source,
            "population": population,
            "readout": readout,
            "dt_ms": info.dt,
            "n_frames": info.n_frames,
            "decimated": decimated,
            "series": series,
            "units": self._units(),
        }

    @staticmethod
    def _stimulus_series(
        ds: h5py.Dataset, info: BundleInfo, ids: Sequence[int], query: Mapping[str, str]
    ) -> np.ndarray:
        """``[T, K]`` frame values at the given sensors of one channel."""
        n_ch = info.layout.n_channels
        channel = _int("channel", query.get("channel"), 0, 0, max(n_ch - 1, 0))
        n_sensors = info.layout.n_sensors
        if ids[-1] >= n_sensors:
            raise ServiceError(
                400, f"sensor {ids[-1]} is out of range: {n_sensors} sensors"
            )
        grid = info.layout.regular_grid
        if grid is None:
            raise ServiceError(422, "the sensors are not a regular grid")
        cols = int(grid["cols"])
        out = np.empty((ds.shape[0], len(ids)), dtype=np.float64)
        for n, k in enumerate(ids):
            i, j = divmod(k, cols)
            out[:, n] = ds[:, i, j] if ds.ndim == 3 else ds[:, channel, i, j]
        return out
