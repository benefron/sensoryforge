"""Writing a static snapshot of a data set (roadmap A12).

The output folder holds ``index.html`` (the draw browser), ``dashboard.html``,
``gallery.html`` and ``checks.html``: the server's own pages with their links made
relative, ``static/`` (the pages' scripts and style plus ``snapshot_shim.js``) and
``snapshot/`` (``index.js``: the recorded answers for the pages and the draw lists; one
``draw_N.js`` per chosen draw: its frames, record and traces). The shim answers each
page's ``fetch`` from those files, so the folder opens from disk (``file://``) with
no server.
Everything the shim does not hold answers with an error that says so.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl

from sensoryforge.evaluate.export.snapshot import (
    Recording,
    local_paths,
    walk_queries,
)
from sensoryforge.evaluate.export.video import FORMATS, ffmpeg_path, write_video
from sensoryforge.evaluate.server import Api, DatasetService, ServiceError
from sensoryforge.evaluate.server.api import STATIC_DIR

#: The page files of the server, and the name each has in the export.
PAGES: Dict[str, str] = {
    "index.html": "index.html",
    "dashboard.html": "dashboard.html",
    "gallery.html": "gallery.html",
    "checks.html": "checks.html",
}

#: The links of the server's pages and the file each becomes.
LINKS: Dict[str, str] = {
    'href="/"': 'href="index.html"',
    'href="/dashboard"': 'href="dashboard.html"',
    'href="/gallery"': 'href="gallery.html"',
    'href="/checks"': 'href="checks.html"',
}

#: Draws per page of a list, as the draw browser asks.
PAGE = 100
#: Frames per request, as the draw browser asks.
CHUNK = 32
#: The ``limit`` choices of the gallery page.
GALLERY_LIMITS = ("12", "24", "60", "200")


@dataclass
class ExportOptions:
    """What to put in a snapshot.

    Attributes:
        draws: Entry ids to include in full. When empty, ``n_draws`` draws spread evenly
            over the manifest are taken.
        n_draws: How many draws to take when ``draws`` is empty.
        sensors: Sensor indices whose stimulus trace is recorded for each draw; ``None``
            takes the middle sensor (an arbitrary but fixed choice), ``[]`` none.
        units: ``(population, unit index)`` pairs whose traces are recorded for each
            draw that has that population; ``None`` takes each population's middle unit.
        max_list_rows: Rows of any one draw list that are recorded (whole pages of
            ``PAGE``); the list shows its true total but cannot be paged past this.
        max_mb: Refuse to write more than this many megabytes of recorded answers.
        video: ``mp4`` or ``gif`` to also write a movie of each chosen draw when
            ffmpeg is installed; ``None`` for none.
        video_speed: Playback speed of the movies (1 is real time).
        made_at: Text shown as the snapshot's time (default: now, UTC).
    """

    draws: Sequence[str] = ()
    n_draws: int = 3
    sensors: Optional[Sequence[int]] = None
    units: Optional[Sequence[Tuple[str, int]]] = None
    max_list_rows: int = 1000
    max_mb: float = 500.0
    video: Optional[str] = None
    video_speed: float = 1.0
    made_at: Optional[str] = None


@dataclass
class ExportResult:
    """What an export wrote.

    Attributes:
        out: The output folder.
        draws: The entry ids included in full.
        pages: The page files written.
        videos: The movie files written (empty when none were asked for or ffmpeg is
            missing).
        notes: Things left out and why, for the caller to show.
        n_bytes: Bytes of recorded answers (before base64 and JSON framing).
    """

    out: Path
    draws: List[str] = field(default_factory=list)
    pages: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    n_bytes: int = 0


def _js_assign(target: str, payload: Any) -> str:
    """JavaScript that stores ``payload`` at ``target`` (``<`` is escaped for HTML)."""
    text = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), ensure_ascii=False
    )
    return f"{target}={text.replace('<', chr(92) + 'u003c')};\n"


def rewrite_page(html: str) -> str:
    """Make one of the server's pages open from disk.

    Args:
        html: The page as the server serves it.

    Returns:
        The page with ``/static/`` and its page links relative, and the snapshot's
        scripts loaded before the page's own.

    Raises:
        ValueError: The page has no script from ``/static/``.
    """
    html = html.replace('href="/static/', 'href="static/').replace(
        'src="/static/', 'src="static/'
    )
    for old, new in LINKS.items():
        html = html.replace(old, new)
    html = html.replace("</head>", '<link rel="icon" href="data:,">\n</head>', 1)
    marker = '<script src="static/'
    at = html.find(marker)
    if at < 0:
        raise ValueError("the page loads no script from /static/")
    head = (
        '<script src="snapshot/index.js"></script>\n'
        '<script src="static/snapshot_shim.js"></script>\n'
    )
    return html[:at] + head + html[at:]


def _evenly(rows: Sequence[str], n: int) -> List[str]:
    """``n`` entries spread evenly over ``rows`` (all of them when there are fewer)."""
    if n <= 0 or not rows:
        return []
    if n >= len(rows):
        return list(rows)
    return [rows[(i * len(rows)) // n] for i in range(n)]


def _record_lists(
    rec: Recording, summary: Dict[str, Any], options: ExportOptions
) -> int:
    """Record the draw lists the pages can ask for; returns how many were cut short."""
    filters: List[Dict[str, str]] = [{}]
    filters += [{"split": k} for k in summary["counts"]["split"]]
    filters += [{"class": k} for k in summary["counts"]["class"]]
    _, dash = rec.get("/api/dashboard")
    if isinstance(dash, dict):
        filters += [dict(parse_qsl(q)) for q in walk_queries(dash)]
    seen, cut = set(), 0
    for base in filters:
        token = json.dumps(sorted(base.items()))
        if token in seen:
            continue
        seen.add(token)
        offset = 0
        while True:
            status, page = rec.get(
                "/api/draws", {**base, "offset": offset, "limit": PAGE}
            )
            if status != 200:
                break
            offset += PAGE
            if offset >= page["total"]:
                break
            if offset >= options.max_list_rows:
                cut += 1
                break
    return cut


def _pins(
    service: DatasetService, entry: str, options: ExportOptions
) -> Tuple[List[int], List[Tuple[str, int]]]:
    """The sensors and units whose traces are recorded for one draw."""
    info = service.info(entry)
    n_sensors = len(info.layout.coords)
    if options.sensors is None:
        sensors = [n_sensors // 2] if n_sensors else []
    else:
        sensors = [k for k in options.sensors if 0 <= k < n_sensors]
    units: List[Tuple[str, int]] = []
    for name, _, n_units in info.populations:
        if options.units is None:
            if n_units:
                units.append((name, n_units // 2))
        else:
            units += [
                (p, i) for p, i in options.units if p == name and 0 <= i < n_units
            ]
    return sensors, units


def _record_draw(
    rec: Recording,
    service: DatasetService,
    entry: str,
    options: ExportOptions,
    budget: int,
) -> None:
    """Record what the draw browser asks for when ``entry`` is opened and pinned."""
    status, draw = rec.get("/api/draw", {"entry": entry})
    bundle = draw.get("bundle") if status == 200 else None
    if bundle is None:
        return
    info = service.info(entry)
    n_channels = info.layout.n_channels
    for channel in range(n_channels):
        for start in range(0, bundle["n_frames"], CHUNK):
            rec.get(
                "/api/draw/frames",
                {"entry": entry, "start": start, "count": CHUNK, "channel": channel},
            )
            if rec.n_bytes > budget:
                raise ValueError(
                    f"{entry}: the frames alone pass the size limit; raise max_mb "
                    "(--max-mb) or choose fewer draws"
                )
    sensors, units = _pins(service, entry, options)
    for k in sensors:
        for channel in range(n_channels):
            rec.get(
                "/api/draw/trace",
                {"entry": entry, "source": "stimulus", "index": k, "channel": channel},
            )
    for pop in bundle["populations"]:
        rec.get("/api/draw/population", {"entry": entry, "population": pop["name"]})
    for name, idx in units:
        for source in ("drive", "filtered", "readout"):
            rec.get(
                "/api/draw/trace",
                {"entry": entry, "source": source, "population": name, "index": idx},
            )
        rec.get("/api/draw/voltage", {"entry": entry, "population": name, "index": idx})
        rec.get(
            "/api/draw/population", {"entry": entry, "population": name, "index": idx}
        )


def _within(spent: int, budget: int, where: str, options: ExportOptions) -> None:
    """Raise once the recorded answers (frames, traces, voltages, lists) pass max_mb."""
    if spent > budget:
        raise ValueError(
            f"{where}: the recorded answers ({spent / (1 << 20):.2f} MB) pass the "
            f"size limit of {options.max_mb:g} MB; raise max_mb (--max-mb) or choose "
            "fewer draws"
        )


def export_snapshot(
    service: DatasetService, out: Path, options: Optional[ExportOptions] = None
) -> ExportResult:
    """Write a snapshot of a data set that opens offline.

    Args:
        service: The service over the data-set folder (and its stage files).
        out: The folder to write; it is created and must be empty or not exist.
        options: What to include; the defaults when omitted.

    Returns:
        What was written.

    Raises:
        ValueError: ``out`` is a non-empty folder, a chosen draw is not in the data set,
            or the recorded answers pass ``max_mb``.
    """
    options = options or ExportOptions()
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; choose a new folder")
    if options.video is not None and options.video not in FORMATS:
        raise ValueError(f"video must be one of {list(FORMATS)}, got {options.video!r}")
    result = ExportResult(out)
    manifest = [r.entry for r in service.rows]
    chosen = list(options.draws) or _evenly(manifest, options.n_draws)
    for entry in chosen:
        try:
            service.draw(entry)
        except ServiceError as exc:
            raise ValueError(f"{entry}: {exc.message}") from exc
    result.draws = chosen
    api = Api(service)
    budget = int(options.max_mb * (1 << 20))

    # a shared snapshot names its folders, never where they are on this machine
    redact = local_paths(service.folder, service.stages)
    rec = Recording(api, redact=redact)
    _, summary = rec.get("/api/dataset")
    rec.get("/api/stages")
    rec.get("/api/stage/checks")
    for limit in GALLERY_LIMITS:
        rec.get("/api/gallery", {"limit": limit})
    cut = _record_lists(rec, summary, options)
    if cut:
        result.notes.append(
            f"{cut} draw lists hold only their first {options.max_list_rows} rows"
        )

    snap = out / "snapshot"
    snap.mkdir(parents=True)
    files: Dict[str, str] = {}
    spent = rec.n_bytes
    _within(spent, budget, "the data set's own answers", options)
    for n, entry in enumerate(chosen):
        part = Recording(api, redact=redact)
        _record_draw(part, service, entry, options, budget - spent)
        spent += part.n_bytes
        _within(spent, budget, entry, options)
        name = f"snapshot/draw_{n}.js"
        parts = "(window.EvalSnapshotParts=window.EvalSnapshotParts||{})"
        target = f"{parts}[{json.dumps(name)}]"
        (out / name).write_text(_js_assign(target, part.entries), encoding="utf-8")
        for k in part.entries:
            files[k] = name
    result.n_bytes = spent
    made = options.made_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    index = {
        "made_at": made,
        "name": summary.get("name") or summary.get("folder"),
        "draws": chosen,
        "entries": rec.entries,
        "files": files,
    }
    (snap / "index.js").write_text(
        _js_assign("window.EvalSnapshot", index), encoding="utf-8"
    )

    static = out / "static"
    shutil.copytree(
        STATIC_DIR, static, ignore=shutil.ignore_patterns("*.html", "__pycache__")
    )
    shutil.copy(Path(__file__).parent / "static" / "snapshot_shim.js", static)
    for source, target in PAGES.items():
        text = (STATIC_DIR / source).read_text(encoding="utf-8")
        (out / target).write_text(rewrite_page(text), encoding="utf-8")
        result.pages.append(target)

    if options.video is not None:
        if ffmpeg_path() is None:
            result.notes.append("no video files: ffmpeg is not installed")
        else:
            movies = out / "videos"
            movies.mkdir()
            for entry in chosen:
                safe = re.sub(r"[^A-Za-z0-9_.-]", "_", entry)
                path = movies / f"{safe}.{options.video}"
                write_video(service, entry, path, options.video, options.video_speed)
                result.videos.append(f"videos/{path.name}")
    return result
