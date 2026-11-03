"""Movie files of chosen draws, only when ffmpeg is installed (roadmap A12, D-7e30f2b).

The colours are the draw browser's: the same viridis-like ramp, the same scale rule
(zero or below up to the audit's peak, else the range of the frames) and the same
orientation (x to the right, y upwards).
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from sensoryforge.evaluate.server import DatasetService

#: Output formats, by file extension.
FORMATS = ("mp4", "gif")

#: The ramp's stops as ``(t, r, g, b)``, as in ``static/geometry.js``.
STOPS = [
    (0.0, 68, 1, 84),
    (0.25, 59, 82, 139),
    (0.5, 33, 145, 140),
    (0.75, 94, 201, 98),
    (1.0, 253, 231, 37),
]

#: Frames per second of the clip, and the smallest side a frame is enlarged to (pixels).
FPS = 30
MIN_SIDE = 360


def ffmpeg_path() -> Optional[str]:
    """The ``ffmpeg`` executable on the search path, or ``None``."""
    return shutil.which("ffmpeg")


def _lut() -> np.ndarray:
    """The 256-entry colour table, ``uint8 [256, 3]``."""
    stops = np.array(STOPS, dtype=float)
    t = np.arange(256) / 255.0
    return np.stack(
        [np.floor(np.interp(t, stops[:, 0], stops[:, c + 1]) + 0.5) for c in range(3)],
        axis=1,
    ).astype(np.uint8)


def scale_for(peak: Optional[float], low: float, high: float) -> "tuple[float, float]":
    """The colour range of a movie, by the draw browser's rule.

    Args:
        peak: The audit's frame peak of the channel, or ``None``.
        low: Lowest finite frame value.
        high: Highest finite frame value.

    Returns:
        ``(low, high)``: zero or below up to the peak when the peak is above that, else
        the range of the frames.
    """
    if peak is not None and math.isfinite(peak):
        lo = min(0.0, low)
        if peak > lo:
            return lo, peak
    if not high >= low:
        return 0.0, 1.0
    return low, high


def to_rgb(frame: np.ndarray, low: float, high: float, lut: np.ndarray) -> np.ndarray:
    """Colour one stored frame.

    Args:
        frame: ``[nx, ny]`` values (axis 0 is x, axis 1 is y).
        low: Value at the bottom of the ramp.
        high: Value at the top.
        lut: The colour table of :func:`_lut`.

    Returns:
        ``uint8 [ny, nx, 3]`` with y upwards (row 0 is the largest y).
    """
    span = high - low if high > low else 1.0
    t = np.where(np.isfinite(frame), np.clip((frame - low) / span, 0.0, 1.0), 0.0)
    idx = np.floor(255.0 * t + 0.5).astype(np.intp)
    return lut[idx].transpose(1, 0, 2)[::-1]


def ffmpeg_command(
    exe: str, width: int, height: int, fps: int, out: Path, fmt: str
) -> List[str]:
    """The ffmpeg command that turns raw RGB frames on stdin into ``out``."""
    k = max(1, math.ceil(MIN_SIDE / max(1, min(width, height))))
    grow = f"scale=iw*{k}:ih*{k}:flags=neighbor"
    if fmt == "mp4":
        vf = grow + ",pad=ceil(iw/2)*2:ceil(ih/2)*2"
        codec = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    elif fmt == "gif":
        vf = grow + ",split[a][b];[a]palettegen[p];[b][p]paletteuse"
        codec = []
    else:
        raise ValueError(f"video format must be one of {list(FORMATS)}, got {fmt!r}")
    return [
        exe, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-framerate", str(fps), "-i", "-", "-vf", vf, *codec, str(out),
    ]  # fmt: skip


def write_video(
    service: "DatasetService",
    entry: str,
    out: Path,
    fmt: str = "mp4",
    speed: float = 1.0,
    channel: int = 0,
) -> Path:
    """Write one draw's stimulus movie.

    Args:
        service: The data-set service.
        entry: Entry id.
        out: The file to write (its extension is not changed).
        fmt: ``mp4`` or ``gif``.
        speed: Playback speed relative to simulated time (1 is real time; the clip keeps
            every ``speed * 1000 / (FPS * dt_ms)``-th frame, at least every frame).
        channel: Stimulus channel.

    Returns:
        ``out``.

    Raises:
        RuntimeError: ffmpeg is not installed, or it failed.
        ValueError: A bad format or speed, or a draw whose bundle does not read.
    """
    exe = ffmpeg_path()
    if exe is None:
        raise RuntimeError("ffmpeg is not installed, so no video file can be written")
    if not speed > 0:
        raise ValueError(f"speed must be above 0, got {speed}")
    draw: Dict[str, Any] = service.draw(entry)
    if draw["bundle"] is None:
        raise ValueError(f"{entry}: {draw['error']}")
    total = int(draw["bundle"]["n_frames"])
    dt = float(draw["bundle"]["dt_ms"])
    step = max(1, round(speed * 1000.0 / (FPS * dt)))
    values = (draw["frame_peak"] or {}).get("values", [])
    peak = next((v["value"] for v in values if v["channel"] == channel), None)
    if peak is None:
        peak = next((v["value"] for v in values if v["channel"] is None), None)
    starts = range(0, total, 32)

    def chunk_at(start: int) -> Any:
        return service.frames(
            entry, {"start": str(start), "count": "32", "channel": str(channel)}
        )

    # one pass reads only each chunk's range, so a long draw is never held whole
    lows, highs, shape = [], [], None
    for start in starts:
        c = chunk_at(start)
        lows.append(c.low)
        highs.append(c.high)
        shape = shape or (c.height, c.width)
    lo, hi = scale_for(peak, min(lows), max(highs))
    lut = _lut()
    width, height = shape  # nx columns, ny rows
    proc = subprocess.Popen(
        ffmpeg_command(exe, width, height, FPS, Path(out), fmt),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    err = ""
    try:
        if proc.stdin is None:
            raise RuntimeError("ffmpeg did not open its input")
        for start in starts:
            c = chunk_at(start)
            block = np.frombuffer(c.body, dtype="<f4").reshape(
                c.count, c.height, c.width
            )
            for i in range(c.count):
                if (c.start + i) % step == 0:
                    proc.stdin.write(to_rgb(block[i], lo, hi, lut).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    finally:
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {entry}: {err.strip()}")
    return Path(out)
