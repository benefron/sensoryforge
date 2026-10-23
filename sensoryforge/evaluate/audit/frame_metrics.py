"""Stimulus statistics read back from the frames (report contract, section 6.1).

Pure functions over one channel's frames ``[T, M]``; no unit is assumed. A frame is
*active* when any of its values has ``|x| > zero_tolerance``; a *bout* is a maximal run
of active frames. Everything is measured on the frames alone, so a quantity such
as motion is the motion of the frame-weighted centroid, not of anything the data
was drawn from.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np

#: Frames per block when a quantity needs a ``[block, M, D]`` temporary.
_BLOCK = 256
#: A centroid displacement below this share of the layout's largest extent is zero
#: (float32 centroid noise on a still shape is at most about 1e-7 of it).
MOVE_FLOOR = 1e-6
#: A signal whose swing is below this share of its own size carries no power.
_POWER_FLOOR = 1e-9
_AXES = ("x", "y", "z")


def active_mask(x: np.ndarray, tol: float) -> np.ndarray:
    """Which frames are active.

    Args:
        x: One channel's frames ``[T, M]``.
        tol: Zero tolerance in the frame unit.

    Returns:
        Boolean ``[T]``, true where any value has ``|x| > tol``.
    """
    return (np.abs(x) > tol).any(axis=1)


def runs(active: np.ndarray) -> np.ndarray:
    """The ``[start, stop)`` frame ranges of the bouts.

    Args:
        active: Boolean ``[T]``.

    Returns:
        Integer ``[B, 2]`` (empty ``[0, 2]`` when no frame is active).
    """
    edge = np.diff(np.concatenate([[0], active.astype(np.int8), [0]]))
    return np.stack([np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)], axis=1)


def centroids(
    x: np.ndarray, frames: np.ndarray, coords: np.ndarray
) -> Dict[str, np.ndarray]:
    """Weighted centroid and spread of the given frames.

    Args:
        x: One channel's frames ``[T, M]``.
        frames: Indices of the (active) frames to measure ``[K]``.
        coords: Sensor coordinates ``[M, D]``.

    Returns:
        ``centroid`` ``[K, D]`` (the ``|x|``-weighted mean position of each frame) and
        ``spread`` ``[K]`` (the weighted RMS distance from it), in the coordinate unit.
    """
    c = np.zeros((frames.size, coords.shape[1]))
    s = np.zeros(frames.size)
    for a in range(0, frames.size, _BLOCK):
        idx = frames[a : a + _BLOCK]
        w = np.abs(x[idx]).astype(np.float64)
        total = w.sum(axis=1)
        cen = (w @ coords) / total[:, None]
        d2 = ((coords[None, :, :] - cen[:, None, :]) ** 2).sum(axis=-1)
        c[a : a + _BLOCK] = cen
        s[a : a + _BLOCK] = np.sqrt((w * d2).sum(axis=1) / total)
    return {"centroid": c, "spread": s}


def dominant_frequency(
    x: np.ndarray, start: int, stop: int, dt: float
) -> Optional[float]:
    """The frequency with the most power in one bout's sensor-mean signal.

    Args:
        x: One channel's frames ``[T, M]``.
        start: First frame of the bout.
        stop: One past its last frame.
        dt: Frame step in the time unit.

    Returns:
        Frequency in 1/time (resolution ``1 / ((stop - start) * dt)``); ``None`` for a
        bout shorter than 4 frames or one whose mean-removed signal has no power.
    """
    n = stop - start
    if n < 4:
        return None
    sig = x[start:stop].mean(axis=1, dtype=np.float64)
    ref = float(np.abs(sig).max())
    sig = sig - sig.mean()
    if ref == 0.0 or float(np.abs(sig).max()) <= _POWER_FLOOR * ref:
        return None
    power = np.abs(np.fft.rfft(sig)) ** 2
    power[0] = 0.0
    k = int(np.argmax(power))
    return k / (n * dt)


def digest(x: np.ndarray) -> str:
    """sha256 of the frames as little-endian float32 in C order.

    Args:
        x: One channel's frames ``[T, M]``.

    Returns:
        The hex digest.
    """
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest()


def measure_channel(
    x: np.ndarray, coords: np.ndarray, dt: float, tol: float
) -> Dict[str, Any]:
    """Every ``core.frames.*`` statistic of one channel (contract, section 6.1).

    Args:
        x: The channel's frames ``[T, M]`` (float32).
        coords: Sensor coordinates ``[M, D]`` in the space unit.
        dt: Frame step in the time unit.
        tol: ``zero_tolerance`` in the frame unit.

    Returns:
        ``{name: value}`` keyed by the metric's last id part (``onset_time``, ...);
        ``None`` where a statistic is undefined, plus ``_active`` (the boolean ``[T]``
        activity mask, for the response metrics).
    """
    n_t, n_m = x.shape
    dims = coords.shape[1]
    active = active_mask(x, tol)
    bouts = runs(active)
    lengths = bouts[:, 1] - bouts[:, 0]
    out: Dict[str, Any] = {"_active": active}
    out["n_frames"] = n_t
    out["dt"] = dt
    out["duration"] = n_t * dt
    out["peak"] = float(x.max())
    out["trough"] = float(x.min())
    out["mean"] = float(x.mean(dtype=np.float64))
    out["rms"] = float(np.sqrt(np.einsum("tm,tm->", x, x, dtype=np.float64) / x.size))
    out["negative_fraction"] = float((x < -tol).mean())
    n_active = int(active.sum())
    out["active_time"] = n_active * dt
    out["active_fraction"] = n_active / n_t
    out["bouts"] = int(bouts.shape[0])
    out["onset_time"] = None if n_active == 0 else float(bouts[0, 0]) * dt
    out["offset_time"] = None if n_active == 0 else float(bouts[-1, 1]) * dt
    out["bout_mean"] = None if n_active == 0 else float(lengths.mean()) * dt
    gaps = bouts[1:, 0] - bouts[:-1, 1]
    out["gap_mean"] = None if gaps.size == 0 else float(gaps.mean()) * dt
    out["gap_max"] = None if gaps.size == 0 else float(gaps.max()) * dt
    for name in ("footprint_mean", "footprint_max", "spread", "path_length"):
        out[name] = None
    for d in range(dims):
        out[f"centroid_{_AXES[d]}"] = None
    out["centroid_speed"] = None
    out["direction"] = None
    out["dominant_frequency"] = None
    if n_active:
        frames = np.flatnonzero(active)
        share = (np.abs(x[frames]) > tol).sum(axis=1) / n_m
        out["footprint_mean"] = float(share.mean())
        out["footprint_max"] = float(share.max())
        cs = centroids(x, frames, coords)
        cen = cs["centroid"]
        for d in range(dims):
            out[f"centroid_{_AXES[d]}"] = float(cen[:, d].mean())
        out["spread"] = float(cs["spread"].mean())
        pos = np.full((n_t, dims), np.nan)
        pos[frames] = cen
        stepped = active[:-1] & active[1:]
        steps = np.linalg.norm(pos[1:][stepped] - pos[:-1][stepped], axis=1)
        out["path_length"] = float(steps.sum())
        if steps.size:
            out["centroid_speed"] = float(steps.sum()) / (steps.size * dt)
        if dims >= 2:
            move = cen[-1, :2] - cen[0, :2]
            extent = float(np.ptp(coords, axis=0).max())
            if float(np.hypot(*move)) > MOVE_FLOOR * max(extent, 1e-300):
                angle = float(np.degrees(np.arctan2(move[1], move[0])) % 360.0)
                out["direction"] = 0.0 if angle >= 360.0 else angle
        longest = int(np.argmax(lengths))
        out["dominant_frequency"] = dominant_frequency(
            x, int(bouts[longest, 0]), int(bouts[longest, 1]), dt
        )
    out["digest"] = digest(x)
    return out
