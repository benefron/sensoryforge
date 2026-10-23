"""Population input and response statistics (report contract, sections 6.2 and 6.3).

Pure functions over one population's arrays ``[T, N]``. The *active* and *quiet* windows
are the stimulus's active and inactive frames over any channel; the onset is the
first of those active frames. Nothing here reads the declaration.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np


def input_stats(drive: np.ndarray, filtered: np.ndarray) -> Dict[str, Any]:
    """``core.input.*`` of one population.

    Args:
        drive: The population's drive ``[T, N]`` before its filter.
        filtered: The drive after its filter ``[T, N]`` (signed).

    Returns:
        ``drive_peak``, ``drive_mean``, ``filtered_max``, ``filtered_min``.
    """
    return {
        "drive_peak": float(drive.max()),
        "drive_mean": float(drive.mean(dtype=np.float64)),
        "filtered_max": float(filtered.max()),
        "filtered_min": float(filtered.min()),
    }


def isi_cv(spikes: np.ndarray) -> Optional[float]:
    """Mean inter-spike-interval CV over the units with at least 3 spikes.

    Args:
        spikes: Boolean ``[T, N]``.

    Returns:
        The mean over such units of ``std / mean`` of their interval lengths (population
        std, in frames; the ratio is dimensionless); ``None`` if no unit has 3 spikes.
    """
    unit, frame = np.nonzero(spikes.T)
    if unit.size < 3:
        return None
    same = unit[1:] == unit[:-1]
    gap = (frame[1:] - frame[:-1])[same].astype(np.float64)
    owner = unit[1:][same]
    n_units = spikes.shape[1]
    count = np.bincount(owner, minlength=n_units)
    keep = count >= 2
    if not keep.any():
        return None
    s1 = np.bincount(owner, weights=gap, minlength=n_units)[keep]
    s2 = np.bincount(owner, weights=gap * gap, minlength=n_units)[keep]
    n = count[keep]
    mean = s1 / n
    var = np.maximum(s2 / n - mean * mean, 0.0)
    return float((np.sqrt(var) / mean).mean())


def response_stats(
    readout: str,
    data: np.ndarray,
    attrs: Dict[str, Any],
    active: np.ndarray,
    dt: float,
) -> Dict[str, Any]:
    """``core.response.*`` of one population.

    Args:
        readout: ``"spikes"``, ``"events"`` or ``"state"``.
        data: The readout array ``[T, N]`` as stored.
        attrs: The readout's allow-listed attributes (unused today; kept for packs that
            read ``polarity``).
        active: Boolean ``[T]``: the stimulus is active in any channel.
        dt: Frame step in the time unit.

    Returns:
        ``{name: value}`` for the metrics that apply to the readout, ``None`` where
        undefined; ``n_units`` and ``digest`` always.
    """
    n_t, n_units = data.shape
    out: Dict[str, Any] = {"n_units": n_units}
    if readout == "spikes":
        per_frame = data.sum(axis=1, dtype=np.int64)
        total = int(per_frame.sum())
        quiet_frames = n_t - int(active.sum())
        out["count"] = total
        out["rate_mean"] = total / (n_units * n_t * dt)
        out["rate_active"] = _rate(
            int(per_frame[active].sum()), n_units, active.sum(), dt
        )
        out["rate_quiet"] = _rate(
            int(per_frame[~active].sum()), n_units, quiet_frames, dt
        )
        out["isi_cv"] = isi_cv(data)
    if readout in ("spikes", "events"):
        fired = (data != 0).any(axis=0)
        out["active_unit_fraction"] = float(fired.mean())
        out["latency"] = _latency(data != 0, active, dt)
    if readout == "events":
        signed = data.astype(np.int64)
        out["events_on"] = int(signed[signed > 0].sum())
        out["events_off"] = int(-signed[signed < 0].sum())
    if readout == "state":
        out["state_mean"] = float(data.mean(dtype=np.float64))
        out["state_std"] = float(data.std(dtype=np.float64))
        out["state_min"] = float(data.min())
        out["state_max"] = float(data.max())
    out["digest"] = hashlib.sha256(
        np.ascontiguousarray(data, dtype=data.dtype.newbyteorder("<")).tobytes()
    ).hexdigest()
    return out


def _rate(count: int, n_units: int, frames: Any, dt: float) -> Optional[float]:
    """Spikes in a window over ``n_units`` times its time; ``None`` for an empty one."""
    if frames == 0:
        return None
    return count / (n_units * float(frames) * dt)


def _latency(fired: np.ndarray, active: np.ndarray, dt: float) -> Optional[float]:
    """First spike or event at or after the onset, minus the onset.

    Args:
        fired: Boolean ``[T, N]``, true where a unit spiked or emitted an event.
        active: Boolean ``[T]``, the stimulus's active frames in any channel.
        dt: Frame step in the time unit.

    Returns:
        Latency in the time unit; ``None`` with no onset or no later spike.
    """
    if not active.any():
        return None
    onset = int(np.argmax(active))
    later = np.flatnonzero(fired[onset:].any(axis=1))
    return None if later.size == 0 else float(later[0]) * dt
