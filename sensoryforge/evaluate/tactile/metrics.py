"""The ``tactile`` metric pack (report contract, section 7): episode phases and speed.

A touch episode rises to a peak, holds (and perhaps slides) and releases. This pack
reads those phases back from the frames, on each bout's envelope
``e(k) = max over sensors of |x[k, m]|``, so they can be set beside what the world
declared (touch, hold plus slide, release). The module is inside the audit's import
wall: it sees frames and nothing else.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from sensoryforge.evaluate.audit import frame_metrics
from sensoryforge.evaluate.audit.summary import finite_or_none
from sensoryforge.evaluate.audit.units import unit_of
from sensoryforge.evaluate.pack import MetricPack, metric_definition

#: ``(name, unit expression, is a time measured on the frame grid)``.
_METRICS: Tuple[Tuple[str, str, bool], ...] = (
    ("episode.rise_time", "time", True),
    ("episode.plateau_time", "time", True),
    ("episode.fall_time", "time", True),
    ("motion.speed", "space/time", False),
)

_UNITS_SOURCE = "docs/user_guide/units_and_gains.md"


def measure_channel(
    x: np.ndarray,
    coords: np.ndarray,
    dt: float,
    plateau_tolerance: float,
    motion_tolerance: float,
) -> Dict[str, Optional[float]]:
    """The episode phases and plateau speed of one channel.

    A bout covers frames ``[a, b)`` and peaks at ``p = max e``. Its plateau is the
    frames with ``e >= (1 - plateau_tolerance) * p``, from the first (``p0``) to the
    last (``p1``) of them. Rise, plateau and fall are ``(p0 - a)``,
    ``(p1 - p0 + 1)`` and ``(b - 1 - p1)`` frames, in time, averaged over the bouts.
    Speed is the mean length of the centroid steps inside the plateaus, divided by
    ``dt``, over the plateaus whose path length exceeds ``motion_tolerance``.

    Args:
        x: One channel's frames ``[T, M]``.
        coords: Sensor coordinates ``[M, D]`` in the space unit.
        dt: Frame step in the time unit.
        plateau_tolerance: Share of a bout's peak a frame may fall short of and stay on
            the plateau.
        motion_tolerance: Least path length (space unit) a plateau needs to count as
            moving.

    Returns:
        ``{"rise_time", "plateau_time", "fall_time", "speed"}`` (time unit; space/time),
        ``None`` where undefined (no bout; no moving plateau).
    """
    out: Dict[str, Optional[float]] = {
        "rise_time": None,
        "plateau_time": None,
        "fall_time": None,
        "speed": None,
    }
    env = np.abs(x).max(axis=1).astype(np.float64)
    bouts = frame_metrics.runs(env > 0.0)
    if bouts.shape[0] == 0:
        return out
    rise, plateau, fall, steps = [], [], [], []
    for a, b in bouts:
        seg = env[a:b]
        floor = (1.0 - plateau_tolerance) * float(seg.max())
        held = np.flatnonzero(seg >= floor)
        p0, p1 = int(a + held[0]), int(a + held[-1])
        rise.append(p0 - a)
        plateau.append(p1 - p0 + 1)
        fall.append(b - 1 - p1)
        if p1 > p0:
            frames = np.arange(p0, p1 + 1)
            cen = frame_metrics.centroids(x, frames, coords)["centroid"]
            length = np.linalg.norm(np.diff(cen, axis=0), axis=1)
            if float(length.sum()) > motion_tolerance:
                steps.append(length)
    out["rise_time"] = float(np.mean(rise)) * dt
    out["plateau_time"] = float(np.mean(plateau)) * dt
    out["fall_time"] = float(np.mean(fall)) * dt
    if steps:
        pooled = np.concatenate(steps)
        out["speed"] = float(pooled.sum()) / (pooled.size * dt)
    return out


class TactilePack(MetricPack):
    """Episode phases and plateau speed, read from the frames.

    Args:
        plateau_tolerance: A frame belongs to a bout's plateau when its envelope is at
            least ``(1 - plateau_tolerance)`` of the bout's peak (default 0.01).
        motion_tolerance: A plateau counts as moving when the path of its centroid is
            longer than this (space unit; default 0, any movement at all).

    Raises:
        ValueError: If ``plateau_tolerance`` is outside [0, 1] or ``motion_tolerance``
            is negative.
    """

    name = "tactile"
    version = "1"

    def __init__(
        self, plateau_tolerance: float = 0.01, motion_tolerance: float = 0.0
    ) -> None:
        if not 0.0 <= plateau_tolerance <= 1.0:
            raise ValueError(
                f"plateau_tolerance must be in [0, 1], got {plateau_tolerance}"
            )
        if motion_tolerance < 0:
            raise ValueError(f"motion_tolerance must be >= 0, got {motion_tolerance}")
        self.plateau_tolerance = float(plateau_tolerance)
        self.motion_tolerance = float(motion_tolerance)

    def to_dict(self) -> Dict[str, Any]:
        """The pack's parameters (``plateau_tolerance``, ``motion_tolerance``)."""
        return {
            "plateau_tolerance": self.plateau_tolerance,
            "motion_tolerance": self.motion_tolerance,
        }

    @classmethod
    def get_param_spec(cls) -> List[Any]:
        """The pack's parameters as ``ParamSpec`` objects."""
        from sensoryforge.stimuli.base import ParamSpec

        return [
            ParamSpec(
                "plateau_tolerance",
                dtype="float",
                default=0.01,
                min_val=0.0,
                max_val=1.0,
                help="A frame is on a bout's plateau when its peak-sensor value is at "
                "least this share below the bout's peak or closer.",
            ),
            ParamSpec(
                "motion_tolerance",
                dtype="float",
                default=0.0,
                min_val=0.0,
                unit="space unit",
                help="A plateau counts as moving when its centroid path is longer.",
            ),
        ]

    def units(self) -> Dict[str, Tuple[str, str]]:
        """``drive`` and ``filtered`` in mA (the units guide)."""
        return {
            "drive": ("mA", _UNITS_SOURCE),
            "filtered": ("mA", _UNITS_SOURCE),
        }

    def metrics(
        self, layout: Any, units: Mapping[str, Any], dt: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """The four definitions (contract, section 7).

        Args:
            layout: The shared layout (unused; the metrics exist once per channel).
            units: The resolved units by quantity.
            dt: The frame step, the resolution of the phase times; ``None`` when the
                bundles' steps differ.

        Returns:
            One definition per metric id, scope ``channel``.
        """
        return [
            metric_definition(
                f"tactile.{name}",
                unit_of(expr, units),
                scope="channel",
                resolution=dt if on_grid else None,
            )
            for name, expr, on_grid in _METRICS
        ]

    def measure(self, view: Any) -> List[Dict[str, Any]]:
        """Measure one entry.

        Args:
            view: The entry's :class:`EntryView`.

        Returns:
            Four values per channel, in catalogue order (the metric, then its channels).
        """
        n_ch = view.frames.shape[1]
        per_channel = [
            measure_channel(
                view.frames[:, c, :],
                view.layout.coords,
                view.dt,
                self.plateau_tolerance,
                self.motion_tolerance,
            )
            for c in range(n_ch)
        ]
        values: List[Dict[str, Any]] = []
        for name, _expr, _grid in _METRICS:
            short = name.split(".")[1]
            for c, stats in enumerate(per_channel):
                value: Dict[str, Any] = {
                    "id": f"tactile.{name}",
                    "value": finite_or_none(stats[short]),
                }
                if n_ch > 1:
                    value["channel"] = c
                values.append(value)
        return values
