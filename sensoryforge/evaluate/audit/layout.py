"""Sensor layout: where each sensor of a frame sits, in the data's space unit.

A world entry's frames are the stimulus rendered on grid 0's canvas, so the sensors are
that canvas's points. The coordinates are computed here from the bundle's
``config.json`` grid block with SensoryForge's centred layout (dim 0 is x,
``indexing="ij"``, sensor
``k = i * cols + j``) and without importing the world package; a test compares the
result with ``sensoryforge.world.render.Canvas.from_grid_config``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Layout:
    """The sensors of one frame and the channels it carries.

    Attributes:
        coords: Sensor coordinates ``[M, D]`` (float64) in the data's space unit; ``D``
            is 1, 2 or 3 and dimension 0 is x.
        channel_names: One name per channel, or ``None`` when the grid names none.
        n_channels: Number of channels ``C`` of a frame ``[C, M]``.
        regular_grid: ``{"rows", "cols", "spacing"}`` when the sensors form a regular
            lattice, else ``None``.
    """

    coords: np.ndarray
    channel_names: Optional[Tuple[str, ...]]
    n_channels: int
    regular_grid: Optional[Dict[str, Any]] = None

    @property
    def n_sensors(self) -> int:
        """Number of sensors ``M``."""
        return int(self.coords.shape[0])

    @property
    def dims(self) -> int:
        """Number of spatial dimensions ``D``."""
        return int(self.coords.shape[1])

    @property
    def bounds(self) -> List[List[float]]:
        """``[lo, hi]`` of the coordinates in each dimension, in the space unit."""
        return [
            [float(self.coords[:, d].min()), float(self.coords[:, d].max())]
            for d in range(self.dims)
        ]

    def same_as(self, other: "Layout") -> bool:
        """True when ``other`` has the same sensors, channels and names, bit for bit."""
        return (
            self.n_channels == other.n_channels
            and self.channel_names == other.channel_names
            and self.regular_grid == other.regular_grid
            and self.coords.shape == other.coords.shape
            and bool(np.array_equal(self.coords, other.coords))
        )

    def summary(self) -> Dict[str, Any]:
        """The ``layout`` object of ``audit.json`` (report contract, section 5)."""
        return {
            "n_sensors": self.n_sensors,
            "dims": self.dims,
            "n_channels": self.n_channels,
            "channel_names": (
                None if self.channel_names is None else list(self.channel_names)
            ),
            "bounds": self.bounds,
            "regular_grid": (
                None if self.regular_grid is None else dict(self.regular_grid)
            ),
        }


def centered_lattice(
    rows: int,
    cols: int,
    spacing: float,
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Coordinates ``[rows * cols, 2]`` of a centred lattice (``k = i * cols + j``).

    Args:
        rows: Points along dimension 0 (x).
        cols: Points along dimension 1 (y).
        spacing: Distance between neighbouring points, in the space unit.
        center: ``(x, y)`` of the lattice centre, in the space unit.

    Returns:
        Float64 coordinates, x in column 0 and y in column 1.

    Raises:
        ValueError: If ``rows`` or ``cols`` is below 1 or ``spacing`` is not positive.
    """
    if rows < 1 or cols < 1 or not spacing > 0:
        raise ValueError(
            f"need rows, cols >= 1 and spacing > 0; got {rows}, {cols}, {spacing}"
        )
    cx, cy = float(center[0]), float(center[1])
    half_x = (rows - 1) * spacing / 2.0
    half_y = (cols - 1) * spacing / 2.0
    x = np.linspace(cx - half_x, cx + half_x, rows, dtype=np.float64)
    y = np.linspace(cy - half_y, cy + half_y, cols, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)


def regular_layout(
    rows: int,
    cols: int,
    spacing: float,
    center: Tuple[float, float],
    channel_names: Optional[Tuple[str, ...]],
    n_channels: int,
) -> Layout:
    """A :class:`Layout` for a centred ``rows x cols`` lattice.

    Args:
        rows: Points along dimension 0.
        cols: Points along dimension 1.
        spacing: Lattice spacing, in the space unit.
        center: ``(x, y)`` of the centre, in the space unit.
        channel_names: Channel names, or ``None``.
        n_channels: Number of channels.

    Returns:
        The layout, with ``regular_grid`` set.
    """
    return Layout(
        coords=centered_lattice(rows, cols, spacing, center),
        channel_names=channel_names,
        n_channels=int(n_channels),
        regular_grid={"rows": int(rows), "cols": int(cols), "spacing": float(spacing)},
    )
