"""Grid construction utilities for tactile encoding experiments."""

from __future__ import annotations

from typing import Tuple

import torch


def create_grid_torch(
    grid_size: int | Tuple[int, int] = 80,
    spacing: float = 0.15,
    center: Tuple[float, float] = (0.0, 0.0),
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a 2D mechanoreceptor lattice as PyTorch tensors.

    Args:
        grid_size: Number of points along each axis or ``(n_x, n_y)`` tuple.
        spacing: Distance between mechanoreceptors in millimetres.
        center: ``(x0, y0)`` coordinates of the grid midpoint.
        device: Torch device identifier for the returned tensors.

    Returns:
        Tuple containing ``xx``, ``yy`` meshgrids plus ``x``/``y`` 1D vectors.
    """
    if isinstance(grid_size, int):
        n_x = n_y = grid_size
    else:
        n_x, n_y = grid_size

    total_x = (n_x - 1) * spacing
    total_y = (n_y - 1) * spacing
    x0, y0 = center

    x = torch.linspace(x0 - total_x / 2, x0 + total_x / 2, n_x, device=device)
    y = torch.linspace(y0 - total_y / 2, y0 + total_y / 2, n_y, device=device)

    xx, yy = torch.meshgrid(x, y, indexing="ij")

    return xx, yy, x, y


def get_grid_spacing(
    xx: torch.Tensor,
    yy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate grid spacing from coordinate meshgrids (ij indexing).

    With ``indexing='ij'``, ``xx`` varies along dim-0 and ``yy`` varies
    along dim-1.  The previous implementation sampled the wrong axes,
    returning ``(0, 0)`` (resolves ReviewFinding#H3).

    Args:
        xx: X-coordinate meshgrid ``[n_x, n_y]``.
        yy: Y-coordinate meshgrid ``[n_x, n_y]``.

    Returns:
        Tuple ``(dx, dy)`` with physical spacing along each axis.
    """
    dx = xx[1, 0] - xx[0, 0]  # x varies along dim-0 (ij indexing)
    dy = yy[0, 1] - yy[0, 0]  # y varies along dim-1 (ij indexing)
    return dx, dy


class GridManager:
    """Manage grid creation and provide coordinate accessors."""

    def __init__(
        self,
        grid_size: int | Tuple[int, int] = 80,
        spacing: float = 0.15,
        center: Tuple[float, float] = (0.0, 0.0),
        device: torch.device | str = "cpu",
    ) -> None:
        """Construct the mechanoreceptor grid and cache properties."""
        if isinstance(grid_size, tuple):
            self.grid_size = grid_size
        else:
            self.grid_size = (grid_size, grid_size)
        self.spacing = spacing
        self.center = center
        self.device = device

        # Create coordinate grids
        self.xx, self.yy, self.x, self.y = create_grid_torch(
            grid_size, spacing, center, device
        )

        # Calculate grid properties
        self.dx, self.dy = get_grid_spacing(self.xx, self.yy)
        self.xlim = (self.x[0].item(), self.x[-1].item())
        self.ylim = (self.y[0].item(), self.y[-1].item())

    def to_device(self, device: torch.device | str) -> "GridManager":
        """Move grid tensors to ``device`` and return ``self`` for chaining."""
        self.device = device
        self.xx = self.xx.to(device)
        self.yy = self.yy.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self

    def get_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 2D coordinate meshgrids ``(xx, yy)``."""
        return self.xx, self.yy

    def get_1d_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 1D coordinate vectors ``(x, y)``."""
        return self.x, self.y

    def get_grid_properties(self) -> dict:
        """Return grid metadata consumed by downstream modules."""
        return {
            "grid_size": self.grid_size,
            "spacing": self.spacing,
            "center": self.center,
            "xlim": self.xlim,
            "ylim": self.ylim,
            "dx": self.dx,
            "dy": self.dy,
            "device": self.device,
        }
