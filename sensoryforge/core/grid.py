"""Grid construction utilities for tactile encoding experiments."""

from __future__ import annotations

from typing import Tuple, Literal, Optional

import torch

# Type alias for arrangement types
ArrangementType = Literal["grid", "poisson", "hex", "jittered_grid"]


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


class ReceptorGrid:
    """Manage receptor grid creation with flexible spatial arrangements.
    
    This class creates spatial grids representing mechanoreceptor positions
    with support for multiple arrangement patterns: regular grid, Poisson-like
    random distribution, hexagonal packing, and jittered grid.
    
    Attributes:
        grid_size: Number of grid points along each axis (rows, cols).
        spacing: Distance between adjacent receptors in mm (for grid arrangement).
        center: Spatial center of the grid (x0, y0) in mm.
        arrangement: Spatial arrangement pattern.
        device: PyTorch device for tensor storage.
        xlim: Spatial bounds along x-axis (min, max).
        ylim: Spatial bounds along y-axis (min, max).
        coordinates: Receptor positions as [N, 2] tensor for non-grid arrangements.
    """

    def __init__(
        self,
        grid_size: int | Tuple[int, int] = 80,
        spacing: float = 0.15,
        center: Tuple[float, float] = (0.0, 0.0),
        arrangement: ArrangementType = "grid",
        density: Optional[float] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Construct receptor grid with specified arrangement pattern.
        
        Args:
            grid_size: Number of points along each axis or (n_x, n_y) tuple.
                Used for 'grid' and 'jittered_grid' arrangements.
            spacing: Distance between receptors in mm (for grid arrangement).
            center: (x0, y0) coordinates of the grid midpoint in mm.
            arrangement: Spatial arrangement type: 'grid' (default), 'poisson',
                'hex', or 'jittered_grid'.
            density: Receptor density in receptors/mm². For 'poisson' and 'hex',
                derived from grid_size and spacing when None. Ignored for grid-based
                arrangements.
            device: PyTorch device identifier for tensors.
        
        """
        if isinstance(grid_size, tuple):
            self.grid_size = grid_size
        else:
            self.grid_size = (grid_size, grid_size)
        self.spacing = spacing
        self.center = center
        self.arrangement = arrangement
        self.device = torch.device(device) if isinstance(device, str) else device

        # For non-grid arrangements, we need density or defer to explicit sizing
        if arrangement in ["grid", "jittered_grid"]:
            # Create coordinate grids using traditional method
            self.xx, self.yy, self.x, self.y = create_grid_torch(
                grid_size, spacing, center, self.device
            )
            
            # Calculate grid properties — store as float for type consistency
            dx_t, dy_t = get_grid_spacing(self.xx, self.yy)
            self.dx: float = dx_t.item()
            self.dy: float = dy_t.item()
            self.xlim = (self.x[0].item(), self.x[-1].item())
            self.ylim = (self.y[0].item(), self.y[-1].item())
            
            if arrangement == "jittered_grid":
                # Apply jitter to the grid coordinates
                base_coords = torch.stack([self.xx.flatten(), self.yy.flatten()], dim=1)
                approximate_spacing = self.spacing
                jitter_magnitude = 0.25 * approximate_spacing
                jitter = torch.randn_like(base_coords) * jitter_magnitude
                jittered = base_coords + jitter
                
                # Clamp to bounds
                jittered[:, 0] = torch.clamp(jittered[:, 0], self.xlim[0], self.xlim[1])
                jittered[:, 1] = torch.clamp(jittered[:, 1], self.ylim[0], self.ylim[1])
                
                self.coordinates = jittered
            else:
                # Regular grid - store flattened coordinates for consistency
                self.coordinates = torch.stack([self.xx.flatten(), self.yy.flatten()], dim=1)
                
        elif arrangement in ["poisson", "hex"]:
            # Compute bounds from grid_size and spacing (same as regular grid)
            n_x, n_y = self.grid_size
            total_x = (n_x - 1) * spacing
            total_y = (n_y - 1) * spacing
            x0, y0 = center

            self.xlim = (x0 - total_x / 2, x0 + total_x / 2)
            self.ylim = (y0 - total_y / 2, y0 + total_y / 2)

            # Derive density from rows×cols and extent (receptors/mm²)
            area = total_x * total_y
            expected_count = n_x * n_y
            density = (expected_count / area) if area > 0 else 100.0

            # Generate coordinates using arrangement-specific methods
            if arrangement == "poisson":
                self.coordinates = self._generate_poisson(density)
            else:  # hex
                self.coordinates = self._generate_hex(density)
            
            # For non-grid arrangements, meshgrids are not defined
            self.xx = None
            self.yy = None
            self.x = None
            self.y = None
            self.dx = spacing
            self.dy = spacing
        else:
            raise ValueError(f"Unknown arrangement type: {arrangement}")

    def to_device(self, device: torch.device | str) -> "ReceptorGrid":
        """Move grid tensors to device and return self for chaining.
        
        Args:
            device: Target PyTorch device.
        
        Returns:
            Self reference for method chaining.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        
        if self.xx is not None:
            self.xx = self.xx.to(self.device)
        if self.yy is not None:
            self.yy = self.yy.to(self.device)
        if self.x is not None:
            self.x = self.x.to(self.device)
        if self.y is not None:
            self.y = self.y.to(self.device)
        if hasattr(self, 'coordinates'):
            self.coordinates = self.coordinates.to(self.device)
            
        return self

    def get_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 2D coordinate meshgrids (xx, yy).
        
        Returns:
            Tuple of (xx, yy) meshgrids for grid-based arrangements.
        
        Raises:
            ValueError: If arrangement does not support meshgrids.
        """
        if self.xx is None or self.yy is None:
            raise ValueError(
                f"Meshgrids not available for '{self.arrangement}' arrangement. "
                "Use get_receptor_coordinates() instead."
            )
        return self.xx, self.yy

    def get_1d_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 1D coordinate vectors (x, y).
        
        Returns:
            Tuple of (x, y) 1D vectors for grid-based arrangements.
        
        Raises:
            ValueError: If arrangement does not support 1D vectors.
        """
        if self.x is None or self.y is None:
            raise ValueError(
                f"1D coordinate vectors not available for '{self.arrangement}' "
                "arrangement. Use get_receptor_coordinates() instead."
            )
        return self.x, self.y
    
    def get_receptor_coordinates(self) -> torch.Tensor:
        """Return receptor positions as [N, 2] tensor.
        
        Returns:
            Tensor of shape (num_receptors, 2) with (x, y) coordinates in mm.
        """
        return self.coordinates

    def get_grid_properties(self) -> dict:
        """Return grid metadata consumed by downstream modules.
        
        Returns:
            Dictionary with grid configuration and bounds.
        """
        return {
            "grid_size": self.grid_size,
            "spacing": self.spacing,
            "center": self.center,
            "arrangement": self.arrangement,
            "xlim": self.xlim,
            "ylim": self.ylim,
            "dx": self.dx,
            "dy": self.dy,
            "device": self.device,
        }
    
    def _compute_area(self) -> float:
        """Calculate total spatial area.
        
        Returns:
            Area in mm².
        """
        width = self.xlim[1] - self.xlim[0]
        height = self.ylim[1] - self.ylim[0]
        return width * height
    
    def _generate_poisson(self, density: float) -> torch.Tensor:
        """Generate approximate Poisson-distributed points via jittered grid.
        
        Creates a random point distribution by starting from a regular grid
        at the target density and applying uniform jitter. This is **not**
        true Poisson-disk sampling but a computationally efficient approximation.
        
        Args:
            density: Target receptor density in receptors per mm².
        
        Returns:
            Tensor of approximately density × area points with shape [N, 2].
        """
        width = self.xlim[1] - self.xlim[0]
        height = self.ylim[1] - self.ylim[0]
        spacing = 1.0 / max(density, 1e-8) ** 0.5

        n_x = max(1, int(width / spacing) + 1)
        n_y = max(1, int(height / spacing) + 1)

        x = torch.linspace(self.xlim[0], self.xlim[1], n_x, device=self.device)
        y = torch.linspace(self.ylim[0], self.ylim[1], n_y, device=self.device)

        xx, yy = torch.meshgrid(x, y, indexing="ij")
        coordinates = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        jitter_scale = 0.5 * spacing
        jitter = (torch.rand_like(coordinates) - 0.5) * jitter_scale
        coordinates = coordinates + jitter

        coordinates[:, 0] = torch.clamp(coordinates[:, 0], self.xlim[0], self.xlim[1])
        coordinates[:, 1] = torch.clamp(coordinates[:, 1], self.ylim[0], self.ylim[1])

        return coordinates
    
    def _generate_hex(self, density: float) -> torch.Tensor:
        """Generate hexagonal lattice arrangement.
        
        Creates optimal packing pattern with hexagonal symmetry.
        
        Args:
            density: Target receptor density in receptors per mm².
        
        Returns:
            Tensor of hexagonally arranged points.
        """
        # Hexagonal packing spacing formula
        spacing = (2.0 / (3.0 ** 0.5 * density)) ** 0.5
        
        # Determine grid dimensions
        width = self.xlim[1] - self.xlim[0]
        height = self.ylim[1] - self.ylim[0]

        n_x = max(1, int(width / spacing) + 1)
        row_spacing = spacing * 3.0 ** 0.5 / 2.0
        n_y = max(1, int(height / row_spacing) + 1)

        x = torch.linspace(self.xlim[0], self.xlim[1], n_x, device=self.device)
        y = torch.linspace(self.ylim[0], self.ylim[1], n_y, device=self.device)

        xx, yy = torch.meshgrid(x, y, indexing="ij")

        row_offsets = (torch.arange(n_y, device=self.device) % 2) * (spacing / 2.0)
        xx = xx + row_offsets.unsqueeze(0)

        coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        mask = (
            (coords[:, 0] >= self.xlim[0])
            & (coords[:, 0] <= self.xlim[1])
            & (coords[:, 1] >= self.ylim[0])
            & (coords[:, 1] <= self.ylim[1])
        )

        return coords[mask]


# Backward compatibility alias
GridManager = ReceptorGrid
