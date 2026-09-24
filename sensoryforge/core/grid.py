"""Grid construction utilities for tactile encoding experiments."""

from __future__ import annotations

from typing import Tuple, Literal, Optional, Dict, Any

import torch

from .grid_base import BaseGrid


def _seeded_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Return a fresh CPU ``torch.Generator`` seeded with ``seed`` (F-050).

    Same pattern as :func:`sensoryforge.core.innervation._seeded_generator`:
    a per-instance generator instead of ``torch.manual_seed``, so building a
    grid never reads or advances the global RNG. ``None`` returns ``None``
    and callers fall back to the global generator (unseeded, as before).
    """
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _randn_seeded(
    like: torch.Tensor, generator: Optional[torch.Generator]
) -> torch.Tensor:
    """``randn_like(like)`` drawn from ``generator`` on CPU, moved to its device."""
    if generator is None:
        return torch.randn_like(like)
    return torch.randn(like.shape, generator=generator, dtype=like.dtype).to(
        like.device
    )


def _rand_seeded(
    like: torch.Tensor, generator: Optional[torch.Generator]
) -> torch.Tensor:
    """``rand_like(like)`` drawn from ``generator`` on CPU, moved to its device."""
    if generator is None:
        return torch.rand_like(like)
    return torch.rand(like.shape, generator=generator, dtype=like.dtype).to(like.device)


# Type alias for arrangement types
ArrangementType = Literal["grid", "poisson", "hex", "jittered_grid", "blue_noise"]


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
    # Guard against degenerate grids (n_x==1 or n_y==1): fall back to spacing
    # derived from the other axis, or return 0.0 if truly a 1×1 grid.
    if xx.shape[0] > 1:
        dx = xx[1, 0] - xx[0, 0]  # x varies along dim-0 (ij indexing)
    else:
        dx = torch.tensor(0.0, dtype=xx.dtype, device=xx.device)
    if yy.shape[1] > 1:
        dy = yy[0, 1] - yy[0, 0]  # y varies along dim-1 (ij indexing)
    else:
        dy = torch.tensor(0.0, dtype=yy.dtype, device=yy.device)
    return dx, dy


class ReceptorGrid(BaseGrid):
    """Manage receptor grid creation with flexible spatial arrangements.

    This class creates spatial grids representing mechanoreceptor positions
    with support for multiple arrangement patterns: regular grid, Poisson-like
    random distribution, hexagonal packing, and jittered grid.

    Attributes:
        grid_size: Number of grid points along each axis (rows, cols).
        spacing: Distance between adjacent receptors in mm (for grid arrangement).
        center: Spatial center of the grid (x0, y0) in mm.
        arrangement: Spatial arrangement pattern.
        seed: Seed of the per-instance jitter generator (``None`` = global RNG).
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
        *,
        seed: Optional[int] = None,
    ) -> None:
        """Construct receptor grid with specified arrangement pattern.

        Args:
            grid_size: Number of points along each axis or (n_x, n_y) tuple.
                Used for 'grid' and 'jittered_grid' arrangements.
            spacing: Distance between receptors in mm (for grid arrangement).
            center: (x0, y0) coordinates of the grid midpoint in mm.
            arrangement: Spatial arrangement type: 'grid' (default), 'poisson',
                'hex', 'jittered_grid', or 'blue_noise'.
            density: Receptor density in receptors/mm², for 'poisson', 'hex'
                and 'blue_noise' (D-88b4b41). Sets the receptor count to
                ``density`` times the ``(rows - 1) * spacing`` by
                ``(cols - 1) * spacing`` extent that ``grid_size`` and
                ``spacing`` define; when ``None`` (default) the count is
                derived from ``grid_size`` instead, unchanged from before
                D-88b4b41. Setting ``density`` on ``'grid'`` or
                ``'jittered_grid'``, where ``spacing`` already fixes the
                receptor count, raises ``ValueError``.
            device: PyTorch device identifier for tensors.
            seed: Seed for the random jitter of the ``jittered_grid``,
                ``blue_noise`` and ``poisson`` arrangements (F-050). Drawn from
                a per-instance CPU generator, so the same seed reproduces the
                same coordinates on every device and building the grid leaves
                the global RNG untouched. ``None`` (default) draws from the
                global RNG, as before. Ignored by ``grid`` and ``hex``.

        Raises:
            ValueError: If ``density`` is not ``None`` and not positive, or
                if ``density`` is set on the ``'grid'`` or ``'jittered_grid'``
                arrangement (D-88b4b41).
        """
        if isinstance(grid_size, tuple):
            self.grid_size = grid_size
        else:
            self.grid_size = (grid_size, grid_size)
        self.spacing = spacing
        self.center = center
        self.arrangement = arrangement
        self.device = torch.device(device) if isinstance(device, str) else device
        # Store the raw constructor argument (may be None) for round-trip
        # fidelity (F-049) -- distinct from the local `density` variable
        # below, which non-grid arrangements overwrite with a *derived*
        # value for coordinate generation only.
        self.density = density
        self.seed = seed
        self._generator = _seeded_generator(seed)

        if density is not None:
            if density <= 0:
                raise ValueError(f"density must be > 0 (receptors/mm²), got {density}")
            if arrangement in ("grid", "jittered_grid"):
                raise ValueError(
                    f"density is not supported for arrangement={arrangement!r}: "
                    "spacing already fixes the receptor count for 'grid' and "
                    "'jittered_grid'. Set density only for 'poisson', 'hex', "
                    "or 'blue_noise' (D-88b4b41)."
                )

        # For non-grid arrangements, we need density or defer to explicit sizing
        if arrangement in ["grid", "jittered_grid", "blue_noise"]:
            # Create coordinate grids using traditional method
            self.xx, self.yy, self.x, self.y = create_grid_torch(
                grid_size, spacing, center, self.device
            )

            # Calculate grid properties — store as float for type consistency
            dx_t, dy_t = get_grid_spacing(self.xx, self.yy)
            self.dx: float = dx_t.item()
            self.dy: float = dy_t.item()
            xlim = (self.x[0].item(), self.x[-1].item())
            ylim = (self.y[0].item(), self.y[-1].item())

            # Initialize base class with computed bounds
            super().__init__(xlim, ylim, device)

            if arrangement == "jittered_grid":
                # Apply jitter to the grid coordinates
                base_coords = torch.stack([self.xx.flatten(), self.yy.flatten()], dim=1)
                approximate_spacing = self.spacing
                jitter_magnitude = 0.25 * approximate_spacing
                jitter = _randn_seeded(base_coords, self._generator) * jitter_magnitude
                jittered = base_coords + jitter
                jittered[:, 0] = torch.clamp(jittered[:, 0], self.xlim[0], self.xlim[1])
                jittered[:, 1] = torch.clamp(jittered[:, 1], self.ylim[0], self.ylim[1])
                self.coordinates = jittered
                self.xx = None
                self.yy = None
                self.x = None
                self.y = None
            elif arrangement == "blue_noise":
                # Blue noise: jittered grid + Lloyd-like relaxation.
                # With density set (D-88b4b41), the base grid is sized from
                # density x extent instead of grid_size x grid_size, using
                # the same isotropic meshgrid as _generate_poisson
                # (spacing = 1/sqrt(density)); the jitter/relaxation below
                # then run on that base grid unchanged. With density unset,
                # this is bit-identical to before D-88b4b41.
                if density is not None:
                    base_coords = self._density_grid_points(density)
                    base_spacing = 1.0 / max(density, 1e-8) ** 0.5
                else:
                    base_coords = torch.stack(
                        [self.xx.flatten(), self.yy.flatten()], dim=1
                    )
                    base_spacing = self.spacing
                jitter_magnitude = 0.4 * base_spacing
                jitter = (
                    (_rand_seeded(base_coords, self._generator) - 0.5)
                    * 2
                    * jitter_magnitude
                )
                points = base_coords + jitter
                for _ in range(3):
                    dists = torch.cdist(points, points)
                    k = min(6, points.shape[0] - 1)
                    _, nearest_idx = torch.topk(dists, k + 1, largest=False, dim=1)
                    for i in range(points.shape[0]):
                        neighbors = points[nearest_idx[i, 1:]]
                        centroid = neighbors.mean(dim=0)
                        points[i] = 0.7 * points[i] + 0.3 * centroid
                points[:, 0] = torch.clamp(points[:, 0], self.xlim[0], self.xlim[1])
                points[:, 1] = torch.clamp(points[:, 1], self.ylim[0], self.ylim[1])
                self.coordinates = points
                self.xx = None
                self.yy = None
                self.x = None
                self.y = None
            else:
                # Regular grid - store flattened coordinates for consistency
                self.coordinates = torch.stack(
                    [self.xx.flatten(), self.yy.flatten()], dim=1
                )

        elif arrangement in ["poisson", "hex"]:
            # Compute bounds from grid_size and spacing (same as regular grid)
            n_x, n_y = self.grid_size
            total_x = (n_x - 1) * spacing
            total_y = (n_y - 1) * spacing
            x0, y0 = center

            xlim = (x0 - total_x / 2, x0 + total_x / 2)
            ylim = (y0 - total_y / 2, y0 + total_y / 2)

            # Initialize base class with computed bounds
            super().__init__(xlim, ylim, device)

            # density set (D-88b4b41) drives the receptor count directly;
            # unset, derive it from rows×cols and extent as before
            # (receptors/mm²) -- bit-identical to pre-D-88b4b41 behaviour.
            if density is not None:
                effective_density = density
            else:
                area = total_x * total_y
                expected_count = n_x * n_y
                effective_density = (expected_count / area) if area > 0 else 100.0

            # Generate coordinates using arrangement-specific methods
            if arrangement == "poisson":
                self.coordinates = self._generate_poisson(effective_density)
            else:  # hex
                self.coordinates = self._generate_hex(effective_density)

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
        if hasattr(self, "coordinates"):
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

    def get_all_coordinates(self) -> torch.Tensor:
        """Get all receptor coordinates (required by BaseGrid).

        Returns:
            Tensor [N_receptors, 2] with (x, y) positions in mm.
        """
        return self.get_receptor_coordinates()

    _CONSTRUCTOR_KEYS = frozenset(
        {"grid_size", "spacing", "center", "arrangement", "density", "device", "seed"}
    )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ReceptorGrid":
        """Create ReceptorGrid from config dict.

        Accepts a constructor-shaped dict or the output of :meth:`to_dict`
        (whose ``type``/``xlim``/``ylim`` keys are derived, not constructor
        arguments, and are dropped).

        Args:
            config: Dictionary with grid_size, spacing, center, arrangement,
                density, device, seed.

        Returns:
            ReceptorGrid instance.
        """
        config = {k: v for k, v in config.items() if k in cls._CONSTRUCTOR_KEYS}
        if isinstance(config.get("grid_size"), list):
            config["grid_size"] = tuple(config["grid_size"])
        if isinstance(config.get("center"), list):
            config["center"] = tuple(config["center"])
        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize grid parameters to dict.

        Returns:
            Dictionary with grid configuration.
        """
        result = super().to_dict()
        result.update(
            {
                "grid_size": self.grid_size,
                "spacing": self.spacing,
                "center": list(self.center),
                "arrangement": self.arrangement,
                "density": self.density,
                "seed": self.seed,
            }
        )
        return result

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
            "seed": self.seed,
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

    def _density_grid_points(self, density: float) -> torch.Tensor:
        """Regular meshgrid of points at ``density`` receptors/mm² over
        ``xlim``/``ylim``.

        Isotropic square meshgrid at spacing ``1/sqrt(density)``, unjittered.
        Shared base-point generator for the density-driven arrangements
        (D-88b4b41): ``_generate_poisson`` jitters this once, ``blue_noise``
        (when ``density`` is set) jitters it and then relaxes it.

        Args:
            density: Target receptor density in receptors per mm².

        Returns:
            Tensor of unjittered ``(x, y)`` coordinates in mm, shape ``[K, 2]``.
        """
        width = self.xlim[1] - self.xlim[0]
        height = self.ylim[1] - self.ylim[0]
        spacing = 1.0 / max(density, 1e-8) ** 0.5

        n_x = max(1, int(width / spacing) + 1)
        n_y = max(1, int(height / spacing) + 1)

        x = torch.linspace(self.xlim[0], self.xlim[1], n_x, device=self.device)
        y = torch.linspace(self.ylim[0], self.ylim[1], n_y, device=self.device)

        xx, yy = torch.meshgrid(x, y, indexing="ij")
        return torch.stack([xx.flatten(), yy.flatten()], dim=1)

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
        coordinates = self._density_grid_points(density)
        spacing = 1.0 / max(density, 1e-8) ** 0.5

        jitter_scale = 0.5 * spacing
        jitter = (_rand_seeded(coordinates, self._generator) - 0.5) * jitter_scale
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
        spacing = (2.0 / (3.0**0.5 * density)) ** 0.5

        # Determine grid dimensions
        width = self.xlim[1] - self.xlim[0]
        height = self.ylim[1] - self.ylim[0]

        n_x = max(1, int(width / spacing) + 1)
        row_spacing = spacing * 3.0**0.5 / 2.0
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


def load_receptor_coords_file(
    path: str, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Load an ``[M, 2]`` receptor coordinate tensor in mm from a file.

    Accepts a CSV (two columns, ``x,y``, optional header row) or a ``.pt``
    file holding an ``[M, 2]`` tensor. Used by ``GridConfig.coords_file``
    (Phase 2, Wave L1) and by composite-grid layer specs (Wave L4) that give
    ``coords_file`` instead of ``coordinates``/``density``.

    Args:
        path: Path to the ``.csv`` or ``.pt`` file.
        device: Target device for the returned tensor.

    Returns:
        ``[M, 2]`` float32 tensor of ``(x, y)`` coordinates in mm.

    Raises:
        ValueError: If the file extension is unsupported or the parsed data
            is not ``[M, 2]``.
    """
    from pathlib import Path as _Path

    p = _Path(path)
    if p.suffix.lower() == ".pt":
        coords = torch.as_tensor(torch.load(p, map_location="cpu", weights_only=False))
    elif p.suffix.lower() == ".csv":
        import csv as _csv

        rows: list = []
        with open(p, newline="") as f:
            reader = _csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    rows.append([float(row[0]), float(row[1])])
                except ValueError:
                    # Header row (non-numeric first cell) -- skip.
                    continue
        coords = torch.tensor(rows, dtype=torch.float32)
    else:
        raise ValueError(
            f"Unsupported receptor coordinate file extension {p.suffix!r} "
            f"for {path!r}; use .csv or .pt"
        )
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"{path}: expected an [M, 2] (x, y) coordinate tensor, got shape "
            f"{list(coords.shape)}"
        )
    return coords.to(device=device, dtype=torch.float32)


# Backward compatibility alias
GridManager = ReceptorGrid
