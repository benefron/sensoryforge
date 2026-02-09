"""Composite grid for multi-population spatial substrates.

This module provides infrastructure for managing multiple named receptor
populations on a shared coordinate system. Each population can have its own
density, spatial arrangement, and optional metadata or filtering attributes.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple, Any
import torch


# Type alias for arrangement types
ArrangementType = Literal["grid", "poisson", "hex", "jittered_grid"]


class CompositeGrid:
    """Multi-population spatial substrate with shared coordinate system.
    
    The CompositeGrid manages multiple named receptor populations, each with
    configurable density and spatial arrangement patterns. All populations
    share a common coordinate system defined by spatial bounds and device.
    
    Arrangement types:
        - grid: Regular rectangular lattice with uniform spacing
        - poisson: Random point distribution following Poisson disk sampling
        - hex: Hexagonal lattice arrangement (optimal packing)
        - jittered_grid: Regular grid with random spatial jitter
    
    Attributes:
        xlim: Spatial bounds along x-axis (min, max) in mm.
        ylim: Spatial bounds along y-axis (min, max) in mm.
        device: PyTorch device for tensor operations.
        populations: Dictionary mapping population names to their configurations
            and generated coordinates.
    
    Example:
        >>> grid = CompositeGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
        >>> grid.add_population(
        ...     name="SA1",
        ...     density=100.0,  # receptors per mm²
        ...     arrangement="grid"
        ... )
        >>> grid.add_population(
        ...     name="RA",
        ...     density=50.0,
        ...     arrangement="hex"
        ... )
        >>> sa1_coords = grid.get_population_coordinates("SA1")
        >>> print(sa1_coords.shape)  # (num_receptors, 2)
    """
    
    def __init__(
        self,
        xlim: Tuple[float, float] = (-5.0, 5.0),
        ylim: Tuple[float, float] = (-5.0, 5.0),
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize composite grid with shared coordinate bounds.
        
        Args:
            xlim: Spatial extent along x-axis as (min, max) in mm.
            ylim: Spatial extent along y-axis as (min, max) in mm.
            device: PyTorch device identifier for tensor storage.
        
        Raises:
            ValueError: If xlim or ylim bounds are invalid (min >= max).
        """
        # Validate spatial bounds
        if xlim[0] >= xlim[1]:
            raise ValueError(
                f"Invalid xlim: min ({xlim[0]}) must be < max ({xlim[1]})"
            )
        if ylim[0] >= ylim[1]:
            raise ValueError(
                f"Invalid ylim: min ({ylim[0]}) must be < max ({ylim[1]})"
            )
        
        self.xlim = xlim
        self.ylim = ylim
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Storage for population data
        # Each entry: {config: {...}, coordinates: Tensor[N, 2]}
        self.populations: Dict[str, Dict[str, Any]] = {}
    
    def add_population(
        self,
        name: str,
        density: float,
        arrangement: ArrangementType = "grid",
        filter: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Add a named receptor population with specified arrangement.
        
        Generates receptor coordinates based on density and arrangement type,
        storing them for later retrieval. Populations are independent and can
        have different densities and arrangements on the same coordinate system.
        
        Args:
            name: Unique identifier for this population.
            density: Target receptor density in receptors per mm².
            arrangement: Spatial arrangement pattern ("grid", "poisson", "hex",
                or "jittered_grid").
            filter: Optional filter specification or metadata tag.
            **metadata: Additional key-value pairs for population-specific
                configuration (e.g., sigma, connection_params, etc.).
        
        Raises:
            ValueError: If population name already exists or density is invalid.
        
        Example:
            >>> grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
            >>> grid.add_population(
            ...     name="mechanoreceptors",
            ...     density=100.0,
            ...     arrangement="grid",
            ...     filter="gaussian",
            ...     sigma=0.5
            ... )
        """
        # Validate inputs
        if name in self.populations:
            raise ValueError(f"Population '{name}' already exists")
        if density <= 0:
            raise ValueError(f"Density must be positive, got {density}")
        
        # Calculate area and expected receptor count
        area = self._compute_area()
        expected_count = int(density * area)
        
        # Generate coordinates based on arrangement type
        coordinates = self._generate_coordinates(
            arrangement=arrangement,
            expected_count=expected_count,
            density=density,
        )
        
        # Store population configuration and coordinates
        self.populations[name] = {
            "config": {
                "density": density,
                "arrangement": arrangement,
                "filter": filter,
                "metadata": metadata,
            },
            "coordinates": coordinates,
            "count": coordinates.shape[0],
        }
    
    def get_population_coordinates(self, name: str) -> torch.Tensor:
        """Retrieve receptor coordinates for a named population.
        
        Args:
            name: Population identifier.
        
        Returns:
            Tensor of shape (num_receptors, 2) containing (x, y) coordinates
            in mm.
        
        Raises:
            KeyError: If population name does not exist.
        
        Example:
            >>> coords = grid.get_population_coordinates("SA1")
            >>> print(coords.shape)
            torch.Size([1000, 2])
        """
        if name not in self.populations:
            raise KeyError(f"Population '{name}' not found")
        return self.populations[name]["coordinates"]
    
    def get_population_config(self, name: str) -> Dict[str, Any]:
        """Retrieve configuration for a named population.
        
        Args:
            name: Population identifier.
        
        Returns:
            Dictionary containing density, arrangement, filter, and metadata.
        
        Raises:
            KeyError: If population name does not exist.
        """
        if name not in self.populations:
            raise KeyError(f"Population '{name}' not found")
        return self.populations[name]["config"]
    
    def get_population_count(self, name: str) -> int:
        """Get the number of receptors in a population.
        
        Args:
            name: Population identifier.
        
        Returns:
            Number of receptors generated for this population.
        
        Raises:
            KeyError: If population name does not exist.
        """
        if name not in self.populations:
            raise KeyError(f"Population '{name}' not found")
        return self.populations[name]["count"]
    
    def list_populations(self) -> list[str]:
        """Get names of all populations in this grid.
        
        Returns:
            List of population name strings.
        """
        return list(self.populations.keys())
    
    def to_device(self, device: torch.device | str) -> CompositeGrid:
        """Move all population coordinates to specified device.
        
        Args:
            device: Target PyTorch device (e.g., "cuda", "cpu").
        
        Returns:
            Self reference for method chaining.
        
        Example:
            >>> grid.to_device("cuda")
            >>> coords = grid.get_population_coordinates("SA1")
            >>> print(coords.device)
            cuda:0
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        for pop_data in self.populations.values():
            pop_data["coordinates"] = pop_data["coordinates"].to(self.device)
        return self
    
    def _compute_area(self) -> float:
        """Calculate total spatial area of the grid.
        
        Returns:
            Area in mm².
        """
        width = self.xlim[1] - self.xlim[0]
        height = self.ylim[1] - self.ylim[0]
        return width * height
    
    def _generate_coordinates(
        self,
        arrangement: ArrangementType,
        expected_count: int,
        density: float,
    ) -> torch.Tensor:
        """Generate receptor coordinates based on arrangement type.
        
        Args:
            arrangement: Type of spatial arrangement.
            expected_count: Target number of receptors.
            density: Receptor density in receptors per mm².
        
        Returns:
            Tensor of shape (N, 2) with (x, y) coordinates.
        
        Raises:
            ValueError: If arrangement type is not recognized.
        """
        if arrangement == "grid":
            return self._generate_grid(expected_count)
        elif arrangement == "poisson":
            return self._generate_poisson(density)
        elif arrangement == "hex":
            return self._generate_hex(density)
        elif arrangement == "jittered_grid":
            return self._generate_jittered_grid(expected_count)
        else:
            raise ValueError(f"Unknown arrangement type: {arrangement}")
    
    def _generate_grid(self, expected_count: int) -> torch.Tensor:
        """Generate regular rectangular grid of points.
        
        Creates a uniform lattice with approximately the expected number of
        points, maintaining square aspect ratio where possible.
        
        Args:
            expected_count: Approximate target number of points.
        
        Returns:
            Tensor of shape (n_x * n_y, 2) with grid coordinates.
        """
        # Compute grid dimensions to match expected count
        # Maintain aspect ratio similar to spatial bounds
        width = self.xlim[1] - self.xlim[0]
        height = self.ylim[1] - self.ylim[0]
        aspect_ratio = width / height
        
        # Solve: n_x * n_y ≈ expected_count, n_x / n_y ≈ aspect_ratio
        n_y = int((expected_count / aspect_ratio) ** 0.5)
        n_x = int(aspect_ratio * n_y)
        
        # Ensure at least 1 point in each dimension
        n_x = max(1, n_x)
        n_y = max(1, n_y)
        
        # Generate uniform coordinates
        x = torch.linspace(
            self.xlim[0], self.xlim[1], n_x, device=self.device
        )
        y = torch.linspace(
            self.ylim[0], self.ylim[1], n_y, device=self.device
        )
        
        # Create meshgrid and flatten
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        coordinates = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        return coordinates
    
    def _generate_poisson(self, density: float) -> torch.Tensor:
        """Generate approximate Poisson-distributed points via jittered grid.
        
        Creates a random point distribution by starting from a regular grid
        at the target density and applying uniform jitter.  This is **not**
        true Poisson-disk sampling (which enforces a hard minimum-distance
        constraint); it is a computationally efficient approximation that
        avoids O(n²) rejection loops while producing visually similar spatial
        distributions (resolves ReviewFinding#M5).
        
        Args:
            density: Target receptor density in receptors per mm².
        
        Returns:
            Tensor of approximately density × area points with shape [N, 2].
        """
        # Approximate Poisson disk sampling via jittered grid at target density.
        # This avoids O(n^2) rejection loops and scales to large grids.
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
        
        Creates optimal packing pattern with hexagonal symmetry, which
        maximizes coverage efficiency for circular receptive fields.
        
        Args:
            density: Target receptor density in receptors per mm².
        
        Returns:
            Tensor of hexagonally arranged points.
        """
        # Hexagonal packing spacing formula:
        # For hexagonal lattice, each hexagon has area = (3√3/2) * spacing²
        # To achieve target density: density = 1 / area_per_point
        # Therefore: spacing = sqrt(2 / (√3 * density))
        # This provides optimal packing density for circular receptive fields
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
    
    def _generate_jittered_grid(self, expected_count: int) -> torch.Tensor:
        """Generate regular grid with random spatial jitter.
        
        Creates a uniform grid then applies random displacement to each point,
        breaking regularity while maintaining approximate uniformity.
        
        Args:
            expected_count: Approximate target number of points.
        
        Returns:
            Tensor of jittered grid coordinates.
        """
        # Start with regular grid
        base_grid = self._generate_grid(expected_count)
        
        # Calculate grid spacing for jitter magnitude
        n_points = base_grid.shape[0]
        if n_points <= 1:
            return base_grid
        
        # Compute approximate spacing
        area = self._compute_area()
        approximate_spacing = (area / n_points) ** 0.5
        
        # Apply jitter: random displacement up to ±25% of spacing
        jitter_magnitude = 0.25 * approximate_spacing
        jitter = torch.randn_like(base_grid) * jitter_magnitude
        
        jittered = base_grid + jitter
        
        # Clamp to stay within bounds
        jittered[:, 0] = torch.clamp(
            jittered[:, 0], self.xlim[0], self.xlim[1]
        )
        jittered[:, 1] = torch.clamp(
            jittered[:, 1], self.ylim[0], self.ylim[1]
        )
        
        return jittered
