"""Innervation tensor builders for SA/RA tactile neuron populations."""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .grid import GridManager


def create_neuron_centers(
    neurons_per_row: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    device: torch.device | str = "cpu",
    edge_offset: Optional[float] = None,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """Compute neuron centre coordinates for a square lattice.

    Args:
        neurons_per_row: Number of neurons along each axis (square grid).
        xlim: ``(min, max)`` spatial bounds along x (mm).
        ylim: ``(min, max)`` spatial bounds along y (mm).
        device: Torch device for the returned tensor.
        edge_offset: Optional margin (mm) to shrink coverage near edges.
        sigma: Optional spatial spread (mm) used by some heuristics.

    Returns:
        Tensor shaped ``(neurons_per_row**2, 2)`` containing ``(x, y)``
        coordinate pairs.
    """
    rows = cols = neurons_per_row
    x_min, x_max = xlim
    y_min, y_max = ylim
    if edge_offset is not None:
        offset = edge_offset
    else:
        offset = 0.0
    x_min_eff = x_min + offset
    x_max_eff = x_max - offset
    y_min_eff = y_min + offset
    y_max_eff = y_max - offset
    x_centers = torch.linspace(x_min_eff, x_max_eff, cols, device=device)
    y_centers = torch.linspace(y_min_eff, y_max_eff, rows, device=device)
    yy_grid, xx_grid = torch.meshgrid(y_centers, x_centers, indexing="ij")
    mesh = torch.stack([xx_grid.flatten(), yy_grid.flatten()], dim=1)
    return mesh


def create_innervation_map_tensor(
    grid_coords: torch.Tensor,
    neuron_centers: torch.Tensor,
    connections_per_neuron: float,
    sigma_d_mm: float,
    grid_spacing_mm: float,
    weight_range: Tuple[float, float] = (0.1, 1.0),
    seed: Optional[int] = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a dense innervation tensor via Gaussian sampling.

    Args:
        grid_coords: ``(grid_h, grid_w, 2)`` coordinate tensor in mm.
        neuron_centers: ``(num_neurons, 2)`` array of neuron centres.
        connections_per_neuron: Target mean number of connections.
        sigma_d_mm: Spatial spread (mm) defining Gaussian falloff.
        grid_spacing_mm: Physical spacing between receptors (unused but
            retained for compatibility).
        weight_range: Tuple ``(min, max)`` for sampled connection weights.
        seed: Optional random seed for reproducibility.
        device: Torch device housing the resulting tensor.

    Returns:
        Dense tensor ``(num_neurons, grid_h, grid_w)`` with connection weights.
    """
    if seed is not None:
        torch.manual_seed(seed)
    grid_h, grid_w, _ = grid_coords.shape
    num_neurons = neuron_centers.shape[0]
    weight_min, weight_max = weight_range

    # Expand grid_coords to (1, grid_h, grid_w, 2)
    # Expand neuron_centers to (num_neurons, 1, 1, 2)
    grid_coords_exp = grid_coords.unsqueeze(0)  # (1, grid_h, grid_w, 2)
    neuron_centers_exp = neuron_centers[:, None, None, :]

    # Compute squared distances: (num_neurons, grid_h, grid_w)
    d2 = ((grid_coords_exp - neuron_centers_exp) ** 2).sum(-1)

    # Compute Gaussian weights for selection only
    gaussian_weights = torch.exp(-d2 / (2 * sigma_d_mm**2))
    flat_weights = gaussian_weights.view(num_neurons, -1)

    # Normalize for probability sampling
    prob_weights = flat_weights / (flat_weights.sum(dim=1, keepdim=True) + 1e-12)

    # Sample K via Poisson draw (CPU for MPS compatibility)
    poisson_tensor = torch.full(
        (num_neurons,), float(connections_per_neuron), device="cpu"
    )
    K_per_neuron = torch.poisson(poisson_tensor).long().to(device)

    # Clamp to at least 1 and at most total number of mechanoreceptors
    max_conn = flat_weights.shape[1]
    K_per_neuron = torch.clamp(K_per_neuron, min=1, max=max_conn)
    rand_weights = torch.zeros_like(flat_weights)
    for n in range(num_neurons):
        K = K_per_neuron[n].item()
        if K > 0:
            # Sample K indices without replacement from probability weights
            idx = torch.multinomial(prob_weights[n], K, replacement=False)
            rand_vals = torch.empty(K, device=device).uniform_(weight_min, weight_max)
            rand_weights[n, idx] = rand_vals

    # Reshape back to (num_neurons, grid_h, grid_w)
    innervation_map = rand_weights.view(num_neurons, grid_h, grid_w)

    return innervation_map


class InnervationModule(nn.Module):
    """Dense innervation operator for SA/RA neuron populations."""

    def __init__(
        self,
        *,
        neuron_type: str = "SA",
        grid_manager: "GridManager",
        neurons_per_row: Optional[int] = None,
        connections_per_neuron: Optional[int] = 28,
        sigma_d_mm: Optional[float] = None,
        weight_range: Optional[Tuple[float, float]] = (0.1, 1.0),
        seed: Optional[int] = None,
        edge_offset: Optional[float] = None,
        neuron_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialise innervation tensors for a tactile neuron population.

        Args:
            neuron_type: ``'SA'`` or ``'RA'``.
            grid_manager: Source grid manager supplying coordinates.
            neurons_per_row: Override for neurons per row (square layout).
            connections_per_neuron: Target mean connections per neuron.
            sigma_d_mm: Spatial spread override in millimetres.
            weight_range: Optional weight range ``(min, max)``.
            seed: Random seed for deterministic sampling.
            edge_offset: Optional edge margin to avoid boundary artefacts.
            neuron_centers: Optional pre-calculated neuron centers. If provided,
                overrides ``neurons_per_row``.
        """
        super().__init__()

        self.neuron_type = neuron_type
        self.grid_manager = grid_manager
        self.connections_per_neuron = connections_per_neuron
        self.weight_range = weight_range
        self.seed = seed

        self.sigma_d_mm = sigma_d_mm or (0.3 if neuron_type == "SA" else 0.39)

        # Get grid properties
        grid_props = grid_manager.get_grid_properties()
        self.device = grid_props["device"]
        self.grid_spacing_mm = grid_props["spacing"]

        if neuron_centers is not None:
            self.neuron_centers = neuron_centers.to(self.device)
            self.num_neurons = len(neuron_centers)
            self.neurons_per_row = None
        else:
            if neurons_per_row is None:
                neurons_per_row = 10 if neuron_type == "SA" else 14
            self.neurons_per_row = neurons_per_row
            self.num_neurons = neurons_per_row**2

            # Create neuron centers
            self.neuron_centers = create_neuron_centers(
                neurons_per_row,
                grid_props["xlim"],
                grid_props["ylim"],
                self.device,
                edge_offset=edge_offset,
                sigma=self.sigma_d_mm,
            )

        # Create coordinate tensor for grid points
        xx, yy = grid_manager.get_coordinates()
        grid_coords = torch.stack([xx, yy], dim=-1)  # (grid_h, grid_w, 2)

        # Create innervation map
        self.innervation_map = create_innervation_map_tensor(
            grid_coords,
            self.neuron_centers,
            connections_per_neuron,
            self.sigma_d_mm,
            self.grid_spacing_mm,
            weight_range,
            seed,
            self.device,
        )

        # Register as buffer so it moves with the module
        self.register_buffer("innervation_weights", self.innervation_map)

    def forward(self, mechanoreceptor_responses):
        """
        Apply innervation to mechanoreceptor responses.
        Args:
            mechanoreceptor_responses: tensor
                - shape (batch_size, grid_h, grid_w) for static
                - shape (batch_size, time_steps, grid_h, grid_w) for temporal
        Returns:
            neuron_inputs: tensor
                - shape (batch_size, num_neurons) for static
                - shape (batch_size, time_steps, num_neurons) for temporal
        """
        original_shape = mechanoreceptor_responses.shape

        if len(original_shape) == 3:
            # Static: (batch_size, grid_h, grid_w)
            batch_size, grid_h, grid_w = original_shape

            # Flatten spatial dimensions for matrix multiplication
            mech_flat = mechanoreceptor_responses.view(batch_size, -1)
            innervation_flat = self.innervation_weights.view(self.num_neurons, -1)

            # Matrix multiplication: (batch_size, num_neurons)
            neuron_inputs = torch.matmul(mech_flat, innervation_flat.T)

        elif len(original_shape) == 4:
            # Temporal: (batch_size, time_steps, grid_h, grid_w)
            batch_size, time_steps, grid_h, grid_w = original_shape

            # Reshape for batch processing
            mech_flat = mechanoreceptor_responses.view(batch_size * time_steps, -1)
            innervation_flat = self.innervation_weights.view(self.num_neurons, -1)

            # Matrix multiplication
            neuron_inputs_flat = torch.matmul(mech_flat, innervation_flat.T)

            # Reshape back: (batch_size, time_steps, num_neurons)
            neuron_inputs = neuron_inputs_flat.view(
                batch_size, time_steps, self.num_neurons
            )
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")

        return neuron_inputs

    def get_connection_density(self):
        """Calculate actual connection density."""
        total_connections = (self.innervation_weights > 0).sum().item()
        total_possible = (
            self.num_neurons
            * self.innervation_weights.shape[1]
            * self.innervation_weights.shape[2]
        )
        return total_connections / total_possible

    def get_weights_per_neuron(self):
        """Get number of connections per neuron."""
        connections_per_neuron = []
        for i in range(self.num_neurons):
            connections = (self.innervation_weights[i] > 0).sum().item()
            connections_per_neuron.append(connections)
        return torch.tensor(connections_per_neuron)

    def visualize_neuron_connections(self, neuron_idx):
        """Get connection pattern for a specific neuron."""
        if neuron_idx >= self.num_neurons:
            raise ValueError(
                f"Neuron index {neuron_idx} exceeds {self.num_neurons - 1}"
            )
        return self.innervation_weights[neuron_idx].detach().cpu().numpy()

    def to_device(self, device):
        """Move module to a different device."""
        self.device = device
        self.neuron_centers = self.neuron_centers.to(device)
        return self.to(device)


def create_sa_innervation(
    grid_manager: "GridManager",
    *,
    seed: Optional[int] = None,
    neurons_per_row: Optional[int] = None,
    connections_per_neuron: Optional[int] = None,
    sigma_d_mm: Optional[float] = None,
    weight_range: Optional[Tuple[float, float]] = None,
    edge_offset: Optional[float] = None,
) -> InnervationModule:
    """Factory for SA innervation modules using project defaults."""

    return InnervationModule(
        neuron_type="SA",
        grid_manager=grid_manager,
        neurons_per_row=neurons_per_row,
        connections_per_neuron=connections_per_neuron or 28,
        sigma_d_mm=sigma_d_mm,
        weight_range=weight_range,
        seed=seed,
        edge_offset=edge_offset,
    )


def create_ra_innervation(
    grid_manager: "GridManager",
    *,
    seed: Optional[int] = None,
    neurons_per_row: Optional[int] = None,
    connections_per_neuron: Optional[int] = None,
    sigma_d_mm: Optional[float] = None,
    weight_range: Optional[Tuple[float, float]] = None,
    edge_offset: Optional[float] = None,
) -> InnervationModule:
    """Factory for RA innervation modules using project defaults."""

    return InnervationModule(
        neuron_type="RA",
        grid_manager=grid_manager,
        neurons_per_row=neurons_per_row,
        connections_per_neuron=connections_per_neuron or 28,
        sigma_d_mm=sigma_d_mm,
        weight_range=weight_range,
        seed=seed,
        edge_offset=edge_offset,
    )
