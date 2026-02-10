"""Innervation tensor builders for SA/RA tactile neuron populations.

This module provides multiple innervation strategies to connect receptor grids
to sensory neuron populations:
- Gaussian: Weighted random sampling with spatial falloff
- One-to-one: Each receptor connects to its nearest neuron
- Distance-weighted: Connection strength based on distance decay
- User-extensible: BaseInnervation class for custom strategies
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, TYPE_CHECKING, Literal

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .grid import GridManager, ReceptorGrid

# Type alias for innervation methods
InnervationMethod = Literal["gaussian", "one_to_one", "distance_weighted"]


# ============================================================================
# Base Innervation Classes (Phase 1.3)
# ============================================================================


class BaseInnervation(ABC):
    """Abstract base class for receptor-to-neuron innervation strategies.
    
    Innervation defines how receptor grid positions connect to sensory neuron
    populations. Different strategies encode different biological assumptions
    about receptive field organization.
    
    Subclasses must implement:
        - compute_weights(): Generate connection weight tensor
    
    Attributes:
        receptor_coords: Receptor positions [N_receptors, 2] in mm
        neuron_centers: Neuron positions [N_neurons, 2] in mm
        device: PyTorch device for tensors
    """
    
    def __init__(
        self,
        receptor_coords: torch.Tensor,
        neuron_centers: torch.Tensor,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize innervation strategy.
        
        Args:
            receptor_coords: Receptor positions [N_receptors, 2] in mm.
            neuron_centers: Neuron center positions [N_neurons, 2] in mm.
            device: PyTorch device identifier.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.receptor_coords = receptor_coords.to(self.device)
        self.neuron_centers = neuron_centers.to(self.device)
        self.num_neurons = neuron_centers.shape[0]
        self.num_receptors = receptor_coords.shape[0]
    
    @abstractmethod
    def compute_weights(self, **kwargs) -> torch.Tensor:
        """Compute connection weight tensor.
        
        Returns:
            Weight tensor [num_neurons, num_receptors] where weights[i, j]
            is the connection strength from receptor j to neuron i.
        """
        pass
    
    def get_connection_density(self, weights: torch.Tensor) -> float:
        """Calculate fraction of nonzero connections.
        
        Args:
            weights: Connection weight tensor [num_neurons, num_receptors].
        
        Returns:
            Density in range [0, 1].
        """
        total_connections = (weights > 0).sum().item()
        total_possible = weights.numel()
        return total_connections / total_possible


class GaussianInnervation(BaseInnervation):
    """Gaussian-weighted random innervation (existing method).
    
    Each neuron connects to a random subset of receptors with connection
    probabilities weighted by spatial distance (Gaussian falloff). This
    produces irregular, overlapping receptive fields.
    
    A hard spatial cutoff at ``max_sigma_distance * sigma_d_mm`` ensures
    biological locality: receptors beyond this distance have zero
    connection probability.
    
    Attributes:
        connections_per_neuron: Mean number of connections per neuron.
        sigma_d_mm: Spatial spread (mm) for Gaussian weighting.
        max_sigma_distance: Hard cutoff in units of sigma. Receptors
            beyond ``max_sigma_distance * sigma_d_mm`` have zero probability.
        weight_range: (min, max) range for sampled connection weights.
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        receptor_coords: torch.Tensor,
        neuron_centers: torch.Tensor,
        connections_per_neuron: float = 28.0,
        sigma_d_mm: float = 0.3,
        max_sigma_distance: float = 3.0,
        weight_range: Tuple[float, float] = (0.1, 1.0),
        seed: Optional[int] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize Gaussian innervation.
        
        Args:
            receptor_coords: Receptor positions [N_receptors, 2].
            neuron_centers: Neuron positions [N_neurons, 2].
            connections_per_neuron: Target mean connections per neuron.
            sigma_d_mm: Gaussian spatial spread in mm.
            max_sigma_distance: Hard cutoff in sigma units (default 3.0).
                Set to 0 or negative to disable cutoff (all-to-all).
            weight_range: (min, max) for sampled weights.
            seed: Optional random seed.
            device: PyTorch device.
        """
        super().__init__(receptor_coords, neuron_centers, device)
        self.connections_per_neuron = connections_per_neuron
        self.sigma_d_mm = sigma_d_mm
        self.max_sigma_distance = max_sigma_distance
        self.weight_range = weight_range
        self.seed = seed
    
    def compute_weights(self, **kwargs) -> torch.Tensor:
        """Compute Gaussian-weighted random connections.
        
        Receptors beyond ``max_sigma_distance * sigma_d_mm`` from a neuron
        have zero connection probability, enforcing spatial locality.
        
        Returns:
            Weight tensor [num_neurons, num_receptors].
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        # Compute pairwise squared distances [num_neurons, num_receptors]
        # receptor_coords: [num_receptors, 2]
        # neuron_centers: [num_neurons, 2]
        receptor_exp = self.receptor_coords.unsqueeze(0)  # [1, num_receptors, 2]
        neuron_exp = self.neuron_centers.unsqueeze(1)     # [num_neurons, 1, 2]
        
        d2 = ((receptor_exp - neuron_exp) ** 2).sum(-1)  # [num_neurons, num_receptors]
        
        # Gaussian weights for probability sampling
        gaussian_weights = torch.exp(-d2 / (2 * self.sigma_d_mm ** 2))
        
        # Apply spatial locality cutoff: zero out receptors beyond max distance
        if self.max_sigma_distance > 0:
            max_dist = self.max_sigma_distance * self.sigma_d_mm
            distances = torch.sqrt(d2)
            gaussian_weights[distances > max_dist] = 0.0
        
        # Handle neurons with no receptors in range: fall back to nearest
        row_sums = gaussian_weights.sum(dim=1)
        empty_rows = row_sums <= 1e-12
        if empty_rows.any():
            # For neurons with no local receptors, use the nearest receptor
            distances_full = torch.sqrt(d2)
            nearest_idx = distances_full[empty_rows].argmin(dim=1)
            for i, neuron_row in enumerate(empty_rows.nonzero(as_tuple=True)[0]):
                gaussian_weights[neuron_row, nearest_idx[i]] = 1.0
        
        prob_weights = gaussian_weights / (gaussian_weights.sum(dim=1, keepdim=True) + 1e-12)
        
        # Sample K connections per neuron (Poisson distribution)
        poisson_tensor = torch.full(
            (self.num_neurons,), float(self.connections_per_neuron), device="cpu"
        )
        K_per_neuron = torch.poisson(poisson_tensor).long().to(self.device)
        K_per_neuron = torch.clamp(K_per_neuron, min=1, max=self.num_receptors)
        
        # Vectorized sampling
        max_K = K_per_neuron.max().item()
        weights = torch.zeros(self.num_neurons, self.num_receptors, device=self.device)
        
        if max_K > 0:
            # Batched multinomial sampling
            all_idx = torch.multinomial(prob_weights, max_K, replacement=False)
            all_vals = torch.empty(self.num_neurons, max_K, device=self.device).uniform_(
                self.weight_range[0], self.weight_range[1]
            )
            
            # Mask out excess samples
            arange = torch.arange(max_K, device=self.device).unsqueeze(0)
            mask = arange < K_per_neuron.unsqueeze(1)
            all_vals[~mask] = 0.0
            
            # Scatter into weight matrix
            weights.scatter_(1, all_idx, all_vals)
        
        return weights


class OneToOneInnervation(BaseInnervation):
    """One-to-one nearest-neighbor innervation.
    
    Each receptor connects to exactly one neuron (its nearest neighbor).
    This creates non-overlapping, Voronoi-like receptive fields. Multiple
    receptors may connect to the same neuron.
    
    Connection weights are uniform (all 1.0).
    """
    
    def compute_weights(self, **kwargs) -> torch.Tensor:
        """Compute one-to-one nearest-neighbor connections.
        
        Returns:
            Weight tensor [num_neurons, num_receptors] with binary weights.
        """
        # Compute pairwise distances [num_receptors, num_neurons]
        receptor_exp = self.receptor_coords.unsqueeze(1)  # [num_receptors, 1, 2]
        neuron_exp = self.neuron_centers.unsqueeze(0)     # [1, num_neurons, 2]
        
        distances = torch.sqrt(((receptor_exp - neuron_exp) ** 2).sum(-1))
        
        # Find nearest neuron for each receptor
        nearest_neuron = distances.argmin(dim=1)  # [num_receptors]
        
        # Create sparse connection matrix
        weights = torch.zeros(self.num_neurons, self.num_receptors, device=self.device)
        
        # Set weights[nearest_neuron[i], i] = 1.0
        neuron_indices = nearest_neuron
        receptor_indices = torch.arange(self.num_receptors, device=self.device)
        weights[neuron_indices, receptor_indices] = 1.0
        
        return weights


class DistanceWeightedInnervation(BaseInnervation):
    """Distance-weighted innervation with decay function.
    
    Each neuron connects to all receptors within a maximum distance, with
    connection strength determined by a decay function (exponential, linear,
    or inverse square).
    
    This creates smooth, continuous receptive fields with controllable
    overlap.
    
    Attributes:
        max_distance_mm: Maximum connection distance in mm.
        decay_function: Type of decay ('exponential', 'linear', 'inverse_square').
        decay_rate: Decay rate parameter (interpretation depends on function).
    """
    
    def __init__(
        self,
        receptor_coords: torch.Tensor,
        neuron_centers: torch.Tensor,
        max_distance_mm: float = 1.0,
        decay_function: Literal["exponential", "linear", "inverse_square"] = "exponential",
        decay_rate: float = 2.0,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize distance-weighted innervation.
        
        Args:
            receptor_coords: Receptor positions [N_receptors, 2].
            neuron_centers: Neuron positions [N_neurons, 2].
            max_distance_mm: Maximum connection distance in mm.
            decay_function: Decay type ('exponential', 'linear', 'inverse_square').
            decay_rate: Decay rate parameter.
            device: PyTorch device.
        """
        super().__init__(receptor_coords, neuron_centers, device)
        self.max_distance_mm = max_distance_mm
        self.decay_function = decay_function
        self.decay_rate = decay_rate
    
    def compute_weights(self, **kwargs) -> torch.Tensor:
        """Compute distance-weighted connections.
        
        Returns:
            Weight tensor [num_neurons, num_receptors].
        """
        # Compute pairwise distances [num_neurons, num_receptors]
        receptor_exp = self.receptor_coords.unsqueeze(0)  # [1, num_receptors, 2]
        neuron_exp = self.neuron_centers.unsqueeze(1)     # [num_neurons, 1, 2]
        
        distances = torch.sqrt(((receptor_exp - neuron_exp) ** 2).sum(-1))
        
        # Apply decay function
        if self.decay_function == "exponential":
            # w = exp(-rate * d / max_d)
            normalized_d = distances / self.max_distance_mm
            weights = torch.exp(-self.decay_rate * normalized_d)
        elif self.decay_function == "linear":
            # w = max(0, 1 - d / max_d)
            weights = torch.clamp(1.0 - distances / self.max_distance_mm, min=0.0)
        elif self.decay_function == "inverse_square":
            # w = 1 / (1 + (rate * d)^2)
            weights = 1.0 / (1.0 + (self.decay_rate * distances) ** 2)
        else:
            raise ValueError(f"Unknown decay function: {self.decay_function}")
        
        # Zero out connections beyond max distance
        weights[distances > self.max_distance_mm] = 0.0
        
        return weights


# ============================================================================
# Factory Function
# ============================================================================


def create_innervation(
    receptor_coords: torch.Tensor,
    neuron_centers: torch.Tensor,
    method: InnervationMethod | str = "gaussian",
    device: torch.device | str = "cpu",
    **method_params,
) -> torch.Tensor:
    """Factory function to create innervation weight tensor.
    
    This is the primary user-facing API for creating receptor-to-neuron
    connections. It instantiates the appropriate innervation strategy and
    returns the weight tensor.
    
    Args:
        receptor_coords: Receptor positions [N_receptors, 2] in mm.
        neuron_centers: Neuron center positions [N_neurons, 2] in mm.
        method: Innervation method: 'gaussian', 'one_to_one', 'distance_weighted'.
        device: PyTorch device for tensors.
        **method_params: Method-specific parameters:
            
            For 'gaussian':
                - connections_per_neuron: float (default: 28.0)
                - sigma_d_mm: float (default: 0.3)
                - weight_range: Tuple[float, float] (default: (0.1, 1.0))
                - seed: Optional[int]
            
            For 'one_to_one':
                - (no parameters)
            
            For 'distance_weighted':
                - max_distance_mm: float (default: 1.0)
                - decay_function: str (default: 'exponential')
                - decay_rate: float (default: 2.0)
    
    Returns:
        Weight tensor [num_neurons, num_receptors] where weights[i, j] is the
        connection strength from receptor j to neuron i.
    
    Raises:
        ValueError: If method is not recognized.
    
    Examples:
        >>> # Gaussian innervation
        >>> W = create_innervation(
        ...     receptor_coords, neuron_centers,
        ...     method="gaussian",
        ...     connections_per_neuron=28.0,
        ...     sigma_d_mm=0.3
        ... )
        
        >>> # One-to-one innervation
        >>> W = create_innervation(
        ...     receptor_coords, neuron_centers,
        ...     method="one_to_one"
        ... )
        
        >>> # Distance-weighted innervation
        >>> W = create_innervation(
        ...     receptor_coords, neuron_centers,
        ...     method="distance_weighted",
        ...     max_distance_mm=1.0,
        ...     decay_function="exponential",
        ...     decay_rate=2.0
        ... )
    """
    if method == "gaussian":
        innervation = GaussianInnervation(
            receptor_coords, neuron_centers, device=device, **method_params
        )
    elif method == "one_to_one":
        innervation = OneToOneInnervation(
            receptor_coords, neuron_centers, device=device
        )
    elif method == "distance_weighted":
        innervation = DistanceWeightedInnervation(
            receptor_coords, neuron_centers, device=device, **method_params
        )
    else:
        raise ValueError(
            f"Unknown innervation method: '{method}'. "
            f"Supported: 'gaussian', 'one_to_one', 'distance_weighted'"
        )
    
    return innervation.compute_weights()


# ============================================================================
# Legacy Helper Functions (maintained for backward compatibility)
# ============================================================================


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

    # Vectorised innervation construction (resolves ReviewFinding#C1)
    # Sample max_K indices per neuron in a single batched multinomial call,
    # then mask out excess samples for neurons with K < max_K.
    max_K = K_per_neuron.max().item()
    rand_weights = torch.zeros_like(flat_weights)

    if max_K > 0:
        # Batched multinomial: [num_neurons, max_K]
        all_idx = torch.multinomial(prob_weights, max_K, replacement=False)
        all_vals = torch.empty(num_neurons, max_K, device=device).uniform_(
            weight_min, weight_max
        )

        # Build mask: only keep first K[n] samples per neuron
        arange = torch.arange(max_K, device=device).unsqueeze(0)  # [1, max_K]
        mask = arange < K_per_neuron.unsqueeze(1)  # [num_neurons, max_K]
        all_vals[~mask] = 0.0

        # Scatter into the flat weight matrix
        rand_weights.scatter_(1, all_idx, all_vals)

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

    def get_weights_per_neuron(self) -> torch.Tensor:
        """Count nonzero connections per neuron (vectorised).

        Returns:
            Tensor of shape ``(num_neurons,)`` with connection counts.
        """
        # Vectorised implementation (resolves ReviewFinding#H2)
        return (self.innervation_weights > 0).view(self.num_neurons, -1).sum(dim=1)

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


class FlatInnervationModule(nn.Module):
    """Innervation operator using flat receptor coordinate arrays.

    Unlike :class:`InnervationModule` which requires a :class:`GridManager`
    (meshgrid-based), this module accepts flat ``[N_receptors, 2]`` coordinate
    tensors.  It is suitable for composite-grid layers and irregular
    (poisson / hex) arrangements where a regular meshgrid does not exist.

    The ``forward()`` method applies the weight matrix to a flat receptor
    response vector ``[batch, N_receptors]`` or temporal
    ``[batch, time, N_receptors]`` to produce neuron inputs.

    Attributes:
        neuron_type: Label for this population (e.g., ``'SA'``, ``'RA'``).
        num_neurons: Number of sensory neurons.
        num_receptors: Number of connected receptors.
        neuron_centers: ``[num_neurons, 2]`` coordinates in mm.
        innervation_method: Which :class:`BaseInnervation` strategy was used.
    """

    def __init__(
        self,
        *,
        neuron_type: str = "SA",
        receptor_coords: torch.Tensor,
        neuron_centers: Optional[torch.Tensor] = None,
        neurons_per_row: Optional[int] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        innervation_method: InnervationMethod = "gaussian",
        connections_per_neuron: float = 28.0,
        sigma_d_mm: Optional[float] = None,
        max_sigma_distance: float = 3.0,
        weight_range: Tuple[float, float] = (0.1, 1.0),
        max_distance_mm: float = 1.0,
        decay_function: str = "exponential",
        decay_rate: float = 2.0,
        seed: Optional[int] = None,
        edge_offset: Optional[float] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Build innervation weights from flat receptor coordinates.

        Either ``neuron_centers`` or ``neurons_per_row`` (+ ``xlim`` / ``ylim``)
        must be supplied.

        Args:
            neuron_type: ``'SA'``, ``'RA'``, or custom label.
            receptor_coords: ``[N_receptors, 2]`` receptor positions in mm.
            neuron_centers: Pre-computed ``[N_neurons, 2]`` centres.  If *None*,
                centres are generated on a square lattice from ``neurons_per_row``
                within ``xlim`` × ``ylim``.
            neurons_per_row: Neurons along each axis (used when ``neuron_centers``
                is *None*).
            xlim: ``(min, max)`` spatial bounds for auto-generated centres.
            ylim: ``(min, max)`` spatial bounds for auto-generated centres.
            innervation_method: ``'gaussian'``, ``'one_to_one'``, or
                ``'distance_weighted'``.
            connections_per_neuron: Target mean connections (Gaussian method).
            sigma_d_mm: Gaussian spatial spread in mm. Default depends on
                ``neuron_type``.
            max_sigma_distance: Hard cutoff in sigma units for Gaussian method.
            weight_range: ``(min, max)`` for sampled connection weights.
            max_distance_mm: Max distance for distance_weighted method (mm).
            decay_function: Decay type for distance_weighted method.
            decay_rate: Decay rate for distance_weighted method.
            seed: Random seed for deterministic sampling.
            edge_offset: Margin in mm to shrink neuron lattice from edges.
            device: Torch device.
        """
        super().__init__()

        self.neuron_type = neuron_type
        self.innervation_method = innervation_method
        self.device = torch.device(device) if isinstance(device, str) else device

        receptor_coords = receptor_coords.to(self.device)
        self.num_receptors = receptor_coords.shape[0]

        self.sigma_d_mm = sigma_d_mm or (0.3 if neuron_type == "SA" else 0.39)

        # Build or accept neuron centres --------------------------------
        if neuron_centers is not None:
            self.neuron_centers = neuron_centers.to(self.device)
            self.num_neurons = self.neuron_centers.shape[0]
            self.neurons_per_row = None
        else:
            if neurons_per_row is None:
                neurons_per_row = 10 if neuron_type == "SA" else 14
            self.neurons_per_row = neurons_per_row
            self.num_neurons = neurons_per_row ** 2

            # Infer bounds from receptor coords when not provided
            if xlim is None:
                xlim = (receptor_coords[:, 0].min().item(),
                        receptor_coords[:, 0].max().item())
            if ylim is None:
                ylim = (receptor_coords[:, 1].min().item(),
                        receptor_coords[:, 1].max().item())

            self.neuron_centers = create_neuron_centers(
                neurons_per_row, xlim, ylim, self.device,
                edge_offset=edge_offset, sigma=self.sigma_d_mm,
            )

        # Compute weight matrix via BaseInnervation subclass -------------
        if innervation_method == "gaussian":
            weights = GaussianInnervation(
                receptor_coords, self.neuron_centers,
                connections_per_neuron=connections_per_neuron,
                sigma_d_mm=self.sigma_d_mm,
                max_sigma_distance=max_sigma_distance,
                weight_range=weight_range,
                seed=seed, device=self.device,
            ).compute_weights()
        elif innervation_method == "one_to_one":
            weights = OneToOneInnervation(
                receptor_coords, self.neuron_centers, device=self.device
            ).compute_weights()
        elif innervation_method == "distance_weighted":
            weights = DistanceWeightedInnervation(
                receptor_coords, self.neuron_centers,
                max_distance_mm=max_distance_mm,
                decay_function=decay_function,
                decay_rate=decay_rate, device=self.device,
            ).compute_weights()
        else:
            raise ValueError(f"Unknown innervation method: {innervation_method}")

        # Store as registered buffer
        self.register_buffer("innervation_weights", weights)

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def forward(self, receptor_responses: torch.Tensor) -> torch.Tensor:
        """Apply innervation weights to receptor responses.

        Args:
            receptor_responses: Flat receptor activations.
                - ``[batch, N_receptors]`` for static input.
                - ``[batch, time, N_receptors]`` for temporal input.

        Returns:
            Neuron inputs:
                - ``[batch, num_neurons]`` for static input.
                - ``[batch, time, num_neurons]`` for temporal input.
        """
        ndim = receptor_responses.ndim
        W = self.innervation_weights  # [num_neurons, num_receptors]

        if ndim == 2:
            # [batch, N_receptors] @ [N_receptors, num_neurons]
            return torch.matmul(receptor_responses, W.T)
        elif ndim == 3:
            batch, time, _ = receptor_responses.shape
            flat = receptor_responses.reshape(batch * time, -1)
            out = torch.matmul(flat, W.T)
            return out.reshape(batch, time, self.num_neurons)
        else:
            raise ValueError(
                f"Expected 2D or 3D input, got {ndim}D with shape "
                f"{receptor_responses.shape}"
            )

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def get_connection_density(self) -> float:
        """Fraction of nonzero connections."""
        total = (self.innervation_weights > 0).sum().item()
        return total / self.innervation_weights.numel()

    def get_weights_per_neuron(self) -> torch.Tensor:
        """Count nonzero connections per neuron.

        Returns:
            ``[num_neurons]`` tensor of connection counts.
        """
        return (self.innervation_weights > 0).sum(dim=1)

    def to_device(self, device: torch.device | str) -> "FlatInnervationModule":
        """Move to a different device."""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.neuron_centers = self.neuron_centers.to(self.device)
        return self.to(self.device)


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
