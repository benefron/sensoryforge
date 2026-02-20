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

# Type alias for innervation methods (distance_weighted removed; use use_distance_weights option)
InnervationMethod = Literal["gaussian", "one_to_one", "uniform", "distance_weighted"]


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
    
    @classmethod
    def from_config(cls, config: dict) -> "BaseInnervation":
        """Create innervation instance from config dict.
        
        Note: This is a base implementation. Subclasses should override
        to handle their specific parameters. The config must include
        receptor_coords and neuron_centers tensors.
        
        Args:
            config: Dictionary with 'receptor_coords', 'neuron_centers',
                'device', and method-specific parameters.
        
        Returns:
            BaseInnervation instance.
        """
        receptor_coords = config.pop("receptor_coords")
        neuron_centers = config.pop("neuron_centers")
        device = config.pop("device", "cpu")
        return cls(receptor_coords, neuron_centers, device=device, **config)
    
    def to_dict(self) -> dict:
        """Serialize innervation parameters to dict.
        
        Note: This does NOT serialize receptor_coords or neuron_centers
        tensors (they are too large). Only serializes configuration parameters.
        Subclasses should override to include their specific parameters.
        
        Returns:
            Dictionary with method name and parameters (excluding tensors).
        """
        return {
            "method": self.__class__.__name__.lower().replace("innervation", ""),
            "num_neurons": self.num_neurons,
            "num_receptors": self.num_receptors,
            "device": str(self.device),
        }


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
        use_distance_weights: If True, weights come from distance falloff
            instead of random in [min, max].
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
        use_distance_weights: bool = False,
        far_connection_fraction: float = 0.0,
        far_sigma_factor: float = 5.0,
        distance_weight_randomness_pct: float = 0.0,
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
            use_distance_weights: If True, weights from distance falloff.
            far_connection_fraction: Fraction of connections from far receptors
                (beyond far_sigma_factor * sigma) to break coherence.
            far_sigma_factor: Receptors beyond this * sigma_d_mm are "far".
            seed: Optional random seed.
            device: PyTorch device.
        """
        super().__init__(receptor_coords, neuron_centers, device)
        self.connections_per_neuron = connections_per_neuron
        self.sigma_d_mm = sigma_d_mm
        self.max_sigma_distance = max_sigma_distance
        self.weight_range = weight_range
        self.use_distance_weights = use_distance_weights
        self.far_connection_fraction = max(0.0, min(1.0, far_connection_fraction))
        self.far_sigma_factor = far_sigma_factor
        self.distance_weight_randomness_pct = max(0.0, min(100.0, distance_weight_randomness_pct))
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
        
        distances = torch.sqrt(d2)
        gaussian_weights = torch.exp(-d2 / (2 * self.sigma_d_mm ** 2))
        
        if self.max_sigma_distance > 0:
            max_dist = self.max_sigma_distance * self.sigma_d_mm
            gaussian_weights[distances > max_dist] = 0.0
        
        far_threshold = self.far_sigma_factor * self.sigma_d_mm
        is_far = distances > far_threshold
        
        if self.far_connection_fraction > 0 and is_far.any():
            local_weights = gaussian_weights.clone()
            local_weights[is_far] = 0.0
            far_weights = torch.zeros_like(gaussian_weights)
            far_weights[is_far] = 1.0
            local_sum = local_weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
            far_sum = far_weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
            local_prob = local_weights / local_sum
            far_prob = far_weights / far_sum
            prob_weights = (
                (1.0 - self.far_connection_fraction) * local_prob
                + self.far_connection_fraction * far_prob
            )
        else:
            prob_weights = gaussian_weights
        
        row_sums = prob_weights.sum(dim=1)
        empty_rows = row_sums <= 1e-12
        if empty_rows.any():
            nearest_idx = distances[empty_rows].argmin(dim=1)
            for i, neuron_row in enumerate(empty_rows.nonzero(as_tuple=True)[0]):
                prob_weights[neuron_row, nearest_idx[i]] = 1.0
        
        prob_weights = prob_weights / (prob_weights.sum(dim=1, keepdim=True) + 1e-12)
        
        # Sample K connections per neuron. Poisson is used because (a) it models
        # count of independent rare events (each receptor has small connection
        # probability), (b) variance equals mean, giving biological variability
        # across neurons. For deterministic K, use floor(connections_per_neuron).
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
            w_min, w_max = self.weight_range
            if self.use_distance_weights:
                # Weights from Gaussian distance falloff, scaled to [w_min, w_max]
                # Gather distances for sampled indices
                d2_sampled = torch.gather(d2, 1, all_idx)
                g_at_sampled = torch.exp(-d2_sampled / (2 * self.sigma_d_mm ** 2))
                # Normalize per row to [0,1] then scale to weight range
                row_max = g_at_sampled.max(dim=1, keepdim=True).values.clamp(min=1e-12)
                norm = g_at_sampled / row_max
                all_vals = w_min + norm * (w_max - w_min)
                if self.distance_weight_randomness_pct > 0:
                    pct = self.distance_weight_randomness_pct / 100.0
                    rand_vals = torch.empty_like(all_vals, device=self.device).uniform_(w_min, w_max)
                    all_vals = (1.0 - pct) * all_vals + pct * rand_vals
            else:
                all_vals = torch.empty(self.num_neurons, max_K, device=self.device).uniform_(
                    w_min, w_max
                )
            
            # Mask out excess samples
            arange = torch.arange(max_K, device=self.device).unsqueeze(0)
            mask = arange < K_per_neuron.unsqueeze(1)
            all_vals[~mask] = 0.0
            
            # Scatter into weight matrix
            weights.scatter_(1, all_idx, all_vals)
        
        return weights


def _decay_weights_from_distances(
    distances: torch.Tensor,
    max_distance_mm: float,
    decay_function: str,
    decay_rate: float,
) -> torch.Tensor:
    """Compute decay weights from distances (exponential, linear, inverse_square)."""
    if decay_function == "exponential":
        normalized_d = distances / (max_distance_mm + 1e-12)
        return torch.exp(-decay_rate * normalized_d)
    if decay_function == "linear":
        return torch.clamp(1.0 - distances / max_distance_mm, min=0.0)
    if decay_function == "inverse_square":
        return 1.0 / (1.0 + (decay_rate * distances) ** 2)
    raise ValueError(f"Unknown decay function: {decay_function}")


class UniformInnervation(BaseInnervation):
    """Uniform nearest-neighbor innervation (Voronoi-like).

    Each receptor connects to exactly one neuron (its nearest neighbor).
    This creates non-overlapping, Voronoi-like receptive fields. Multiple
    receptors may connect to the same neuron. Connection weights are uniform (1.0)
    or distance-weighted when use_distance_weights=True. Supports far_connection_fraction
    to add a fraction of connections from distant receptors.
    """

    def __init__(
        self,
        receptor_coords: torch.Tensor,
        neuron_centers: torch.Tensor,
        sigma_d_mm: float = 0.3,
        weight_range: Tuple[float, float] = (0.1, 1.0),
        use_distance_weights: bool = False,
        far_connection_fraction: float = 0.0,
        far_sigma_factor: float = 5.0,
        max_distance_mm: float = 1.0,
        decay_function: str = "exponential",
        decay_rate: float = 2.0,
        distance_weight_randomness_pct: float = 0.0,
        seed: Optional[int] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(receptor_coords, neuron_centers, device)
        self.sigma_d_mm = sigma_d_mm
        self.weight_range = weight_range
        self.use_distance_weights = use_distance_weights
        self.far_connection_fraction = max(0.0, min(1.0, far_connection_fraction))
        self.far_sigma_factor = far_sigma_factor
        self.max_distance_mm = max_distance_mm
        self.decay_function = decay_function
        self.decay_rate = decay_rate
        self.distance_weight_randomness_pct = max(0.0, min(100.0, distance_weight_randomness_pct))
        self.seed = seed

    def compute_weights(self, **kwargs) -> torch.Tensor:
        """Compute uniform nearest-neighbor connections."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        receptor_exp = self.receptor_coords.unsqueeze(1)
        neuron_exp = self.neuron_centers.unsqueeze(0)
        d2 = ((receptor_exp - neuron_exp) ** 2).sum(-1)
        distances = torch.sqrt(d2)
        nearest_neuron = distances.argmin(dim=1)
        weights = torch.zeros(self.num_neurons, self.num_receptors, device=self.device)
        receptor_indices = torch.arange(self.num_receptors, device=self.device)
        w_min, w_max = self.weight_range
        if self.use_distance_weights:
            decay = _decay_weights_from_distances(
                distances[receptor_indices, nearest_neuron],
                self.max_distance_mm,
                self.decay_function,
                self.decay_rate,
            )
            decay_max = decay.max().clamp(min=1e-12)
            norm = decay / decay_max
            vals = w_min + norm * (w_max - w_min)
            if self.distance_weight_randomness_pct > 0:
                pct = self.distance_weight_randomness_pct / 100.0
                rand_vals = torch.empty_like(vals, device=self.device).uniform_(w_min, w_max)
                vals = (1.0 - pct) * vals + pct * rand_vals
        else:
            vals = torch.full(
                (self.num_receptors,), (w_min + w_max) / 2, device=self.device
            )
        weights[nearest_neuron, receptor_indices] = vals

        if self.far_connection_fraction > 0:
            far_threshold = self.far_sigma_factor * self.sigma_d_mm
            is_far = distances > far_threshold
            if is_far.any():
                far_weights = torch.zeros_like(weights)
                far_weights.T[is_far] = 1.0
                far_sum = far_weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
                far_prob = far_weights / far_sum
                n_far_per_neuron = torch.poisson(
                    torch.full(
                        (self.num_neurons,),
                        self.far_connection_fraction * self.num_receptors / max(1, self.num_neurons),
                        device="cpu",
                    )
                ).long().to(self.device)
                n_far_per_neuron = torch.clamp(n_far_per_neuron, min=0)
                max_far = n_far_per_neuron.max().item()
                if max_far > 0:
                    all_idx = torch.multinomial(
                        far_prob + 1e-12, max_far, replacement=False
                    )
                    far_vals = torch.empty(
                        self.num_neurons, max_far, device=self.device
                    ).uniform_(w_min, w_max)
                    arange = torch.arange(max_far, device=self.device).unsqueeze(0)
                    mask = arange < n_far_per_neuron.unsqueeze(1)
                    far_vals[~mask] = 0.0
                    weights.scatter_add_(1, all_idx, far_vals)
        return weights


class OneToOneInnervation(BaseInnervation):
    """One-to-one: each neuron connects to exactly K nearest receptors.

    Each neuron selects exactly ``connections_per_neuron`` nearest receptors.
    Connection weights are distance-weighted when use_distance_weights=True,
    else uniform in weight_range. Supports far_connection_fraction to mix in
    connections from distant receptors (uses sampling when far_frac > 0).
    """

    def __init__(
        self,
        receptor_coords: torch.Tensor,
        neuron_centers: torch.Tensor,
        connections_per_neuron: float = 28.0,
        sigma_d_mm: float = 0.3,
        weight_range: Tuple[float, float] = (0.1, 1.0),
        use_distance_weights: bool = True,
        far_connection_fraction: float = 0.0,
        far_sigma_factor: float = 5.0,
        max_distance_mm: float = 1.0,
        decay_function: str = "exponential",
        decay_rate: float = 2.0,
        distance_weight_randomness_pct: float = 0.0,
        seed: Optional[int] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(receptor_coords, neuron_centers, device)
        self.connections_per_neuron = int(max(1, connections_per_neuron))
        self.sigma_d_mm = sigma_d_mm
        self.weight_range = weight_range
        self.use_distance_weights = use_distance_weights
        self.far_connection_fraction = max(0.0, min(1.0, far_connection_fraction))
        self.far_sigma_factor = far_sigma_factor
        self.max_distance_mm = max_distance_mm
        self.decay_function = decay_function
        self.decay_rate = decay_rate
        self.distance_weight_randomness_pct = max(0.0, min(100.0, distance_weight_randomness_pct))
        self.seed = seed

    def compute_weights(self, **kwargs) -> torch.Tensor:
        """Each neuron gets exactly K connections, optionally distance-weighted."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        K = self.connections_per_neuron
        receptor_exp = self.receptor_coords.unsqueeze(0)
        neuron_exp = self.neuron_centers.unsqueeze(1)
        d2 = ((receptor_exp - neuron_exp) ** 2).sum(-1)
        distances = torch.sqrt(d2)
        decay_weights = _decay_weights_from_distances(
            distances, self.max_distance_mm, self.decay_function, self.decay_rate
        )
        w_min, w_max = self.weight_range

        far_threshold = self.far_sigma_factor * self.sigma_d_mm
        is_far = distances > far_threshold
        weights = torch.zeros(self.num_neurons, self.num_receptors, device=self.device)
        for i in range(self.num_neurons):
            local_mask = ~is_far[i]
            far_mask = is_far[i]
            n_local_avail = local_mask.sum().item()
            n_far_avail = far_mask.sum().item()
            n_local = min(
                round((1.0 - self.far_connection_fraction) * K),
                n_local_avail,
            )
            n_far = min(K - n_local, n_far_avail)
            n_local = K - n_far  # Ensure we pick exactly K total

            local_idx = torch.where(local_mask)[0]
            far_idx = torch.where(far_mask)[0]
            if n_local > 0 and len(local_idx) > 0:
                local_probs = decay_weights[i, local_idx] + 1e-12
                local_probs = local_probs / local_probs.sum()
                k_local = min(n_local, len(local_idx))
                chosen_local = torch.multinomial(
                    local_probs.unsqueeze(0), k_local, replacement=False
                )[0]
                idx_local = local_idx[chosen_local]
            else:
                idx_local = torch.tensor([], dtype=torch.long, device=self.device)
                k_local = 0
            if n_far > 0 and len(far_idx) > 0:
                far_probs = torch.ones(len(far_idx), device=self.device) / len(far_idx)
                k_far = min(n_far, len(far_idx))
                chosen_far = torch.multinomial(
                    far_probs.unsqueeze(0), k_far, replacement=False
                )[0]
                idx_far = far_idx[chosen_far]
            else:
                idx_far = torch.tensor([], dtype=torch.long, device=self.device)
                k_far = 0
            all_idx = torch.cat([idx_local, idx_far])
            if len(all_idx) < K and len(local_idx) > 0:
                remaining = K - len(all_idx)
                already = set(all_idx.tolist())
                extra = [j for j in local_idx.tolist() if j not in already][:remaining]
                if extra:
                    all_idx = torch.cat([all_idx, torch.tensor(extra, device=self.device)])
            k_actual = min(K, len(all_idx))
            all_idx = all_idx[:k_actual]
            if self.use_distance_weights:
                d_at_sampled = distances[i, all_idx]
                decay_at = _decay_weights_from_distances(
                    d_at_sampled, self.max_distance_mm,
                    self.decay_function, self.decay_rate,
                )
                row_max = decay_at.max().clamp(min=1e-12)
                norm = decay_at / row_max
                vals = w_min + norm * (w_max - w_min)
                if self.distance_weight_randomness_pct > 0:
                    pct = self.distance_weight_randomness_pct / 100.0
                    rand_vals = torch.empty(k_actual, device=self.device).uniform_(w_min, w_max)
                    vals = (1.0 - pct) * vals + pct * rand_vals
            else:
                vals = torch.empty(k_actual, device=self.device).uniform_(w_min, w_max)
            weights[i, all_idx] = vals
        return weights


class DistanceWeightedInnervation(BaseInnervation):
    """Distance-weighted innervation with mean connections and decay function.
    
    Behaves like Gaussian: mean connections per neuron (Poisson), but weights
    come from distance decay (exponential, linear, inverse square) instead of
    random. Uses sigma_d_mm for spatial spread and max_distance_mm as cutoff.
    
    Attributes:
        connections_per_neuron: Mean number of connections per neuron.
        sigma_d_mm: Spatial spread (mm) for probability weighting.
        max_distance_mm: Maximum connection distance in mm (hard cutoff).
        decay_function: Type of decay ('exponential', 'linear', 'inverse_square').
        decay_rate: Decay rate parameter.
        weight_range: (min, max) for weight scaling.
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        receptor_coords: torch.Tensor,
        neuron_centers: torch.Tensor,
        connections_per_neuron: float = 28.0,
        sigma_d_mm: float = 0.3,
        max_distance_mm: float = 1.0,
        decay_function: Literal["exponential", "linear", "inverse_square"] = "exponential",
        decay_rate: float = 2.0,
        distance_weight_randomness_pct: float = 0.0,
        weight_range: Tuple[float, float] = (0.1, 1.0),
        seed: Optional[int] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize distance-weighted innervation.
        
        Args:
            receptor_coords: Receptor positions [N_receptors, 2].
            neuron_centers: Neuron positions [N_neurons, 2].
            connections_per_neuron: Target mean connections per neuron.
            sigma_d_mm: Spatial spread for probability weighting.
            max_distance_mm: Maximum connection distance in mm.
            decay_function: Decay type ('exponential', 'linear', 'inverse_square').
            decay_rate: Decay rate parameter.
            weight_range: (min, max) for weight scaling.
            seed: Optional random seed.
            device: PyTorch device.
        """
        super().__init__(receptor_coords, neuron_centers, device)
        self.connections_per_neuron = connections_per_neuron
        self.sigma_d_mm = sigma_d_mm
        self.max_distance_mm = max_distance_mm
        self.decay_function = decay_function
        self.decay_rate = decay_rate
        self.distance_weight_randomness_pct = max(0.0, min(100.0, distance_weight_randomness_pct))
        self.weight_range = weight_range
        self.seed = seed
    
    def _decay_weights(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute decay weights from distances."""
        if self.decay_function == "exponential":
            normalized_d = distances / (self.max_distance_mm + 1e-12)
            return torch.exp(-self.decay_rate * normalized_d)
        elif self.decay_function == "linear":
            return torch.clamp(1.0 - distances / self.max_distance_mm, min=0.0)
        elif self.decay_function == "inverse_square":
            return 1.0 / (1.0 + (self.decay_rate * distances) ** 2)
        else:
            raise ValueError(f"Unknown decay function: {self.decay_function}")
    
    def compute_weights(self, **kwargs) -> torch.Tensor:
        """Compute distance-weighted connections with mean K per neuron.
        
        Returns:
            Weight tensor [num_neurons, num_receptors].
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        receptor_exp = self.receptor_coords.unsqueeze(0)
        neuron_exp = self.neuron_centers.unsqueeze(1)
        distances = torch.sqrt(((receptor_exp - neuron_exp) ** 2).sum(-1))
        
        decay_weights = self._decay_weights(distances)
        decay_weights[distances > self.max_distance_mm] = 0.0
        
        # Fallback for neurons with no receptors in range
        row_sums = decay_weights.sum(dim=1)
        empty_rows = row_sums <= 1e-12
        if empty_rows.any():
            nearest_idx = distances[empty_rows].argmin(dim=1)
            for i, neuron_row in enumerate(empty_rows.nonzero(as_tuple=True)[0]):
                decay_weights[neuron_row, nearest_idx[i]] = 1.0
        
        prob_weights = decay_weights / (decay_weights.sum(dim=1, keepdim=True) + 1e-12)
        
        K_per_neuron = torch.poisson(
            torch.full((self.num_neurons,), float(self.connections_per_neuron), device="cpu")
        ).long().to(self.device)
        K_per_neuron = torch.clamp(K_per_neuron, min=1, max=self.num_receptors)
        
        max_K = K_per_neuron.max().item()
        weights = torch.zeros(self.num_neurons, self.num_receptors, device=self.device)
        
        if max_K > 0:
            all_idx = torch.multinomial(prob_weights, max_K, replacement=False)
            dist_sampled = torch.gather(distances, 1, all_idx)
            decay_sampled = self._decay_weights(dist_sampled)
            w_min, w_max = self.weight_range
            row_max = decay_sampled.max(dim=1, keepdim=True).values.clamp(min=1e-12)
            norm = decay_sampled / row_max
            all_vals = w_min + norm * (w_max - w_min)
            if self.distance_weight_randomness_pct > 0:
                pct = self.distance_weight_randomness_pct / 100.0
                rand_vals = torch.empty_like(all_vals, device=self.device).uniform_(w_min, w_max)
                all_vals = (1.0 - pct) * all_vals + pct * rand_vals
            
            arange = torch.arange(max_K, device=self.device).unsqueeze(0)
            mask = arange < K_per_neuron.unsqueeze(1)
            all_vals[~mask] = 0.0
            weights.scatter_(1, all_idx, all_vals)
        
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
    returns the weight tensor. Uses registry pattern for extensibility.
    
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
        KeyError: If method is not registered in INNERVATION_REGISTRY.
    
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
    # Import registry (ensure components are registered)
    from sensoryforge.register_components import register_all
    register_all()
    from sensoryforge.registry import INNERVATION_REGISTRY
    
    # Use registry to create innervation instance
    try:
        innervation = INNERVATION_REGISTRY.create(
            method,
            receptor_coords=receptor_coords,
            neuron_centers=neuron_centers,
            device=device,
            **method_params,
        )
    except KeyError:
        available = ", ".join(INNERVATION_REGISTRY.list_registered())
        raise ValueError(
            f"Unknown innervation method: '{method}'. "
            f"Registered methods: {available}"
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
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    arrangement: str = "grid",
    seed: Optional[int] = None,
    jitter_factor: float = 1.0,
) -> torch.Tensor:
    """Compute neuron centre coordinates.

    All arrangements respect a regular grid base (like receptor grid).
    - grid: Regular lattice.
    - jittered_grid: Grid + small jitter (0.25 * spacing).
    - blue_noise: Grid + jitter (0.4 * spacing) + Lloyd relaxation.
    - poisson: Grid + moderate jitter (0.5 * spacing); grid-respecting, no clustering.
    - hex: Hexagonal lattice (grid-like).

    Args:
        neurons_per_row: Default for square layout (rows=cols).
        xlim: ``(min, max)`` spatial bounds along x (mm).
        ylim: ``(min, max)`` spatial bounds along y (mm).
        device: Torch device for the returned tensor.
        edge_offset: Optional margin (mm) to shrink coverage near edges.
        sigma: Optional spatial spread (mm) used by some heuristics.
        rows: Override rows (vertical). Default: neurons_per_row.
        cols: Override cols (horizontal). Default: neurons_per_row.
        arrangement: 'grid', 'poisson', 'hex', 'jittered_grid', 'blue_noise'.
        seed: Random seed for jittered arrangements.
        jitter_factor: Scale for jitter magnitude (1.0 = default). Use >1 for more
            irregularity in advanced options.

    Returns:
        Tensor shaped ``(N, 2)`` containing ``(x, y)`` coordinate pairs.
    """
    n_rows = rows if rows is not None else neurons_per_row
    n_cols = cols if cols is not None else neurons_per_row
    x_min, x_max = xlim
    y_min, y_max = ylim
    offset = edge_offset if edge_offset is not None else 0.0
    x_min_eff = x_min + offset
    x_max_eff = x_max - offset
    y_min_eff = y_min + offset
    y_max_eff = y_max - offset

    # Base regular grid (used by all arrangements)
    x_centers = torch.linspace(x_min_eff, x_max_eff, n_cols, device=device)
    y_centers = torch.linspace(y_min_eff, y_max_eff, n_rows, device=device)
    yy_grid, xx_grid = torch.meshgrid(y_centers, x_centers, indexing="ij")
    base_coords = torch.stack([xx_grid.flatten(), yy_grid.flatten()], dim=1)

    if arrangement == "grid":
        return base_coords

    if seed is not None:
        torch.manual_seed(seed)

    # Grid-based jitter arrangements (match receptor grid logic)
    width = x_max_eff - x_min_eff
    height = y_max_eff - y_min_eff
    spacing_x = width / max(n_cols - 1, 1)
    spacing_y = height / max(n_rows - 1, 1)
    spacing = min(spacing_x, spacing_y)

    if arrangement == "jittered_grid":
        jitter_mag = 0.25 * spacing * jitter_factor
        jitter = torch.randn_like(base_coords, device=device) * jitter_mag
        coords = base_coords + jitter
    elif arrangement == "blue_noise":
        jitter_mag = 0.4 * spacing * jitter_factor
        jitter = (torch.rand_like(base_coords, device=device) - 0.5) * 2 * jitter_mag
        points = base_coords + jitter
        for _ in range(3):
            dists = torch.cdist(points, points)
            k = min(6, points.shape[0] - 1)
            _, nearest_idx = torch.topk(dists, k + 1, largest=False, dim=1)
            for i in range(points.shape[0]):
                neighbors = points[nearest_idx[i, 1:]]
                centroid = neighbors.mean(dim=0)
                points[i] = 0.7 * points[i] + 0.3 * centroid
        coords = points
    elif arrangement == "poisson":
        # Grid-respecting: moderate jitter, no pure random clustering
        jitter_mag = 0.5 * spacing * jitter_factor
        jitter = torch.randn_like(base_coords, device=device) * jitter_mag
        coords = base_coords + jitter
    elif arrangement == "hex":
        from .grid import ReceptorGrid
        center = ((x_min_eff + x_max_eff) / 2, (y_min_eff + y_max_eff) / 2)
        grid = ReceptorGrid(
            grid_size=(n_cols, n_rows),
            spacing=spacing,
            center=center,
            arrangement="hex",
            density=None,
            device=device,
        )
        hex_coords = grid.get_receptor_coordinates()
        expected = n_rows * n_cols
        if hex_coords.shape[0] >= expected:
            return hex_coords[:expected]
        return base_coords
    else:
        return base_coords

    coords[:, 0] = torch.clamp(coords[:, 0], x_min_eff, x_max_eff)
    coords[:, 1] = torch.clamp(coords[:, 1], y_min_eff, y_max_eff)
    return coords


def create_innervation_map_tensor(
    grid_coords: torch.Tensor,
    neuron_centers: torch.Tensor,
    connections_per_neuron: float,
    sigma_d_mm: float,
    grid_spacing_mm: float,
    weight_range: Tuple[float, float] = (0.1, 1.0),
    use_distance_weights: bool = False,
    far_connection_fraction: float = 0.0,
    far_sigma_factor: float = 5.0,
    distance_weight_randomness_pct: float = 0.0,
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

    d2 = ((grid_coords_exp - neuron_centers_exp) ** 2).sum(-1)
    distances = torch.sqrt(d2)
    gaussian_weights = torch.exp(-d2 / (2 * sigma_d_mm**2))
    flat_gaussian = gaussian_weights.view(num_neurons, -1)

    far_threshold = far_sigma_factor * sigma_d_mm
    flat_distances = distances.view(num_neurons, -1)
    is_far = flat_distances > far_threshold

    if far_connection_fraction > 0 and is_far.any():
        local_weights = flat_gaussian.clone()
        local_weights[is_far] = 0.0
        far_weights = torch.zeros_like(flat_gaussian)
        far_weights[is_far] = 1.0
        local_sum = local_weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        far_sum = far_weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        local_prob = local_weights / local_sum
        far_prob = far_weights / far_sum
        flat_weights = (
            (1.0 - far_connection_fraction) * local_prob
            + far_connection_fraction * far_prob
        )
    else:
        flat_weights = flat_gaussian

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
        all_idx = torch.multinomial(prob_weights, max_K, replacement=False)
        if use_distance_weights:
            d2_flat = d2.view(num_neurons, -1)
            d2_sampled = torch.gather(d2_flat, 1, all_idx)
            g_at_sampled = torch.exp(-d2_sampled / (2 * sigma_d_mm ** 2))
            row_max = g_at_sampled.max(dim=1, keepdim=True).values.clamp(min=1e-12)
            norm = g_at_sampled / row_max
            all_vals = weight_min + norm * (weight_max - weight_min)
            if distance_weight_randomness_pct > 0:
                pct = distance_weight_randomness_pct / 100.0
                rand_vals = torch.empty(num_neurons, max_K, device=device).uniform_(
                    weight_min, weight_max
                )
                all_vals = (1.0 - pct) * all_vals + pct * rand_vals
        else:
            all_vals = torch.empty(num_neurons, max_K, device=device).uniform_(
                weight_min, weight_max
            )

        arange = torch.arange(max_K, device=device).unsqueeze(0)
        mask = arange < K_per_neuron.unsqueeze(1)
        all_vals[~mask] = 0.0
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
        neuron_rows: Optional[int] = None,
        neuron_cols: Optional[int] = None,
        neuron_arrangement: str = "grid",
        connections_per_neuron: Optional[int] = 28,
        sigma_d_mm: Optional[float] = None,
        weight_range: Optional[Tuple[float, float]] = (0.1, 1.0),
        use_distance_weights: bool = False,
        far_connection_fraction: float = 0.0,
        far_sigma_factor: float = 5.0,
        distance_weight_randomness_pct: float = 0.0,
        seed: Optional[int] = None,
        edge_offset: Optional[float] = None,
        neuron_centers: Optional[torch.Tensor] = None,
        neuron_jitter_factor: float = 1.0,
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
            n_rows = neuron_rows if neuron_rows is not None else neurons_per_row
            n_cols = neuron_cols if neuron_cols is not None else neurons_per_row
            self.neurons_per_row = neurons_per_row
            self.num_neurons = n_rows * n_cols

            self.neuron_centers = create_neuron_centers(
                neurons_per_row,
                grid_props["xlim"],
                grid_props["ylim"],
                self.device,
                edge_offset=edge_offset,
                sigma=self.sigma_d_mm,
                rows=n_rows,
                cols=n_cols,
                arrangement=neuron_arrangement,
                seed=seed,
                jitter_factor=neuron_jitter_factor,
            )

        # Create coordinate tensor for grid points
        if hasattr(grid_manager, "xx") and grid_manager.xx is not None:
            xx, yy = grid_manager.get_coordinates()
            grid_coords = torch.stack([xx, yy], dim=-1)  # (grid_h, grid_w, 2)
        else:
            # Non-meshgrid arrangements: create regular grid from bounds for innervation
            grid_props = grid_manager.get_grid_properties()
            n_x, n_y = grid_manager.grid_size
            xlim, ylim = grid_props["xlim"], grid_props["ylim"]
            x = torch.linspace(xlim[0], xlim[1], n_x, device=self.device)
            y = torch.linspace(ylim[0], ylim[1], n_y, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            grid_coords = torch.stack([xx, yy], dim=-1)  # (grid_h, grid_w, 2)

        # Create innervation map
        self.innervation_map = create_innervation_map_tensor(
            grid_coords,
            self.neuron_centers,
            connections_per_neuron,
            self.sigma_d_mm,
            self.grid_spacing_mm,
            weight_range,
            use_distance_weights,
            far_connection_fraction,
            far_sigma_factor,
            distance_weight_randomness_pct,
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
        neuron_rows: Optional[int] = None,
        neuron_cols: Optional[int] = None,
        neuron_arrangement: str = "grid",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        innervation_method: InnervationMethod = "gaussian",
        connections_per_neuron: float = 28.0,
        sigma_d_mm: Optional[float] = None,
        max_sigma_distance: float = 3.0,
        weight_range: Tuple[float, float] = (0.1, 1.0),
        use_distance_weights: bool = False,
        far_connection_fraction: float = 0.0,
        far_sigma_factor: float = 5.0,
        max_distance_mm: float = 1.0,
        decay_function: str = "exponential",
        decay_rate: float = 2.0,
        distance_weight_randomness_pct: float = 0.0,
        seed: Optional[int] = None,
        edge_offset: Optional[float] = None,
        device: torch.device | str = "cpu",
        neuron_jitter_factor: float = 1.0,
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
        self.register_buffer("receptor_coords", receptor_coords)
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
            n_rows = neuron_rows if neuron_rows is not None else neurons_per_row
            n_cols = neuron_cols if neuron_cols is not None else neurons_per_row
            self.neurons_per_row = neurons_per_row
            self.num_neurons = n_rows * n_cols

            if xlim is None:
                xlim = (receptor_coords[:, 0].min().item(),
                        receptor_coords[:, 0].max().item())
            if ylim is None:
                ylim = (receptor_coords[:, 1].min().item(),
                        receptor_coords[:, 1].max().item())

            self.neuron_centers = create_neuron_centers(
                neurons_per_row, xlim, ylim, self.device,
                edge_offset=edge_offset, sigma=self.sigma_d_mm,
                rows=n_rows, cols=n_cols, arrangement=neuron_arrangement,
                seed=seed,
                jitter_factor=neuron_jitter_factor,
            )

        # Compute weight matrix via BaseInnervation subclass -------------
        if innervation_method == "gaussian":
            weights = GaussianInnervation(
                receptor_coords, self.neuron_centers,
                connections_per_neuron=connections_per_neuron,
                sigma_d_mm=self.sigma_d_mm,
                max_sigma_distance=max_sigma_distance,
                weight_range=weight_range,
                use_distance_weights=use_distance_weights,
                far_connection_fraction=far_connection_fraction,
                far_sigma_factor=far_sigma_factor,
                distance_weight_randomness_pct=distance_weight_randomness_pct,
                seed=seed, device=self.device,
            ).compute_weights()
        elif innervation_method == "one_to_one":
            weights = OneToOneInnervation(
                receptor_coords, self.neuron_centers,
                connections_per_neuron=connections_per_neuron,
                sigma_d_mm=self.sigma_d_mm,
                weight_range=weight_range,
                use_distance_weights=use_distance_weights,
                far_connection_fraction=far_connection_fraction,
                far_sigma_factor=far_sigma_factor,
                max_distance_mm=max_distance_mm,
                decay_function=decay_function,
                decay_rate=decay_rate,
                distance_weight_randomness_pct=distance_weight_randomness_pct,
                seed=seed,
                device=self.device,
            ).compute_weights()
        elif innervation_method == "uniform":
            weights = UniformInnervation(
                receptor_coords, self.neuron_centers,
                sigma_d_mm=self.sigma_d_mm,
                weight_range=weight_range,
                use_distance_weights=use_distance_weights,
                far_connection_fraction=far_connection_fraction,
                far_sigma_factor=far_sigma_factor,
                max_distance_mm=max_distance_mm,
                decay_function=decay_function,
                decay_rate=decay_rate,
                distance_weight_randomness_pct=distance_weight_randomness_pct,
                seed=seed,
                device=self.device,
            ).compute_weights()
        elif innervation_method == "distance_weighted":
            weights = DistanceWeightedInnervation(
                receptor_coords, self.neuron_centers,
                connections_per_neuron=connections_per_neuron,
                sigma_d_mm=self.sigma_d_mm,
                max_distance_mm=max_distance_mm,
                decay_function=decay_function,
                decay_rate=decay_rate,
                distance_weight_randomness_pct=distance_weight_randomness_pct,
                weight_range=weight_range,
                seed=seed,
                device=self.device,
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
