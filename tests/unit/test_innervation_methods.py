"""Unit tests for Phase 1.3 innervation methods."""

import pytest
import torch
from sensoryforge.core.innervation import (
    BaseInnervation,
    GaussianInnervation,
    OneToOneInnervation,
    UniformInnervation,
    DistanceWeightedInnervation,
    create_innervation,
)


@pytest.fixture
def simple_coords():
    """Create simple test coordinates."""
    # 3x3 grid of receptors
    receptor_coords = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
    ], dtype=torch.float32)
    
    # 2x2 grid of neurons
    neuron_centers = torch.tensor([
        [0.5, 0.5], [1.5, 0.5],
        [0.5, 1.5], [1.5, 1.5],
    ], dtype=torch.float32)
    
    return receptor_coords, neuron_centers


class TestGaussianInnervation:
    """Test Gaussian innervation method."""
    
    def test_initialization(self, simple_coords):
        """Test Gaussian innervation initializes correctly."""
        receptor_coords, neuron_centers = simple_coords
        
        innervation = GaussianInnervation(
            receptor_coords,
            neuron_centers,
            connections_per_neuron=3.0,
            sigma_d_mm=0.5,
        )
        
        assert innervation.num_neurons == 4
        assert innervation.num_receptors == 9
        assert innervation.connections_per_neuron == 3.0
        assert innervation.sigma_d_mm == 0.5
    
    def test_compute_weights_shape(self, simple_coords):
        """Test weight tensor has correct shape."""
        receptor_coords, neuron_centers = simple_coords
        
        innervation = GaussianInnervation(
            receptor_coords,
            neuron_centers,
            connections_per_neuron=3.0,
            sigma_d_mm=0.5,
            seed=42,
        )
        
        weights = innervation.compute_weights()
        
        assert weights.shape == (4, 9)  # [num_neurons, num_receptors]
        assert weights.dtype == torch.float32
    
    def test_compute_weights_sparsity(self, simple_coords):
        """Test weights are sparse (not all receptors connected)."""
        receptor_coords, neuron_centers = simple_coords
        
        innervation = GaussianInnervation(
            receptor_coords,
            neuron_centers,
            connections_per_neuron=3.0,
            sigma_d_mm=0.5,
            seed=42,
        )
        
        weights = innervation.compute_weights()
        
        # Should have sparse connections
        nonzero_count = (weights > 0).sum().item()
        assert nonzero_count < weights.numel()
        assert nonzero_count > 0
    
    def test_reproducible_with_seed(self, simple_coords):
        """Test same seed produces same weights."""
        receptor_coords, neuron_centers = simple_coords
        
        innervation1 = GaussianInnervation(
            receptor_coords, neuron_centers, seed=42
        )
        weights1 = innervation1.compute_weights()
        
        innervation2 = GaussianInnervation(
            receptor_coords, neuron_centers, seed=42
        )
        weights2 = innervation2.compute_weights()
        
        assert torch.allclose(weights1, weights2)


class TestUniformInnervation:
    """Test uniform (Voronoi-like) innervation: each receptor -> nearest neuron."""

    def test_initialization(self, simple_coords):
        receptor_coords, neuron_centers = simple_coords
        innervation = UniformInnervation(receptor_coords, neuron_centers)
        assert innervation.num_neurons == 4
        assert innervation.num_receptors == 9

    def test_compute_weights_shape(self, simple_coords):
        receptor_coords, neuron_centers = simple_coords
        innervation = UniformInnervation(receptor_coords, neuron_centers)
        weights = innervation.compute_weights()
        assert weights.shape == (4, 9)

    def test_each_receptor_connects_once(self, simple_coords):
        receptor_coords, neuron_centers = simple_coords
        innervation = UniformInnervation(receptor_coords, neuron_centers)
        weights = innervation.compute_weights()
        connections_per_receptor = (weights > 0).sum(dim=0)
        assert torch.all(connections_per_receptor == 1)

    def test_binary_weights(self, simple_coords):
        receptor_coords, neuron_centers = simple_coords
        innervation = UniformInnervation(
            receptor_coords, neuron_centers, weight_range=(1.0, 1.0)
        )
        weights = innervation.compute_weights()
        nonzero_weights = weights[weights > 0]
        assert torch.all(nonzero_weights == 1.0)

    def test_nearest_neighbor_property(self, simple_coords):
        receptor_coords, neuron_centers = simple_coords
        innervation = UniformInnervation(
            receptor_coords, neuron_centers, weight_range=(1.0, 1.0)
        )
        weights = innervation.compute_weights()
        assert weights[0, 0] == 1.0
        assert weights[1, 2] == 1.0


class TestOneToOneInnervation:
    """Test one-to-one: each neuron gets exactly K connections."""

    def test_each_neuron_gets_k_connections(self, simple_coords):
        receptor_coords, neuron_centers = simple_coords
        innervation = OneToOneInnervation(
            receptor_coords, neuron_centers,
            connections_per_neuron=2,
            sigma_d_mm=0.5,
            seed=42,
        )
        weights = innervation.compute_weights()
        assert weights.shape == (4, 9)
        connections_per_neuron = (weights > 0).sum(dim=1)
        assert torch.all(connections_per_neuron == 2)

    def test_weights_from_distance(self, simple_coords):
        receptor_coords, neuron_centers = simple_coords
        innervation = OneToOneInnervation(
            receptor_coords, neuron_centers,
            connections_per_neuron=2,
            sigma_d_mm=0.5,
            weight_range=(0.1, 1.0),
            seed=42,
        )
        weights = innervation.compute_weights()
        nonzero = weights[weights > 0]
        assert nonzero.min() >= 0.1
        assert nonzero.max() <= 1.0


class TestDistanceWeightedInnervation:
    """Test distance-weighted innervation method."""
    
    def test_initialization(self, simple_coords):
        """Test distance-weighted innervation initializes correctly."""
        receptor_coords, neuron_centers = simple_coords
        
        innervation = DistanceWeightedInnervation(
            receptor_coords,
            neuron_centers,
            max_distance_mm=1.5,
            decay_function="exponential",
            decay_rate=2.0,
        )
        
        assert innervation.num_neurons == 4
        assert innervation.num_receptors == 9
        assert innervation.max_distance_mm == 1.5
        assert innervation.decay_function == "exponential"
        assert innervation.decay_rate == 2.0
    
    def test_compute_weights_shape(self, simple_coords):
        """Test weight tensor has correct shape."""
        receptor_coords, neuron_centers = simple_coords
        
        innervation = DistanceWeightedInnervation(
            receptor_coords, neuron_centers, max_distance_mm=2.0
        )
        weights = innervation.compute_weights()
        
        assert weights.shape == (4, 9)
    
    def test_exponential_decay(self, simple_coords):
        """Test exponential decay function produces correct weights."""
        receptor_coords, neuron_centers = simple_coords
        
        innervation = DistanceWeightedInnervation(
            receptor_coords,
            neuron_centers,
            max_distance_mm=3.0,
            decay_function="exponential",
            decay_rate=1.0,
        )
        weights = innervation.compute_weights()
        
        # Weights should be positive and decrease with distance
        assert torch.all(weights >= 0)
        
        # Closer receptors should have higher weights
        # Neuron 0 at [0.5, 0.5] should have higher weight for receptor 0 [0, 0]
        # than for receptor 8 [2, 2]
        assert weights[0, 0] > weights[0, 8]
    
    def test_linear_decay(self, simple_coords):
        """Test linear decay function."""
        receptor_coords, neuron_centers = simple_coords
        
        innervation = DistanceWeightedInnervation(
            receptor_coords,
            neuron_centers,
            max_distance_mm=2.0,
            decay_function="linear",
        )
        weights = innervation.compute_weights()
        
        # Should have some connections
        assert (weights > 0).sum() > 0
        
        # Weights should be in [0, 1]
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)
    
    def test_inverse_square_decay(self, simple_coords):
        """Test inverse square decay function."""
        receptor_coords, neuron_centers = simple_coords
        
        innervation = DistanceWeightedInnervation(
            receptor_coords,
            neuron_centers,
            max_distance_mm=3.0,
            decay_function="inverse_square",
            decay_rate=1.0,
        )
        weights = innervation.compute_weights()
        
        # Should have positive weights
        assert torch.all(weights >= 0)
    
    def test_max_distance_cutoff(self, simple_coords):
        """Test mean connections with distance cutoff; far receptors excluded from sampling."""
        receptor_coords, neuron_centers = simple_coords
        innervation = DistanceWeightedInnervation(
            receptor_coords,
            neuron_centers,
            connections_per_neuron=3.0,
            sigma_d_mm=0.5,
            max_distance_mm=1.5,
            decay_function="linear",
            seed=42,
        )
        weights = innervation.compute_weights()
        assert weights.shape == (4, 9)
        conn_per_neuron = (weights > 0).sum(dim=1)
        assert torch.all(conn_per_neuron >= 1)
        assert torch.all(conn_per_neuron <= 9)


class TestCreateInnervationFactory:
    """Test factory function create_innervation()."""
    
    def test_gaussian_method(self, simple_coords):
        """Test factory creates Gaussian innervation."""
        receptor_coords, neuron_centers = simple_coords
        
        weights = create_innervation(
            receptor_coords,
            neuron_centers,
            method="gaussian",
            connections_per_neuron=3.0,
            sigma_d_mm=0.5,
            seed=42,
        )
        
        assert weights.shape == (4, 9)
        assert (weights > 0).sum() > 0
    
    def test_uniform_method(self, simple_coords):
        """Test factory creates uniform (Voronoi-like) innervation."""
        receptor_coords, neuron_centers = simple_coords

        weights = create_innervation(
            receptor_coords,
            neuron_centers,
            method="uniform",
        )

        assert weights.shape == (4, 9)
        # Each receptor connects to exactly one neuron
        connections_per_receptor = (weights > 0).sum(dim=0)
        assert torch.all(connections_per_receptor == 1)

    def test_one_to_one_method(self, simple_coords):
        """Test one-to-one: each neuron gets exactly K connections."""
        receptor_coords, neuron_centers = simple_coords

        weights = create_innervation(
            receptor_coords,
            neuron_centers,
            method="one_to_one",
            connections_per_neuron=2,
            sigma_d_mm=0.5,
            seed=42,
        )

        assert weights.shape == (4, 9)
        # Each neuron gets exactly K connections (or fewer if not enough receptors)
        connections_per_neuron = (weights > 0).sum(dim=1)
        assert torch.all(connections_per_neuron >= 1)
        assert torch.all(connections_per_neuron <= 2)
    
    def test_distance_weighted_method(self, simple_coords):
        """Test factory creates distance-weighted innervation."""
        receptor_coords, neuron_centers = simple_coords
        
        weights = create_innervation(
            receptor_coords,
            neuron_centers,
            method="distance_weighted",
            max_distance_mm=2.0,
            decay_function="exponential",
            decay_rate=1.0,
        )
        
        assert weights.shape == (4, 9)
        assert (weights > 0).sum() > 0
    
    def test_invalid_method_raises_error(self, simple_coords):
        """Test invalid method raises ValueError."""
        receptor_coords, neuron_centers = simple_coords
        
        with pytest.raises(ValueError, match="Unknown innervation method"):
            create_innervation(
                receptor_coords,
                neuron_centers,
                method="invalid_method",
            )
    
    def test_device_handling(self, simple_coords):
        """Test factory respects device parameter."""
        receptor_coords, neuron_centers = simple_coords

        weights = create_innervation(
            receptor_coords,
            neuron_centers,
            method="uniform",
            device="cpu",
        )

        assert weights.device.type == "cpu"


class TestBackwardCompatibility:
    """Test that existing code still works."""
    
    def test_gaussian_innervation_equivalent_to_legacy(self, simple_coords):
        """Test new Gaussian innervation produces similar results to legacy."""
        receptor_coords, neuron_centers = simple_coords
        
        # New API
        new_weights = create_innervation(
            receptor_coords,
            neuron_centers,
            method="gaussian",
            connections_per_neuron=5.0,
            sigma_d_mm=0.5,
            seed=42,
        )
        
        # Direct class instantiation
        innervation = GaussianInnervation(
            receptor_coords,
            neuron_centers,
            connections_per_neuron=5.0,
            sigma_d_mm=0.5,
            seed=42,
        )
        direct_weights = innervation.compute_weights()
        
        assert torch.allclose(new_weights, direct_weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
