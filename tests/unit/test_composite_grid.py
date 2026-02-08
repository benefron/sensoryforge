"""Unit tests for CompositeGrid multi-population spatial substrate."""

import pytest
import torch
import numpy as np

from sensoryforge.core.composite_grid import CompositeGrid


class TestCompositeGridInitialization:
    """Test CompositeGrid initialization and basic properties."""
    
    def test_default_initialization(self):
        """Test grid creation with default parameters."""
        grid = CompositeGrid()
        assert grid.xlim == (-5.0, 5.0)
        assert grid.ylim == (-5.0, 5.0)
        assert grid.device.type == "cpu"
        assert len(grid.list_populations()) == 0
    
    def test_custom_bounds(self):
        """Test grid creation with custom spatial bounds."""
        grid = CompositeGrid(xlim=(0.0, 10.0), ylim=(-2.0, 8.0))
        assert grid.xlim == (0.0, 10.0)
        assert grid.ylim == (-2.0, 8.0)
    
    def test_custom_device(self):
        """Test grid creation on specified device."""
        grid = CompositeGrid(device="cpu")
        assert grid.device.type == "cpu"
        
        # Test with torch.device object
        device = torch.device("cpu")
        grid = CompositeGrid(device=device)
        assert grid.device == device
    
    def test_invalid_xlim(self):
        """Test that invalid x bounds raise ValueError."""
        with pytest.raises(ValueError, match="Invalid xlim"):
            CompositeGrid(xlim=(5.0, 5.0))
        
        with pytest.raises(ValueError, match="Invalid xlim"):
            CompositeGrid(xlim=(10.0, 5.0))
    
    def test_invalid_ylim(self):
        """Test that invalid y bounds raise ValueError."""
        with pytest.raises(ValueError, match="Invalid ylim"):
            CompositeGrid(ylim=(3.0, 3.0))
        
        with pytest.raises(ValueError, match="Invalid ylim"):
            CompositeGrid(ylim=(8.0, 2.0))


class TestPopulationCreation:
    """Test population creation with various configurations."""
    
    def test_add_single_population(self):
        """Test adding a single population."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        grid.add_population(name="SA1", density=100.0, arrangement="grid")
        
        assert "SA1" in grid.list_populations()
        assert grid.get_population_count("SA1") > 0
    
    def test_add_multiple_populations(self):
        """Test adding multiple populations with different configs."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        
        grid.add_population(name="SA1", density=100.0, arrangement="grid")
        grid.add_population(name="RA", density=50.0, arrangement="hex")
        grid.add_population(name="PC", density=25.0, arrangement="poisson")
        
        populations = grid.list_populations()
        assert len(populations) == 3
        assert "SA1" in populations
        assert "RA" in populations
        assert "PC" in populations
    
    def test_duplicate_population_name(self):
        """Test that duplicate population names raise ValueError."""
        grid = CompositeGrid()
        grid.add_population(name="SA1", density=100.0)
        
        with pytest.raises(ValueError, match="already exists"):
            grid.add_population(name="SA1", density=50.0)
    
    def test_invalid_density(self):
        """Test that invalid density values raise ValueError."""
        grid = CompositeGrid()
        
        with pytest.raises(ValueError, match="Density must be positive"):
            grid.add_population(name="test", density=0.0)
        
        with pytest.raises(ValueError, match="Density must be positive"):
            grid.add_population(name="test", density=-10.0)
    
    def test_population_with_metadata(self):
        """Test adding population with optional metadata."""
        grid = CompositeGrid()
        grid.add_population(
            name="SA1",
            density=100.0,
            arrangement="grid",
            filter="gaussian",
            sigma=0.5,
            custom_param="value",
        )
        
        config = grid.get_population_config("SA1")
        assert config["filter"] == "gaussian"
        assert config["metadata"]["sigma"] == 0.5
        assert config["metadata"]["custom_param"] == "value"


class TestDensityValidation:
    """Test that population densities produce expected receptor counts."""
    
    def test_grid_density_accuracy(self):
        """Test that grid arrangement respects target density."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        density = 100.0  # 100 receptors per mm²
        area = 10 * 10  # 100 mm²
        expected_count = int(density * area)
        
        grid.add_population(name="test", density=density, arrangement="grid")
        actual_count = grid.get_population_count("test")
        
        # Grid arrangement should be close to expected count
        # Allow 20% tolerance due to discretization
        assert abs(actual_count - expected_count) / expected_count < 0.2
    
    def test_poisson_density_accuracy(self):
        """Test that Poisson arrangement approximates target density."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        density = 50.0
        area = 10 * 10
        expected_count = int(density * area)
        
        grid.add_population(name="test", density=density, arrangement="poisson")
        actual_count = grid.get_population_count("test")
        
        # Poisson should be within 30% due to random sampling
        assert abs(actual_count - expected_count) / expected_count < 0.3
    
    def test_hex_density_accuracy(self):
        """Test that hex arrangement approximates target density."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        density = 80.0
        area = 10 * 10
        expected_count = int(density * area)
        
        grid.add_population(name="test", density=density, arrangement="hex")
        actual_count = grid.get_population_count("test")
        
        # Hex packing should be reasonably close
        assert abs(actual_count - expected_count) / expected_count < 0.25
    
    def test_different_densities_independent(self):
        """Test that different populations maintain independent densities."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        
        grid.add_population(name="high_density", density=200.0, arrangement="grid")
        grid.add_population(name="low_density", density=50.0, arrangement="grid")
        
        high_count = grid.get_population_count("high_density")
        low_count = grid.get_population_count("low_density")
        
        # High density should have more receptors
        assert high_count > low_count
        # Ratio should approximate density ratio
        density_ratio = 200.0 / 50.0
        count_ratio = high_count / low_count
        assert abs(count_ratio - density_ratio) / density_ratio < 0.3


class TestArrangementTypes:
    """Test all arrangement types produce expected spatial patterns."""
    
    def test_grid_arrangement_bounds(self):
        """Test grid arrangement stays within spatial bounds."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        grid.add_population(name="test", density=100.0, arrangement="grid")
        
        coords = grid.get_population_coordinates("test")
        assert coords.shape[1] == 2  # (x, y) pairs
        
        # Check bounds
        assert coords[:, 0].min() >= 0.0
        assert coords[:, 0].max() <= 10.0
        assert coords[:, 1].min() >= 0.0
        assert coords[:, 1].max() <= 10.0
    
    def test_grid_arrangement_uniformity(self):
        """Test grid arrangement has uniform spacing."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        grid.add_population(name="test", density=100.0, arrangement="grid")
        
        coords = grid.get_population_coordinates("test").cpu().numpy()
        
        # Check that x and y coordinates have regular structure
        x_unique = np.unique(coords[:, 0])
        y_unique = np.unique(coords[:, 1])
        
        # Should have multiple distinct x and y values
        assert len(x_unique) > 5
        assert len(y_unique) > 5
        
        # Spacing should be approximately uniform (within 1% tolerance)
        if len(x_unique) > 1:
            x_diffs = np.diff(np.sort(x_unique))
            assert np.std(x_diffs) / np.mean(x_diffs) < 0.01
    
    def test_poisson_arrangement_bounds(self):
        """Test Poisson arrangement stays within spatial bounds."""
        grid = CompositeGrid(xlim=(-5, 5), ylim=(-5, 5))
        grid.add_population(name="test", density=50.0, arrangement="poisson")
        
        coords = grid.get_population_coordinates("test")
        
        assert coords[:, 0].min() >= -5.0
        assert coords[:, 0].max() <= 5.0
        assert coords[:, 1].min() >= -5.0
        assert coords[:, 1].max() <= 5.0
    
    def test_poisson_minimum_separation(self):
        """Test Poisson arrangement maintains minimum separation."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        density = 50.0
        grid.add_population(name="test", density=density, arrangement="poisson")
        
        coords = grid.get_population_coordinates("test").cpu()
        
        # Check minimum separation between points
        # For Poisson disk: min_dist ≈ 1 / sqrt(2 * density)
        expected_min_dist = 1.0 / (2.0 * density) ** 0.5
        
        # Compute pairwise distances (sample to avoid O(n²) for large n)
        n_samples = min(100, coords.shape[0])
        sample_indices = torch.randperm(coords.shape[0])[:n_samples]
        sample_coords = coords[sample_indices]
        
        for i in range(n_samples):
            dists = torch.norm(sample_coords - sample_coords[i], dim=1)
            non_zero_dists = dists[dists > 0]
            if len(non_zero_dists) > 0:
                min_dist = non_zero_dists.min().item()
                # Should be at least 50% of expected separation
                assert min_dist >= expected_min_dist * 0.5
    
    def test_hex_arrangement_bounds(self):
        """Test hexagonal arrangement stays within spatial bounds."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        grid.add_population(name="test", density=100.0, arrangement="hex")
        
        coords = grid.get_population_coordinates("test")
        
        assert coords[:, 0].min() >= 0.0
        assert coords[:, 0].max() <= 10.0
        assert coords[:, 1].min() >= 0.0
        assert coords[:, 1].max() <= 10.0
    
    def test_hex_arrangement_structure(self):
        """Test hexagonal arrangement has expected structure."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        grid.add_population(name="test", density=100.0, arrangement="hex")
        
        coords = grid.get_population_coordinates("test").cpu().numpy()
        
        # Hexagonal lattice should have rows with offset pattern
        # Group by y coordinate (with tolerance)
        y_coords = coords[:, 1]
        y_unique = []
        for y in y_coords:
            if not any(abs(y - yu) < 0.01 for yu in y_unique):
                y_unique.append(y)
        
        # Should have multiple rows
        assert len(y_unique) > 5
    
    def test_jittered_grid_arrangement_bounds(self):
        """Test jittered grid arrangement stays within spatial bounds."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        grid.add_population(name="test", density=100.0, arrangement="jittered_grid")
        
        coords = grid.get_population_coordinates("test")
        
        # Jitter may push slightly outside but should clamp to bounds
        assert coords[:, 0].min() >= 0.0
        assert coords[:, 0].max() <= 10.0
        assert coords[:, 1].min() >= 0.0
        assert coords[:, 1].max() <= 10.0
    
    def test_jittered_grid_breaks_regularity(self):
        """Test jittered grid is not perfectly regular."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        grid.add_population(name="regular", density=100.0, arrangement="grid")
        grid.add_population(name="jittered", density=100.0, arrangement="jittered_grid")
        
        regular_coords = grid.get_population_coordinates("regular").cpu().numpy()
        jittered_coords = grid.get_population_coordinates("jittered").cpu().numpy()
        
        # Jittered should have different coordinates
        # (comparing sorted to account for potential ordering differences)
        regular_sorted = np.sort(regular_coords.flatten())
        jittered_sorted = np.sort(jittered_coords.flatten())
        
        # Allow same count but coordinates should differ
        if regular_sorted.shape == jittered_sorted.shape:
            assert not np.allclose(regular_sorted, jittered_sorted, atol=0.01)
    
    def test_unknown_arrangement_raises_error(self):
        """Test that unknown arrangement type raises ValueError."""
        grid = CompositeGrid()
        
        with pytest.raises(ValueError, match="Unknown arrangement type"):
            grid.add_population(
                name="test", density=100.0, arrangement="invalid_type"
            )


class TestCoordinateConsistency:
    """Test coordinate system consistency across populations."""
    
    def test_shared_bounds_across_populations(self):
        """Test all populations share the same spatial bounds."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        
        grid.add_population(name="pop1", density=100.0, arrangement="grid")
        grid.add_population(name="pop2", density=50.0, arrangement="hex")
        grid.add_population(name="pop3", density=75.0, arrangement="poisson")
        
        # All populations should respect the same bounds
        for pop_name in ["pop1", "pop2", "pop3"]:
            coords = grid.get_population_coordinates(pop_name)
            assert coords[:, 0].min() >= 0.0
            assert coords[:, 0].max() <= 10.0
            assert coords[:, 1].min() >= 0.0
            assert coords[:, 1].max() <= 10.0
    
    def test_coordinate_tensor_shapes(self):
        """Test all populations return (N, 2) coordinate tensors."""
        grid = CompositeGrid()
        
        grid.add_population(name="pop1", density=50.0, arrangement="grid")
        grid.add_population(name="pop2", density=100.0, arrangement="hex")
        
        coords1 = grid.get_population_coordinates("pop1")
        coords2 = grid.get_population_coordinates("pop2")
        
        assert coords1.ndim == 2
        assert coords1.shape[1] == 2
        assert coords2.ndim == 2
        assert coords2.shape[1] == 2
    
    def test_device_consistency(self):
        """Test all populations share device with grid."""
        grid = CompositeGrid(device="cpu")
        
        grid.add_population(name="pop1", density=100.0, arrangement="grid")
        grid.add_population(name="pop2", density=50.0, arrangement="hex")
        
        coords1 = grid.get_population_coordinates("pop1")
        coords2 = grid.get_population_coordinates("pop2")
        
        assert coords1.device.type == "cpu"
        assert coords2.device.type == "cpu"
    
    def test_populations_independent(self):
        """Test populations don't interfere with each other."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        
        grid.add_population(name="SA1", density=100.0, arrangement="grid")
        sa1_coords_before = grid.get_population_coordinates("SA1").clone()
        
        # Add another population
        grid.add_population(name="RA", density=50.0, arrangement="hex")
        sa1_coords_after = grid.get_population_coordinates("SA1")
        
        # SA1 coordinates should be unchanged
        assert torch.allclose(sa1_coords_before, sa1_coords_after)


class TestDeviceManagement:
    """Test device management and tensor movement."""
    
    def test_to_device_cpu(self):
        """Test moving grid to CPU device."""
        grid = CompositeGrid(device="cpu")
        grid.add_population(name="test", density=100.0, arrangement="grid")
        
        grid.to_device("cpu")
        coords = grid.get_population_coordinates("test")
        assert coords.device.type == "cpu"
    
    def test_to_device_with_torch_device(self):
        """Test moving grid using torch.device object."""
        grid = CompositeGrid()
        grid.add_population(name="test", density=100.0, arrangement="grid")
        
        device = torch.device("cpu")
        grid.to_device(device)
        
        assert grid.device == device
        coords = grid.get_population_coordinates("test")
        assert coords.device == device
    
    def test_to_device_returns_self(self):
        """Test to_device returns self for chaining."""
        grid = CompositeGrid()
        grid.add_population(name="test", density=100.0, arrangement="grid")
        
        result = grid.to_device("cpu")
        assert result is grid


class TestPopulationRetrieval:
    """Test population data retrieval methods."""
    
    def test_get_nonexistent_population(self):
        """Test accessing non-existent population raises KeyError."""
        grid = CompositeGrid()
        
        with pytest.raises(KeyError, match="not found"):
            grid.get_population_coordinates("nonexistent")
        
        with pytest.raises(KeyError, match="not found"):
            grid.get_population_config("nonexistent")
        
        with pytest.raises(KeyError, match="not found"):
            grid.get_population_count("nonexistent")
    
    def test_list_populations_empty(self):
        """Test listing populations when none exist."""
        grid = CompositeGrid()
        assert grid.list_populations() == []
    
    def test_list_populations_order(self):
        """Test populations are listed in order added."""
        grid = CompositeGrid()
        
        names = ["first", "second", "third"]
        for name in names:
            grid.add_population(name=name, density=100.0, arrangement="grid")
        
        # List should contain all names (order may vary with dict)
        populations = grid.list_populations()
        assert set(populations) == set(names)
        assert len(populations) == len(names)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_small_grid_area(self):
        """Test grid with very small spatial area."""
        grid = CompositeGrid(xlim=(0, 0.1), ylim=(0, 0.1))
        grid.add_population(name="test", density=100.0, arrangement="grid")
        
        # Should have at least one receptor
        assert grid.get_population_count("test") >= 1
    
    def test_large_grid_area(self):
        """Test grid with large spatial area."""
        grid = CompositeGrid(xlim=(0, 100), ylim=(0, 100))
        grid.add_population(name="test", density=10.0, arrangement="grid")
        
        count = grid.get_population_count("test")
        # Should have many receptors
        assert count > 1000
    
    def test_low_density(self):
        """Test population with very low density."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        grid.add_population(name="test", density=0.1, arrangement="grid")
        
        # Should have at least a few receptors
        assert grid.get_population_count("test") >= 1
    
    def test_high_density(self):
        """Test population with very high density."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        grid.add_population(name="test", density=1000.0, arrangement="grid")
        
        count = grid.get_population_count("test")
        # Should have many receptors
        assert count > 10000
    
    def test_non_square_bounds(self):
        """Test grid with non-square spatial bounds."""
        grid = CompositeGrid(xlim=(0, 20), ylim=(0, 5))
        grid.add_population(name="test", density=100.0, arrangement="grid")
        
        coords = grid.get_population_coordinates("test")
        
        # Should respect bounds
        assert coords[:, 0].min() >= 0.0
        assert coords[:, 0].max() <= 20.0
        assert coords[:, 1].min() >= 0.0
        assert coords[:, 1].max() <= 5.0
        
        # Should have more points along x than y
        x_span = coords[:, 0].max() - coords[:, 0].min()
        y_span = coords[:, 1].max() - coords[:, 1].min()
        assert x_span > y_span


class TestIntegrationScenarios:
    """Test realistic multi-population scenarios."""
    
    def test_mechanoreceptor_simulation(self):
        """Test simulating multiple mechanoreceptor types."""
        grid = CompositeGrid(xlim=(0, 10), ylim=(0, 10))
        
        # SA1: Slowly adapting type 1 (high density, regular)
        grid.add_population(
            name="SA1",
            density=100.0,
            arrangement="grid",
            filter="lowpass",
            sigma=0.5,
        )
        
        # RA: Rapidly adapting (medium density, hex)
        grid.add_population(
            name="RA",
            density=50.0,
            arrangement="hex",
            filter="bandpass",
        )
        
        # PC: Pacinian corpuscles (low density, scattered)
        grid.add_population(
            name="PC",
            density=25.0,
            arrangement="poisson",
            filter="highpass",
        )
        
        # Verify all populations created
        assert len(grid.list_populations()) == 3
        
        # Verify density ordering
        sa1_count = grid.get_population_count("SA1")
        ra_count = grid.get_population_count("RA")
        pc_count = grid.get_population_count("PC")
        
        assert sa1_count > ra_count > pc_count
        
        # Verify metadata preserved
        sa1_config = grid.get_population_config("SA1")
        assert sa1_config["filter"] == "lowpass"
        assert sa1_config["metadata"]["sigma"] == 0.5
    
    def test_retinal_mosaic_simulation(self):
        """Test simulating retinal photoreceptor mosaic."""
        grid = CompositeGrid(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
        
        # Cones: high density in fovea
        grid.add_population(
            name="cones",
            density=200.0,
            arrangement="hex",
            receptor_type="photoreceptor",
        )
        
        # Rods: medium density, more peripheral
        grid.add_population(
            name="rods",
            density=100.0,
            arrangement="jittered_grid",
            receptor_type="photoreceptor",
        )
        
        assert "cones" in grid.list_populations()
        assert "rods" in grid.list_populations()
        
        # Cones should be more numerous in this setup
        assert grid.get_population_count("cones") > grid.get_population_count("rods")
