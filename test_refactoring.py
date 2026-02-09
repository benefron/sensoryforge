#!/usr/bin/env python
"""Quick test script for Phase 1 refactoring."""

from sensoryforge.core import (
    ReceptorGrid, GridManager,
    CompositeGrid, CompositeReceptorGrid,
    create_innervation,
)
import torch


def test_receptor_grid():
    """Test ReceptorGrid with different arrangements."""
    print("=" * 60)
    print("Testing ReceptorGrid (Phase 1.1)")
    print("=" * 60)
    
    # Test 1: Regular grid
    print("\n1. Regular grid arrangement")
    grid1 = ReceptorGrid(grid_size=10, spacing=0.15, arrangement='grid')
    coords1 = grid1.get_receptor_coordinates()
    print(f"   ✓ Shape: {coords1.shape}")
    assert coords1.shape == (100, 2), f"Expected (100, 2), got {coords1.shape}"
    
    # Test 2: Jittered grid
    print("\n2. Jittered grid arrangement")
    grid2 = ReceptorGrid(grid_size=10, spacing=0.15, arrangement='jittered_grid')
    coords2 = grid2.get_receptor_coordinates()
    print(f"   ✓ Shape: {coords2.shape}")
    assert coords2.shape == (100, 2), f"Expected (100, 2), got {coords2.shape}"
    
    # Test 3: Poisson arrangement
    print("\n3. Poisson arrangement")
    grid3 = ReceptorGrid(grid_size=20, spacing=0.15, arrangement='poisson', density=100.0)
    coords3 = grid3.get_receptor_coordinates()
    print(f"   ✓ Shape: {coords3.shape}")
    assert coords3.shape[1] == 2, f"Expected Nx2, got {coords3.shape}"
    print(f"   ✓ Generated {coords3.shape[0]} receptors")
    
    # Test 4: Hexagonal arrangement
    print("\n4. Hexagonal arrangement")
    grid4 = ReceptorGrid(grid_size=20, spacing=0.15, arrangement='hex', density=100.0)
    coords4 = grid4.get_receptor_coordinates()
    print(f"   ✓ Shape: {coords4.shape}")
    assert coords4.shape[1] == 2, f"Expected Nx2, got {coords4.shape}"
    print(f"   ✓ Generated {coords4.shape[0]} receptors")
    
    # Test 5: Backward compatibility with GridManager
    print("\n5. Backward compatibility (GridManager)")
    grid5 = GridManager(grid_size=10, spacing=0.15)
    coords5 = grid5.get_receptor_coordinates()
    print(f"   ✓ Type: {type(grid5).__name__}")
    print(f"   ✓ Shape: {coords5.shape}")
    assert type(grid5).__name__ == 'ReceptorGrid', "GridManager should be alias for ReceptorGrid"
    
    print("\n✅ All ReceptorGrid tests passed!")


def test_composite_receptor_grid():
    """Test new CompositeReceptorGrid API with add_layer()."""
    print("\n" + "=" * 60)
    print("Testing CompositeReceptorGrid (Phase 1.2 - New API)")
    print("=" * 60)
    
    # New API: CompositeReceptorGrid with add_layer()
    grid = CompositeReceptorGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
    
    print("\n1. Adding layers with add_layer() method")
    grid.add_layer("layer1", density=100.0, arrangement="grid")
    grid.add_layer("layer2", density=80.0, arrangement="hex")
    grid.add_layer("layer3", density=50.0, arrangement="poisson")
    
    print(f"   ✓ Created {len(grid.list_layers())} layers")
    
    for name in grid.list_layers():
        coords = grid.get_layer_coordinates(name)
        print(f"   - {name}: {coords.shape[0]} receptors")
    
    # Test new accessor methods
    print("\n2. Testing new accessor methods")
    layer1_coords = grid.get_layer_coordinates("layer1")
    layer1_config = grid.get_layer_config("layer1")
    layer1_count = grid.get_layer_count("layer1")
    
    print(f"   ✓ get_layer_coordinates(): {layer1_coords.shape}")
    print(f"   ✓ get_layer_config(): {layer1_config}")
    print(f"   ✓ get_layer_count(): {layer1_count}")
    
    # Verify filter is NOT in config
    print("\n3. Verifying filter parameter removed")
    assert "filter" not in layer1_config, "Filter should not be in layer config"
    print("   ✓ Filter parameter correctly removed from config")
    
    print("\n✅ CompositeReceptorGrid (new API) tests passed!")


def test_composite_grid_backward_compat():
    """Test backward compatibility with old CompositeGrid API."""
    print("\n" + "=" * 60)
    print("Testing Backward Compatibility (Phase 1.2)")
    print("=" * 60)
    
    # Old API: CompositeGrid with add_population()
    grid = CompositeGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
    
    print("\n1. Using legacy add_population() method")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid.add_population("pop1", density=100.0, arrangement="grid", filter="SA")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        print("   ✓ Deprecation warning raised for 'filter' parameter")
    
    print("\n2. Using legacy accessor methods")
    coords = grid.get_population_coordinates("pop1")
    config = grid.get_population_config("pop1")
    count = grid.get_population_count("pop1")
    pops = grid.list_populations()
    
    print(f"   ✓ get_population_coordinates(): {coords.shape}")
    print(f"   ✓ get_population_config(): works")
    print(f"   ✓ get_population_count(): {count}")
    print(f"   ✓ list_populations(): {pops}")
    
    print("\n3. Verifying CompositeGrid is alias for CompositeReceptorGrid")
    assert CompositeGrid is CompositeReceptorGrid
    print("   ✓ CompositeGrid === CompositeReceptorGrid")
    
    print("\n✅ Backward compatibility tests passed!")


def test_innervation_methods():
    """Test new innervation methods (Phase 1.3)."""
    print("\n" + "=" * 60)
    print("Testing Innervation Methods (Phase 1.3 - New API)")
    print("=" * 60)
    
    # Create test coordinates
    receptor_coords = torch.randn(100, 2) * 2.0  # 100 receptors
    neuron_centers = torch.randn(20, 2) * 2.0    # 20 neurons
    
    print("\n1. Gaussian innervation (refactored existing method)")
    weights_gaussian = create_innervation(
        receptor_coords,
        neuron_centers,
        method="gaussian",
        connections_per_neuron=10.0,
        sigma_d_mm=0.5,
        seed=42,
    )
    print(f"   ✓ Shape: {weights_gaussian.shape}")
    print(f"   ✓ Nonzero connections: {(weights_gaussian > 0).sum().item()}")
    assert weights_gaussian.shape == (20, 100)
    
    print("\n2. One-to-one innervation (new method)")
    weights_one_to_one = create_innervation(
        receptor_coords,
        neuron_centers,
        method="one_to_one",
    )
    print(f"   ✓ Shape: {weights_one_to_one.shape}")
    print(f"   ✓ Nonzero connections: {(weights_one_to_one > 0).sum().item()}")
    # Each receptor connects to exactly one neuron
    connections_per_receptor = (weights_one_to_one > 0).sum(dim=0)
    assert torch.all(connections_per_receptor == 1), "Each receptor should connect to 1 neuron"
    print("   ✓ Each receptor connects to exactly 1 neuron")
    
    print("\n3. Distance-weighted innervation (new method)")
    weights_distance = create_innervation(
        receptor_coords,
        neuron_centers,
        method="distance_weighted",
        max_distance_mm=2.0,
        decay_function="exponential",
        decay_rate=1.5,
    )
    print(f"   ✓ Shape: {weights_distance.shape}")
    print(f"   ✓ Nonzero connections: {(weights_distance > 0).sum().item()}")
    assert weights_distance.shape == (20, 100)
    
    print("\n4. Testing different decay functions")
    for decay_fn in ["exponential", "linear", "inverse_square"]:
        weights = create_innervation(
            receptor_coords,
            neuron_centers,
            method="distance_weighted",
            max_distance_mm=2.0,
            decay_function=decay_fn,
            decay_rate=1.0,
        )
        print(f"   ✓ {decay_fn}: {(weights > 0).sum().item()} connections")
    
    print("\n✅ Innervation methods tests passed!")


if __name__ == "__main__":
    test_receptor_grid()
    test_composite_receptor_grid()
    test_composite_grid_backward_compat()
    test_innervation_methods()
    print("\n" + "=" * 60)
    print("🎉 Phase 1.1, 1.2 & 1.3 Implementation Verified!")
    print("=" * 60)
