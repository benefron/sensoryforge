"""Integration tests for registry system.

Tests that registry-based component creation works correctly throughout
the codebase and that refactoring didn't break existing functionality.
"""

import pytest
import torch

from sensoryforge.register_components import register_all
from sensoryforge.registry import (
    NEURON_REGISTRY,
    FILTER_REGISTRY,
    INNERVATION_REGISTRY,
    STIMULUS_REGISTRY,
    SOLVER_REGISTRY,
)


class TestRegistryIntegration:
    """Test registry integration across codebase."""
    
    def test_all_components_registered(self):
        """Test that all expected components are registered."""
        register_all()
        
        # Neurons
        assert NEURON_REGISTRY.is_registered("izhikevich")
        assert NEURON_REGISTRY.is_registered("adex")
        assert NEURON_REGISTRY.is_registered("mqif")
        assert NEURON_REGISTRY.is_registered("fa")
        assert NEURON_REGISTRY.is_registered("sa")
        
        # Filters
        assert FILTER_REGISTRY.is_registered("sa")
        assert FILTER_REGISTRY.is_registered("ra")
        assert FILTER_REGISTRY.is_registered("none")
        
        # Innervation
        assert INNERVATION_REGISTRY.is_registered("gaussian")
        assert INNERVATION_REGISTRY.is_registered("uniform")
        assert INNERVATION_REGISTRY.is_registered("one_to_one")
        assert INNERVATION_REGISTRY.is_registered("distance_weighted")
    
    def test_neuron_creation_via_registry(self):
        """Test creating neurons via registry matches direct creation."""
        register_all()
        
        # Create via registry
        neuron_cls = NEURON_REGISTRY.get_class("izhikevich")
        neuron_registry = neuron_cls(dt=1.0, a=0.02, b=0.2, c=-65.0, d=8.0)
        
        # Create directly
        from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
        neuron_direct = IzhikevichNeuronTorch(dt=1.0, a=0.02, b=0.2, c=-65.0, d=8.0)
        
        # Both should be same type
        assert type(neuron_registry) == type(neuron_direct)
        
        # Both should produce similar outputs
        input_current = torch.randn(1, 100, 10)
        output_registry = neuron_registry(input_current)
        output_direct = neuron_direct(input_current)
        
        # Neuron models return tuples (v_trace, spikes)
        if isinstance(output_registry, tuple):
            assert isinstance(output_direct, tuple)
            assert output_registry[1].shape == output_direct[1].shape  # Compare spikes
        else:
            assert output_registry.shape == output_direct.shape
    
    def test_filter_creation_via_registry(self):
        """Test creating filters via registry matches direct creation."""
        register_all()
        
        # Create via registry
        filter_cls = FILTER_REGISTRY.get_class("sa")
        filter_registry = filter_cls(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=1.0)
        
        # Create directly
        from sensoryforge.filters.sa_ra import SAFilterTorch
        filter_direct = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=1.0)
        
        # Both should be same type
        assert type(filter_registry) == type(filter_direct)
        
        # Both should produce similar outputs
        input_current = torch.randn(1, 100, 10)
        output_registry = filter_registry(input_current)
        output_direct = filter_direct(input_current)
        
        assert output_registry.shape == output_direct.shape
    
    def test_innervation_creation_via_registry(self):
        """Test creating innervation via registry."""
        register_all()
        
        receptor_coords = torch.randn(100, 2)
        neuron_centers = torch.randn(10, 2)
        
        # Create via registry
        innervation = INNERVATION_REGISTRY.create(
            "gaussian",
            receptor_coords=receptor_coords,
            neuron_centers=neuron_centers,
            connections_per_neuron=28.0,
            sigma_d_mm=0.3,
            device="cpu",
        )
        
        assert innervation is not None
        assert hasattr(innervation, 'compute_weights')
        
        # Should be able to compute weights
        weights = innervation.compute_weights()
        assert weights.shape == (10, 100)  # [num_neurons, num_receptors]


class TestRegistryBackwardCompatibility:
    """Test that registry doesn't break backward compatibility."""
    
    def test_direct_imports_still_work(self):
        """Test that direct imports still work."""
        # These should all work without registry
        from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
        from sensoryforge.neurons.adex import AdExNeuronTorch
        from sensoryforge.filters.sa_ra import SAFilterTorch, RAFilterTorch
        from sensoryforge.core.innervation import GaussianInnervation
        
        # Should be able to instantiate directly
        neuron = IzhikevichNeuronTorch(dt=1.0)
        filter_module = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=1.0)
        
        assert neuron is not None
        assert filter_module is not None
    
    def test_registry_and_direct_are_compatible(self):
        """Test that registry-created and directly-created components are compatible."""
        register_all()
        
        # Create via registry
        neuron_cls = NEURON_REGISTRY.get_class("izhikevich")
        neuron_registry = neuron_cls(dt=1.0)
        
        # Create directly
        from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
        neuron_direct = IzhikevichNeuronTorch(dt=1.0)
        
        # Both should work identically
        input_current = torch.randn(1, 100, 10)
        output_registry = neuron_registry(input_current)
        output_direct = neuron_direct(input_current)
        
        # Neuron models return tuples (v_trace, spikes)
        if isinstance(output_registry, tuple):
            assert isinstance(output_direct, tuple)
            assert output_registry[1].shape == output_direct[1].shape  # Compare spikes
            assert output_registry[1].dtype == output_direct[1].dtype
        else:
            assert output_registry.shape == output_direct.shape
            assert output_registry.dtype == output_direct.dtype


class TestRegistryErrorHandling:
    """Test error handling in registry system."""
    
    def test_unregistered_component_raises_error(self):
        """Test that unregistered components raise KeyError."""
        register_all()
        
        with pytest.raises(KeyError):
            NEURON_REGISTRY.get_class("nonexistent_neuron")
        
        with pytest.raises(KeyError):
            FILTER_REGISTRY.get_class("nonexistent_filter")
        
        with pytest.raises(KeyError):
            INNERVATION_REGISTRY.get_class("nonexistent_innervation")
    
    def test_registry_list_registered(self):
        """Test that list_registered returns all registered components."""
        register_all()
        
        neurons = NEURON_REGISTRY.list_registered()
        filters = FILTER_REGISTRY.list_registered()
        innervations = INNERVATION_REGISTRY.list_registered()
        
        assert len(neurons) > 0
        assert len(filters) > 0
        assert len(innervations) > 0
        
        assert "izhikevich" in neurons
        assert "sa" in filters
        assert "gaussian" in innervations
