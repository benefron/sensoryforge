"""Regression tests for refactoring changes.

These tests ensure that the major refactoring (registry system, canonical config,
SimulationEngine) didn't break existing functionality.
"""

import pytest
import torch

from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.register_components import register_all
from sensoryforge.registry import NEURON_REGISTRY, FILTER_REGISTRY


class TestPipelineBackwardCompatibility:
    """Test that existing pipeline usage still works."""
    
    def test_legacy_config_format_still_works(self):
        """Test that legacy config format still works."""
        legacy_config = {
            "pipeline": {
                "device": "cpu",
                "grid_size": 40,
                "spacing": 0.15,
                "center": [0.0, 0.0],
            },
            "neurons": {
                "sa_neurons": 10,
                "ra_neurons": 14,
                "dt": 1.0,
            },
            "innervation": {
                "receptors_per_neuron": 28,
                "sa_spread": 0.3,
                "ra_spread": 0.39,
            },
            "filters": {
                "sa_tau_r": 5.0,
                "sa_tau_d": 30.0,
                "ra_tau_ra": 30.0,
            },
        }
        
        pipeline = GeneralizedTactileEncodingPipeline(config_dict=legacy_config)
        
        # Verify pipeline was created
        assert pipeline is not None
        assert hasattr(pipeline, 'sa_innervation')
        assert hasattr(pipeline, 'ra_innervation')
        assert hasattr(pipeline, 'sa_filter')
        assert hasattr(pipeline, 'ra_filter')
    
    def test_pipeline_forward_pass_still_works(self):
        """Test that pipeline forward pass still works."""
        config = {
            "pipeline": {
                "device": "cpu",
                "grid_size": 40,
                "spacing": 0.15,
            },
            "neurons": {
                "sa_neurons": 10,
                "ra_neurons": 14,
                "dt": 1.0,
            },
        }
        
        pipeline = GeneralizedTactileEncodingPipeline(config_dict=config)
        
        results = pipeline.forward(
            stimulus_type="gaussian",
            amplitude=10.0,
            sigma=1.0,
            duration=100.0,
        )
        
        # Verify results structure
        assert "sa_spikes" in results
        assert "ra_spikes" in results
        assert isinstance(results["sa_spikes"], torch.Tensor)
        assert isinstance(results["ra_spikes"], torch.Tensor)
        assert results["sa_spikes"].shape[1] == 100  # time dimension
        assert results["ra_spikes"].shape[1] == 100


class TestRegistryBackwardCompatibility:
    """Test that registry doesn't break direct imports."""
    
    def test_direct_neuron_imports_still_work(self):
        """Test that direct neuron imports still work."""
        from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
        from sensoryforge.neurons.adex import AdExNeuronTorch
        
        neuron1 = IzhikevichNeuronTorch(dt=1.0)
        neuron2 = AdExNeuronTorch(dt=1.0)
        
        assert neuron1 is not None
        assert neuron2 is not None
        
        # Both should work
        input_current = torch.randn(1, 100, 10)
        output1 = neuron1(input_current)
        output2 = neuron2(input_current)
        
        assert isinstance(output1, tuple)  # (v_trace, spikes)
        assert isinstance(output2, tuple)
    
    def test_direct_filter_imports_still_work(self):
        """Test that direct filter imports still work."""
        from sensoryforge.filters.sa_ra import SAFilterTorch, RAFilterTorch
        
        sa_filter = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=1.0)
        ra_filter = RAFilterTorch(tau_RA=30.0, k3=2.0, dt=1.0)
        
        assert sa_filter is not None
        assert ra_filter is not None
        
        # Both should work
        input_current = torch.randn(1, 100, 10)
        output_sa = sa_filter(input_current)
        output_ra = ra_filter(input_current)
        
        assert output_sa.shape == input_current.shape
        assert output_ra.shape == input_current.shape


class TestRegistryVsDirectEquivalence:
    """Test that registry-created components match direct creation."""
    
    def test_neuron_registry_matches_direct(self):
        """Test that registry-created neurons match direct creation."""
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
        
        # Compare spikes (second element of tuple)
        assert output_registry[1].shape == output_direct[1].shape
    
    def test_filter_registry_matches_direct(self):
        """Test that registry-created filters match direct creation."""
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


class TestCanonicalConfigAdapter:
    """Test that canonical config adapter works correctly."""
    
    def test_canonical_config_loads_via_adapter(self):
        """Test that canonical config loads via adapter."""
        canonical_config = {
            "grids": [
                {
                    "name": "Grid 1",
                    "arrangement": "grid",
                    "rows": 40,
                    "cols": 40,
                    "spacing": 0.15,
                }
            ],
            "populations": [
                {
                    "name": "SA Population",
                    "neuron_type": "SA",
                    "neuron_model": "izhikevich",
                    "filter_method": "sa",
                    "innervation_method": "gaussian",
                    "neurons_per_row": 10,
                    "connections_per_neuron": 28,
                    "sigma_d_mm": 0.3,
                }
            ],
            "stimulus": {
                "type": "gaussian",
                "amplitude": 10.0,
            },
            "simulation": {
                "device": "cpu",
                "dt": 1.0,
            },
        }
        
        # Pipeline should accept canonical config via adapter
        pipeline = GeneralizedTactileEncodingPipeline(config_dict=canonical_config)
        
        assert pipeline is not None
        assert hasattr(pipeline, 'sa_innervation')


class TestComponentCreation:
    """Test that component creation works via registry."""
    
    def test_all_registered_components_creatable(self):
        """Test that all registered components can be created."""
        register_all()
        
        # Test neurons
        neuron_names = ["izhikevich", "adex", "mqif", "fa", "sa"]
        for name in neuron_names:
            neuron_cls = NEURON_REGISTRY.get_class(name)
            neuron = neuron_cls(dt=1.0)
            assert neuron is not None
        
        # Test filters
        filter_names = ["sa", "ra"]
        for name in filter_names:
            filter_cls = FILTER_REGISTRY.get_class(name)
            if name == "sa":
                filter_module = filter_cls(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=1.0)
            else:
                filter_module = filter_cls(tau_RA=30.0, k3=2.0, dt=1.0)
            assert filter_module is not None
