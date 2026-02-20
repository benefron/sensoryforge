"""Integration tests for GUI-CLI parity after refactoring.

These tests ensure that configurations created in the GUI can be loaded and executed
via CLI, and that results are consistent across execution paths.
"""

import pytest
import torch
import tempfile
import yaml
from pathlib import Path

from sensoryforge.config.schema import SensoryForgeConfig, GridConfig, PopulationConfig, StimulusConfig, SimulationConfig
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.core.simulation_engine import SimulationEngine
# GUI imports are optional - only test if available
try:
    from sensoryforge.gui.main import MainWindow
    from PyQt5.QtWidgets import QApplication
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
import sys


class TestGUICLIParity:
    """Test that GUI configs work with CLI execution."""
    
    def test_canonical_config_round_trip(self):
        """Test that canonical config can be saved and loaded."""
        config = SensoryForgeConfig(
            grids=[
                GridConfig(
                    name="Test Grid",
                    arrangement="grid",
                    rows=20,
                    cols=20,
                    spacing=0.15,
                )
            ],
            populations=[
                PopulationConfig(
                    name="SA Population",
                    neuron_type="SA",
                    neuron_model="izhikevich",
                    filter_method="sa",
                    innervation_method="gaussian",
                    neurons_per_row=10,
                    connections_per_neuron=28,
                    sigma_d_mm=0.3,
                )
            ],
            stimulus=StimulusConfig(
                type="gaussian",
                amplitude=10.0,
                sigma=1.0,
            ),
            simulation=SimulationConfig(
                device="cpu",
                dt=1.0,
            ),
        )
        
        # Save to YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml_path = f.name
            f.write(config.to_yaml())
        
        try:
            # Load from YAML
            config2 = SensoryForgeConfig.from_yaml(yaml_path)
            
            # Verify round-trip
            assert len(config2.grids) == len(config.grids)
            assert config2.grids[0].name == config.grids[0].name
            assert len(config2.populations) == len(config.populations)
            assert config2.populations[0].name == config.populations[0].name
        finally:
            Path(yaml_path).unlink()
    
    def test_pipeline_accepts_canonical_config(self):
        """Test that pipeline can accept canonical config via adapter."""
        config = SensoryForgeConfig(
            grids=[
                GridConfig(
                    name="Test Grid",
                    arrangement="grid",
                    rows=20,
                    cols=20,
                    spacing=0.15,
                )
            ],
            populations=[
                PopulationConfig(
                    name="SA Population",
                    neuron_type="SA",
                    neuron_model="izhikevich",
                    filter_method="sa",
                    innervation_method="gaussian",
                    neurons_per_row=10,
                    connections_per_neuron=28,
                    sigma_d_mm=0.3,
                )
            ],
            stimulus=StimulusConfig(
                type="gaussian",
                amplitude=10.0,
            ),
            simulation=SimulationConfig(
                device="cpu",
                dt=1.0,
            ),
        )
        
        # Convert to dict and pass to pipeline
        config_dict = config.to_dict()
        pipeline = GeneralizedTactileEncodingPipeline(config_dict=config_dict)
        
        # Verify pipeline was created
        assert pipeline is not None
        assert hasattr(pipeline, 'sa_innervation')
    
    def test_simulation_engine_accepts_canonical_config(self):
        """Test that SimulationEngine accepts canonical config."""
        config = SensoryForgeConfig(
            grids=[
                GridConfig(
                    name="Test Grid",
                    arrangement="grid",
                    rows=20,
                    cols=20,
                    spacing=0.15,
                )
            ],
            populations=[
                PopulationConfig(
                    name="SA Population",
                    neuron_type="SA",
                    neuron_model="izhikevich",
                    filter_method="sa",
                    innervation_method="gaussian",
                    neurons_per_row=10,
                    connections_per_neuron=28,
                    sigma_d_mm=0.3,
                )
            ],
            stimulus=StimulusConfig(
                type="gaussian",
                amplitude=10.0,
            ),
            simulation=SimulationConfig(
                device="cpu",
                dt=1.0,
            ),
        )
        
        engine = SimulationEngine(config)
        
        # Verify engine was created
        assert engine is not None
        assert len(engine.grids) > 0
        assert len(engine.populations) > 0
    
    def test_registry_components_accessible(self):
        """Test that all registered components are accessible."""
        from sensoryforge.register_components import register_all
        from sensoryforge.registry import (
            NEURON_REGISTRY,
            FILTER_REGISTRY,
            INNERVATION_REGISTRY,
        )
        
        register_all()
        
        # Check neurons
        assert NEURON_REGISTRY.is_registered("izhikevich")
        assert NEURON_REGISTRY.is_registered("adex")
        assert NEURON_REGISTRY.is_registered("mqif")
        
        # Check filters
        assert FILTER_REGISTRY.is_registered("sa")
        assert FILTER_REGISTRY.is_registered("ra")
        
        # Check innervation
        assert INNERVATION_REGISTRY.is_registered("gaussian")
        assert INNERVATION_REGISTRY.is_registered("uniform")
        assert INNERVATION_REGISTRY.is_registered("one_to_one")


class TestConfigAdapter:
    """Test config adapter between GUI and pipeline."""
    
    def test_legacy_config_still_works(self):
        """Test that legacy config format still works."""
        legacy_config = {
            "pipeline": {
                "device": "cpu",
                "grid_size": 80,
                "spacing": 0.15,
                "center": [0.0, 0.0],
            },
            "neurons": {
                "sa_neurons": 100,
                "ra_neurons": 196,
                "dt": 0.5,
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
        assert pipeline is not None
    
    def test_canonical_to_legacy_conversion(self):
        """Test that canonical config converts to legacy format."""
        canonical = {
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
                    "num_neurons": 100,
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
        
        pipeline = GeneralizedTactileEncodingPipeline(config_dict=canonical)
        
        # Verify pipeline was created (adapter should convert)
        assert pipeline is not None


class TestRegistryIntegration:
    """Test that registry-based component creation works end-to-end."""
    
    def test_neuron_creation_via_registry(self):
        """Test creating neurons via registry."""
        from sensoryforge.register_components import register_all
        from sensoryforge.registry import NEURON_REGISTRY
        
        register_all()
        
        # Create neuron via registry
        neuron_cls = NEURON_REGISTRY.get_class("izhikevich")
        neuron = neuron_cls(dt=1.0)
        
        assert neuron is not None
        assert hasattr(neuron, 'forward')
    
    def test_filter_creation_via_registry(self):
        """Test creating filters via registry."""
        from sensoryforge.register_components import register_all
        from sensoryforge.registry import FILTER_REGISTRY
        
        register_all()
        
        # Create filter via registry
        filter_cls = FILTER_REGISTRY.get_class("sa")
        filter_module = filter_cls(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=1.0)
        
        assert filter_module is not None
        assert hasattr(filter_module, 'forward')
    
    def test_innervation_creation_via_registry(self):
        """Test creating innervation via registry."""
        from sensoryforge.register_components import register_all
        from sensoryforge.registry import INNERVATION_REGISTRY
        import torch
        
        register_all()
        
        # Create receptor and neuron coordinates
        receptor_coords = torch.randn(100, 2)
        neuron_centers = torch.randn(10, 2)
        
        # Create innervation via registry
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


class TestBackwardCompatibility:
    """Test that existing functionality still works after refactoring."""
    
    def test_pipeline_forward_pass(self):
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
        
        # Run forward pass
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
    
    def test_pipeline_with_intermediates(self):
        """Test that pipeline returns intermediates correctly."""
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
            return_intermediates=True,
        )
        
        # Verify intermediate results
        assert "sa_inputs" in results
        assert "ra_inputs" in results
        assert "sa_filtered" in results
        assert "ra_filtered" in results
        assert "sa_v_trace" in results
        assert "ra_v_trace" in results
