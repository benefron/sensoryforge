"""Integration tests for SimulationEngine.

Tests the unified simulation execution engine to ensure it works correctly
with canonical configs and produces consistent results.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from sensoryforge.config.schema import (
    SensoryForgeConfig,
    GridConfig,
    PopulationConfig,
    StimulusConfig,
    SimulationConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine


class TestSimulationEngineBasic:
    """Basic functionality tests for SimulationEngine."""
    
    def test_engine_initialization(self):
        """Test that engine initializes correctly."""
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
        
        assert engine is not None
        assert len(engine.grids) == 1
        assert len(engine.populations) == 1
        assert engine.populations[0]["name"] == "SA Population"
    
    def test_engine_with_multiple_populations(self):
        """Test engine with multiple populations."""
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
                ),
                PopulationConfig(
                    name="RA Population",
                    neuron_type="RA",
                    neuron_model="izhikevich",
                    filter_method="ra",
                    innervation_method="gaussian",
                    neurons_per_row=14,
                ),
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
        
        assert len(engine.populations) == 2
        assert engine.populations[0]["name"] == "SA Population"
        assert engine.populations[1]["name"] == "RA Population"
    
    def test_engine_run_basic(self):
        """Test basic simulation run."""
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
        
        # Create simple stimulus [batch, time, height, width] or [time, height, width]
        # SimulationEngine will add batch dimension if needed
        stimulus = torch.randn(100, 20, 20)  # [time, height, width]
        
        results = engine.run(stimulus, return_intermediates=False)
        
        assert "SA Population" in results
        assert "spikes" in results["SA Population"]
        assert isinstance(results["SA Population"]["spikes"], torch.Tensor)
    
    def test_engine_run_with_intermediates(self):
        """Test simulation run with intermediate results."""
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
        
        stimulus = torch.randn(100, 20, 20)
        results = engine.run(stimulus, return_intermediates=True)
        
        assert "SA Population" in results
        pop_results = results["SA Population"]
        assert "spikes" in pop_results
        assert "drive" in pop_results
        assert "filtered" in pop_results


class TestSimulationEngineNeuronArrangements:
    """Test different neuron arrangements."""
    
    @pytest.mark.parametrize("arrangement", ["grid", "poisson", "hex", "jittered_grid", "blue_noise"])
    def test_neuron_arrangements(self, arrangement):
        """Test that different neuron arrangements work."""
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
                    name="Test Population",
                    neuron_type="SA",
                    neuron_model="izhikevich",
                    filter_method="none",
                    innervation_method="gaussian",
                    neurons_per_row=10,
                    neuron_arrangement=arrangement,
                    seed=42,
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
        
        # Verify neuron centers were generated
        assert len(engine.populations) == 1
        pop = engine.populations[0]
        assert "neuron_centers" in pop
        assert pop["neuron_centers"] is not None
        assert pop["neuron_centers"].shape[0] > 0  # Should have neurons


class TestSimulationEngineInnervationMethods:
    """Test different innervation methods."""
    
    @pytest.mark.parametrize("method", ["gaussian", "uniform", "one_to_one", "distance_weighted"])
    def test_innervation_methods(self, method):
        """Test that different innervation methods work."""
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
                    name="Test Population",
                    neuron_type="SA",
                    neuron_model="izhikevich",
                    filter_method="none",
                    innervation_method=method,
                    neurons_per_row=10,
                    connections_per_neuron=28,
                    sigma_d_mm=0.3,
                    seed=42,
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
        
        # Verify innervation module was created
        assert len(engine.populations) == 1
        pop = engine.populations[0]
        assert "innervation" in pop
        assert pop["innervation"] is not None


class TestSimulationEngineFilters:
    """Test different filter methods."""
    
    @pytest.mark.parametrize("filter_method", ["none", "sa", "ra"])
    def test_filter_methods(self, filter_method):
        """Test that different filter methods work."""
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
                    name="Test Population",
                    neuron_type="SA",
                    neuron_model="izhikevich",
                    filter_method=filter_method,
                    innervation_method="gaussian",
                    neurons_per_row=10,
                    filter_params={"tau_r": 5.0, "tau_d": 30.0, "k1": 0.05, "k2": 3.0} if filter_method == "sa" else {"tau_RA": 30.0, "k3": 2.0} if filter_method == "ra" else {},
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
        
        # Verify filter was created (or None if "none")
        assert len(engine.populations) == 1
        pop = engine.populations[0]
        assert "filter" in pop
        if filter_method == "none":
            assert pop["filter"] is None
        else:
            assert pop["filter"] is not None


class TestSimulationEngineNeuronModels:
    """Test different neuron models."""
    
    @pytest.mark.parametrize("model", ["izhikevich", "adex", "mqif", "fa", "sa"])
    def test_neuron_models(self, model):
        """Test that different neuron models work."""
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
                    name="Test Population",
                    neuron_type="SA",
                    neuron_model=model,
                    filter_method="none",
                    innervation_method="gaussian",
                    neurons_per_row=10,
                    model_params={},
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
        
        # Verify neuron model was created
        assert len(engine.populations) == 1
        pop = engine.populations[0]
        assert "neuron" in pop
        assert pop["neuron"] is not None


class TestSimulationEngineErrorHandling:
    """Test error handling in SimulationEngine."""
    
    def test_missing_grid_raises_error(self):
        """Test that missing grid raises appropriate error."""
        config = SensoryForgeConfig(
            grids=[],  # No grids
            populations=[
                PopulationConfig(
                    name="Test Population",
                    neuron_type="SA",
                    neuron_model="izhikevich",
                    filter_method="none",
                    innervation_method="gaussian",
                    neurons_per_row=10,
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
        
        with pytest.raises(ValueError, match="has no target grid"):
            SimulationEngine(config)
    
    def test_invalid_innervation_method_raises_error(self):
        """Test that invalid innervation method raises error."""
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
                    name="Test Population",
                    neuron_type="SA",
                    neuron_model="izhikevich",
                    filter_method="none",
                    innervation_method="invalid_method",
                    neurons_per_row=10,
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
        
        with pytest.raises(ValueError, match="Unknown innervation method"):
            SimulationEngine(config)
    
    def test_invalid_neuron_model_raises_error(self):
        """Test that invalid neuron model raises error."""
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
                    name="Test Population",
                    neuron_type="SA",
                    neuron_model="invalid_model",
                    filter_method="none",
                    innervation_method="gaussian",
                    neurons_per_row=10,
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
        
        with pytest.raises(ValueError, match="Unknown neuron model"):
            SimulationEngine(config)
