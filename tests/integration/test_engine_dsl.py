"""The engine builds DSL neurons (Phase 2, Wave N, N3; F-010 DSL half).

Before this, ``SimulationEngine._build_populations`` constructed every
neuron model the same way -- ``neuron_cls(dt=..., noise_std=..., **params)``
-- which raises ``TypeError`` for the DSL model (``NeuronModel.__init__``
takes ``equations``/``threshold``/``reset``/..., not ``dt=``/``noise_std=``);
resolving ``neuron_model="dsl"`` also passed through
``resolve_neuron_params``, which raises ``ValueError: Unknown neuron
model``. Now a DSL population is built from ``dsl_config`` and compiled.
"""

import pytest
import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.neurons.model_dsl import SYMPY_AVAILABLE

pytestmark = pytest.mark.skipif(
    not SYMPY_AVAILABLE, reason="SymPy is required for DSL tests but is not installed"
)


def _make_config(dsl_config, readout="auto"):
    return SensoryForgeConfig(
        grids=[
            GridConfig(
                name="Test Grid",
                arrangement="grid",
                rows=12,
                cols=12,
                spacing=0.15,
            )
        ],
        populations=[
            PopulationConfig(
                name="DSL Population",
                neuron_type="SA",
                neuron_model="dsl",
                innervation_method="gaussian",
                neurons_per_row=3,
                connections_per_neuron=8,
                sigma_d_mm=0.3,
                dsl_config=dsl_config,
                readout=readout,
            )
        ],
        stimulus=StimulusConfig(type="gaussian", amplitude=10.0),
        simulation=SimulationConfig(device="cpu", dt_ms=1.0, integrate_dt_ms=1.0),
    )


class TestEngineBuildsDSLNeurons:
    def test_leaky_integrator_runs_and_returns_state(self):
        dsl_config = {
            "equations": "dv/dt = (-(v - v_rest) + R*I) / tau_m",
            "parameters": {"v_rest": -65.0, "R": 1.0, "tau_m": 10.0},
            "state_vars": {"v": -65.0},
        }
        config = _make_config(dsl_config)
        engine = SimulationEngine(config)
        stimulus = torch.rand(20, 12, 12) * 5.0

        results = engine.run(stimulus)
        pop_results = results["DSL Population"]

        assert "state" in pop_results
        assert "spikes" not in pop_results
        assert pop_results["state"].shape[0] == 1
        assert pop_results["state"].shape[2] == 9  # 3x3 neurons

    def test_thresholded_dsl_returns_spikes(self):
        dsl_config = {
            "equations": "dv/dt = (-(v - (-65.0)) + I) / 10.0",
            "threshold": "v >= -50.0",
            "reset": "v = -65.0",
            "state_vars": {"v": -65.0},
        }
        config = _make_config(dsl_config)
        engine = SimulationEngine(config)
        stimulus = torch.rand(20, 12, 12) * 500.0

        results = engine.run(stimulus)
        pop_results = results["DSL Population"]

        assert "spikes" in pop_results
        assert "state" not in pop_results
        assert pop_results["spikes"].shape[0] == 1
        assert pop_results["spikes"].shape[2] == 9

    def test_missing_dsl_config_raises(self):
        config = _make_config(dsl_config=None)
        with pytest.raises(ValueError, match="dsl_config"):
            SimulationEngine(config)

    def test_readout_analog_forced_on_thresholded_config_raises(self):
        dsl_config = {
            "equations": "dv/dt = (-(v - (-65.0)) + I) / 10.0",
            "threshold": "v >= -50.0",
            "reset": "v = -65.0",
            "state_vars": {"v": -65.0},
        }
        config = _make_config(dsl_config, readout="analog")
        with pytest.raises(ValueError, match="readout='analog'"):
            SimulationEngine(config)

    def test_readout_spiking_forced_on_thresholdless_config_raises(self):
        dsl_config = {
            "equations": "dv/dt = (-(v - v_rest)) / tau_m",
            "parameters": {"v_rest": -65.0, "tau_m": 10.0},
            "state_vars": {"v": -65.0},
        }
        config = _make_config(dsl_config, readout="spiking")
        with pytest.raises(ValueError, match="readout='spiking'"):
            SimulationEngine(config)
