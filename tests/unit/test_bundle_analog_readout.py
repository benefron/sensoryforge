"""A run that writes a bundle returns an analog population's readout."""

import torch

from sensoryforge.config.schema import GridConfig, PopulationConfig, SensoryForgeConfig
from sensoryforge.core.simulation_engine import SimulationEngine


def _config():
    return SensoryForgeConfig(
        grids=[GridConfig(name="g", rows=8, cols=8)],
        populations=[
            PopulationConfig(
                name="analog",
                target_grid="g",
                neuron_model="dsl",
                filter_method="sa",
                seed=1,
                dsl_config={
                    "equations": "dv/dt = (-v + I) / 10.0",
                    "state_vars": {"v": 0.0},
                    "parameters": {},
                },
            )
        ],
    )


def test_a_bundle_run_without_intermediates_returns_the_analog_state(tmp_path):
    stimulus = torch.rand(1, 20, 8, 8)
    with_bundle = SimulationEngine(_config()).run(
        stimulus, return_intermediates=False, bundle_dir=tmp_path / "run"
    )
    without = SimulationEngine(_config()).run(stimulus, return_intermediates=False)
    assert set(with_bundle["analog"]) == set(without["analog"]) == {"state"}
    assert torch.equal(with_bundle["analog"]["state"], without["analog"]["state"])
