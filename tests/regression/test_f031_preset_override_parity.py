"""Regression tests for ledger finding F-031 (task A8).

``resolve_neuron_params`` used to expand the neuron-type preset only when no
``a``/``b``/``c``/``d`` override was present. Overriding a single parameter
(e.g. ``{"d": 4.0}`` on an RA population) therefore silently dropped the
other three back to whatever ``IzhikevichNeuronTorch`` defaults to on its
own (RS), diverging between the GUI (which pre-resolves the RA/FS preset
before merging overrides) and ``SimulationEngine``/the legacy adapter
(which called the resolver with the override and got a partial dict back --
the legacy adapter then raised ``KeyError: 'a'`` reading the missing key).

These tests fail on commit d194cd4 (the tip of Wave A, before A8):
``resolve_neuron_params("Izhikevich", "RA", {"d": 4.0})`` returns
``{"d": 4.0, "threshold": ..., "noise_std": ...}`` with no "a" key, and
``GeneralizedTactileEncodingPipeline.from_config`` raises ``KeyError: 'a'``
building the equivalent canonical config.
"""

import pytest

from sensoryforge.config.defaults import resolve_neuron_params
from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
)
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.core.pipeline import TactileEncodingPipelineTorch
from sensoryforge.core.simulation_engine import SimulationEngine

OVERRIDE_CASES = [
    {},
    {"d": 4.0},
    {"a": 0.05},
    {"preset": "IB"},
    {"preset": "IB", "d": 1.0},
]


def _canonical_config(overrides: dict) -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[
            GridConfig(name="grid", arrangement="grid", rows=10, cols=10, spacing=0.15)
        ],
        populations=[
            PopulationConfig(
                name="RA Pop",
                target_grid="grid",
                neuron_type="RA",
                neuron_model="izhikevich",
                filter_method="ra",
                innervation_method="gaussian",
                neurons_per_row=3,
                model_params=dict(overrides),
            )
        ],
        stimulus=StimulusConfig(type="gaussian", amplitude=10.0, sigma=1.0),
        simulation=SimulationConfig(device="cpu", dt=0.1),
    )


@pytest.mark.parametrize("overrides", OVERRIDE_CASES, ids=lambda o: str(o) or "empty")
def test_resolver_engine_and_adapter_agree_on_ra_overrides(overrides):
    """resolve_neuron_params, SimulationEngine and the legacy adapter must
    build identical a/b/c/d for every combination of neuron type, explicit
    preset, and individual a/b/c/d overrides -- and the adapter must never
    raise KeyError.
    """
    resolved = resolve_neuron_params("izhikevich", "RA", overrides)
    for key in ("a", "b", "c", "d"):
        assert (
            key in resolved
        ), f"resolved params missing {key!r} for overrides={overrides}"

    config = _canonical_config(overrides)

    engine = SimulationEngine(config)
    engine_neuron = engine.populations[0]["neuron"]
    assert engine_neuron.a == resolved["a"]
    assert engine_neuron.b == resolved["b"]
    assert engine_neuron.c == resolved["c"]
    assert engine_neuron.d == resolved["d"]

    # Must not raise KeyError (the F-031 adapter crash).
    legacy_pipeline = GeneralizedTactileEncodingPipeline.from_config(config.to_dict())
    legacy_neuron_params = legacy_pipeline.config["neuron_params"]
    assert legacy_neuron_params["ra_a"] == resolved["a"]
    assert legacy_neuron_params["ra_b"] == resolved["b"]
    assert legacy_neuron_params["ra_c"] == resolved["c"]
    assert legacy_neuron_params["ra_d"] == resolved["d"]


def test_tactile_encoding_pipeline_torch_ra_neurons_are_fast_spiking():
    """Regression for F-032: TactileEncodingPipelineTorch's RA neurons must
    resolve to the fast-spiking preset, not its old hard-coded RS default.
    """
    pipeline = TactileEncodingPipelineTorch(
        overrides={"pipeline": {"device": "cpu", "seed": 7}},
    )
    assert pipeline.ra_neurons.a == 0.1
    assert pipeline.ra_neurons.d == 2.0
    assert pipeline.sa_neurons.a == 0.02
    assert pipeline.sa_neurons.d == 8.0
