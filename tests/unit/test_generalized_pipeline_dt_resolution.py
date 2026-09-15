"""Regression for task E7 (F-039, F-040, F-024): one resolved record step.

Before this fix, GeneralizedTactileEncodingPipeline's stimulus generators
read one of two independent dt keys (neurons.dt or temporal.dt, chosen
per-generator) that could silently disagree, and the canonical adapter
only ever wrote neurons.dt from the legacy "dt" key -- never dt_ms, and
never temporal.dt. A canonical config's dt_ms therefore never reached
stimulus generation at all.
"""

import pytest

from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.stimuli.builder import StaticStimulus


def _canonical_config(sim_overrides: dict) -> dict:
    return {
        "grids": [
            {"name": "g", "rows": 8, "cols": 8, "spacing": 1.0, "arrangement": "grid"}
        ],
        "populations": [
            {
                "name": "SA Pop",
                "target_grid": "g",
                "neuron_type": "SA",
                "neurons_per_row": 2,
                "innervation_method": "gaussian",
                "connections_per_neuron": 4,
                "sigma_d_mm": 2.0,
                "filter_method": "none",
                "neuron_model": "Izhikevich",
                "seed": 42,
            }
        ],
        "simulation": {"device": "cpu", **sim_overrides},
    }


# ---------------------------------------------------------------------------
# Adapter: dt_ms and legacy dt keys
# ---------------------------------------------------------------------------


def test_adapter_reads_dt_ms():
    pipeline = GeneralizedTactileEncodingPipeline.__new__(
        GeneralizedTactileEncodingPipeline
    )
    canonical = _canonical_config({"dt_ms": 0.5})
    legacy = pipeline._canonical_to_legacy_config(canonical)
    assert legacy["neurons"]["dt"] == 0.5
    assert legacy["temporal"]["dt"] == 0.5


def test_adapter_falls_back_to_legacy_dt_key():
    pipeline = GeneralizedTactileEncodingPipeline.__new__(
        GeneralizedTactileEncodingPipeline
    )
    canonical = _canonical_config({"dt": 0.25})
    legacy = pipeline._canonical_to_legacy_config(canonical)
    assert legacy["neurons"]["dt"] == 0.25
    assert legacy["temporal"]["dt"] == 0.25


# ---------------------------------------------------------------------------
# Legacy hand-written configs: one key set, the other must follow
# ---------------------------------------------------------------------------


def test_legacy_config_only_neurons_dt_set_temporal_follows():
    config = {
        "pipeline": {"device": "cpu"},
        "neurons": {"sa_neurons": 2, "ra_neurons": 2, "dt": 2.0},
    }
    pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
    assert pipeline.config["neurons"]["dt"] == 2.0
    assert pipeline.config["temporal"]["dt"] == 2.0


def test_legacy_config_only_temporal_dt_set_neurons_follows():
    config = {
        "pipeline": {"device": "cpu"},
        "neurons": {"sa_neurons": 2, "ra_neurons": 2},
        "temporal": {"dt": 3.0},
    }
    pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
    assert pipeline.config["neurons"]["dt"] == 3.0
    assert pipeline.config["temporal"]["dt"] == 3.0


# ---------------------------------------------------------------------------
# Each stimulus generator honours the resolved dt / duration
# ---------------------------------------------------------------------------


def _timeline_sub_stimuli():
    return [
        {
            "stimulus": StaticStimulus("gaussian", {"amplitude": 10.0, "sigma": 1.0}),
            "onset_ms": 0.0,
            "duration_ms": 40.0,
        }
    ]


@pytest.mark.parametrize(
    "stimulus_type,extra_params",
    [
        ("gaussian", {}),
        ("texture", {}),
        ("moving", {}),
        ("timeline", {}),
    ],
)
def test_stimulus_generator_uses_resolved_dt_ms(stimulus_type, extra_params):
    pipeline = GeneralizedTactileEncodingPipeline.from_config(
        _canonical_config({"dt_ms": 2.0})
    )
    if stimulus_type == "timeline":
        extra_params = {"sub_stimuli": _timeline_sub_stimuli()}
    stimulus_sequence, time_array, _ = pipeline.generate_stimulus(
        stimulus_type=stimulus_type, duration=40.0, **extra_params
    )
    expected_steps = int(40.0 / 2.0)
    assert stimulus_sequence.shape[1] == expected_steps
    assert time_array.shape[0] == expected_steps
