"""Unit tests for Izhikevich (2003) presets and constructor contract.

Covers task A3 of docs/development/handover/phase1_tasks.md: preset values
match Izhikevich (2003) Fig. 2, explicit a/b/c/d overrides win over the
preset, unknown presets raise, preset is keyword-only, and the historical
default construction is unchanged.
"""

import pytest

from sensoryforge.neurons.izhikevich import IZHIKEVICH_PRESETS, IzhikevichNeuronTorch

EXPECTED_PRESETS = {
    "RS": {"a": 0.02, "b": 0.20, "c": -65.0, "d": 8.0},
    "FS": {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.0},
    "IB": {"a": 0.02, "b": 0.20, "c": -55.0, "d": 4.0},
    "CH": {"a": 0.02, "b": 0.20, "c": -50.0, "d": 2.0},
    "LTS": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0},
}


@pytest.mark.parametrize("name,expected", EXPECTED_PRESETS.items())
def test_preset_matches_izhikevich_2003_fig2(name, expected):
    assert IZHIKEVICH_PRESETS[name] == expected
    neuron = IzhikevichNeuronTorch(preset=name)
    assert neuron.a == expected["a"]
    assert neuron.b == expected["b"]
    assert neuron.c == expected["c"]
    assert neuron.d == expected["d"]


def test_explicit_d_override_wins_over_preset():
    neuron = IzhikevichNeuronTorch(preset="FS", d=4.0)
    assert neuron.a == 0.1
    assert neuron.d == 4.0


def test_unknown_preset_raises_value_error():
    with pytest.raises(ValueError, match="Unknown Izhikevich preset"):
        IzhikevichNeuronTorch(preset="not-a-real-preset")


def test_positional_abcd_v_init_construction():
    """preset is keyword-only, so positional a,b,c,d,v_init must still work."""
    neuron = IzhikevichNeuronTorch(0.02, 0.2, -65.0, 8.0, -70.0)
    assert neuron.v_init == -70.0
    assert neuron.a == 0.02
    assert neuron.d == 8.0


def test_default_construction_equals_historical_rs_values():
    neuron = IzhikevichNeuronTorch()
    assert neuron.a == 0.02
    assert neuron.b == 0.20
    assert neuron.c == -65.0
    assert neuron.d == 8.0
    assert neuron.preset == "RS"
