"""The neuron input floor silences tactile afferents (D-43dc520)."""

import math

import torch

from sensoryforge.config.defaults import resolve_input_floor
from sensoryforge.config.schema import PopulationConfig, SensoryForgeConfig
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch


def test_the_floor_resolves_to_zero_for_tactile_afferents_only():
    for neuron_type in ("SA", "RA", "SA2", "sa"):
        assert resolve_input_floor(neuron_type, "izhikevich", None) == 0.0
        assert resolve_input_floor(neuron_type, "adex", None) == 0.0
    # A DSL model may be an analog readout of a signed signal.
    assert resolve_input_floor("SA", "dsl", None) is None
    assert resolve_input_floor("ON", "izhikevich", None) is None


def test_an_explicit_floor_wins_and_minus_inf_disables_it():
    assert resolve_input_floor("SA", "izhikevich", 0.5) == 0.5
    assert resolve_input_floor("ON", "izhikevich", -2.0) == -2.0
    assert resolve_input_floor("SA", "izhikevich", -math.inf) is None


def test_the_floor_round_trips_through_yaml():
    config = SensoryForgeConfig(
        populations=[
            PopulationConfig(name="a", input_floor=-math.inf),
            PopulationConfig(name="b"),
        ]
    )
    again = SensoryForgeConfig.from_yaml(config.to_yaml())
    assert again.populations[0].input_floor == -math.inf
    assert again.populations[1].input_floor is None


def _run(input_floor):
    drive = torch.zeros(1, 60, 1)
    drive[0, 10:30, 0] = -40.0  # strongly negative, as on a trailing edge
    neuron = IzhikevichNeuronTorch(dt=0.05)
    return SimulationEngine._run_pop_from_drive(
        drive=drive,
        filter_module=None,
        neuron_model=neuron,
        input_gain=1.0,
        return_intermediates=True,
        dt_ms=1.0,
        integrate_dt_ms=0.05,
        input_floor=input_floor,
    )


def test_the_neuron_sees_the_floored_drive_while_filtered_stays_signed():
    floored = _run(0.0)
    unfloored = _run(None)
    # The recorded signal keeps its sign in both cases.
    assert float(floored["filtered"].min()) == -40.0
    assert torch.equal(floored["filtered"], unfloored["filtered"])
    # Without the floor the neuron is driven far below rest; with it, it rests.
    assert float(unfloored["voltages"].min()) < -85.0
    assert float(floored["voltages"].min()) > -75.0
