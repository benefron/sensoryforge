"""Contract test for LeakyThresholdNeuronTorch (from `sensoryforge new-component`).

Calls the same three checks as SensoryForge's own in-repo contract sweep
(`tests/contract/test_component_contracts.py`): a `get_param_spec()` shape
check, one forward pass with the `neuron` kind's canonical tensor shape,
and a `from_config(to_dict())` round trip. See
`sensoryforge.testing.contracts.check_component`.
"""

from sensoryforge.testing.contracts import check_component

from sensoryforge_leaky_threshold.component import LeakyThresholdNeuronTorch


def test_leaky_threshold_satisfies_component_contract():
    check_component("neuron", LeakyThresholdNeuronTorch)


def test_a_silent_input_gives_no_spikes_and_a_strong_one_does():
    import torch

    neuron = LeakyThresholdNeuronTorch(
        tau_m=10.0, v_thresh=-50.0, v_reset=-65.0, dt=1.0
    )
    quiet = torch.zeros(1, 20, 1)
    _, spikes = neuron(quiet)
    assert not spikes.any()  # no drive: the membrane stays at v_reset
    # One step of drive 150 moves v by 150 / 10 * 1 = 15 mV, from -65 to -50: a spike.
    _, spikes = neuron(torch.full((1, 1, 1), 150.0))
    assert spikes[0, 1, 0]
