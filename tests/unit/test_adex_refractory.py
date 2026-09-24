"""AdEx's absolute refractory period (D-f5853a4)."""

import torch

from sensoryforge.neurons.adex import ADEX_PRESETS, AdExNeuronTorch


def _rate_hz(neuron, current_ma, duration_ms=200.0):
    steps = int(duration_ms / neuron.dt)
    _, spikes = neuron(torch.full((1, steps, 1), current_ma))
    return float(spikes.sum()) / (duration_ms / 1000.0)


def test_the_refractory_period_caps_the_rate():
    fast = AdExNeuronTorch(preset="SA1_tonic", t_ref=0.0, dt=0.05)
    capped = AdExNeuronTorch(preset="SA1_tonic", t_ref=2.0, dt=0.05)
    assert _rate_hz(fast, 400.0) > 700.0
    # One spike per (t_ref + at least one step): below 1 / 2 ms = 500 Hz.
    assert _rate_hz(capped, 400.0) <= 500.0


def test_no_two_spikes_fall_within_t_ref():
    neuron = AdExNeuronTorch(preset="RA1_phasic", t_ref=2.0, dt=0.05)
    _, spikes = neuron(torch.full((1, 4000, 1), 400.0))
    times = torch.nonzero(spikes[0, :, 0]).flatten().float() * neuron.dt
    onsets = times[torch.cat([torch.tensor([True]), times.diff() > neuron.dt])]
    assert float(onsets.diff().min()) >= 2.0


def test_the_tactile_presets_use_two_ms_and_the_class_default_is_zero():
    assert ADEX_PRESETS["SA1_tonic"]["t_ref"] == 2.0
    assert ADEX_PRESETS["RA1_phasic"]["t_ref"] == 2.0
    assert AdExNeuronTorch().t_ref == 0.0


def test_t_ref_round_trips():
    neuron = AdExNeuronTorch(preset="SA1_tonic")
    again = AdExNeuronTorch.from_config(neuron.to_dict())
    assert again.t_ref == 2.0
