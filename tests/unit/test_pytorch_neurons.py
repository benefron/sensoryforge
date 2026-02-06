import math

import torch

from neurons import (
    AdExNeuronTorch,
    FANeuronTorch,
    IzhikevichNeuronTorch,
    MQIFNeuronTorch,
    SANeuronTorch,
)


def _constant_drive(steps: int, features: int, current: float) -> torch.Tensor:
    return torch.full((1, steps, features), current, dtype=torch.float32)


def test_izhikevich_neuron_emits_spikes_under_constant_drive():
    neuron = IzhikevichNeuronTorch(dt=0.1, threshold=30.0)
    drive = _constant_drive(steps=50, features=4, current=12.0)

    _, spikes = neuron(drive)
    assert spikes.shape == (1, 51, 4)
    assert spikes[:, 1:].any()


def test_adex_neuron_runs_without_nan():
    neuron = AdExNeuronTorch(dt=0.1)
    drive = _constant_drive(steps=40, features=2, current=1.5)

    voltages, spikes = neuron(drive)
    assert torch.isfinite(voltages).all()
    assert torch.isfinite(spikes.float()).all()


def test_mqif_neuron_supports_zero_drive():
    neuron = MQIFNeuronTorch(dt=0.1)
    drive = _constant_drive(steps=30, features=1, current=0.0)

    voltages, spikes = neuron(drive)
    assert voltages.shape[1] == drive.shape[1] + 1
    assert spikes.sum() == 0


def test_fa_neuron_threshold_behavior():
    neuron = FANeuronTorch(dt=0.1, theta=0.2)
    drive = torch.zeros(1, 20, 3)
    drive[:, :10, :] = 0.0
    drive[:, 10:, :] = 0.5

    _, spikes = neuron(drive)
    assert spikes.any()


def test_sa_neuron_supports_parameter_sampling():
    neuron = SANeuronTorch(dt=0.2)
    drive = _constant_drive(steps=25, features=2, current=5e-10)

    traces, spikes = neuron(drive)
    assert traces.shape[1] == drive.shape[1] + 1
    assert spikes.dtype == torch.bool
    assert math.isclose(float(neuron.dt_ms), 0.2, rel_tol=1e-6)
