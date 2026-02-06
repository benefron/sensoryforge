import torch

from neurons.izhikevich import IzhikevichNeuronTorch


def test_vectorized_and_scalar_runs_match():
    params = dict(a=0.02, b=0.2, c=-65.0, d=8.0, v_init=-65.0, dt=0.1)
    neuron = IzhikevichNeuronTorch(**params)

    drive_scalar = torch.full((1, 40, 1), 15.0, dtype=torch.float32)
    drive_vector = torch.cat([drive_scalar, drive_scalar], dim=2)

    _, spikes_scalar = neuron(drive_scalar)
    _, spikes_vector = neuron(drive_vector)

    left = spikes_vector[:, :, 0:1]
    right = spikes_vector[:, :, 1:2]

    assert torch.equal(spikes_scalar, left)
    assert torch.equal(spikes_scalar, right)
