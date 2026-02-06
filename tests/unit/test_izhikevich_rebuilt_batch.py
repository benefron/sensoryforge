import torch

from neurons.izhikevich import IzhikevichNeuronTorch


def test_batched_integration_matches_iterating_batches():
    neuron = IzhikevichNeuronTorch(dt=0.1, threshold=30.0)
    drive = torch.full((3, 50, 2), 12.0, dtype=torch.float32)

    _, batched_spikes = neuron(drive)

    sequential = []
    for sample in drive:
        _, spikes = neuron(sample.unsqueeze(0))
        sequential.append(spikes)

    stacked = torch.cat(sequential, dim=0)
    assert torch.equal(batched_spikes, stacked)
