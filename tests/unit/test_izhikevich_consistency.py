import numpy as np
import torch

from neurons.izhikevich import IzhikevichNeuronTorch


def _numpy_reference(a, b, c, d, v_init, dt, current, steps):
    v = np.full(steps + 1, v_init, dtype=np.float32)
    u = np.full(steps + 1, b * v_init, dtype=np.float32)
    spikes = np.zeros(steps + 1, dtype=bool)

    for t in range(steps):
        if v[t] >= 30:
            v[t] = 30.0
            v[t + 1] = c
            u[t + 1] = u[t] + d
            spikes[t] = True
            continue

        dv = 0.04 * v[t] ** 2 + 5 * v[t] + 140 - u[t] + current
        du = a * (b * v[t] - u[t])
        v[t + 1] = v[t] + dt * dv
        u[t + 1] = u[t] + dt * du

    return v, spikes


def test_izhikevich_torch_matches_reference_integration():
    a, b, c, d = 0.02, 0.2, -65.0, 8.0
    v_init = -65.0
    dt = 0.05
    steps = 500
    current = 10.0

    neuron = IzhikevichNeuronTorch(a=a, b=b, c=c, d=d, v_init=v_init, dt=dt)
    drive = torch.full((1, steps, 1), current, dtype=torch.float32)
    v_trace, spikes = neuron(drive)

    ref_v, ref_spikes = _numpy_reference(a, b, c, d, v_init, dt, current, steps)

    np.testing.assert_allclose(
        v_trace[0, :-1, 0].cpu().numpy(), ref_v[:-1], atol=1e-3, rtol=1e-3
    )
    np.testing.assert_array_equal(spikes[0, :-1, 0].cpu().numpy(), ref_spikes[:-1])
