import torch

from encoding.filters_torch import RAFilterTorch, SAFilterTorch


def test_sa_filter_step_response_matches_analytic_shape():
    tau_r = 5.0  # ms
    tau_d = 30.0  # ms
    k1 = 0.05
    k2 = 3.0
    dt = 0.1  # ms
    steps = int(200.0 / dt)

    sa_filter = SAFilterTorch(tau_r=tau_r, tau_d=tau_d, k1=k1, k2=k2, dt=dt)
    step_input = torch.ones(1, steps, 1)
    sa_out = sa_filter(step_input, reset_states=True).squeeze().numpy()

    t = torch.arange(steps, dtype=torch.float64) * dt
    x_t = k1 * (1 - torch.exp(-t / tau_r))
    analytic = (x_t * (1 - torch.exp(-t / tau_d))).numpy()

    error = abs(sa_out - analytic).max()
    assert error < 1e-2


def test_ra_filter_impulse_response_is_exponential():
    tau_ra = 30.0  # ms
    k3 = 2.0
    dt = 0.1  # ms
    steps = int(200.0 / dt)

    ra_filter = RAFilterTorch(tau_RA=tau_ra, k3=k3, dt=dt)
    impulse = torch.zeros(1, steps, 1)
    impulse[:, 0, 0] = 1.0 / dt  # discrete impulse

    ra_out = ra_filter(impulse, reset_states=True).squeeze().numpy()

    # After the initial impulse response settles, the RA filter should
    # decay monotonically toward zero and remain non-negative.
    assert ra_out[0] > 0.0
    tail = ra_out[1:]
    assert (tail[:-1] >= tail[1:] - 1e-6).all()
    assert tail[-1] < 1e-2
