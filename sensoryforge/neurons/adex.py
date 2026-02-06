import math
import torch
import torch.nn as nn


class AdExNeuronTorch(nn.Module):
    r"""Adaptive exponential integrate-and-fire neuron (batched PyTorch).

    Per feature and time step the model evaluates:

    1. **Membrane voltage**
        \( \frac{dv}{dt} = \frac{-(v-EL) + \Delta_T e^{(v-VT)/\Delta_T}
        - w + R I}{\tau_m} \).
        Euler update: ``v_{t+1} = v_t + dt * dv/dt`` plus optional Langevin
        noise ``η_t ~ N(0, noise_std · sqrt(dt))``.
    2. **Adaptation current**
        \( \frac{dw}{dt} = \frac{a (v-EL) - w}{\tau_w} \) with Euler update
        ``w_{t+1} = w_t + dt * dw/dt``.
    3. **Spike/reset**
        When ``v >= v_spike`` a spike is emitted, ``v`` is set to
        ``v_reset`` and ``w`` increments by ``b``. Spikes persist for one
        step in ``spikes``.

    Inputs ``I`` have shape ``[batch, steps, features]`` (currents in mA). The
    forward pass returns ``(v_trace, spikes)`` where ``v_trace`` stores the
    membrane trajectory ``[batch, steps+1, features]`` before resets and
    ``spikes`` provides boolean events of the same shape.
    """

    def __init__(
        self,
        EL=-70.0,
        VT=-50.0,
        DeltaT=2.0,
        tau_m=20.0,
        tau_w=100.0,
        a=2.0,
        b=0.0,
        v_reset=-58.0,
        v_spike=20.0,
        R=1.0,
        v_init=None,
        w_init=None,
        dt=0.05,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.EL = EL
        self.VT = VT
        self.DeltaT = DeltaT
        self.tau_m = tau_m
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.v_reset = v_reset
        self.v_spike = v_spike
        self.R = R
        self.dt = dt
        self.v_init = EL if v_init is None else v_init
        self.w_init = 0.0 if w_init is None else w_init
        # Langevin noise intensity (additive, mV/sqrt(ms))
        self.noise_std = noise_std

    def forward(self, input_current):
        batch, steps, features = input_current.shape
        device, dtype = input_current.device, input_current.dtype
        v = torch.full((batch, features), self.v_init, dtype=dtype, device=device)
        w = torch.full((batch, features), self.w_init, dtype=dtype, device=device)
        v_trace = torch.zeros((batch, steps + 1, features), dtype=dtype, device=device)
        spikes = torch.zeros(
            (batch, steps + 1, features), dtype=torch.bool, device=device
        )
        v_trace[:, 0, :] = v
        for t in range(steps):
            fired = v >= self.v_spike
            v_vis = v.clone()
            v_vis[fired] = self.v_spike
            v_trace[:, t, :] = v_vis
            spikes[:, t, :] = fired
            v_next = torch.where(fired, torch.full_like(v, self.v_reset), v)
            w_next = torch.where(fired, w + self.b, w)
            not_fired = ~fired
            exp_term = self.DeltaT * torch.exp((v - self.VT) / self.DeltaT)
            dv_input = self.R * input_current[:, t, :]
            dv = (-(v - self.EL) + exp_term - w + dv_input) / self.tau_m
            dw = (self.a * (v - self.EL) - w) / self.tau_w
            # Add Langevin noise to v integration (not during reset)
            if self.noise_std != 0.0:
                # scale by sqrt(dt) for discrete-time white noise
                sqrt_dt = math.sqrt(max(self.dt, 1e-6))
                eta = torch.randn_like(v) * (self.noise_std * sqrt_dt)
                v_next = torch.where(not_fired, v + self.dt * dv + eta, v_next)
            else:
                v_next = torch.where(not_fired, v + self.dt * dv, v_next)
            w_next = torch.where(not_fired, w + self.dt * dw, w_next)
            v = v_next
            w = w_next
            v_trace[:, t + 1, :] = v
            spikes[:, t + 1, :] = v >= self.v_spike
        return v_trace, spikes
