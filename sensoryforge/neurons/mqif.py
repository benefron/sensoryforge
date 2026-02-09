import math
import torch
import torch.nn as nn


class MQIFNeuronTorch(nn.Module):
    r"""Modified quadratic integrate-and-fire neuron (batched PyTorch).

    Continuous dynamics per feature ``f`` and time ``t``:

    - Membrane: ``dv/dt = (a (v-v_r)(v-v_t) - u + I) / tau_m``
    - Adaptation: ``du/dt = (b (v-v_r) - u) / tau_u``

    Both states advance via Euler integration ``state_{t+1} = state_t + dt *``
    ``dstate/dt`` plus optional Langevin noise ``η_t ~ N(0, noise_std ·
    sqrt(dt))`` on the membrane voltage.

    A spike occurs when ``v >= v_peak``; the model records the event, sets
    ``v = v_reset`` and increments ``u`` by ``d`` for that feature. Noise is
    suppressed during the reset branch so that refractory segments remain
    deterministic.

    Inputs ``I`` have shape ``[batch, steps, features]`` (currents in mA). The
    forward pass returns ``(v_trace, spikes)`` capturing voltages and boolean
    spike events with shape ``[batch, steps+1, features]``.
    """

    def __init__(
        self,
        a=0.04,
        b=0.2,
        vr=-60.0,
        vt=-40.0,
        v_reset=-60.0,
        v_peak=30.0,
        d=2.0,
        tau_m=10.0,
        tau_u=100.0,
        v_init=None,
        u_init=None,
        dt=0.05,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.a = a
        self.b = b
        self.vr = vr
        self.vt = vt
        self.v_reset = v_reset
        self.v_peak = v_peak
        self.d = d
        self.tau_m = tau_m
        self.tau_u = tau_u
        self.dt = dt
        self.v_init = vr if v_init is None else v_init
        self.u_init = 0.0 if u_init is None else u_init
        # Langevin noise intensity (additive, mV/sqrt(ms))
        self.noise_std = noise_std

    def reset_state(self) -> None:
        """Reset internal state (no-op for stateless MQIF).

        The MQIF model re-initialises v and u each forward pass so
        there is no persistent state to clear, but the method exists to
        satisfy the BaseNeuron contract (resolves ReviewFinding#H6).
        """
        pass

    def forward(self, input_current):
        batch, steps, features = input_current.shape
        device, dtype = input_current.device, input_current.dtype
        v = torch.full((batch, features), self.v_init, dtype=dtype, device=device)
        u = torch.full((batch, features), self.u_init, dtype=dtype, device=device)
        v_trace = torch.zeros((batch, steps + 1, features), dtype=dtype, device=device)
        spikes = torch.zeros(
            (batch, steps + 1, features), dtype=torch.bool, device=device
        )
        v_trace[:, 0, :] = v
        for t in range(steps):
            fired = v >= self.v_peak
            v_vis = v.clone()
            v_vis[fired] = self.v_peak
            v_trace[:, t, :] = v_vis
            spikes[:, t, :] = fired
            v_next = torch.where(fired, torch.full_like(v, self.v_reset), v)
            u_next = torch.where(fired, u + self.d, u)
            not_fired = ~fired
            quad_term = self.a * (v - self.vr) * (v - self.vt)
            dv = (quad_term - u + input_current[:, t, :]) / self.tau_m
            du = (self.b * (v - self.vr) - u) / self.tau_u
            # Add Langevin noise to v integration (not during reset)
            if self.noise_std != 0.0:
                sqrt_dt = math.sqrt(max(self.dt, 1e-6))
                eta = torch.randn_like(v) * (self.noise_std * sqrt_dt)
                v_next = torch.where(not_fired, v + self.dt * dv + eta, v_next)
            else:
                v_next = torch.where(not_fired, v + self.dt * dv, v_next)
            u_next = torch.where(not_fired, u + self.dt * du, u_next)
            v = v_next
            u = u_next
            v_trace[:, t + 1, :] = v
            spikes[:, t + 1, :] = v >= self.v_peak
        return v_trace, spikes
