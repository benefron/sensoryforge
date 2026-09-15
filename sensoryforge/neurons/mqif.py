import math
from typing import Any, Dict, List

import torch
import torch.nn as nn

from sensoryforge.neurons.base import BaseNeuron
from sensoryforge.stimuli.base import ParamSpec


class MQIFNeuronTorch(BaseNeuron):
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
        v_floor: float = -120.0,
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
        # Physiological voltage floor (mV). Prevents non-physical hyperpolarization
        # from large negative drive (e.g., noisy SA filter output). Default: -120 mV.
        self.v_floor = v_floor

    def reset_state(self) -> None:
        """Reset internal state (no-op for stateless MQIF).

        The MQIF model re-initialises v and u each forward pass so
        there is no persistent state to clear, but the method exists to
        satisfy the BaseNeuron contract (resolves ReviewFinding#H6).
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialise every constructor parameter's current value (F-045).

        Returns:
            Dictionary with every ``__init__`` parameter.
        """
        return {
            "a": self.a,
            "b": self.b,
            "vr": self.vr,
            "vt": self.vt,
            "v_reset": self.v_reset,
            "v_peak": self.v_peak,
            "d": self.d,
            "tau_m": self.tau_m,
            "tau_u": self.tau_u,
            "v_init": self.v_init,
            "u_init": self.u_init,
            "dt": self.dt,
            "noise_std": self.noise_std,
            "v_floor": self.v_floor,
        }

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """Return parameter specifications for UI auto-generation (F-045)."""
        return [
            ParamSpec(
                "a",
                dtype="float",
                default=0.04,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                unit="",
                tooltip="Quadratic term gain",
            ),
            ParamSpec(
                "b",
                dtype="float",
                default=0.2,
                min_val=0.0,
                max_val=1.0,
                step=0.01,
                unit="",
                tooltip="Adaptation coupling to v",
            ),
            ParamSpec(
                "vr",
                dtype="float",
                default=-60.0,
                min_val=-100.0,
                max_val=0.0,
                step=1.0,
                unit="mV",
                tooltip="Resting/quadratic root voltage",
            ),
            ParamSpec(
                "vt",
                dtype="float",
                default=-40.0,
                min_val=-80.0,
                max_val=0.0,
                step=1.0,
                unit="mV",
                tooltip="Instantaneous threshold voltage",
            ),
            ParamSpec(
                "v_reset",
                dtype="float",
                default=-60.0,
                min_val=-100.0,
                max_val=0.0,
                step=1.0,
                unit="mV",
                tooltip="Post-spike reset voltage",
            ),
            ParamSpec(
                "v_peak",
                dtype="float",
                default=30.0,
                min_val=-20.0,
                max_val=60.0,
                step=1.0,
                unit="mV",
                tooltip="Spike detection voltage",
            ),
            ParamSpec(
                "d",
                dtype="float",
                default=2.0,
                min_val=0.0,
                max_val=20.0,
                step=0.5,
                unit="",
                tooltip="Post-spike adaptation increment",
            ),
            ParamSpec(
                "tau_m",
                dtype="float",
                default=10.0,
                min_val=0.5,
                max_val=200.0,
                step=0.5,
                unit="ms",
                tooltip="Membrane time constant",
            ),
            ParamSpec(
                "tau_u",
                dtype="float",
                default=100.0,
                min_val=1.0,
                max_val=500.0,
                step=5.0,
                unit="ms",
                tooltip="Adaptation time constant",
            ),
            ParamSpec(
                "v_init",
                dtype="float",
                default=-60.0,
                min_val=-100.0,
                max_val=50.0,
                step=1.0,
                unit="mV",
                tooltip="Initial membrane voltage (default: vr)",
            ),
            ParamSpec(
                "u_init",
                dtype="float",
                default=0.0,
                min_val=-20.0,
                max_val=20.0,
                step=0.5,
                unit="",
                tooltip="Initial adaptation variable",
            ),
            ParamSpec(
                "dt",
                dtype="float",
                default=0.05,
                min_val=0.001,
                max_val=5.0,
                step=0.01,
                unit="ms",
                tooltip="Integration time step",
            ),
            ParamSpec(
                "noise_std",
                dtype="float",
                default=0.0,
                min_val=0.0,
                max_val=20.0,
                step=0.1,
                unit="mV/sqrt(ms)",
                tooltip="Additive Langevin noise intensity on v",
            ),
            ParamSpec(
                "v_floor",
                dtype="float",
                default=-120.0,
                min_val=-200.0,
                max_val=-50.0,
                step=1.0,
                unit="mV",
                tooltip="Physiological voltage floor (clamp)",
            ),
        ]

    def _dynamics(self, v, u, input_t):
        """Compute MQIF state derivatives (subthreshold dynamics only).

        Args:
            v: Membrane voltage tensor [batch, features] in mV.
            u: Adaptation variable tensor [batch, features].
            input_t: External current at this time step [batch, features] in mA.

        Returns:
            Tuple (dv, du) of derivative tensors, each [batch, features].
        """
        quad_term = self.a * (v - self.vr) * (v - self.vt)
        dv = (quad_term - u + input_t) / self.tau_m
        du = (self.b * (v - self.vr) - u) / self.tau_u
        return dv, du

    def forward(self, input_current, solver=None):
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
            dv, du = self._dynamics(v, u, input_current[:, t, :])
            # Add Langevin noise to v integration (not during reset)
            if self.noise_std != 0.0:
                sqrt_dt = math.sqrt(max(self.dt, 1e-6))
                eta = torch.randn_like(v) * (self.noise_std * sqrt_dt)
                v_next = torch.where(not_fired, v + self.dt * dv + eta, v_next)
            else:
                v_next = torch.where(not_fired, v + self.dt * dv, v_next)
            u_next = torch.where(not_fired, u + self.dt * du, u_next)
            if self.v_floor is not None:
                v_next = v_next.clamp(min=self.v_floor)
            v = v_next
            u = u_next
            v_trace[:, t + 1, :] = v
            spikes[:, t + 1, :] = v >= self.v_peak
        return v_trace, spikes
