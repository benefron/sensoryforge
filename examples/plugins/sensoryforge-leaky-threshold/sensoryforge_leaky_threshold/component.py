"""LeakyThresholdNeuronTorch: a new spiking neuron model.

Implements the ``BaseNeuron`` contract.
Full guide: https://benefron.github.io/sensoryforge/how-to/extend/add_neuron/
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from sensoryforge.neurons.base import BaseNeuron
from sensoryforge.stimuli.base import ParamSpec


class LeakyThresholdNeuronTorch(BaseNeuron):
    """LeakyThresholdNeuronTorch neuron model.

    Args:
        tau_m: Membrane time constant in ms.
        v_thresh: Spike threshold in mV.
        v_reset: Reset potential in mV.
        dt: Integration time step in ms.
        noise_std: Additive noise intensity on the membrane voltage
            (mV/sqrt(ms)). ``SimulationEngine`` unconditionally passes
            ``noise_std`` to every neuron's constructor, so every neuron
            model -- including this one -- must accept it even if the
            placeholder dynamics below do not use it yet.
    """

    def __init__(
        self,
        tau_m: float = 20.0,
        v_thresh: float = -50.0,
        v_reset: float = -65.0,
        dt: float = 0.05,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__(dt=dt)
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.noise_std = noise_std

    def reset_state(self) -> None:
        """No persistent state: forward() re-initialises v each call."""

    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate a leaky membrane over ``input_current`` (placeholder LIF).

        Args:
            input_current: Input currents ``[batch, steps, features]`` in mA.

        Returns:
            Tuple of ``v_trace`` ``[batch, steps+1, features]`` in mV and
            ``spikes`` ``[batch, steps+1, features]`` (bool).
        """
        batch, steps, features = input_current.shape
        device, dtype = input_current.device, input_current.dtype
        v = torch.full((batch, features), self.v_reset, device=device, dtype=dtype)
        v_trace = [v]
        spikes = [torch.zeros_like(v, dtype=torch.bool)]
        for t in range(steps):
            dv = (-(v - self.v_reset) + input_current[:, t, :]) / self.tau_m * self.dt
            v = v + dv
            if self.noise_std > 0.0:
                v = v + torch.randn_like(v) * (self.noise_std * self.dt**0.5)
            spiked = v >= self.v_thresh
            v = torch.where(spiked, torch.full_like(v, self.v_reset), v)
            v_trace.append(v)
            spikes.append(spiked)
        return torch.stack(v_trace, dim=1), torch.stack(spikes, dim=1)

    def to_dict(self) -> dict:
        return {
            "tau_m": self.tau_m,
            "v_thresh": self.v_thresh,
            "v_reset": self.v_reset,
            "dt": self.dt,
            "noise_std": self.noise_std,
        }

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("tau_m", dtype="float", default=20.0, min_val=0.1, unit="ms"),
            ParamSpec("v_thresh", dtype="float", default=-50.0, unit="mV"),
            ParamSpec("v_reset", dtype="float", default=-65.0, unit="mV"),
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
        ]


def register() -> None:
    """Entry-point target: register LeakyThresholdNeuronTorch.

    Called with no arguments by
    :func:`sensoryforge.plugins.discover_entry_point_plugins` when this
    distribution is installed and its
    ``[project.entry-points."sensoryforge.components"]`` entry loads.
    """
    from sensoryforge.registry import NEURON_REGISTRY

    NEURON_REGISTRY.register("leaky_threshold", LeakyThresholdNeuronTorch)
