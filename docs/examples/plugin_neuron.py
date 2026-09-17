"""Worked example: define, register, and run a minimal neuron component (U3).

This is the runnable companion to ``docs/developer_guide/add_neuron.md``'s
leaky integrate-and-fire (LIF) walkthrough -- the guide's code block is
excerpted from this file so what a reader copies is what actually runs.

1. Define a neuron class inheriting :class:`~sensoryforge.neurons.base.BaseNeuron`.
2. Implement ``get_param_spec()`` (required on every component, G1) and a
   ``noise_std`` constructor parameter (every neuron model must accept one --
   ``SimulationEngine._build_populations`` passes it unconditionally).
3. Register it directly with ``NEURON_REGISTRY``.
4. Run it through the shared contract check
   (``sensoryforge.testing.contracts.check_component``) and a tiny standalone
   forward pass.

Run it directly: ``python docs/examples/plugin_neuron.py``. It is also
executed by ``tests/docs/test_docs_examples.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from sensoryforge.neurons.base import BaseNeuron
from sensoryforge.registry import NEURON_REGISTRY
from sensoryforge.stimuli.base import ParamSpec


class LIFNeuronTorch(BaseNeuron):
    """Leaky integrate-and-fire neuron, vectorised over a population.

    Args:
        tau_m: Membrane time constant in ms.
        v_thresh: Spike threshold in mV.
        v_reset: Reset potential in mV.
        v_rest: Resting potential in mV.
        r_m: Membrane resistance in MOhm.
        dt: Integration time step in ms.
        noise_std: Additive membrane noise (mV / sqrt(ms)). Required: every
            neuron model must accept it, or a real ``SimulationEngine`` run
            raises ``TypeError`` the moment this population is used.
    """

    def __init__(
        self,
        tau_m: float = 20.0,
        v_thresh: float = -50.0,
        v_reset: float = -65.0,
        v_rest: float = -70.0,
        r_m: float = 10.0,
        dt: float = 1.0,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        if tau_m <= 0:
            raise ValueError(f"tau_m must be positive, got {tau_m}")
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.r_m = r_m
        self.dt = dt
        self.noise_std = noise_std
        self._v: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Clear membrane state between runs (BaseNeuron contract)."""
        self._v = None

    def forward(self, I: torch.Tensor) -> tuple:
        """Integrate LIF dynamics over a batch of current traces.

        Args:
            I: Input current ``[batch, time, N_neurons]`` in mA.

        Returns:
            ``(v_trace, spikes)``, both ``[batch, time + 1, N_neurons]``:
            ``v_trace`` in mV, ``spikes`` as float ``{0., 1.}``. The leading
            extra step is the initial condition, matching the built-in
            neuron models' convention.
        """
        batch, steps, n = I.shape
        device = I.device
        v = (
            self._v
            if self._v is not None
            else torch.full((batch, n), self.v_rest, device=device)
        )
        v_traces = [v.unsqueeze(1)]
        spike_traces = [torch.zeros_like(v).unsqueeze(1)]

        for t in range(steps):
            I_t = I[:, t, :]
            dv = (-(v - self.v_rest) + self.r_m * I_t) / self.tau_m * self.dt
            v_next = v + dv
            if self.noise_std > 0.0:
                v_next = v_next + torch.randn_like(v_next) * self.noise_std * (
                    self.dt**0.5
                )
            spike = (v_next >= self.v_thresh).float()
            v_next = torch.where(
                spike.bool(), torch.full_like(v_next, self.v_reset), v_next
            )
            v_traces.append(v_next.unsqueeze(1))
            spike_traces.append(spike.unsqueeze(1))
            v = v_next

        self._v = v.detach()
        return torch.cat(v_traces, dim=1), torch.cat(spike_traces, dim=1)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LIFNeuronTorch":
        return cls(
            tau_m=config.get("tau_m", 20.0),
            v_thresh=config.get("v_thresh", -50.0),
            v_reset=config.get("v_reset", -65.0),
            v_rest=config.get("v_rest", -70.0),
            r_m=config.get("r_m", 10.0),
            dt=config.get("dt", 1.0),
            noise_std=config.get("noise_std", 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau_m": self.tau_m,
            "v_thresh": self.v_thresh,
            "v_reset": self.v_reset,
            "v_rest": self.v_rest,
            "r_m": self.r_m,
            "dt": self.dt,
            "noise_std": self.noise_std,
        }

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                "tau_m",
                dtype="float",
                default=20.0,
                min_val=0.1,
                max_val=200.0,
                unit="ms",
                choices=None,
                help="Membrane time constant.",
                group="Dynamics",
                advanced=False,
            ),
            ParamSpec(
                "v_thresh",
                dtype="float",
                default=-50.0,
                min_val=-100.0,
                max_val=50.0,
                unit="mV",
                choices=None,
                help="Spike threshold.",
                group="Dynamics",
                advanced=False,
            ),
        ]


def main() -> None:
    NEURON_REGISTRY.register("lif_demo", LIFNeuronTorch)

    from sensoryforge.testing.contracts import check_component

    check_component("neuron", LIFNeuronTorch)

    neuron = NEURON_REGISTRY.create("lif_demo", tau_m=15.0, noise_std=0.0)
    current = torch.zeros(1, 20, 4)
    current[:, 5:, :] = 5.0  # step current onto 4 neurons
    v_trace, spikes = neuron(current)
    print(f"v_trace shape: {tuple(v_trace.shape)}, spikes: {int(spikes.sum())}")
    assert v_trace.shape == (1, 21, 4)
    assert spikes.shape == (1, 21, 4)


if __name__ == "__main__":
    main()
