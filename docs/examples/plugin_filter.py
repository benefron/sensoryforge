"""Worked example: define, register, and run a minimal filter component (H5).

This is a runnable script (not a plugin package) that shows the smallest
complete example of the in-process extension path described in
``docs/developer_guide/add_filter.md`` and ``docs/developer_guide/plugins.md``:

1. Define a filter class inheriting :class:`~sensoryforge.filters.base.BaseFilter`.
2. Implement ``get_param_spec()`` (required on every component, G1).
3. Register it directly with ``FILTER_REGISTRY`` (the same registry a
   plugin package's ``register()`` function or ``register_components.py``
   would call into).
4. Build the filter through the registry and run a tiny simulation: a
   synthetic step-current input through ``forward()``.

Run it directly: ``python docs/examples/plugin_filter.py``. It is also
executed by ``tests/docs/test_docs_examples.py``, which runs every ``.py``
file under ``docs/examples/`` as a subprocess and asserts exit code 0.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from sensoryforge.filters.base import BaseFilter
from sensoryforge.registry import FILTER_REGISTRY
from sensoryforge.stimuli.base import ParamSpec


class LeakyFilterTorch(BaseFilter):
    """Minimal first-order leaky filter: dI_out/dt = (I_in - I_out) / tau.

    Args:
        tau: Time constant in ms.
        dt: Integration time step in ms.
    """

    def __init__(self, tau: float = 10.0, dt: float = 1.0) -> None:
        super().__init__(dt=dt)
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau
        self._I_out: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Clear hidden state (BaseFilter contract)."""
        self._I_out = None

    def forward(self, x: torch.Tensor, dt: Optional[float] = None) -> torch.Tensor:
        """Apply the leaky filter over the time dimension.

        Args:
            x: Input drive [batch, time, N_neurons] in mA.
            dt: Optional override for the integration step in ms.

        Returns:
            Filtered current [batch, time, N_neurons] in mA, same shape as ``x``.
        """
        step = dt if dt is not None else self.dt
        batch, steps, n = x.shape
        I_out = (
            self._I_out
            if self._I_out is not None
            else torch.zeros(batch, n, device=x.device)
        )

        outputs = []
        for t in range(steps):
            I_out = I_out + (x[:, t, :] - I_out) / self.tau * step
            outputs.append(I_out.unsqueeze(1))

        self._I_out = I_out.detach()
        return torch.cat(outputs, dim=1)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LeakyFilterTorch":
        return cls(tau=config.get("tau", 10.0), dt=config.get("dt", 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return {"tau": self.tau, "dt": self.dt}

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                "tau", dtype="float", default=10.0,
                min_val=0.1, max_val=500.0, unit="ms",
                help="Leaky filter time constant.",
            ),
        ]


def main() -> None:
    # Register directly with the registry -- exactly what a plugin
    # package's entry-point register() function does under the hood.
    FILTER_REGISTRY.register("leaky_example", LeakyFilterTorch)

    # Build through the registry, like any config-driven component.
    filt = FILTER_REGISTRY.create("leaky_example", tau=5.0, dt=1.0)

    # Tiny synthetic simulation: a step current into 4 neurons over 50 ms.
    batch, steps, n_neurons = 1, 50, 4
    step_current = torch.ones(batch, steps, n_neurons) * 2.0

    filtered = filt(step_current)

    print(f"input shape:  {tuple(step_current.shape)}")
    print(f"output shape: {tuple(filtered.shape)}")
    print(f"final value:  {filtered[0, -1, 0].item():.4f} mA")

    assert filtered.shape == step_current.shape
    assert filtered[0, -1, 0].item() > 0.0


if __name__ == "__main__":
    main()
