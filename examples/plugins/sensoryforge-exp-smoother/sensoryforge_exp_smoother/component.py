"""ExpSmootherFilterTorch: a new temporal filter.

Implements the ``BaseFilter`` contract.
Full guide: https://benefron.github.io/sensoryforge/how-to/extend/add_filter/
"""

from __future__ import annotations

from typing import List, Optional

import torch

from sensoryforge.filters.base import BaseFilter
from sensoryforge.stimuli.base import ParamSpec


class ExpSmootherFilterTorch(BaseFilter):
    """ExpSmootherFilterTorch temporal filter (placeholder single-pole lowpass).

    Args:
        tau: Time constant in ms.
        dt: Integration time step in ms.
    """

    def __init__(self, tau: float = 10.0, dt: float = 0.1) -> None:
        super().__init__(dt=dt)
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau

    def reset_state(self) -> None:
        """No state is kept between calls."""

    def forward(self, x: torch.Tensor, dt: Optional[float] = None) -> torch.Tensor:
        """Apply a single-pole lowpass over the time dimension.

        Args:
            x: Input drive ``[batch, time, N_neurons]`` in mA.
            dt: Optional override for the integration step (ms).

        Returns:
            Filtered current, same shape as ``x``.
        """
        step = dt if dt is not None else self.dt
        batch, steps, n = x.shape
        # Start from rest on every call: the engine runs one forward per stimulus
        # and a filter that carries state between calls gives a different answer
        # for the same stimulus the second time (the determinism check fails).
        state = torch.zeros(batch, n, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(steps):
            state = state + (x[:, t, :] - state) / self.tau * step
            outputs.append(state.unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def to_dict(self) -> dict:
        return {"tau": self.tau, "dt": self.dt}

    @classmethod
    def from_config(cls, config: dict) -> "ExpSmootherFilterTorch":
        return cls(tau=config.get("tau", 10.0), dt=config.get("dt", 0.1))

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("tau", dtype="float", default=10.0, min_val=0.01, unit="ms"),
        ]


def register() -> None:
    """Entry-point target: register ExpSmootherFilterTorch with the component registry.

    Called with no arguments by
    :func:`sensoryforge.plugins.discover_entry_point_plugins` when this
    distribution is installed and its
    ``[project.entry-points."sensoryforge.components"]`` entry loads.
    """
    from sensoryforge.registry import FILTER_REGISTRY

    FILTER_REGISTRY.register("exp_smoother", ExpSmootherFilterTorch)
