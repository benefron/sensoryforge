"""Worked example: define, register, and run a minimal stimulus component (U3).

Runnable companion to ``docs/developer_guide/add_stimulus.md``.

1. Define a stimulus class inheriting :class:`~sensoryforge.stimuli.base.BaseStimulus`.
2. Implement ``get_param_spec()`` (required on every component, G1).
3. Register it with ``STIMULUS_REGISTRY``.
4. Run it through the shared contract check
   (``sensoryforge.testing.contracts.check_component``) and generate one
   frame directly with the ``(xx, yy)`` meshgrid ``forward()`` takes.

Run it directly: ``python docs/examples/plugin_stimulus.py``. It is also
executed by ``tests/docs/test_docs_examples.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from sensoryforge.registry import STIMULUS_REGISTRY
from sensoryforge.stimuli.base import BaseStimulus, ParamSpec


class AnnulusStimulus(BaseStimulus):
    """A ring (annulus) of activation: high between ``r_inner`` and
    ``r_outer`` from the origin, zero elsewhere.

    Args:
        amplitude: Peak activation.
        r_inner: Inner radius in mm.
        r_outer: Outer radius in mm.
    """

    def __init__(
        self, amplitude: float = 1.0, r_inner: float = 0.5, r_outer: float = 1.0
    ) -> None:
        super().__init__()
        if r_outer <= r_inner:
            raise ValueError(f"r_outer ({r_outer}) must exceed r_inner ({r_inner})")
        self.amplitude = amplitude
        self.r_inner = r_inner
        self.r_outer = r_outer

    def forward(
        self, xx: torch.Tensor, yy: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Generate the ring frame.

        Args:
            xx: X-coordinates, mm, any shape (typically ``[H, W]``).
            yy: Y-coordinates, mm, same shape as ``xx``.

        Returns:
            Same shape as ``xx``: ``amplitude`` inside the annulus, else 0.
        """
        r = torch.sqrt(xx**2 + yy**2)
        mask = (r >= self.r_inner) & (r <= self.r_outer)
        return torch.where(
            mask, torch.full_like(r, self.amplitude), torch.zeros_like(r)
        )

    def reset_state(self) -> None:
        """No internal state to reset (BaseStimulus contract)."""
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AnnulusStimulus":
        return cls(
            amplitude=config.get("amplitude", 1.0),
            r_inner=config.get("r_inner", 0.5),
            r_outer=config.get("r_outer", 1.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "amplitude": self.amplitude,
            "r_inner": self.r_inner,
            "r_outer": self.r_outer,
        }

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                "amplitude",
                dtype="float",
                default=1.0,
                min_val=0.0,
                max_val=500.0,
                unit="mA",
                choices=None,
                help="Peak activation inside the ring.",
                group="Amplitude",
                advanced=False,
            ),
            ParamSpec(
                "r_inner",
                dtype="float",
                default=0.5,
                min_val=0.0,
                max_val=50.0,
                unit="mm",
                choices=None,
                help="Inner radius of the ring.",
                group="Spatial",
                advanced=False,
            ),
            ParamSpec(
                "r_outer",
                dtype="float",
                default=1.0,
                min_val=0.0,
                max_val=50.0,
                unit="mm",
                choices=None,
                help="Outer radius of the ring.",
                group="Spatial",
                advanced=False,
            ),
        ]


def main() -> None:
    STIMULUS_REGISTRY.register("annulus_demo", AnnulusStimulus)

    from sensoryforge.testing.contracts import check_component

    check_component("stimulus", AnnulusStimulus)

    stim = STIMULUS_REGISTRY.create("annulus_demo", r_inner=0.5, r_outer=1.5)
    xx, yy = torch.meshgrid(
        torch.linspace(-2, 2, 40), torch.linspace(-2, 2, 40), indexing="ij"
    )
    frame = stim(xx, yy)
    print(f"frame shape: {tuple(frame.shape)}, nonzero: {int((frame > 0).sum())}")
    assert frame.shape == xx.shape
    assert (frame > 0).any()


if __name__ == "__main__":
    main()
