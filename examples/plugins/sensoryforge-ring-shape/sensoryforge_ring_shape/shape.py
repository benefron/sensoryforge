"""The ``ring`` shape: a Gaussian-profile ring around the contact's centre.

The profile at distance r (mm) from the centre is::

    exp(-(r - radius_mm)^2 / (2 * width_mm^2))

so it is 1 on the circle of radius ``radius_mm`` and falls off with the
standard deviation ``width_mm`` on either side. The world engine multiplies the
profile by the contact's ``amplitude`` itself, so the function does not read it.
The formula is a teaching example, not a model of any real stimulus; the ranges
below are placeholders for a world to override.
"""

from __future__ import annotations

import torch

from sensoryforge.stimuli.base import ParamSpec


def ring(x: torch.Tensor, y: torch.Tensor, params: dict) -> torch.Tensor:
    """Ring profile.

    Args:
        x: Offsets from the centre, mm, shape ``[..., *S]``.
        y: Offsets from the centre, mm, same shape as ``x``.
        params: ``radius_mm`` and ``width_mm`` (mm), tensors broadcasting
            against ``x``.

    Returns:
        The ring values, same shape and dtype as ``x``.
    """
    r = torch.sqrt(x**2 + y**2)
    return torch.exp(
        -((r - params["radius_mm"]) ** 2) / (2.0 * params["width_mm"] ** 2)
    )


RING_SPECS = [
    ParamSpec("amplitude", dtype="float", default=1.0, min_val=0.0, max_val=1.0e4),
    ParamSpec(
        "radius_mm", dtype="float", default=0.3, min_val=0.0, max_val=1000.0, unit="mm"
    ),
    ParamSpec(
        "width_mm", dtype="float", default=0.1, min_val=0.01, max_val=1000.0, unit="mm"
    ),
]


def register() -> None:
    """Entry-point target: register the ring shape with the world engine.

    Registering the same function and specs again is a no-op, so this may run
    from both an entry point and a config.
    """
    from sensoryforge.world import register_shape

    register_shape("ring", ring, RING_SPECS)
