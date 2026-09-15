"""Concrete, registrable grid-arrangement classes (G3).

``ReceptorGrid`` (``core/grid.py``) implements every arrangement's coordinate
math internally, dispatched on its ``arrangement`` constructor argument. Before
this module, ``GRID_REGISTRY`` registered each arrangement name against the
placeholder ``str`` (see ``register_components.py``'s old comment "these are
string identifiers, not classes"), so ``list-components``, the contract tests
(G4), and plugin discovery (G2) had nothing to introspect for grid arrangements.

Each class here is a thin ``ReceptorGrid`` subclass that fixes ``arrangement``
to one value and exposes ``get_param_spec()``. They delegate all coordinate
generation to ``ReceptorGrid.__init__`` — no arrangement math is duplicated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.stimuli.base import ParamSpec


def _shared_param_spec() -> List[ParamSpec]:
    return [
        ParamSpec(
            "grid_size",
            dtype="int",
            default=80,
            min_val=1,
            max_val=1000,
            tooltip="Number of points along each axis (rows == cols).",
        ),
        ParamSpec(
            "spacing",
            dtype="float",
            default=0.15,
            min_val=0.001,
            max_val=100.0,
            unit="mm",
            tooltip="Distance between adjacent receptors.",
        ),
        ParamSpec(
            "center_x",
            dtype="float",
            default=0.0,
            unit="mm",
            group="Position",
        ),
        ParamSpec(
            "center_y",
            dtype="float",
            default=0.0,
            unit="mm",
            group="Position",
        ),
        ParamSpec(
            "seed",
            dtype="int",
            default=None,
            min_val=0,
            max_val=2**31 - 1,
            tooltip=(
                "Seed for the random jitter of jittered_grid, blue_noise and "
                "poisson (F-050); None draws from the global RNG."
            ),
            advanced=True,
        ),
    ]


class _FixedArrangementGrid(ReceptorGrid):
    """Base for a ``ReceptorGrid`` subclass that pins one ``arrangement``."""

    ARRANGEMENT: str = "grid"

    def __init__(
        self,
        grid_size: int | Tuple[int, int] = 80,
        spacing: float = 0.15,
        center: Tuple[float, float] = (0.0, 0.0),
        density: Optional[float] = None,
        device: torch.device | str = "cpu",
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            grid_size=grid_size,
            spacing=spacing,
            center=center,
            arrangement=self.ARRANGEMENT,
            density=density,
            device=device,
            seed=seed,
        )

    _CONSTRUCTOR_KEYS = frozenset(
        {"grid_size", "spacing", "center", "density", "device", "seed"}
    )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "_FixedArrangementGrid":
        """Create an instance from a config dict.

        Accepts either a constructor-shaped config or the ``to_dict()`` output
        (which adds ``type``/``xlim``/``ylim``/``arrangement`` -- derived, not
        constructor arguments); both are filtered down to the actual
        constructor keywords.
        """
        config = {k: v for k, v in config.items() if k in cls._CONSTRUCTOR_KEYS}
        if "center" in config and isinstance(config["center"], list):
            config["center"] = tuple(config["center"])
        if isinstance(config.get("grid_size"), list):
            config["grid_size"] = tuple(config["grid_size"])
        return cls(**config)

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return _shared_param_spec()


class GridArrangement(_FixedArrangementGrid):
    """Regular rectangular receptor grid."""

    ARRANGEMENT = "grid"


class PoissonArrangement(_FixedArrangementGrid):
    """Poisson-like (jittered-density) receptor arrangement."""

    ARRANGEMENT = "poisson"


class HexArrangement(_FixedArrangementGrid):
    """Hexagonally packed receptor arrangement."""

    ARRANGEMENT = "hex"


class JitteredGridArrangement(_FixedArrangementGrid):
    """Regular grid with per-receptor positional jitter."""

    ARRANGEMENT = "jittered_grid"


class BlueNoiseArrangement(_FixedArrangementGrid):
    """Blue-noise (jitter + Lloyd relaxation) receptor arrangement."""

    ARRANGEMENT = "blue_noise"
