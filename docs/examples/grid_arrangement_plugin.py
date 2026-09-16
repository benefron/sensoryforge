"""Worked example: a new receptor grid arrangement as a plugin (Wave L5).

The smallest complete example of the extension path described in
``docs/extending/add_grid_arrangement.md``:

1. Define a grid arrangement inheriting
   :class:`~sensoryforge.core.grid_base.BaseGrid`.
2. Implement ``get_all_coordinates()``, ``to_dict()``/``from_config()`` (a
   full round trip -- every constructor parameter reappears in ``to_dict()``)
   and ``get_param_spec()``.
3. Register it with ``GRID_REGISTRY`` (what a plugin package's entry point
   or ``register_components.py`` would do).
4. Check the component contract, then use it as a ``GridConfig.coords_file``
   grid (Wave L1/L4's coordinate-import path) so it drives a real
   ``SimulationEngine`` run without touching engine code.

Run it directly: ``python docs/examples/grid_arrangement_plugin.py``. It is
also executed by ``tests/docs/test_docs_examples.py``.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.grid_base import BaseGrid
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.registry import GRID_REGISTRY
from sensoryforge.stimuli.base import ParamSpec
from sensoryforge.testing.contracts import check_component


class SpiralArrangement(BaseGrid):
    """Receptors on an Archimedean spiral -- a fovea-style radial layout.

    Point ``k`` (``k = 0, ..., n_points - 1``) sits at radius
    ``r_k = pitch_mm * sqrt(k)`` (equal-area rings, like a fovea's receptor
    density falling off from the centre) and angle
    ``theta_k = k * golden_angle``, the golden angle giving the same
    non-repeating packing sunflower seed heads use. Purely spatial, no
    randomness: two instances with the same parameters are bit-identical.

    Args:
        n_points: Number of receptors.
        pitch_mm: Radial spacing scale in mm (``r_k = pitch_mm * sqrt(k)``).
        center: ``(x0, y0)`` spiral centre in mm.
        device: PyTorch device for the returned coordinates.
    """

    _GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

    def __init__(
        self,
        n_points: int = 200,
        pitch_mm: float = 0.1,
        center: Tuple[float, float] = (0.0, 0.0),
        device: torch.device | str = "cpu",
    ) -> None:
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1, got {n_points}")
        self.n_points = int(n_points)
        self.pitch_mm = float(pitch_mm)
        self.center = tuple(center)
        k = torch.arange(self.n_points, dtype=torch.float64)
        r = self.pitch_mm * torch.sqrt(k)
        theta = k * self._GOLDEN_ANGLE
        x = self.center[0] + r * torch.cos(theta)
        y = self.center[1] + r * torch.sin(theta)
        coords = torch.stack([x, y], dim=1).to(torch.float32)
        x_max = float(coords[:, 0].abs().max()) if self.n_points else 0.0
        y_max = float(coords[:, 1].abs().max()) if self.n_points else 0.0
        xlim = (self.center[0] - x_max, self.center[0] + x_max)
        ylim = (self.center[1] - y_max, self.center[1] + y_max)
        super().__init__(xlim, ylim, device)
        self._coords = coords.to(self.device)

    def get_all_coordinates(self) -> torch.Tensor:
        return self._coords

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "n_points": self.n_points,
                "pitch_mm": self.pitch_mm,
                "center": list(self.center),
            }
        )
        return result

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpiralArrangement":
        keys = {"n_points", "pitch_mm", "center", "device"}
        kwargs = {k: v for k, v in config.items() if k in keys}
        if isinstance(kwargs.get("center"), list):
            kwargs["center"] = tuple(kwargs["center"])
        return cls(**kwargs)

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("n_points", dtype="int", default=200, min_val=1, max_val=100000),
            ParamSpec("pitch_mm", dtype="float", default=0.1, min_val=1e-4, unit="mm"),
        ]


def main() -> None:
    # 3. Register -- exactly what a plugin package's register() does.
    GRID_REGISTRY.register("spiral_example", SpiralArrangement)

    # 4a. The shared component contract (shape, round trip, param spec).
    check_component("grid", SpiralArrangement, SpiralArrangement(n_points=50))

    # 4b. Build the arrangement directly and inspect it.
    spiral = SpiralArrangement(n_points=64, pitch_mm=0.12)
    coords = spiral.get_all_coordinates()
    print(f"spiral coords shape: {tuple(coords.shape)}")
    reconstructed = SpiralArrangement.from_config(spiral.to_dict())
    if not torch.equal(reconstructed.get_all_coordinates(), coords):
        raise RuntimeError("from_config(to_dict()) did not reproduce coordinates")

    # 4c. Feed it into SimulationEngine via GridConfig.coords_file (Wave L1),
    # the same import path a hand-authored or imported receptor layout uses
    # -- no engine code needs to know about SpiralArrangement itself.
    with tempfile.TemporaryDirectory() as tmp:
        coords_path = Path(tmp) / "spiral_coords.pt"
        torch.save(coords, coords_path)

        config = SensoryForgeConfig(
            grids=[GridConfig(name="spiral_grid", coords_file=str(coords_path))],
            populations=[
                PopulationConfig(
                    name="spiral population",
                    neuron_type="SA",
                    innervation_method="gaussian",
                    neurons_per_row=3,
                    filter_method="sa",
                    seed=1,
                )
            ],
            simulation=SimulationConfig(device="cpu", dt_ms=1.0),
        )
        engine = SimulationEngine(config)
        bank = engine.populations[0]["bank"]
        if bank.num_receptors != 64:
            raise RuntimeError(
                f"expected 64 receptors from the spiral, got {bank.num_receptors}"
            )

        # A composite grid (which SpiralArrangement enters through here,
        # via coords_file) has no fixed pixel raster of its own -- the
        # stimulus is a spatial field over the grid's bounding box, sampled
        # at each receptor's own (x, y) position (Wave L3). Any resolution
        # works; 40x40 here.
        stimulus = torch.zeros(1, 30, 40, 40)  # [batch, time, H, W]
        stimulus[:, 10:] = 30.0
        results = engine.run(stimulus, return_intermediates=True)
        spikes = results["spiral population"]["spikes"]
        print(f"spikes shape: {tuple(spikes.shape)}, total spikes: {int(spikes.sum())}")


if __name__ == "__main__":
    main()
