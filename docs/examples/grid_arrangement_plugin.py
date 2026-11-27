"""Worked example: a custom sensor and neuron layout as a plugin.

The smallest complete example of the extension path described in
``docs/how-to/extend/add_grid_arrangement.md``:

1. Define a layout inheriting :class:`~sensoryforge.core.grid_base.BaseGrid`
   whose constructor takes the area to fill (``xlim``, ``ylim``) and ``device``.
2. Implement ``get_all_coordinates()``, ``to_dict()``/``from_config()`` (a
   full round trip -- every constructor parameter reappears in ``to_dict()``)
   and ``get_param_spec()``.
3. Register it with ``GRID_REGISTRY`` (what a plugin package's entry point
   or ``register_components.py`` would do).
4. Check the component contract, then name it in a config as the sensor
   layout and as the neuron layout of a real ``SimulationEngine`` run.

Run it directly: ``python docs/examples/grid_arrangement_plugin.py``. It is
also executed by ``tests/docs/test_docs_examples.py``.
"""

from __future__ import annotations

import math
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
    """A sunflower spiral filling the largest circle that fits in the area.

    Point ``k`` (``k = 0, ..., N - 1``) sits at radius
    ``r_k = pitch_mm * sqrt(k)`` from the area's centre and angle
    ``theta_k = k * golden_angle``. The layout decides ``N``: every ``k``
    whose radius still fits. Purely spatial, no randomness.

    Args:
        xlim: ``(min, max)`` of the area along x, in mm.
        ylim: ``(min, max)`` of the area along y, in mm.
        device: PyTorch device for the returned coordinates.
        pitch_mm: Radial spacing scale in mm.

    Raises:
        ValueError: If ``pitch_mm`` is not positive.
    """

    _GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

    def __init__(
        self,
        xlim: Tuple[float, float] = (-1.0, 1.0),
        ylim: Tuple[float, float] = (-1.0, 1.0),
        device: torch.device | str = "cpu",
        pitch_mm: float = 0.1,
    ) -> None:
        if not pitch_mm > 0:
            raise ValueError(f"pitch_mm must be > 0, got {pitch_mm}")
        super().__init__(tuple(xlim), tuple(ylim), device)
        self.pitch_mm = float(pitch_mm)
        radius = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2
        n_points = int((radius / self.pitch_mm) ** 2 + 1e-9) + 1
        k = torch.arange(n_points, dtype=torch.float64)
        r = self.pitch_mm * torch.sqrt(k)
        theta = k * self._GOLDEN_ANGLE
        x = (xlim[0] + xlim[1]) / 2 + r * torch.cos(theta)
        y = (ylim[0] + ylim[1]) / 2 + r * torch.sin(theta)
        self._coords = torch.stack([x, y], dim=1).to(self.device, torch.float32)

    def get_all_coordinates(self) -> torch.Tensor:
        return self._coords

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["pitch_mm"] = self.pitch_mm
        return result

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpiralArrangement":
        keys = {"xlim", "ylim", "device", "pitch_mm"}
        return cls(**{k: v for k, v in config.items() if k in keys})

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                "pitch_mm",
                label="Pitch",
                dtype="float",
                default=0.1,
                min_val=1e-4,
                unit="mm",
            ),
        ]


def main() -> None:
    # 3. Register -- exactly what a plugin package's register() does.
    GRID_REGISTRY.register("spiral_example", SpiralArrangement)

    # 4a. The shared component contract (shape, round trip, param spec, and
    # that the points fill the area given, the same way twice).
    check_component("grid", SpiralArrangement, SpiralArrangement(pitch_mm=0.2))

    # 4b. Name it in a config: as the sensors and as the neurons.
    config = SensoryForgeConfig(
        grids=[
            GridConfig(
                name="spiral_grid",
                arrangement="spiral_example",
                rows=12,  # rows, cols and spacing give the area (2.2 x 1.4 mm)
                cols=8,
                spacing=0.2,
                layout_params={"pitch_mm": 0.1},
            )
        ],
        populations=[
            PopulationConfig(
                name="spiral population",
                neuron_type="SA",
                neuron_arrangement="spiral_example",
                neuron_layout_params={"pitch_mm": 0.25},
                innervation_method="gaussian",
                filter_method="sa",
                seed=1,
            )
        ],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0),
    )
    engine = SimulationEngine(config)
    bank = engine.populations[0]["bank"]
    print(f"sensors: {bank.num_receptors}, neurons: {bank.neuron_centers.shape[0]}")

    # The stimulus is a raster over the area; the engine samples it at each
    # sensor's own (x, y) position.
    stimulus = torch.zeros(1, 30, 12, 8)  # [batch, time, x, y]
    stimulus[:, 10:] = 30.0
    results = engine.run(stimulus, return_intermediates=True)
    spikes = results["spiral population"]["spikes"]
    print(f"spikes shape: {tuple(spikes.shape)}, total spikes: {int(spikes.sum())}")


if __name__ == "__main__":
    main()
