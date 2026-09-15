"""Worked example: define, register and simulate a receptive-field builder (I8).

The smallest complete example of the extension path described in
``docs/developer_guide/add_rf_builder.md``:

1. Define a builder inheriting :class:`~sensoryforge.core.innervation.BaseInnervation`.
2. Implement ``compute_weights()``, ``to_dict()`` and ``get_param_spec()``.
3. Register it with ``INNERVATION_REGISTRY`` (what a plugin package's entry
   point or ``register_components.py`` would do).
4. Check the component contract, build a bank, and run ``SimulationEngine``
   with ``innervation_method`` set to the new builder.

Run it directly: ``python docs/examples/rf_builder_plugin.py``. It is also
executed by ``tests/docs/test_docs_examples.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.innervation import BaseInnervation, build_population_bank
from sensoryforge.core.rf_bank import ReceptiveFieldBank
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.registry import INNERVATION_REGISTRY
from sensoryforge.stimuli.base import ParamSpec
from sensoryforge.testing.contracts import check_component


class RingRFBuilder(BaseInnervation):
    """Annular receptive fields: unit weight on receptors within a ring.

    Every receptor whose distance from a neuron's centre lies in
    ``[r_in_mm, r_out_mm]`` gets weight 1; all others 0. Deterministic.

    Args:
        receptor_coords: ``[M, 2]`` receptor positions (x, y) in mm.
        neuron_centers: ``[N, 2]`` neuron centres in mm.
        r_in_mm: Inner radius of the ring in mm.
        r_out_mm: Outer radius of the ring in mm.
        device: Device for the weights.
    """

    def __init__(
        self,
        receptor_coords: torch.Tensor,
        neuron_centers: torch.Tensor,
        r_in_mm: float = 0.1,
        r_out_mm: float = 0.4,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(receptor_coords, neuron_centers, device)
        if r_out_mm <= r_in_mm:
            raise ValueError(
                f"r_out_mm must exceed r_in_mm, got r_in_mm={r_in_mm}, "
                f"r_out_mm={r_out_mm}"
            )
        self.r_in_mm = float(r_in_mm)
        self.r_out_mm = float(r_out_mm)

    def compute_weights(self, **kwargs: Any) -> torch.Tensor:
        """Return ``[N, M]`` ring-indicator weights (vectorised, no loops)."""
        distances = torch.cdist(self.neuron_centers, self.receptor_coords)  # [N, M]
        inside = (distances >= self.r_in_mm) & (distances <= self.r_out_mm)
        return inside.to(torch.float32)

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({"r_in_mm": self.r_in_mm, "r_out_mm": self.r_out_mm})
        return result

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("r_in_mm", dtype="float", default=0.1, min_val=0.0, unit="mm"),
            ParamSpec("r_out_mm", dtype="float", default=0.4, min_val=0.0, unit="mm"),
        ]


def main() -> None:
    # 3. Register -- exactly what a plugin package's register() does.
    INNERVATION_REGISTRY.register("ring_example", RingRFBuilder)

    # 4a. The shared component contract (shapes, round trip, build()).
    check_component(
        "innervation",
        RingRFBuilder,
        RingRFBuilder(torch.rand(30, 2), torch.rand(4, 2)),
    )

    # 4b. Build a bank directly on a small regular grid.
    from sensoryforge.core.grid import ReceptorGrid

    coords = ReceptorGrid(grid_size=(8, 8), spacing=0.15).get_receptor_coordinates()
    bank = build_population_bank(
        receptor_coords=coords,
        innervation_method="ring_example",
        neurons_per_row=2,
        r_in_mm=0.1,
        r_out_mm=0.4,
    )
    if not isinstance(bank, ReceptiveFieldBank):
        raise RuntimeError(f"expected a ReceptiveFieldBank, got {type(bank)!r}")
    print(f"bank: {bank.num_neurons} neurons x {bank.num_receptors} receptors")
    print(f"provenance builder: {bank.provenance['builder']}")

    # 4c. Simulate through the engine with the new method and its parameters.
    config = SensoryForgeConfig(
        grids=[
            GridConfig(name="grid", arrangement="grid", rows=8, cols=8, spacing=0.15)
        ],
        populations=[
            PopulationConfig(
                name="ring population",
                neuron_type="SA",
                innervation_method="ring_example",
                innervation_params={"r_in_mm": 0.1, "r_out_mm": 0.4},
                neurons_per_row=2,
                filter_method="sa",
                seed=1,
            )
        ],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0),
    )
    engine = SimulationEngine(config)
    stimulus = torch.zeros(60, 8, 8)
    stimulus[10:] = 30.0
    results = engine.run(stimulus, return_intermediates=True)
    spikes = results["ring population"]["spikes"]
    drive = results["ring population"]["drive"]
    print(f"drive shape:  {tuple(drive.shape)}")
    print(f"spikes shape: {tuple(spikes.shape)}, total spikes: {int(spikes.sum())}")

    engine_bank = engine.populations[0]["bank"]
    if engine_bank.provenance["builder"] != "ring_example":
        raise RuntimeError("engine did not build the population with ring_example")
    if not torch.equal(engine_bank.weights, bank.weights):
        raise RuntimeError("engine bank differs from the directly built bank")
    if tuple(spikes.shape) != (1, 60, 4):
        raise RuntimeError(f"unexpected spikes shape {tuple(spikes.shape)}")


if __name__ == "__main__":
    main()
