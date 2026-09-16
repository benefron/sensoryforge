"""Worked example: processing layers, using OnOffLayer as the model (M3).

The smallest complete example of the extension path described in
``docs/extending/add_processing_layer.md``:

1. Define a custom layer inheriting
   :class:`~sensoryforge.core.processing.BaseProcessingLayer` (a simple gain
   layer -- the contract a real plugin follows).
2. Register it with ``PROCESSING_REGISTRY`` (what a plugin package's entry
   point or ``register_components.py`` would do).
3. Use the shipped :class:`~sensoryforge.core.processing.OnOffLayer` -- a
   centre-surround difference-of-Gaussians filter that splits receptor
   responses into a rectified ON plane and a rectified OFF plane -- both
   standalone, and wired into a ``PopulationInput.processing`` list and run
   through a real :class:`~sensoryforge.core.simulation_engine.SimulationEngine`.
4. Show *which* plane responds to a bright vs. a dark stimulus -- the
   anti-plausibility check every OnOffLayer test in the suite makes,
   because a plane swap or a collapse to a single plane looks fine in any
   shape check.

Run it directly: ``python docs/examples/onoff_processing_layer.py``. It is
also executed by ``tests/docs/test_docs_examples.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    PopulationInput,
    RFBuilderConfig,
    SensoryForgeConfig,
)
from sensoryforge.core.processing import (
    BaseProcessingLayer,
    OnOffLayer,
    ProcessingPipeline,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.registry import PROCESSING_REGISTRY
from sensoryforge.stimuli.base import ParamSpec

# --------------------------------------------------------------------------- #
# 1-2. A minimal custom processing layer, registered the same way OnOffLayer
#      is (sensoryforge/register_components.py).
# --------------------------------------------------------------------------- #


class GainLayer(BaseProcessingLayer):
    """Multiply every receptor response by a fixed gain."""

    def __init__(self, gain: float = 2.0) -> None:
        super().__init__()
        self.gain = float(gain)

    def forward(
        self,
        receptor_responses: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        return receptor_responses * self.gain

    def to_dict(self) -> Dict[str, Any]:
        return {"method": "gain", "params": {"gain": self.gain}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GainLayer":
        params = dict(config.get("params") or {})
        return cls(gain=config.get("gain", params.get("gain", 2.0)))

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                "gain",
                dtype="float",
                default=2.0,
                min_val=0.0,
                max_val=100.0,
                unit=None,
                choices=None,
                help="Multiplier applied to every receptor response.",
                group="Gain",
                advanced=False,
            )
        ]


PROCESSING_REGISTRY.register("gain", GainLayer)

pipeline = ProcessingPipeline.from_config([{"method": "gain", "params": {"gain": 3.0}}])
gained = pipeline(torch.ones(1, 1, 4))
assert torch.equal(gained, torch.full((1, 1, 4), 3.0))
print("GainLayer registered and applied: x3 on a unit input ->", gained.tolist())

# --------------------------------------------------------------------------- #
# 3. OnOffLayer standalone: a difference-of-Gaussians centre-surround split.
# --------------------------------------------------------------------------- #

coords = torch.stack(
    [torch.arange(6, dtype=torch.float32) * 0.2, torch.zeros(6)], dim=-1
)  # 6 receptors on a line, 0.2 mm apart
onoff = OnOffLayer(coords, sigma_center_mm=0.15, sigma_surround_mm=0.45)

bright = torch.zeros(1, 1, 6)
bright[0, 0, 3] = 5.0  # a bright spot at receptor 3
out_bright = onoff(bright)
on_bright, off_bright = out_bright[..., :6], out_bright[..., 6:]
assert on_bright[0, 0, 3] > 0 and off_bright[0, 0, 3] == 0
print(
    "Bright spot (receptor 3): ON =",
    round(on_bright[0, 0, 3].item(), 3),
    "OFF =",
    off_bright[0, 0, 3].item(),
)

dark = torch.zeros(1, 1, 6)
dark[0, 0, 3] = -5.0  # a dark spot at the same receptor
out_dark = onoff(dark)
on_dark, off_dark = out_dark[..., :6], out_dark[..., 6:]
assert off_dark[0, 0, 3] > 0 and on_dark[0, 0, 3] == 0
print(
    "Dark spot   (receptor 3): ON =",
    on_dark[0, 0, 3].item(),
    "OFF =",
    round(off_dark[0, 0, 3].item(), 3),
)

# --------------------------------------------------------------------------- #
# 4. OnOffLayer wired into a population input, through a real SimulationEngine.
# --------------------------------------------------------------------------- #

config = SensoryForgeConfig(
    grids=[GridConfig(name="Grid", arrangement="grid", rows=10, cols=10, spacing=0.2)],
    populations=[
        PopulationConfig(
            name="onoff_pop",
            neuron_type="SA",
            neurons_per_row=4,
            inputs=[
                PopulationInput(
                    grid="Grid",
                    rf=RFBuilderConfig(method="gaussian"),
                    processing=[{"method": "onoff"}],
                )
            ],
        )
    ],
)
engine = SimulationEngine(config)
pop = engine.populations[0]
ctx = pop["inputs"][0]
# 10x10 = 100 raw receptors; the bank is built on 200 (ON + OFF planes).
assert ctx["receptor_coords"].shape[0] == 100
assert ctx["bank"].num_receptors == 200

stimulus = torch.zeros(1, 3, 10, 10)
stimulus[:, :, 5, 5] = 30.0  # a bright spot in the middle of the frame
results = engine.run(stimulus, return_intermediates=True)
drive = results["onoff_pop"]["drive"]
print(
    f"\nOnOffLayer through SimulationEngine: bank M={ctx['bank'].num_receptors}, "
    f"drive shape={list(drive.shape)}, finite={bool(torch.isfinite(drive).all())}"
)

print("\nDone.")
