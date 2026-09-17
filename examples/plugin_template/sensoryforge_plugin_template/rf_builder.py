"""``radial_falloff``: a receptive-field builder with a linear radial falloff.

Every neuron-receptor pair within ``radius_mm`` gets weight
``max(0, 1 - distance / radius_mm)``; pairs farther apart get 0. This is the
plugin-package analogue of ``docs/examples/rf_builder_plugin.py``'s
``RingRFBuilder`` (an indicator kernel) -- here the kernel decays smoothly
instead of switching on/off, to show a second, independent worked example of
the same extension point (``sensoryforge.core.innervation.BaseInnervation``,
see ``docs/developer_guide/add_rf_builder.md``).
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from sensoryforge.core.innervation import BaseInnervation
from sensoryforge.stimuli.base import ParamSpec


class RadialFalloffRFBuilder(BaseInnervation):
    """Linear radial-falloff receptive fields.

    ``weights[i, j] = max(0, 1 - distance(neuron_i, receptor_j) / radius_mm)``,
    vectorised with :func:`torch.cdist` (no per-neuron/per-receptor loops).

    Attributes:
        radius_mm: Falloff radius in mm; weight reaches 0 at this distance.
    """

    def __init__(
        self,
        receptor_coords: torch.Tensor,
        neuron_centers: torch.Tensor,
        radius_mm: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the radial-falloff builder.

        Args:
            receptor_coords: Receptor positions ``[M, 2]`` (x, y) in mm.
            neuron_centers: Neuron centre positions ``[N, 2]`` (x, y) in mm.
            radius_mm: Falloff radius in mm. Must be positive.
            device: PyTorch device.

        Raises:
            ValueError: If ``radius_mm`` is not positive.
        """
        super().__init__(receptor_coords, neuron_centers, device)
        if radius_mm <= 0:
            raise ValueError(f"radius_mm must be positive, got {radius_mm}")
        self.radius_mm = float(radius_mm)

    def compute_weights(self, **kwargs: Any) -> torch.Tensor:
        """Return ``[N, M]`` linear radial-falloff weights.

        Returns:
            Weight tensor ``[num_neurons, num_receptors]`` in ``[0, 1]``.
        """
        distances = torch.cdist(self.neuron_centers, self.receptor_coords)  # [N, M]
        weights = torch.clamp(1.0 - distances / self.radius_mm, min=0.0)
        return weights

    def to_dict(self) -> Dict[str, Any]:
        """Serialise builder parameters (F-049: every constructor kwarg but
        the receptor/neuron tensors, per ``_TO_DICT_EXCLUDE_PARAMS``)."""
        result = super().to_dict()
        result["radius_mm"] = self.radius_mm
        return result

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """Return this builder's configurable parameters (G1)."""
        return [
            ParamSpec(
                "radius_mm",
                dtype="float",
                default=0.5,
                min_val=1e-6,
                unit="mm",
                help="Distance at which the linear falloff reaches zero.",
                group="Receptive Field",
            ),
        ]


def register() -> None:
    """Entry-point target: register ``RadialFalloffRFBuilder``.

    Called with no arguments by
    :func:`sensoryforge.plugins.discover_entry_point_plugins` when this
    distribution is installed and its ``[project.entry-points
    ."sensoryforge.components"]`` entry loads.
    """
    from sensoryforge.registry import INNERVATION_REGISTRY

    INNERVATION_REGISTRY.register("radial_falloff", RadialFalloffRFBuilder)
