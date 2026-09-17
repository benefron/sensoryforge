"""``gain_threshold``: a processing layer applying a gain and rectifying
threshold to receptor responses.

``output = relu(receptor_responses * gain - threshold)``. This is a minimal,
stateless worked example of ``sensoryforge.core.processing.BaseProcessingLayer``
(see ``IdentityLayer`` and ``OnOffLayer`` in ``sensoryforge/core/processing.py``
for the built-in layers this follows the same contract as). It does not
change the receptor axis (``M`` in, ``M`` out), so it needs no coordinate
transform and ``REQUIRES_RECEPTOR_COORDS`` stays at its default, ``False``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from sensoryforge.core.processing import BaseProcessingLayer
from sensoryforge.stimuli.base import ParamSpec


class GainThresholdLayer(BaseProcessingLayer):
    """Gain then rectifying threshold: ``relu(x * gain - threshold)``.

    Attributes:
        gain: Multiplicative gain applied before thresholding.
        threshold: Subtracted before the rectifying non-linearity.
    """

    def __init__(self, gain: float = 1.0, threshold: float = 0.0) -> None:
        """Initialize the layer.

        Args:
            gain: Multiplicative gain. Must be non-negative.
            threshold: Value subtracted before rectification.

        Raises:
            ValueError: If ``gain`` is negative.
        """
        super().__init__()
        if gain < 0:
            raise ValueError(f"gain must be non-negative, got {gain}")
        self.gain = float(gain)
        self.threshold = float(threshold)

    def forward(
        self,
        receptor_responses: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Apply gain and rectifying threshold.

        Args:
            receptor_responses: Any shape ending in the receptor axis, e.g.
                ``[batch, time, M]``.
            metadata: Ignored.

        Returns:
            Same shape as ``receptor_responses``.
        """
        return torch.relu(receptor_responses * self.gain - self.threshold)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the ``{"method": ..., ...}`` shape
        ``ProcessingPipeline.from_config`` dispatches on."""
        return {
            "method": "gain_threshold",
            "gain": self.gain,
            "threshold": self.threshold,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GainThresholdLayer":
        """Reconstruct from :meth:`to_dict`'s output.

        Args:
            config: A dict with (at least) ``gain`` and ``threshold``; a
                ``method``/``type`` dispatch key is tolerated and ignored.

        Returns:
            A new :class:`GainThresholdLayer`.
        """
        return cls(
            gain=config.get("gain", 1.0),
            threshold=config.get("threshold", 0.0),
        )

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """Return this layer's configurable parameters (G1)."""
        return [
            ParamSpec(
                "gain",
                dtype="float",
                default=1.0,
                min_val=0.0,
                help="Multiplicative gain applied before thresholding.",
                group="Processing",
            ),
            ParamSpec(
                "threshold",
                dtype="float",
                default=0.0,
                help="Value subtracted before the rectifying non-linearity.",
                group="Processing",
            ),
        ]


def register() -> None:
    """Entry-point target: register ``GainThresholdLayer``.

    Called with no arguments by
    :func:`sensoryforge.plugins.discover_entry_point_plugins` when this
    distribution is installed and its ``[project.entry-points
    ."sensoryforge.components"]`` entry loads.
    """
    from sensoryforge.registry import PROCESSING_REGISTRY

    PROCESSING_REGISTRY.register("gain_threshold", GainThresholdLayer)
