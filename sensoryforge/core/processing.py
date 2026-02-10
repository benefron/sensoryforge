"""Processing layer base classes for intermediate signal transformations.

This module defines the ``BaseProcessingLayer`` interface for inserting
composable transformations between the receptor grid and the innervation /
neuron stages.  Future layers include:

- ON/OFF centre-surround fields
- Lateral inhibition
- Cross-grid fusion

Current implementations:

- ``IdentityLayer``: Pass-through (no-op) — used as the default when no
  processing layers are specified.

All layers are ``nn.Module`` subclasses so they participate in PyTorch's
parameter/buffer tracking, device management, and backpropagation graph.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class BaseProcessingLayer(nn.Module):
    """Abstract base class for processing layers.

    A processing layer sits between the receptor grid output and the
    innervation → neuron stages.  It receives receptor responses (possibly
    with spatial metadata) and returns transformed responses of the same
    or different shape.

    Subclasses must implement :meth:`forward`.

    Example:
        >>> class GainLayer(BaseProcessingLayer):
        ...     def __init__(self, gain: float = 2.0):
        ...         super().__init__()
        ...         self.gain = gain
        ...     def forward(self, receptor_responses, metadata=None):
        ...         return receptor_responses * self.gain
    """

    @abstractmethod
    def forward(
        self,
        receptor_responses: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Apply processing to receptor responses.

        Args:
            receptor_responses: Receptor activations.  Typical shapes:
                - ``[batch, grid_h, grid_w]`` (static, grid-based)
                - ``[batch, time, grid_h, grid_w]`` (temporal, grid-based)
                - ``[batch, N_receptors]`` (static, flat coordinates)
                - ``[batch, time, N_receptors]`` (temporal, flat)
            metadata: Optional dictionary carrying layer-specific extra
                information (e.g., grid coordinates, layer names).

        Returns:
            Transformed responses (same or different shape — downstream
            modules must be aware of the contract).
        """
        ...

    def reset_state(self) -> None:
        """Reset internal state (e.g., running averages, adaptation vars).

        Override in stateful layers.  Default is a no-op.
        """
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseProcessingLayer":
        """Construct from a YAML-compatible configuration dict.

        Override in subclasses for YAML pipeline integration.  The default
        implementation returns a bare ``IdentityLayer`` regardless of
        config content.

        Args:
            config: Configuration dictionary.

        Returns:
            Configured processing layer instance.
        """
        return IdentityLayer()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize layer to a YAML-compatible dict.

        Override in subclasses.  Default returns ``{'type': 'identity'}``.
        """
        return {"type": "identity"}


class IdentityLayer(BaseProcessingLayer):
    """Pass-through layer — returns input unchanged.

    Used as the default when no processing layers are configured in the
    pipeline.  Incurs negligible overhead.
    """

    def forward(
        self,
        receptor_responses: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Return input unchanged.

        Args:
            receptor_responses: Any tensor.
            metadata: Ignored.

        Returns:
            Same tensor, unmodified.
        """
        return receptor_responses

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "identity"}


class ProcessingPipeline(nn.Module):
    """Sequential chain of processing layers.

    Wraps an ordered list of :class:`BaseProcessingLayer` instances and
    applies them in sequence.  If the list is empty a single
    :class:`IdentityLayer` is used.

    Example:
        >>> pipe = ProcessingPipeline([GainLayer(2.0), IdentityLayer()])
        >>> out = pipe(receptor_data)
    """

    def __init__(self, layers: Optional[List[BaseProcessingLayer]] = None):
        """Initialize the pipeline.

        Args:
            layers: Ordered processing layers.  ``None`` or empty →
                single :class:`IdentityLayer`.
        """
        super().__init__()
        if not layers:
            layers = [IdentityLayer()]
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        receptor_responses: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Apply all layers in order.

        Args:
            receptor_responses: Input activations.
            metadata: Passed to every layer.

        Returns:
            Transformed activations.
        """
        x = receptor_responses
        for layer in self.layers:
            x = layer(x, metadata)
        return x

    def reset_state(self) -> None:
        """Reset state of all constituent layers."""
        for layer in self.layers:
            layer.reset_state()

    @classmethod
    def from_config(cls, configs: List[Dict[str, Any]]) -> "ProcessingPipeline":
        """Build pipeline from a list of layer configs.

        Each config dict must have a ``'type'`` key.  Currently only
        ``'identity'`` is supported; future types will be registered
        here.

        Args:
            configs: List of layer configuration dicts.

        Returns:
            ProcessingPipeline instance.
        """
        _registry = {
            "identity": IdentityLayer,
        }
        layers: List[BaseProcessingLayer] = []
        for cfg in configs:
            layer_type = cfg.get("type", "identity")
            layer_cls = _registry.get(layer_type)
            if layer_cls is None:
                raise ValueError(
                    f"Unknown processing layer type: '{layer_type}'. "
                    f"Available: {list(_registry.keys())}"
                )
            layers.append(layer_cls.from_config(cfg))
        return cls(layers)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize all layers."""
        return [layer.to_dict() for layer in self.layers]
