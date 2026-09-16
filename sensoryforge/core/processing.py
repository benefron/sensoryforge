"""Processing layer base classes for intermediate signal transformations.

This module defines the ``BaseProcessingLayer`` interface for inserting
composable transformations between the receptor grid and the innervation /
neuron stages.  Future layers include:

- Lateral inhibition
- Cross-grid fusion

Current implementations:

- ``IdentityLayer``: Pass-through (no-op) — used as the default when no
  processing layers are specified.
- ``OnOffLayer`` (Wave M3): a centre-surround difference-of-Gaussians over
  receptor coordinates, splitting the receptor response into an ON plane
  and an OFF plane.

All layers are ``nn.Module`` subclasses so they participate in PyTorch's
parameter/buffer tracking, device management, and backpropagation graph.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from sensoryforge.stimuli.base import ParamSpec


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
    def expand_receptor_coords(cls, receptor_coords: "torch.Tensor") -> "torch.Tensor":
        """Coordinate-space image of this layer's receptor-axis transform.

        The receptive-field bank for a population input with processing
        must be built on the *post-processing* receptor axis (Wave M3): if
        a layer changes the receptor count ``M`` (e.g. :class:`OnOffLayer`
        emitting ON and OFF planes, doubling it), the bank's
        ``receptor_coords`` must double the same way, in the same order,
        so the bank's weight columns line up with
        :meth:`ProcessingPipeline.forward`'s output.

        Default: identity (the layer does not change ``M``).

        Args:
            receptor_coords: ``[M, 2]`` receptor positions ``(x, y)`` in mm.

        Returns:
            ``[M', 2]`` receptor positions in the post-processing axis.
        """
        return receptor_coords

    #: Whether this layer's :meth:`from_config` needs the population
    #: input's own receptor coordinates (Wave M3) -- e.g. a centre-surround
    #: layer computing distances between receptors. ``ProcessingPipeline
    #: .from_config`` passes ``receptor_coords`` to a layer's
    #: ``from_config`` only when this is ``True``.
    REQUIRES_RECEPTOR_COORDS: bool = False

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

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """Return this layer's configurable parameters (Wave M3, G1 contract).

        Default: none. Override in a subclass that takes constructor
        parameters (e.g. ``OnOffLayer``'s ``sigma_center_mm``/
        ``sigma_surround_mm``).
        """
        return []


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


class OnOffLayer(BaseProcessingLayer):
    """Centre-surround ON/OFF split over receptor coordinates (Wave M3).

    A difference-of-Gaussians (DoG) centre-surround filter is computed
    directly over the *receptor coordinate* space (not spatial grid
    indices, so it works for any arrangement -- grid, hex, Poisson,
    imported): ``dog[j, k] = g(d_jk; sigma_center) - g(d_jk; sigma_surround)``
    where ``d_jk`` is the mm distance between receptors ``j`` and ``k`` and
    ``g`` is a normalised 2-D Gaussian. ``forward()`` applies this kernel to
    the receptor responses, then splits the result into a rectified ON
    plane (positive part) and a rectified OFF plane (negative part,
    sign-flipped to be non-negative), concatenated along the receptor axis:
    output ``[..., 2M]`` is ``[on (M), off (M)]``, same receptor order both
    halves. :meth:`expand_receptor_coords` mirrors this in coordinate space
    (``[receptor_coords; receptor_coords]``) so a receptive-field bank built
    on the doubled coordinates lines up column-for-column with this output.

    This is the worked example for ``docs/extending/add_processing_layer.md``
    and the vision demo (M4): with ``sigma_center_mm < sigma_surround_mm``,
    a bright spot on the stimulus drives the ON plane and a dark spot
    drives the OFF plane, never both at the same receptor.
    """

    REQUIRES_RECEPTOR_COORDS = True

    def __init__(
        self,
        receptor_coords: torch.Tensor,
        *,
        sigma_center_mm: float = 0.15,
        sigma_surround_mm: float = 0.45,
    ) -> None:
        """Precompute the DoG kernel over one set of receptor coordinates.

        Args:
            receptor_coords: ``[M, 2]`` receptor positions ``(x, y)`` in mm.
            sigma_center_mm: Centre Gaussian's standard deviation, mm.
            sigma_surround_mm: Surround Gaussian's standard deviation, mm.
                Must exceed ``sigma_center_mm`` for a conventional (excitatory
                centre, inhibitory surround) receptive field.

        Raises:
            ValueError: If ``receptor_coords`` is not ``[M, 2]``, or either
                sigma is not positive.
        """
        super().__init__()
        if receptor_coords.ndim != 2 or receptor_coords.shape[1] != 2:
            raise ValueError(
                "receptor_coords must be [M, 2] (x, y) in mm, got shape "
                f"{list(receptor_coords.shape)}"
            )
        if sigma_center_mm <= 0 or sigma_surround_mm <= 0:
            raise ValueError(
                "sigma_center_mm and sigma_surround_mm must be positive, got "
                f"{sigma_center_mm}, {sigma_surround_mm}"
            )
        self.sigma_center_mm = float(sigma_center_mm)
        self.sigma_surround_mm = float(sigma_surround_mm)
        coords = receptor_coords.detach().to(torch.float32)
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(dim=-1)  # [M, M]

        def _gaussian(d2: torch.Tensor, sigma: float) -> torch.Tensor:
            return torch.exp(-d2 / (2.0 * sigma * sigma)) / (
                2.0 * math.pi * sigma * sigma
            )

        dog = _gaussian(d2, self.sigma_center_mm) - _gaussian(
            d2, self.sigma_surround_mm
        )
        self.register_buffer("dog_kernel", dog.contiguous())

    def forward(
        self,
        receptor_responses: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Apply the DoG kernel and split into rectified ON/OFF planes.

        Args:
            receptor_responses: ``[..., M]`` receptor activations, ``M``
                matching the receptor coordinates this layer was built with.
            metadata: Ignored.

        Returns:
            ``[..., 2 * M]``: the ON plane (``[..., :M]``) then the OFF
            plane (``[..., M:]``), both ``>= 0``.

        Raises:
            ValueError: If the last dimension is not ``M``.
        """
        m = self.dog_kernel.shape[0]
        if receptor_responses.shape[-1] != m:
            raise ValueError(
                f"OnOffLayer built for M={m} receptors, got input with last "
                f"dimension {receptor_responses.shape[-1]}"
            )
        center_surround = torch.matmul(receptor_responses, self.dog_kernel.T)
        on = torch.clamp(center_surround, min=0.0)
        off = torch.clamp(-center_surround, min=0.0)
        return torch.cat([on, off], dim=-1)

    @classmethod
    def expand_receptor_coords(cls, receptor_coords: torch.Tensor) -> torch.Tensor:
        """``[receptor_coords; receptor_coords]`` -- ON plane then OFF plane."""
        return torch.cat([receptor_coords, receptor_coords], dim=0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to ``{"method": "onoff", "params": {...}}``."""
        return {
            "method": "onoff",
            "params": {
                "sigma_center_mm": self.sigma_center_mm,
                "sigma_surround_mm": self.sigma_surround_mm,
            },
        }

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], *, receptor_coords: Optional[torch.Tensor] = None
    ) -> "OnOffLayer":
        """Construct from a config dict plus the receptor coordinates it
        needs (``REQUIRES_RECEPTOR_COORDS``).

        Args:
            config: May carry ``sigma_center_mm``/``sigma_surround_mm``
                directly, or nested under ``params`` (``to_dict()``'s shape).
            receptor_coords: ``[M, 2]`` receptor positions; required.

        Raises:
            ValueError: If ``receptor_coords`` is not given.
        """
        if receptor_coords is None:
            raise ValueError("OnOffLayer.from_config requires receptor_coords")
        params = dict(config.get("params") or {})
        return cls(
            receptor_coords,
            sigma_center_mm=config.get(
                "sigma_center_mm", params.get("sigma_center_mm", 0.15)
            ),
            sigma_surround_mm=config.get(
                "sigma_surround_mm", params.get("sigma_surround_mm", 0.45)
            ),
        )

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """This layer's configurable parameters (G1 contract)."""
        return [
            ParamSpec(
                "sigma_center_mm",
                dtype="float",
                default=0.15,
                min_val=0.001,
                max_val=10.0,
                unit="mm",
                choices=None,
                help="Centre Gaussian standard deviation of the DoG kernel.",
                group="OnOff",
                advanced=False,
            ),
            ParamSpec(
                "sigma_surround_mm",
                dtype="float",
                default=0.45,
                min_val=0.001,
                max_val=20.0,
                unit="mm",
                choices=None,
                help="Surround Gaussian standard deviation of the DoG kernel.",
                group="OnOff",
                advanced=False,
            ),
        ]


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
    def from_config(
        cls,
        configs: List[Dict[str, Any]],
        *,
        receptor_coords: Optional["torch.Tensor"] = None,
    ) -> "ProcessingPipeline":
        """Build pipeline from a list of layer configs.

        Each config dict names its layer with either ``'type'`` (the
        original key, still accepted for
        :class:`~sensoryforge.core.generalized_pipeline.GeneralizedTactileEncodingPipeline`
        configs) or ``'method'`` (``PopulationInput.processing``'s key,
        Wave M3, matching :class:`~sensoryforge.config.schema.RFBuilderConfig`'s
        convention) -- looked up in ``PROCESSING_REGISTRY`` (Wave M3; every
        built-in and plugin processing layer, not just ``identity``).
        Remaining keys plus a nested ``'params'`` dict (if present) are
        passed to the layer's own ``from_config``.

        Args:
            configs: List of layer configuration dicts.
            receptor_coords: ``[M, 2]`` receptor positions ``(x, y)`` in mm,
                required only by a layer whose class sets
                ``REQUIRES_RECEPTOR_COORDS = True`` (e.g. ``OnOffLayer``).

        Returns:
            ProcessingPipeline instance.

        Raises:
            ValueError: If a layer's ``type``/``method`` is not registered,
                or it requires ``receptor_coords`` and none was given.
        """
        from sensoryforge.registry import PROCESSING_REGISTRY

        layers: List[BaseProcessingLayer] = []
        for cfg in configs:
            layer_type = cfg.get("type") or cfg.get("method", "identity")
            try:
                layer_cls = PROCESSING_REGISTRY.get_class(layer_type)
            except KeyError:
                raise ValueError(
                    f"Unknown processing layer type: '{layer_type}'. "
                    f"Available: {sorted(PROCESSING_REGISTRY.list_registered())}"
                ) from None
            layer_config = {**cfg, **dict(cfg.get("params") or {})}
            if layer_cls.REQUIRES_RECEPTOR_COORDS:
                if receptor_coords is None:
                    raise ValueError(
                        f"Processing layer '{layer_type}' requires "
                        "receptor_coords, none was given"
                    )
                layers.append(
                    layer_cls.from_config(layer_config, receptor_coords=receptor_coords)
                )
            else:
                layers.append(layer_cls.from_config(layer_config))
        return cls(layers)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize all layers."""
        return [layer.to_dict() for layer in self.layers]

    @staticmethod
    def expand_receptor_coords(
        configs: List[Dict[str, Any]], receptor_coords: "torch.Tensor"
    ) -> "torch.Tensor":
        """Apply every layer spec's :meth:`BaseProcessingLayer.expand_receptor_coords`
        in order (Wave M3), without constructing the layers themselves --
        used at bank-build time, before a layer requiring
        ``receptor_coords`` (e.g. ``OnOffLayer``) can even be built.

        Args:
            configs: Same layer-spec list :meth:`from_config` takes.
            receptor_coords: ``[M, 2]`` receptor positions ``(x, y)`` in mm.

        Returns:
            ``[M', 2]`` receptor positions after every layer's expansion.
        """
        from sensoryforge.registry import PROCESSING_REGISTRY

        coords = receptor_coords
        for cfg in configs:
            layer_type = cfg.get("type") or cfg.get("method", "identity")
            try:
                layer_cls = PROCESSING_REGISTRY.get_class(layer_type)
            except KeyError:
                raise ValueError(
                    f"Unknown processing layer type: '{layer_type}'. "
                    f"Available: {sorted(PROCESSING_REGISTRY.list_registered())}"
                ) from None
            coords = layer_cls.expand_receptor_coords(coords)
        return coords
