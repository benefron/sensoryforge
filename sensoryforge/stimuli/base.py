"""Abstract base class for stimulus generators in SensoryForge.

All stimulus implementations (Gaussian, texture, moving, etc.) should
inherit from :class:`BaseStimulus` to ensure a consistent API across
the framework (resolves ReviewFinding#H1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Parameter spec dataclass
# ---------------------------------------------------------------------------

class ParamSpec:
    """Descriptor for a single stimulus parameter, used for UI auto-generation.

    Attributes:
        name: Python attribute name.
        label: Human-readable label for display.
        dtype: ``"float"``, ``"int"``, or ``"bool"``.
        default: Default value.
        min_val: Minimum allowed value (float/int; ignored for bool).
        max_val: Maximum allowed value (float/int; ignored for bool).
        step: Suggested spin-box step (None → auto).
        unit: Physical unit string (e.g. ``"mm"``, ``"mA"``).  Empty string = dimensionless.
        tooltip: Optional help text shown as a tooltip in the UI.
    """

    __slots__ = ("name", "label", "dtype", "default", "min_val", "max_val",
                 "step", "unit", "tooltip")

    def __init__(
        self,
        name: str,
        label: str = "",
        dtype: str = "float",
        default: Any = 0.0,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        step: Optional[float] = None,
        unit: str = "",
        tooltip: str = "",
    ) -> None:
        self.name = name
        self.label = label or name.replace("_", " ").title()
        self.dtype = dtype
        self.default = default
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.unit = unit
        self.tooltip = tooltip

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self) -> str:
        return (
            f"ParamSpec(name={self.name!r}, dtype={self.dtype!r}, "
            f"default={self.default!r}, unit={self.unit!r})"
        )


class BaseStimulus(nn.Module, ABC):
    """Abstract base class for stimulus generators.

    All stimulus generators must:
    1. Inherit from ``nn.Module`` (for PyTorch compatibility)
    2. Implement ``forward()`` that generates a spatial stimulus frame
    3. Implement ``reset_state()`` to clear any internal state
    4. Provide ``from_config()`` class method for YAML instantiation
    5. Provide ``to_dict()`` method for serialization

    Example:
        >>> class MyStimulus(BaseStimulus):
        ...     def __init__(self, amplitude=1.0):
        ...         super().__init__()
        ...         self.amplitude = amplitude
        ...
        ...     def forward(self, xx, yy):
        ...         return self.amplitude * torch.exp(-(xx**2 + yy**2))
        ...
        ...     def reset_state(self):
        ...         pass
        ...
        ...     @classmethod
        ...     def from_config(cls, config):
        ...         return cls(**config)
        ...
        ...     def to_dict(self):
        ...         return {'amplitude': self.amplitude}
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        xx: torch.Tensor,
        yy: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate a spatial stimulus pattern.

        Args:
            xx: X-coordinate meshgrid from the spatial grid.
            yy: Y-coordinate meshgrid from the spatial grid.
            **kwargs: Additional parameters (e.g., time index, center).

        Returns:
            Stimulus pressure field with shape matching ``xx``.
        """
        ...

    @abstractmethod
    def reset_state(self) -> None:
        """Reset any internal state (e.g., trajectory position)."""
        ...

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseStimulus":
        """Construct a stimulus instance from a configuration dictionary.

        Args:
            config: Dictionary of stimulus parameters (typically from YAML).

        Returns:
            Initialised stimulus instance.
        """
        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise stimulus parameters to a dictionary.

        Returns:
            Dictionary suitable for YAML/JSON serialisation.
        """
        return {}

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """Return parameter specifications for UI auto-generation.

        Override in subclasses to expose stimulus parameters as
        :class:`ParamSpec` descriptors.  The GUI uses these to build
        parameter forms dynamically.

        Returns:
            Ordered list of :class:`ParamSpec` instances describing
            every user-configurable parameter.  Empty list by default.

        Example::

            @classmethod
            def get_param_spec(cls):
                return [
                    ParamSpec("amplitude", dtype="float", default=1.0,
                              min_val=0.0, max_val=100.0, unit="mA"),
                    ParamSpec("sigma", dtype="float", default=0.5,
                              min_val=0.01, max_val=20.0, unit="mm"),
                ]
        """
        return []
