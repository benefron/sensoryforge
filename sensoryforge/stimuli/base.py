"""Abstract base class for stimulus generators in SensoryForge.

All stimulus implementations (Gaussian, texture, moving, etc.) should
inherit from :class:`BaseStimulus` to ensure a consistent API across
the framework (resolves ReviewFinding#H1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


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
