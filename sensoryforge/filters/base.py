"""Abstract base class for temporal filters in SensoryForge.

All filter implementations (SA, RA, etc.) should inherit from
:class:`BaseFilter` to ensure a consistent API across the framework.
This enables interchangeable filter usage in pipelines, YAML-driven
construction, and plugin discovery (resolves ReviewFinding#H1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseFilter(nn.Module, ABC):
    """Abstract base class for temporal filters.

    All filters must:
    1. Inherit from ``nn.Module`` (for PyTorch compatibility)
    2. Implement ``forward()`` for filtering input signals
    3. Implement ``reset_state()`` to clear internal temporal state
    4. Provide ``from_config()`` class method for YAML instantiation
    5. Provide ``to_dict()`` method for serialization

    Attributes:
        dt: Time step in seconds used by the filter.

    Example:
        >>> class MyFilter(BaseFilter):
        ...     def __init__(self, config):
        ...         super().__init__(dt=config.get('dt', 0.001))
        ...         self.gain = config.get('gain', 1.0)
        ...
        ...     def forward(self, x, dt=None):
        ...         return x * self.gain
        ...
        ...     def reset_state(self):
        ...         pass
        ...
        ...     @classmethod
        ...     def from_config(cls, config):
        ...         return cls(config)
        ...
        ...     def to_dict(self):
        ...         return {'dt': self.dt, 'gain': self.gain}
    """

    def __init__(self, dt: float = 0.001) -> None:
        """Initialise the filter.

        Args:
            dt: Integration time step in seconds.
        """
        super().__init__()
        self.dt = dt

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply the filter to an input signal.

        Args:
            x: Input tensor. Shape depends on implementation but typically
               ``[batch, time, num_neurons]`` for temporal sequences or
               ``[batch, num_neurons]`` for single frames.
            dt: Optional override for the integration time step (seconds).

        Returns:
            Filtered output tensor with shape matching the input convention.
        """
        ...

    @abstractmethod
    def reset_state(self) -> None:
        """Reset all internal temporal state.

        Must be called between independent sequences to avoid temporal
        carryover from previous inputs.
        """
        ...

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseFilter":
        """Construct a filter instance from a configuration dictionary.

        Args:
            config: Dictionary of filter parameters (typically from YAML).

        Returns:
            Initialised filter instance.
        """
        return cls(config)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise filter parameters to a dictionary.

        Returns:
            Dictionary suitable for YAML/JSON serialisation.
        """
        return {"dt": self.dt}
