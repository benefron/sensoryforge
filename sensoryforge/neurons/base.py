"""Abstract base class for spiking neuron models in SensoryForge.

All neuron implementations (Izhikevich, AdEx, MQIF, FA, SA, DSL-compiled)
should inherit from :class:`BaseNeuron` to ensure a consistent API across
the framework. This enables interchangeable neuron usage in pipelines,
YAML-driven construction, and plugin discovery (resolves ReviewFinding#H1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class BaseNeuron(nn.Module, ABC):
    """Abstract base class for spiking neuron models.

    All neuron models must:
    1. Inherit from ``nn.Module`` (for PyTorch compatibility)
    2. Implement ``forward(input_current)`` returning ``(v_trace, spikes)``
    3. Implement ``reset_state()`` to clear internal membrane state
    4. Provide ``from_config()`` class method for YAML instantiation
    5. Provide ``to_dict()`` method for serialization

    The canonical ``forward()`` signature accepts a single ``input_current``
    tensor of shape ``[batch, steps, features]`` and returns a tuple of
    ``(v_trace, spikes)`` both of shape ``[batch, steps+1, features]``.

    Attributes:
        dt: Integration time step in milliseconds.

    Example:
        >>> class MyNeuron(BaseNeuron):
        ...     def __init__(self, dt=0.05):
        ...         super().__init__(dt=dt)
        ...
        ...     def forward(self, input_current):
        ...         # ... dynamics ...
        ...         return v_trace, spikes
        ...
        ...     def reset_state(self):
        ...         pass
        ...
        ...     @classmethod
        ...     def from_config(cls, config):
        ...         return cls(**config)
        ...
        ...     def to_dict(self):
        ...         return {'dt': self.dt}
    """

    def __init__(self, dt: float = 0.05) -> None:
        """Initialise the neuron model.

        Args:
            dt: Integration time step in milliseconds.
        """
        super().__init__()
        self.dt = dt

    @abstractmethod
    def forward(
        self,
        input_current: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate the neuron dynamics for a batch of input currents.

        Args:
            input_current: Input currents with shape
                ``[batch, steps, features]`` in mA.

        Returns:
            Tuple of:
                - ``v_trace``: Membrane voltage trajectory
                  ``[batch, steps+1, features]`` in mV.
                - ``spikes``: Boolean spike events
                  ``[batch, steps+1, features]``.
        """
        ...

    @abstractmethod
    def reset_state(self) -> None:
        """Reset all internal state (membrane voltage, adaptation, etc.).

        Must be called between independent sequences to avoid temporal
        carryover from previous simulations.
        """
        ...

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseNeuron":
        """Construct a neuron instance from a configuration dictionary.

        Args:
            config: Dictionary of neuron parameters (typically from YAML).

        Returns:
            Initialised neuron instance.
        """
        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise neuron parameters to a dictionary.

        Returns:
            Dictionary suitable for YAML/JSON serialisation.
        """
        return {"dt": self.dt}
