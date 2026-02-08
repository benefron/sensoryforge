"""Base solver interface for ODE integration in SensoryForge.

This module defines the abstract base class for all ODE solvers used in the
framework. Solvers are responsible for numerical integration of neuron dynamics
and other differential equations.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Tuple
import torch


class BaseSolver(ABC):
    """Abstract base class for ODE solvers.
    
    All solvers must implement methods for single-step integration and
    trajectory integration over a time span. Solvers can be configured
    from dictionaries for easy serialization and deserialization.
    
    The solver interface is designed to be compatible with PyTorch tensors
    and supports batched operations across devices (CPU, CUDA, MPS).
    
    Attributes:
        dt: Default time step size in milliseconds (ms).
    """
    
    def __init__(self, dt: float = 0.05):
        """Initialize the base solver.
        
        Args:
            dt: Time step size in milliseconds. Defaults to 0.05 ms (Forward Euler default).
        """
        self.dt = dt
    
    @abstractmethod
    def step(
        self,
        ode_func: Callable[[torch.Tensor, float], torch.Tensor],
        state: torch.Tensor,
        t: float,
        dt: float
    ) -> torch.Tensor:
        """Perform a single integration step.
        
        Advances the ODE state by one time step using the solver's numerical
        integration scheme.
        
        Args:
            ode_func: Function computing the time derivative of the state.
                      Signature: f(state, t) -> dstate_dt
                      where state has shape [batch, ...] and dstate_dt has the same shape.
            state: Current state tensor with shape [batch, ...].
            t: Current time in milliseconds.
            dt: Time step size in milliseconds for this step.
        
        Returns:
            Updated state tensor with the same shape as input state [batch, ...].
        
        Example:
            >>> def ode_func(state, t):
            ...     # Simple decay: dstate/dt = -state
            ...     return -state
            >>> solver = EulerSolver(dt=0.1)
            >>> state = torch.tensor([[1.0, 2.0]])
            >>> new_state = solver.step(ode_func, state, t=0.0, dt=0.1)
        """
        pass
    
    @abstractmethod
    def integrate(
        self,
        ode_func: Callable[[torch.Tensor, float], torch.Tensor],
        state: torch.Tensor,
        t_span: Tuple[float, float],
        dt: float
    ) -> torch.Tensor:
        """Integrate the ODE over a time span.
        
        Computes the solution trajectory by repeatedly applying the solver's
        step method over the specified time interval.
        
        Args:
            ode_func: Function computing the time derivative of the state.
                      Signature: f(state, t) -> dstate_dt.
            state: Initial state tensor with shape [batch, ...].
            t_span: Tuple of (t_start, t_end) in milliseconds defining the
                    integration interval.
            dt: Time step size in milliseconds.
        
        Returns:
            Trajectory tensor with shape [batch, num_steps+1, ...] where
            num_steps = ceil((t_end - t_start) / dt). The first time slice
            contains the initial state.
        
        Example:
            >>> def ode_func(state, t):
            ...     return -state
            >>> solver = EulerSolver(dt=0.1)
            >>> state = torch.tensor([[1.0, 2.0]])
            >>> trajectory = solver.integrate(ode_func, state, t_span=(0.0, 1.0), dt=0.1)
            >>> trajectory.shape
            torch.Size([1, 11, 2])
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseSolver':
        """Create a solver instance from a configuration dictionary.
        
        This factory method enables easy construction of solvers from
        serialized configurations, supporting flexible solver selection
        in model pipelines.
        
        Args:
            config: Dictionary containing solver configuration.
                    Must include a 'type' key specifying the solver class.
                    Other keys are solver-specific parameters.
        
        Returns:
            Configured solver instance.
        
        Raises:
            ValueError: If the config is invalid or missing required keys.
        
        Example:
            >>> config = {'type': 'euler', 'dt': 0.1}
            >>> solver = BaseSolver.from_config(config)
        """
        pass
