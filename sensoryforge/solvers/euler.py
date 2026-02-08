"""Forward Euler ODE solver implementation.

This module provides the EulerSolver class, which implements the explicit
(forward) Euler method for numerical integration of ODEs. This is the default
solver used throughout SensoryForge for neuron dynamics integration.
"""

import math
from typing import Callable, Dict, Any, Tuple
import torch

from .base import BaseSolver


class EulerSolver(BaseSolver):
    """Forward Euler method for ODE integration.
    
    The explicit Euler method is a first-order numerical procedure for solving
    ordinary differential equations with a given initial value. It is the simplest
    Runge-Kutta method and provides a good balance of simplicity and performance
    for neuron dynamics.
    
    The integration scheme is:
        state_{t+1} = state_t + dt * f(state_t, t)
    
    where f(state, t) is the ODE function computing the time derivative.
    
    This solver matches the current neuron forward-Euler behavior in the
    SensoryForge framework and is the default choice for all simulations.
    
    Attributes:
        dt: Default time step size in milliseconds (ms).
    
    Example:
        >>> # Create a solver with default time step
        >>> solver = EulerSolver(dt=0.05)
        >>> 
        >>> # Define a simple ODE: dv/dt = -v (exponential decay)
        >>> def ode_func(state, t):
        ...     return -state
        >>> 
        >>> # Initial state
        >>> state = torch.tensor([[1.0, 2.0, 3.0]])  # [batch=1, features=3]
        >>> 
        >>> # Single step
        >>> new_state = solver.step(ode_func, state, t=0.0, dt=0.05)
        >>> 
        >>> # Full trajectory over time span
        >>> trajectory = solver.integrate(ode_func, state, t_span=(0.0, 1.0), dt=0.05)
        >>> trajectory.shape
        torch.Size([1, 21, 3])  # [batch, time_steps+1, features]
    """
    
    def __init__(self, dt: float = 0.05):
        """Initialize the Forward Euler solver.
        
        Args:
            dt: Time step size in milliseconds. Defaults to 0.05 ms,
                which matches the default used in SensoryForge neuron models.
        """
        super().__init__(dt=dt)
    
    def step(
        self,
        ode_func: Callable[[torch.Tensor, float], torch.Tensor],
        state: torch.Tensor,
        t: float,
        dt: float
    ) -> torch.Tensor:
        """Perform a single Forward Euler integration step.
        
        Computes the next state using the explicit Euler formula:
            state_{t+1} = state_t + dt * f(state_t, t)
        
        Args:
            ode_func: Function computing the time derivative of the state.
                      Signature: f(state, t) -> dstate_dt.
                      Both state and dstate_dt have shape [batch, ...].
            state: Current state tensor with shape [batch, ...].
                   Can have arbitrary trailing dimensions (features, spatial dims, etc.).
            t: Current time in milliseconds.
            dt: Time step size in milliseconds for this step.
        
        Returns:
            Updated state tensor with the same shape as input state [batch, ...].
            The computation preserves device and dtype of the input state.
        
        Example:
            >>> solver = EulerSolver()
            >>> def decay(state, t):
            ...     return -0.1 * state  # dv/dt = -0.1 * v
            >>> v = torch.tensor([[10.0]])
            >>> v_next = solver.step(decay, v, t=0.0, dt=0.1)
            >>> v_next
            tensor([[9.9000]])  # 10.0 + 0.1 * (-0.1 * 10.0) = 9.9
        """
        # Compute the derivative at the current state and time
        dstate_dt = ode_func(state, t)
        
        # Apply Forward Euler update: state_{t+1} = state_t + dt * dstate/dt
        new_state = state + dt * dstate_dt
        
        return new_state
    
    def integrate(
        self,
        ode_func: Callable[[torch.Tensor, float], torch.Tensor],
        state: torch.Tensor,
        t_span: Tuple[float, float],
        dt: float
    ) -> torch.Tensor:
        """Integrate the ODE over a time span using Forward Euler method.
        
        Repeatedly applies the Forward Euler step to compute the solution
        trajectory from t_start to t_end. The trajectory includes the initial
        state as the first time slice.
        
        Args:
            ode_func: Function computing the time derivative of the state.
                      Signature: f(state, t) -> dstate_dt.
            state: Initial state tensor with shape [batch, ...].
            t_span: Tuple of (t_start, t_end) in milliseconds defining the
                    integration interval. Must have t_end > t_start.
            dt: Time step size in milliseconds. Must be positive.
        
        Returns:
            Trajectory tensor with shape [batch, num_steps+1, ...] where
            num_steps = ceil((t_end - t_start) / dt). The first time slice
            (trajectory[:, 0, ...]) contains the initial state, and subsequent
            slices contain the integrated states at times t_start + k*dt.
        
        Raises:
            ValueError: If t_span is invalid (t_end <= t_start) or dt <= 0.
        
        Example:
            >>> solver = EulerSolver(dt=0.1)
            >>> def ode(state, t):
            ...     return -state
            >>> initial = torch.tensor([[1.0, 2.0]])  # [batch=1, features=2]
            >>> trajectory = solver.integrate(ode, initial, t_span=(0.0, 0.5), dt=0.1)
            >>> trajectory.shape
            torch.Size([1, 6, 2])  # 6 time points: t=0.0, 0.1, 0.2, 0.3, 0.4, 0.5
            >>> trajectory[:, 0, :]  # Initial state
            tensor([[1., 2.]])
        """
        t_start, t_end = t_span
        
        # Validate inputs
        if t_end <= t_start:
            raise ValueError(
                f"Invalid time span: t_end ({t_end}) must be greater than "
                f"t_start ({t_start})"
            )
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        
        # Compute number of steps needed
        num_steps = math.ceil((t_end - t_start) / dt)
        
        # Get state shape and device/dtype information
        batch_size = state.shape[0]
        state_shape = state.shape[1:]  # All dimensions after batch
        device = state.device
        dtype = state.dtype
        
        # Allocate trajectory tensor: [batch, num_steps+1, ...]
        # The +1 accounts for the initial state
        trajectory_shape = (batch_size, num_steps + 1) + state_shape
        trajectory = torch.zeros(trajectory_shape, dtype=dtype, device=device)
        
        # Store initial state at t=t_start
        trajectory[:, 0, ...] = state
        
        # Current state and time
        current_state = state
        current_time = t_start
        
        # Integrate forward in time
        for step_idx in range(num_steps):
            # Compute next state using Forward Euler
            current_state = self.step(ode_func, current_state, current_time, dt)
            
            # Store the computed state in trajectory
            trajectory[:, step_idx + 1, ...] = current_state
            
            # Advance time
            current_time += dt
        
        return trajectory
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'EulerSolver':
        """Create an EulerSolver instance from a configuration dictionary.
        
        This factory method allows easy construction of the solver from
        serialized configurations, enabling flexible solver selection in
        model pipelines.
        
        Args:
            config: Dictionary containing solver configuration. Expected keys:
                    - 'dt' (optional): Time step size in milliseconds.
                                       Defaults to 0.05 if not provided.
                    The 'type' key is typically used by higher-level factories
                    but is ignored here since the class is already known.
        
        Returns:
            Configured EulerSolver instance.
        
        Example:
            >>> config = {'dt': 0.1}
            >>> solver = EulerSolver.from_config(config)
            >>> solver.dt
            0.1
            >>> 
            >>> # Using default dt
            >>> solver_default = EulerSolver.from_config({})
            >>> solver_default.dt
            0.05
        """
        # Extract dt from config, use default if not specified
        dt = config.get('dt', 0.05)
        
        return cls(dt=dt)
