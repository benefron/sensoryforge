"""Adaptive ODE solver implementation using optional dependencies.

This module provides the AdaptiveSolver class, which wraps high-quality adaptive
ODE solvers from torchdiffeq or torchode libraries. These are optional dependencies
that provide more accurate integration for stiff systems or when high precision is required.
"""

from typing import Callable, Dict, Any, Tuple, Optional
import torch

from .base import BaseSolver


# Attempt to import optional adaptive solver dependencies
try:
    import torchdiffeq
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


class AdaptiveSolver(BaseSolver):
    """Adaptive ODE solver using torchdiffeq or torchode.
    
    This solver wraps high-quality adaptive integration methods that automatically
    adjust step sizes to maintain error tolerances. It provides more accurate
    solutions than fixed-step methods like Euler, especially for stiff systems
    or when high precision is required.
    
    The solver requires optional dependencies (torchdiffeq or torchode) to be
    installed. If these packages are not available, instantiation will fail with
    a clear error message and installation instructions.
    
    Supported methods (when torchdiffeq is installed):
        - 'dopri5': Dormand-Prince 5th order (default, recommended)
        - 'dopri8': Dormand-Prince 8th order (higher accuracy)
        - 'adams': Adams-Bashforth-Moulton (good for smooth problems)
        - 'bosh3': Bogacki-Shampine 3rd order
    
    Attributes:
        dt: Default time step size in milliseconds (used as initial step hint).
        method: Integration method name (e.g., 'dopri5').
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
    
    Example:
        >>> # This will raise ImportError if torchdiffeq is not installed
        >>> try:
        ...     solver = AdaptiveSolver(method='dopri5', rtol=1e-5, atol=1e-7)
        ...     
        ...     def ode_func(state, t):
        ...         return -0.1 * state
        ...     
        ...     state = torch.tensor([[1.0, 2.0]])
        ...     trajectory = solver.integrate(ode_func, state, t_span=(0.0, 10.0), dt=0.1)
        ... except ImportError as e:
        ...     print(f"Optional dependencies not available: {e}")
    """
    
    def __init__(
        self,
        dt: float = 0.05,
        method: str = 'dopri5',
        rtol: float = 1e-5,
        atol: float = 1e-7
    ):
        """Initialize the adaptive solver.
        
        Args:
            dt: Initial time step size hint in milliseconds. Defaults to 0.05 ms.
                The adaptive solver may use smaller or larger steps as needed.
            method: Integration method. Defaults to 'dopri5'. Options include:
                    'dopri5', 'dopri8', 'adams', 'bosh3'.
            rtol: Relative error tolerance. Defaults to 1e-5.
            atol: Absolute error tolerance. Defaults to 1e-7.
        
        Raises:
            ImportError: If neither torchdiffeq nor torchode is installed,
                        with instructions on how to install the required packages.
        """
        super().__init__(dt=dt)
        
        # Check if required dependencies are available
        if not HAS_TORCHDIFFEQ:
            raise ImportError(
                "AdaptiveSolver requires optional dependencies that are not installed.\n"
                "\n"
                "To use adaptive solvers, install torchdiffeq:\n"
                "\n"
                "Option 1 - Install torchdiffeq (recommended):\n"
                "    pip install torchdiffeq>=0.2.3\n"
                "\n"
                "Or install with the 'solvers' extra:\n"
                "    pip install sensoryforge[solvers]\n"
                "\n"
                "For more information, see the SensoryForge documentation."
            )
        
        self.method = method
        self.rtol = rtol
        self.atol = atol
    
    def step(
        self,
        ode_func: Callable[[torch.Tensor, float], torch.Tensor],
        state: torch.Tensor,
        t: float,
        dt: float
    ) -> torch.Tensor:
        """Perform a single adaptive integration step.
        
        Note: For adaptive solvers, the single-step interface is less efficient
        than trajectory integration because it doesn't allow the solver to
        choose optimal step sizes. This method integrates from t to t+dt and
        returns only the final state.
        
        Args:
            ode_func: Function computing the time derivative of the state.
                      Signature: f(state, t) -> dstate_dt.
            state: Current state tensor with shape [batch, ...].
            t: Current time in milliseconds.
            dt: Time step size in milliseconds for this step.
        
        Returns:
            Updated state tensor with the same shape as input state [batch, ...].
        
        Example:
            >>> solver = AdaptiveSolver(method='dopri5')
            >>> def ode(state, t):
            ...     return -state
            >>> state = torch.tensor([[1.0]])
            >>> next_state = solver.step(ode, state, t=0.0, dt=0.1)
        """
        if HAS_TORCHDIFFEQ:
            # Wrap the ODE function to match torchdiffeq's expected signature
            # torchdiffeq expects f(t, state) but we use f(state, t)
            def wrapped_ode(t_val: torch.Tensor, state_val: torch.Tensor) -> torch.Tensor:
                # Convert tensor time to float
                t_float = t_val.item() if isinstance(t_val, torch.Tensor) else float(t_val)
                return ode_func(state_val, t_float)
            
            # Define time points: start and end
            t_points = torch.tensor([t, t + dt], dtype=state.dtype, device=state.device)
            
            # Integrate using torchdiffeq
            # odeint returns trajectory with shape [2, batch, ...]
            # We only need the final state (index 1)
            trajectory = torchdiffeq.odeint(
                wrapped_ode,
                state,
                t_points,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol
            )
            
            # Return the final state
            return trajectory[1]
        else:
            # This should not be reached due to __init__ check, but just in case
            raise ImportError("No adaptive solver backend available")
    
    def integrate(
        self,
        ode_func: Callable[[torch.Tensor, float], torch.Tensor],
        state: torch.Tensor,
        t_span: Tuple[float, float],
        dt: float
    ) -> torch.Tensor:
        """Integrate the ODE over a time span using adaptive methods.
        
        This method uses the adaptive solver to compute the solution trajectory
        with automatic step size control. The dt parameter serves as a hint for
        output sampling, not as a fixed step size.
        
        Args:
            ode_func: Function computing the time derivative of the state.
                      Signature: f(state, t) -> dstate_dt.
            state: Initial state tensor with shape [batch, ...].
            t_span: Tuple of (t_start, t_end) in milliseconds defining the
                    integration interval.
            dt: Time step size in milliseconds for output sampling.
                The actual integration steps may be smaller or larger.
        
        Returns:
            Trajectory tensor with shape [batch, num_steps+1, ...] where
            num_steps is determined by the dt sampling interval.
        
        Raises:
            ValueError: If t_span is invalid (t_end <= t_start) or dt <= 0.
        
        Example:
            >>> solver = AdaptiveSolver(method='dopri5', rtol=1e-6)
            >>> def stiff_ode(state, t):
            ...     # Stiff ODE example
            ...     return -100 * state
            >>> initial = torch.tensor([[1.0, 2.0]])
            >>> trajectory = solver.integrate(stiff_ode, initial, t_span=(0.0, 1.0), dt=0.1)
        """
        import math
        
        t_start, t_end = t_span
        
        # Validate inputs
        if t_end <= t_start:
            raise ValueError(
                f"Invalid time span: t_end ({t_end}) must be greater than "
                f"t_start ({t_start})"
            )
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        
        if HAS_TORCHDIFFEQ:
            # Compute number of output points
            num_steps = math.ceil((t_end - t_start) / dt)
            
            # Create output time points
            t_points = torch.linspace(
                t_start,
                t_start + num_steps * dt,
                num_steps + 1,
                dtype=state.dtype,
                device=state.device
            )
            
            # Wrap the ODE function to match torchdiffeq's expected signature
            def wrapped_ode(t_val: torch.Tensor, state_val: torch.Tensor) -> torch.Tensor:
                # Convert tensor time to float
                t_float = t_val.item() if isinstance(t_val, torch.Tensor) else float(t_val)
                return ode_func(state_val, t_float)
            
            # Integrate using torchdiffeq
            # odeint returns trajectory with shape [num_steps+1, batch, ...]
            trajectory = torchdiffeq.odeint(
                wrapped_ode,
                state,
                t_points,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol
            )
            
            # Transpose to match expected output format: [batch, num_steps+1, ...]
            # trajectory is currently [num_steps+1, batch, ...]
            # We need to permute dimensions to get [batch, num_steps+1, ...]
            num_time_steps = trajectory.shape[0]
            batch_size = trajectory.shape[1]
            
            # For 3D tensors: [time, batch, features] -> [batch, time, features]
            if trajectory.ndim == 3:
                trajectory = trajectory.permute(1, 0, 2)
            # For higher dimensional tensors, use general permutation
            else:
                # Build permutation: (1, 0, 2, 3, ..., ndim-1)
                perm = [1, 0] + list(range(2, trajectory.ndim))
                trajectory = trajectory.permute(*perm)
            
            return trajectory
        else:
            # This should not be reached due to __init__ check
            raise ImportError("No adaptive solver backend available")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AdaptiveSolver':
        """Create an AdaptiveSolver instance from a configuration dictionary.
        
        Args:
            config: Dictionary containing solver configuration. Expected keys:
                    - 'dt' (optional): Time step size hint. Defaults to 0.05.
                    - 'method' (optional): Integration method. Defaults to 'dopri5'.
                    - 'rtol' (optional): Relative tolerance. Defaults to 1e-5.
                    - 'atol' (optional): Absolute tolerance. Defaults to 1e-7.
        
        Returns:
            Configured AdaptiveSolver instance.
        
        Raises:
            ImportError: If required optional dependencies are not installed.
        
        Example:
            >>> config = {
            ...     'method': 'dopri8',
            ...     'rtol': 1e-6,
            ...     'atol': 1e-8,
            ...     'dt': 0.1
            ... }
            >>> solver = AdaptiveSolver.from_config(config)
        """
        dt = config.get('dt', 0.05)
        method = config.get('method', 'dopri5')
        rtol = config.get('rtol', 1e-5)
        atol = config.get('atol', 1e-7)
        
        return cls(dt=dt, method=method, rtol=rtol, atol=atol)
