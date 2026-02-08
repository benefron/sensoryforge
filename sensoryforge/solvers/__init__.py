"""ODE solver infrastructure for SensoryForge.

This package provides pluggable ODE solvers for numerical integration of neuron
dynamics and other differential equations. The package includes:

- BaseSolver: Abstract base class defining the solver interface
- EulerSolver: Forward Euler method (default, matches existing neuron behavior)
- AdaptiveSolver: High-quality adaptive methods (requires optional dependencies)

The default solver is EulerSolver with dt=0.05 ms, which maintains backward
compatibility with existing neuron model implementations.

Example:
    >>> from sensoryforge.solvers import EulerSolver, AdaptiveSolver
    >>> 
    >>> # Create a Forward Euler solver (default)
    >>> euler = EulerSolver(dt=0.05)
    >>> 
    >>> # Create an adaptive solver (requires torchdiffeq)
    >>> try:
    ...     adaptive = AdaptiveSolver(method='dopri5', rtol=1e-6)
    ... except ImportError:
    ...     print("Install torchdiffeq for adaptive solvers")
    >>> 
    >>> # Use from_config for flexible configuration
    >>> config = {'type': 'euler', 'dt': 0.1}
    >>> solver = get_solver(config)
"""

from typing import Dict, Any

from .base import BaseSolver
from .euler import EulerSolver
from .adaptive import AdaptiveSolver


# Public API exports
__all__ = [
    'BaseSolver',
    'EulerSolver',
    'AdaptiveSolver',
    'get_solver',
]


def get_solver(config: Dict[str, Any]) -> BaseSolver:
    """Factory function to create a solver from a configuration dictionary.
    
    This convenience function provides a unified interface for creating any
    solver type based on a configuration dictionary. It dispatches to the
    appropriate solver class based on the 'type' field.
    
    Args:
        config: Dictionary containing solver configuration. Must include:
                - 'type': Solver type ('euler' or 'adaptive')
                Additional keys are solver-specific parameters passed to
                the solver's from_config method.
    
    Returns:
        Configured solver instance (BaseSolver subclass).
    
    Raises:
        ValueError: If the solver type is unknown or missing.
        ImportError: If an adaptive solver is requested but dependencies
                    are not installed.
    
    Example:
        >>> # Create an Euler solver
        >>> config = {'type': 'euler', 'dt': 0.1}
        >>> solver = get_solver(config)
        >>> isinstance(solver, EulerSolver)
        True
        >>> 
        >>> # Create an adaptive solver
        >>> config = {'type': 'adaptive', 'method': 'dopri5', 'rtol': 1e-6}
        >>> solver = get_solver(config)
        >>> isinstance(solver, AdaptiveSolver)
        True
        >>> 
        >>> # Invalid type raises error
        >>> try:
        ...     solver = get_solver({'type': 'unknown'})
        ... except ValueError as e:
        ...     print(e)
        Unknown solver type: unknown
    """
    solver_type = config.get('type')
    
    if solver_type is None:
        raise ValueError(
            "Solver configuration must include a 'type' field. "
            "Valid types are: 'euler', 'adaptive'"
        )
    
    # Normalize to lowercase for case-insensitive matching
    solver_type = solver_type.lower()
    
    # Dispatch to appropriate solver class
    if solver_type == 'euler':
        return EulerSolver.from_config(config)
    elif solver_type == 'adaptive':
        return AdaptiveSolver.from_config(config)
    else:
        raise ValueError(
            f"Unknown solver type: {solver_type}. "
            f"Valid types are: 'euler', 'adaptive'"
        )
