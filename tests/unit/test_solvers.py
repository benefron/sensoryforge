"""Unit tests for ODE solver infrastructure.

This module tests the solver API, including:
- BaseSolver abstract interface
- EulerSolver implementation (step and integrate methods)
- AdaptiveSolver wrapper and import error handling
- Configuration-based solver creation (from_config)
"""

import pytest
import torch
import math

from sensoryforge.solvers import (
    BaseSolver,
    EulerSolver,
    AdaptiveSolver,
    get_solver,
)


class TestEulerSolver:
    """Test suite for the Forward Euler solver."""
    
    def test_euler_initialization(self):
        """Test that EulerSolver initializes with correct default parameters."""
        solver = EulerSolver()
        assert solver.dt == 0.05, "Default dt should be 0.05"
        
        # Test custom dt
        solver = EulerSolver(dt=0.1)
        assert solver.dt == 0.1, "Custom dt should be set correctly"
    
    def test_euler_step_simple_decay(self):
        """Test single Euler step with exponential decay ODE."""
        # Define a simple exponential decay ODE: dv/dt = -v
        def decay_ode(state, t):
            return -state
        
        solver = EulerSolver(dt=0.1)
        
        # Initial state
        state = torch.tensor([[1.0, 2.0, 3.0]])  # [batch=1, features=3]
        
        # Perform one step
        next_state = solver.step(decay_ode, state, t=0.0, dt=0.1)
        
        # Expected: state + dt * (-state) = state * (1 - dt)
        expected = state * (1 - 0.1)
        
        assert next_state.shape == state.shape, "Output shape should match input"
        assert torch.allclose(next_state, expected), "Euler step calculation incorrect"
    
    def test_euler_step_preserves_device_dtype(self):
        """Test that Euler step preserves device and dtype."""
        def simple_ode(state, t):
            return state * 0.1
        
        solver = EulerSolver(dt=0.05)
        
        # Test float32
        state_f32 = torch.tensor([[1.0]], dtype=torch.float32)
        result_f32 = solver.step(simple_ode, state_f32, t=0.0, dt=0.05)
        assert result_f32.dtype == torch.float32, "dtype should be preserved"
        
        # Test float64
        state_f64 = torch.tensor([[1.0]], dtype=torch.float64)
        result_f64 = solver.step(simple_ode, state_f64, t=0.0, dt=0.05)
        assert result_f64.dtype == torch.float64, "dtype should be preserved"
        
        # Test device (if CUDA available)
        if torch.cuda.is_available():
            state_cuda = torch.tensor([[1.0]], device='cuda')
            result_cuda = solver.step(simple_ode, state_cuda, t=0.0, dt=0.05)
            assert result_cuda.device.type == 'cuda', "device should be preserved"
    
    def test_euler_step_batched(self):
        """Test Euler step works correctly with batched inputs."""
        def linear_ode(state, t):
            # dv/dt = 2*v
            return 2.0 * state
        
        solver = EulerSolver(dt=0.1)
        
        # Batched state: [batch=3, features=2]
        state = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])
        
        next_state = solver.step(linear_ode, state, t=0.0, dt=0.1)
        
        # Expected: state + 0.1 * 2 * state = state * 1.2
        expected = state * 1.2
        
        assert next_state.shape == (3, 2), "Batch shape should be preserved"
        assert torch.allclose(next_state, expected), "Batched computation incorrect"
    
    def test_euler_integrate_shape(self):
        """Test that integrate produces correct output shape."""
        def simple_ode(state, t):
            return -0.1 * state
        
        solver = EulerSolver(dt=0.1)
        
        # Initial state: [batch=2, features=3]
        state = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        # Integrate from t=0 to t=1 with dt=0.1
        # Expected: 10 steps + initial state = 11 time points
        trajectory = solver.integrate(simple_ode, state, t_span=(0.0, 1.0), dt=0.1)
        
        assert trajectory.shape == (2, 11, 3), \
            "Trajectory shape should be [batch, num_steps+1, features]"
        
        # First time point should be initial state
        assert torch.allclose(trajectory[:, 0, :], state), \
            "First time point should match initial state"
    
    def test_euler_integrate_exponential_decay(self):
        """Test integrate accuracy with known analytical solution."""
        # ODE: dv/dt = -k*v, solution: v(t) = v0 * exp(-k*t)
        k = 0.5
        
        def decay_ode(state, t):
            return -k * state
        
        solver = EulerSolver(dt=0.01)  # Small dt for better accuracy
        
        state = torch.tensor([[10.0]])
        t_end = 2.0
        trajectory = solver.integrate(decay_ode, state, t_span=(0.0, t_end), dt=0.01)
        
        # Check final value against analytical solution
        expected_final = 10.0 * math.exp(-k * t_end)
        numerical_final = trajectory[0, -1, 0].item()
        
        # Allow 5% relative error due to Euler approximation
        rel_error = abs(numerical_final - expected_final) / expected_final
        assert rel_error < 0.05, \
            f"Euler integration error too large: {rel_error*100:.2f}%"
    
    def test_euler_integrate_invalid_time_span(self):
        """Test that integrate raises error for invalid time spans."""
        def dummy_ode(state, t):
            return state
        
        solver = EulerSolver()
        state = torch.tensor([[1.0]])
        
        # Test t_end <= t_start
        with pytest.raises(ValueError, match="t_end.*must be greater"):
            solver.integrate(dummy_ode, state, t_span=(1.0, 1.0), dt=0.1)
        
        with pytest.raises(ValueError, match="t_end.*must be greater"):
            solver.integrate(dummy_ode, state, t_span=(2.0, 1.0), dt=0.1)
    
    def test_euler_integrate_invalid_dt(self):
        """Test that integrate raises error for invalid dt."""
        def dummy_ode(state, t):
            return state
        
        solver = EulerSolver()
        state = torch.tensor([[1.0]])
        
        # Test dt <= 0
        with pytest.raises(ValueError, match="dt must be positive"):
            solver.integrate(dummy_ode, state, t_span=(0.0, 1.0), dt=0.0)
        
        with pytest.raises(ValueError, match="dt must be positive"):
            solver.integrate(dummy_ode, state, t_span=(0.0, 1.0), dt=-0.1)
    
    def test_euler_from_config(self):
        """Test creation from configuration dictionary."""
        # Test with custom dt
        config = {'dt': 0.2}
        solver = EulerSolver.from_config(config)
        assert solver.dt == 0.2, "dt should be set from config"
        
        # Test with default dt (empty config)
        solver_default = EulerSolver.from_config({})
        assert solver_default.dt == 0.05, "Default dt should be used"
        
        # Test with type field (should be ignored)
        config_with_type = {'type': 'euler', 'dt': 0.15}
        solver = EulerSolver.from_config(config_with_type)
        assert solver.dt == 0.15, "dt should be extracted correctly"


class TestAdaptiveSolver:
    """Test suite for the adaptive solver wrapper."""
    
    def test_adaptive_import_error(self):
        """Test that AdaptiveSolver raises clear ImportError when deps missing."""
        # We can't guarantee dependencies are missing, but we can test the error message
        # This test checks the error message format when dependencies are absent
        try:
            solver = AdaptiveSolver()
            # If we get here, dependencies are installed - skip detailed message check
            assert solver is not None
        except ImportError as e:
            error_msg = str(e)
            # Check that the error message contains installation instructions
            assert "pip install" in error_msg, \
                "Error should contain pip install instructions"
            assert "torchdiffeq" in error_msg or "torchode" in error_msg, \
                "Error should mention required packages"
            assert "sensoryforge[solvers]" in error_msg, \
                "Error should mention extras install option"
    
    def test_adaptive_from_config(self):
        """Test creation from configuration dictionary."""
        # Test that from_config either works or raises ImportError
        config = {
            'method': 'dopri5',
            'rtol': 1e-6,
            'atol': 1e-8,
            'dt': 0.1
        }
        
        try:
            solver = AdaptiveSolver.from_config(config)
            # If dependencies are installed, verify attributes
            assert solver.method == 'dopri5'
            assert solver.rtol == 1e-6
            assert solver.atol == 1e-8
            assert solver.dt == 0.1
        except ImportError:
            # This is expected if optional dependencies are not installed
            pass
    
    def test_adaptive_default_params(self):
        """Test that AdaptiveSolver uses correct default parameters."""
        try:
            solver = AdaptiveSolver()
            assert solver.dt == 0.05, "Default dt should be 0.05"
            assert solver.method == 'dopri5', "Default method should be dopri5"
            assert solver.rtol == 1e-5, "Default rtol should be 1e-5"
            assert solver.atol == 1e-7, "Default atol should be 1e-7"
        except ImportError:
            # Expected when dependencies not installed
            pytest.skip("Optional dependencies not installed")
    
    def test_adaptive_step_and_integrate(self):
        """Test adaptive solver step and integrate methods (if available)."""
        try:
            solver = AdaptiveSolver(method='dopri5', rtol=1e-6, atol=1e-8)
        except ImportError:
            pytest.skip("Optional dependencies (torchdiffeq) not installed")
        
        # Simple exponential decay ODE
        def decay_ode(state, t):
            return -0.5 * state
        
        # Test step method
        state = torch.tensor([[1.0, 2.0]])
        next_state = solver.step(decay_ode, state, t=0.0, dt=0.1)
        
        assert next_state.shape == state.shape, "Step output shape should match input"
        assert next_state.dtype == state.dtype, "Step should preserve dtype"
        
        # Test integrate method
        trajectory = solver.integrate(decay_ode, state, t_span=(0.0, 1.0), dt=0.1)
        
        assert trajectory.shape[0] == state.shape[0], "Batch dimension preserved"
        assert trajectory.shape[2] == state.shape[1], "Feature dimension preserved"
        assert trajectory.shape[1] == 11, "Should have 11 time points (0 to 1, dt=0.1)"
        assert torch.allclose(trajectory[:, 0, :], state), \
            "First time point should be initial state"


class TestSolverFactory:
    """Test suite for the get_solver factory function."""
    
    def test_get_solver_euler(self):
        """Test factory creates EulerSolver correctly."""
        config = {'type': 'euler', 'dt': 0.1}
        solver = get_solver(config)
        
        assert isinstance(solver, EulerSolver), "Should create EulerSolver"
        assert solver.dt == 0.1, "Should set dt from config"
    
    def test_get_solver_adaptive(self):
        """Test factory creates AdaptiveSolver correctly."""
        config = {
            'type': 'adaptive',
            'method': 'dopri5',
            'rtol': 1e-6,
            'dt': 0.05
        }
        
        try:
            solver = get_solver(config)
            assert isinstance(solver, AdaptiveSolver), "Should create AdaptiveSolver"
            assert solver.method == 'dopri5', "Should set method from config"
            assert solver.rtol == 1e-6, "Should set rtol from config"
        except ImportError:
            # Expected when optional dependencies not installed
            pytest.skip("Optional dependencies not installed")
    
    def test_get_solver_case_insensitive(self):
        """Test that solver type matching is case-insensitive."""
        config_upper = {'type': 'EULER', 'dt': 0.1}
        solver_upper = get_solver(config_upper)
        assert isinstance(solver_upper, EulerSolver)
        
        config_mixed = {'type': 'Euler', 'dt': 0.1}
        solver_mixed = get_solver(config_mixed)
        assert isinstance(solver_mixed, EulerSolver)
    
    def test_get_solver_missing_type(self):
        """Test that factory raises error when type is missing."""
        config = {'dt': 0.1}
        
        with pytest.raises(ValueError, match="must include a 'type' field"):
            get_solver(config)
    
    def test_get_solver_unknown_type(self):
        """Test that factory raises error for unknown solver type."""
        config = {'type': 'runge_kutta', 'dt': 0.1}
        
        with pytest.raises(ValueError, match="Unknown solver type"):
            get_solver(config)


class TestSolverAPI:
    """Test suite for general solver API compliance."""
    
    def test_euler_implements_base_interface(self):
        """Test that EulerSolver properly implements BaseSolver interface."""
        solver = EulerSolver()
        
        # Check that it's an instance of BaseSolver
        assert isinstance(solver, BaseSolver)
        
        # Check that it has required methods
        assert hasattr(solver, 'step')
        assert hasattr(solver, 'integrate')
        assert hasattr(solver, 'from_config')
        
        # Check that methods are callable
        assert callable(solver.step)
        assert callable(solver.integrate)
        assert callable(solver.from_config)
    
    def test_solver_consistency(self):
        """Test that step and integrate produce consistent results."""
        def linear_ode(state, t):
            return 0.5 * state
        
        solver = EulerSolver(dt=0.1)
        initial_state = torch.tensor([[1.0, 2.0]])
        
        # Manual stepping
        state = initial_state.clone()
        states_manual = [state.clone()]
        for _ in range(5):
            state = solver.step(linear_ode, state, t=0.0, dt=0.1)
            states_manual.append(state.clone())
        
        # Using integrate
        trajectory = solver.integrate(
            linear_ode,
            initial_state,
            t_span=(0.0, 0.5),
            dt=0.1
        )
        
        # Compare results
        for i, manual_state in enumerate(states_manual):
            assert torch.allclose(trajectory[0, i, :], manual_state[0]), \
                f"Step {i}: integrate and manual step should match"
    
    def test_multidimensional_state(self):
        """Test solvers work with multi-dimensional state tensors."""
        def spatial_ode(state, t):
            # Simple diffusion-like: dstate/dt = Laplacian-like operator
            # Just return negative state for simplicity
            return -0.1 * state
        
        solver = EulerSolver(dt=0.05)
        
        # 2D spatial state: [batch=2, height=4, width=4]
        state = torch.ones((2, 4, 4))
        
        # Test step
        next_state = solver.step(spatial_ode, state, t=0.0, dt=0.05)
        assert next_state.shape == state.shape
        
        # Test integrate
        trajectory = solver.integrate(spatial_ode, state, t_span=(0.0, 0.5), dt=0.05)
        assert trajectory.shape == (2, 11, 4, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
