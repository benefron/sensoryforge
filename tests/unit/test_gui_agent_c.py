"""Tests for GUI Agent C: DSL neuron model and solver selection integration.

This test module verifies that the DSL neuron model editor and solver selection
functionality work correctly. It tests:
- NeuronModel compilation with default Izhikevich equations
- Solver imports (EulerSolver and AdaptiveSolver)
- Compiled DSL module interface
- Expected output shapes from DSL models
"""

import pytest
import torch


class TestDSLNeuronIntegration:
    """Test DSL neuron model compilation and integration."""
    
    def test_neuron_model_import(self):
        """NeuronModel can be imported without error."""
        from sensoryforge.neurons.model_dsl import NeuronModel
        assert NeuronModel is not None
    
    def test_euler_solver_import(self):
        """EulerSolver can be imported without error."""
        from sensoryforge.solvers.euler import EulerSolver
        assert EulerSolver is not None
    
    def test_adaptive_solver_import(self):
        """AdaptiveSolver can be imported without error."""
        from sensoryforge.solvers.adaptive import AdaptiveSolver
        assert AdaptiveSolver is not None
    
    def test_compile_default_izhikevich(self):
        """DSL model compiles successfully with default Izhikevich equations."""
        from sensoryforge.neurons.model_dsl import NeuronModel
        
        # Note: ms and mV are unit conversion constants
        equations = "dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms\ndu/dt = (a * (b*v - u)) / ms"
        threshold = "v >= 30 * mV"
        reset = "v = c\nu = u + d"
        parameters = {
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 8.0,
            "ms": 1.0,  # millisecond time constant
            "mV": 1.0,  # millivolt conversion
        }
        
        model = NeuronModel(
            equations=equations,
            threshold=threshold,
            reset=reset,
            parameters=parameters,
        )
        
        assert model is not None
        assert model.parameters["a"] == 0.02
    
    def test_compiled_module_has_forward(self):
        """Compiled DSL module has forward() method."""
        from sensoryforge.neurons.model_dsl import NeuronModel
        
        equations = "dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms\ndu/dt = (a * (b*v - u)) / ms"
        threshold = "v >= 30 * mV"
        reset = "v = c\nu = u + d"
        parameters = {
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 8.0,
            "ms": 1.0,
            "mV": 1.0,
        }
        
        model = NeuronModel(
            equations=equations,
            threshold=threshold,
            reset=reset,
            parameters=parameters,
        )
        
        neuron = model.compile(solver='euler', dt=0.5, device='cpu')
        
        assert hasattr(neuron, 'forward')
        assert callable(neuron.forward)
    
    def test_compiled_module_produces_output(self):
        """Compiled DSL module produces output with expected shape."""
        from sensoryforge.neurons.model_dsl import NeuronModel
        
        equations = "dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms\ndu/dt = (a * (b*v - u)) / ms"
        threshold = "v >= 30 * mV"
        reset = "v = c\nu = u + d"
        parameters = {
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 8.0,
            "ms": 1.0,
            "mV": 1.0,
        }
        
        model = NeuronModel(
            equations=equations,
            threshold=threshold,
            reset=reset,
            parameters=parameters,
        )
        
        neuron = model.compile(solver='euler', dt=0.5, device='cpu')
        
        # Create dummy input current: [batch=2, time_steps=10, features=1]
        input_current = torch.randn(2, 10, 1)
        
        # Run forward pass
        v_trace, spikes = neuron(input_current)
        
        # Verify output shapes
        assert v_trace.shape[0] == 2  # batch dimension
        assert spikes.shape[0] == 2  # batch dimension
        assert v_trace.ndim == 3  # [batch, time, features]
        assert spikes.ndim == 3  # [batch, time, features]
    
    def test_euler_solver_instantiation(self):
        """EulerSolver can be instantiated with default parameters."""
        from sensoryforge.solvers.euler import EulerSolver
        
        solver = EulerSolver(dt=0.5)
        assert solver is not None
        assert solver.dt == 0.5
    
    def test_adaptive_solver_instantiation(self):
        """AdaptiveSolver import is available (may require torchdiffeq)."""
        pytest.importorskip("torchdiffeq", reason="torchdiffeq not installed")
        from sensoryforge.solvers.adaptive import AdaptiveSolver
        
        solver = AdaptiveSolver(method="dopri5", dt=0.5, rtol=1e-5, atol=1e-7)
        assert solver is not None
        assert solver.method == "dopri5"
    
    def test_dsl_model_from_config(self):
        """NeuronModel.from_config() works correctly."""
        from sensoryforge.neurons.model_dsl import NeuronModel
        
        config = {
            "equations": "dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms\ndu/dt = (a * (b*v - u)) / ms",
            "threshold": "v >= 30 * mV",
            "reset": "v = c\nu = u + d",
            "parameters": {
                "a": 0.02,
                "b": 0.2,
                "c": -65.0,
                "d": 8.0,
                "ms": 1.0,
                "mV": 1.0,
            },
        }
        
        model = NeuronModel.from_config(config)
        assert model is not None
        assert model.parameters["a"] == 0.02
        assert model.parameters["d"] == 8.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

