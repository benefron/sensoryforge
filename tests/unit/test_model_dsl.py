"""Unit tests for the equation DSL (model_dsl.py).

Tests cover:
- Valid equation parsing
- Error handling for malformed equations
- Compiled Izhikevich model matches hand-written implementation
- Configuration serialization and deserialization
"""

import pytest
import torch
import numpy as np

# Import the module to test
from sensoryforge.neurons.model_dsl import NeuronModel, SYMPY_AVAILABLE

# Import hand-written Izhikevich for comparison
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch


# Skip all tests if sympy is not available
pytestmark = pytest.mark.skipif(
    not SYMPY_AVAILABLE,
    reason="SymPy is required for DSL tests but is not installed"
)


class TestNeuronModelParsing:
    """Test equation parsing and validation."""
    
    def test_simple_equation_parsing(self):
        """Test parsing a simple differential equation."""
        model = NeuronModel(
            equations='dv/dt = -v + I',
            threshold='v >= 1.0',
            reset='v = 0.0',
            state_vars={'v': 0.0}
        )
        
        assert 'v' in model.derivatives
        assert 'v' in model.state_var_list
        assert model.threshold_var == 'v'
        assert model.threshold_op == '>='
        assert 'v' in model.reset_rules
    
    def test_multi_variable_equations(self):
        """Test parsing equations with multiple state variables."""
        equations = '''
        dv/dt = 0.04*v**2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
        '''
        model = NeuronModel(
            equations=equations,
            threshold='v >= 30',
            reset='''
            v = c
            u = u + d
            ''',
            parameters={'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0},
            state_vars={'v': -65.0, 'u': -13.0}
        )
        
        assert 'v' in model.derivatives
        assert 'u' in model.derivatives
        assert len(model.state_var_list) == 2
        assert set(model.parameters.keys()) == {'a', 'b', 'c', 'd'}
    
    def test_malformed_equation_error(self):
        """Test that malformed equations raise clear errors."""
        # Missing 'd/dt'
        with pytest.raises(ValueError, match="Invalid equation format"):
            NeuronModel(
                equations='v = -v + I',  # Should be dv/dt
                threshold='v >= 1.0',
                reset='v = 0.0'
            )
        
        # Invalid expression syntax
        with pytest.raises(ValueError, match="Failed to parse expression"):
            NeuronModel(
                equations='dv/dt = v***2',  # Invalid syntax (triple asterisk)
                threshold='v >= 1.0',
                reset='v = 0.0'
            )
    
    def test_malformed_threshold_error(self):
        """Test that malformed threshold conditions raise errors."""
        with pytest.raises(ValueError, match="Invalid threshold format"):
            NeuronModel(
                equations='dv/dt = -v + I',
                threshold='v = 1.0',  # Should use comparison operator
                reset='v = 0.0'
            )
    
    def test_malformed_reset_error(self):
        """Test that malformed reset rules raise errors."""
        with pytest.raises(ValueError, match="Invalid reset rule format"):
            NeuronModel(
                equations='dv/dt = -v + I',
                threshold='v >= 1.0',
                reset='v := 0.0'  # Wrong assignment operator
            )
    
    def test_undefined_variable_error(self):
        """Test that undefined variables in equations raise errors."""
        with pytest.raises(ValueError, match="Undefined symbols"):
            NeuronModel(
                equations='dv/dt = -v + unknown_var + I',  # unknown_var not defined
                threshold='v >= 1.0',
                reset='v = 0.0'
            )
    
    def test_threshold_variable_not_state_var_error(self):
        """Test error when threshold variable is not a state variable."""
        with pytest.raises(ValueError, match="Threshold variable.*is not a state variable"):
            NeuronModel(
                equations='dv/dt = -v + I',
                threshold='x >= 1.0',  # x is not a state variable
                reset='v = 0.0'
            )
    
    def test_reset_variable_not_state_var_error(self):
        """Test error when reset rule references non-existent variable."""
        with pytest.raises(ValueError, match="Reset rule variable.*is not a state variable"):
            NeuronModel(
                equations='dv/dt = -v + I',
                threshold='v >= 1.0',
                reset='x = 0.0'  # x is not a state variable
            )
    
    def test_auto_infer_state_vars(self):
        """Test automatic inference of state variables from equations."""
        model = NeuronModel(
            equations='''
            dv/dt = -v + I
            du/dt = v - u
            ''',
            threshold='v >= 1.0',
            reset='v = 0.0'
        )
        
        # State vars should be auto-initialized to 0.0
        assert 'v' in model.state_vars
        assert 'u' in model.state_vars
        assert model.state_vars['v'] == 0.0
        assert model.state_vars['u'] == 0.0


class TestNeuronModelCompilation:
    """Test compilation to PyTorch modules."""
    
    def test_compile_returns_module(self):
        """Test that compile() returns an nn.Module."""
        model = NeuronModel(
            equations='dv/dt = -v + I',
            threshold='v >= 1.0',
            reset='v = 0.0',
            state_vars={'v': 0.0}
        )
        
        neuron = model.compile(dt=0.1, device='cpu')
        assert isinstance(neuron, torch.nn.Module)
    
    def test_compiled_module_forward_shape(self):
        """Test that compiled module returns correct tensor shapes."""
        model = NeuronModel(
            equations='dv/dt = -v + I',
            threshold='v >= 1.0',
            reset='v = 0.0',
            state_vars={'v': 0.0}
        )
        
        neuron = model.compile(dt=0.1, device='cpu')
        
        # Test with dummy input
        batch, steps, features = 2, 100, 3
        input_current = torch.randn(batch, steps, features)
        
        v_trace, spikes = neuron(input_current)
        
        # Check shapes
        assert v_trace.shape == (batch, steps + 1, features)
        assert spikes.shape == (batch, steps + 1, features)
        assert spikes.dtype == torch.bool
    
    def test_unsupported_solver_error(self):
        """Test that unsupported solvers raise errors."""
        model = NeuronModel(
            equations='dv/dt = -v + I',
            threshold='v >= 1.0',
            reset='v = 0.0'
        )
        
        with pytest.raises(ValueError, match="Unsupported solver"):
            model.compile(solver='rk4')  # Only euler is supported


class TestIzhikevichDSLvsHandWritten:
    """Test that DSL-compiled Izhikevich matches hand-written implementation."""
    
    def test_izhikevich_dsl_matches_handwritten(self):
        """Validate DSL output against hand-written Izhikevich model."""
        # Parameters for regular spiking neuron
        a, b, c, d = 0.02, 0.2, -65.0, 8.0
        v_init_val = -65.0
        u_init = b * v_init_val
        dt = 0.05
        threshold = 30.0
        
        # Create DSL version
        equations = '''
        dv/dt = 0.04*v**2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
        '''
        reset = '''
        v = c
        u = u + d
        '''
        
        dsl_model = NeuronModel(
            equations=equations,
            threshold=f'v >= {threshold}',
            reset=reset,
            parameters={'a': a, 'b': b, 'c': c, 'd': d},
            state_vars={'v': v_init_val, 'u': u_init}
        )
        dsl_neuron = dsl_model.compile(dt=dt, device='cpu')
        
        # Create hand-written version
        handwritten_neuron = IzhikevichNeuronTorch(
            a=a, b=b, c=c, d=d,
            v_init=v_init_val, u_init=u_init,
            dt=dt, threshold=threshold
        )
        
        # Test with constant current
        steps = 500
        current = 10.0
        input_current = torch.full((1, steps, 1), current, dtype=torch.float32)
        
        # Run both models
        v_dsl, spikes_dsl = dsl_neuron(input_current)
        v_hand, spikes_hand = handwritten_neuron(input_current)
        
        # Compare results with reasonable tolerance
        # Note: Small numerical differences are expected due to implementation details
        np.testing.assert_allclose(
            v_dsl.cpu().numpy(),
            v_hand.cpu().numpy(),
            rtol=1e-4,
            atol=1e-3,
            err_msg="DSL voltage trace does not match hand-written implementation"
        )
        
        # Spike times should match exactly
        np.testing.assert_array_equal(
            spikes_dsl.cpu().numpy(),
            spikes_hand.cpu().numpy(),
            err_msg="DSL spike times do not match hand-written implementation"
        )
    
    def test_izhikevich_batched(self):
        """Test DSL Izhikevich with batched inputs."""
        a, b, c, d = 0.02, 0.2, -65.0, 8.0
        v_init_val = -65.0
        u_init = b * v_init_val
        
        equations = '''
        dv/dt = 0.04*v**2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
        '''
        
        model = NeuronModel(
            equations=equations,
            threshold='v >= 30',
            reset='v = c\nu = u + d',
            parameters={'a': a, 'b': b, 'c': c, 'd': d},
            state_vars={'v': v_init_val, 'u': u_init}
        )
        neuron = model.compile(dt=0.05, device='cpu')
        
        # Test with multiple batches and features
        batch, steps, features = 4, 200, 5
        input_current = torch.randn(batch, steps, features) * 5 + 10
        
        v_trace, spikes = neuron(input_current)
        
        # Check shapes
        assert v_trace.shape == (batch, steps + 1, features)
        assert spikes.shape == (batch, steps + 1, features)
        
        # Check that some spikes occurred (with this input, neurons should spike)
        assert spikes.any(), "No spikes detected with positive current"


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""
    
    def test_to_dict_contains_all_fields(self):
        """Test that to_dict() includes all necessary fields."""
        model = NeuronModel(
            equations='dv/dt = -v + I',
            threshold='v >= 1.0',
            reset='v = 0.0',
            parameters={'tau': 10.0},
            state_vars={'v': 0.0}
        )
        
        config = model.to_dict()
        
        assert 'equations' in config
        assert 'threshold' in config
        assert 'reset' in config
        assert 'parameters' in config
        assert 'state_vars' in config
        assert config['parameters']['tau'] == 10.0
        assert config['state_vars']['v'] == 0.0
    
    def test_from_config_reconstruction(self):
        """Test that from_config() correctly reconstructs a model."""
        config = {
            'equations': 'dv/dt = -v/tau + I',
            'threshold': 'v >= 1.0',
            'reset': 'v = 0.0',
            'parameters': {'tau': 10.0},
            'state_vars': {'v': 0.0}
        }
        
        model = NeuronModel.from_config(config)
        
        assert model.equations_str == config['equations']
        assert model.threshold_str == config['threshold']
        assert model.reset_str == config['reset']
        assert model.parameters == config['parameters']
        assert model.state_vars == config['state_vars']
    
    def test_round_trip_serialization(self):
        """Test that model survives serialization round-trip."""
        original_model = NeuronModel(
            equations='''
            dv/dt = 0.04*v**2 + 5*v + 140 - u + I
            du/dt = a*(b*v - u)
            ''',
            threshold='v >= 30',
            reset='v = c\nu = u + d',
            parameters={'a': 0.02, 'b': 0.2, 'c': -65.0, 'd': 8.0},
            state_vars={'v': -65.0, 'u': -13.0}
        )
        
        # Serialize and deserialize
        config = original_model.to_dict()
        restored_model = NeuronModel.from_config(config)
        
        # Compare configurations
        assert original_model.equations_str == restored_model.equations_str
        assert original_model.threshold_str == restored_model.threshold_str
        assert original_model.reset_str == restored_model.reset_str
        assert original_model.parameters == restored_model.parameters
        assert original_model.state_vars == restored_model.state_vars
        
        # Test that both produce same output
        input_current = torch.full((1, 100, 1), 10.0, dtype=torch.float32)
        
        neuron_orig = original_model.compile(dt=0.05, device='cpu')
        neuron_rest = restored_model.compile(dt=0.05, device='cpu')
        
        v_orig, spikes_orig = neuron_orig(input_current)
        v_rest, spikes_rest = neuron_rest(input_current)
        
        np.testing.assert_allclose(v_orig.numpy(), v_rest.numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(spikes_orig.numpy(), spikes_rest.numpy())
    
    def test_from_config_missing_keys_error(self):
        """Test that from_config() raises error for missing required keys."""
        incomplete_config = {
            'equations': 'dv/dt = -v + I',
            'threshold': 'v >= 1.0',
            # Missing 'reset' key
        }
        
        with pytest.raises(ValueError, match="Missing required keys"):
            NeuronModel.from_config(incomplete_config)


class TestDifferentThresholdOperators:
    """Test different threshold comparison operators."""
    
    def test_greater_than_threshold(self):
        """Test '>' threshold operator."""
        model = NeuronModel(
            equations='dv/dt = I',
            threshold='v > 1.0',  # Strict greater than
            reset='v = 0.0',
            state_vars={'v': 0.0}
        )
        
        neuron = model.compile(dt=0.1, device='cpu')
        
        # Constant positive current should eventually cause spike
        # With dv/dt = I = 1.0 and dt=0.1, need 11+ steps to reach v > 1.0
        input_current = torch.full((1, 20, 1), 1.0, dtype=torch.float32)
        v_trace, spikes = neuron(input_current)
        
        # Should spike when v crosses 1.0
        assert spikes.any(), f"No spikes detected. Max v: {v_trace.max()}"
    
    def test_less_than_threshold(self):
        """Test '<' threshold operator (inverted threshold)."""
        model = NeuronModel(
            equations='dv/dt = -I',  # Negative derivative
            threshold='v < -1.0',  # Fire when v drops below -1.0
            reset='v = 0.0',
            state_vars={'v': 0.0}
        )
        
        neuron = model.compile(dt=0.1, device='cpu')
        
        # Constant positive current with negative derivative
        # With dv/dt = -I = -1.0 and dt=0.1, need 11+ steps to reach v < -1.0
        input_current = torch.full((1, 20, 1), 1.0, dtype=torch.float32)
        v_trace, spikes = neuron(input_current)
        
        # Should spike when v goes below -1.0
        assert spikes.any(), f"No spikes detected. Min v: {v_trace.min()}"


class TestNoiseIntegration:
    """Test Langevin noise in integration."""
    
    def test_noise_std_parameter(self):
        """Test that noise_std parameter is accepted and used."""
        model = NeuronModel(
            equations='dv/dt = -v + I',
            threshold='v >= 1.0',
            reset='v = 0.0',
            state_vars={'v': 0.0}
        )
        
        # Compile with noise
        neuron = model.compile(dt=0.1, device='cpu', noise_std=0.1)
        
        # Should run without error
        input_current = torch.zeros((1, 100, 1))
        v_trace, spikes = neuron(input_current)
        
        assert v_trace.shape == (1, 101, 1)
    
    def test_noise_affects_trajectory(self):
        """Test that noise actually affects the voltage trajectory."""
        model = NeuronModel(
            equations='dv/dt = 0.0',  # No dynamics, only noise
            threshold='v >= 10.0',  # High threshold
            reset='v = 0.0',
            state_vars={'v': 0.0}
        )
        
        # Without noise - should stay at 0
        neuron_no_noise = model.compile(dt=0.1, device='cpu', noise_std=0.0)
        input_current = torch.zeros((1, 100, 1))
        v_no_noise, _ = neuron_no_noise(input_current)
        
        # With noise - should fluctuate
        torch.manual_seed(42)
        neuron_with_noise = model.compile(dt=0.1, device='cpu', noise_std=1.0)
        v_with_noise, _ = neuron_with_noise(input_current)
        
        # Trajectories should be different
        assert not torch.allclose(v_no_noise, v_with_noise, atol=0.1)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_equations_error(self):
        """Test that empty equations raise error."""
        with pytest.raises(ValueError, match="No valid differential equations found"):
            NeuronModel(
                equations='',
                threshold='v >= 1.0',
                reset='v = 0.0'
            )
    
    def test_single_timestep(self):
        """Test with single timestep input."""
        model = NeuronModel(
            equations='dv/dt = I',
            threshold='v >= 1.0',
            reset='v = 0.0',
            state_vars={'v': 0.0}
        )
        
        neuron = model.compile(dt=0.1, device='cpu')
        input_current = torch.zeros((1, 1, 1))  # Single timestep
        
        v_trace, spikes = neuron(input_current)
        
        assert v_trace.shape == (1, 2, 1)  # steps + 1
        assert spikes.shape == (1, 2, 1)
    
    def test_zero_dt_warning(self):
        """Test that very small dt is handled gracefully in noise."""
        model = NeuronModel(
            equations='dv/dt = I',
            threshold='v >= 1.0',
            reset='v = 0.0',
            state_vars={'v': 0.0}
        )
        
        # Very small dt with noise should not crash (uses max(dt, 1e-6))
        neuron = model.compile(dt=1e-8, device='cpu', noise_std=0.1)
        input_current = torch.zeros((1, 10, 1))
        
        # Should not raise error
        v_trace, spikes = neuron(input_current)
        assert v_trace.shape == (1, 11, 1)


class TestSymPyNotInstalled:
    """Test behavior when SymPy is not installed."""
    
    def test_import_error_message(self, monkeypatch):
        """Test that helpful error message is shown when SymPy is missing."""
        # Temporarily make SYMPY_AVAILABLE False
        import sensoryforge.neurons.model_dsl as dsl_module
        original_value = dsl_module.SYMPY_AVAILABLE
        
        try:
            monkeypatch.setattr(dsl_module, 'SYMPY_AVAILABLE', False)
            
            with pytest.raises(ImportError, match="SymPy is required"):
                NeuronModel(
                    equations='dv/dt = -v + I',
                    threshold='v >= 1.0',
                    reset='v = 0.0'
                )
        finally:
            # Restore original value
            monkeypatch.setattr(dsl_module, 'SYMPY_AVAILABLE', original_value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
