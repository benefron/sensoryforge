"""Tests for abstract base classes (resolves ReviewFinding#H1).

Verifies that BaseFilter, BaseNeuron, and BaseStimulus enforce the
required abstract interface contracts.
"""

import pytest
import torch
import torch.nn as nn
from typing import Any, Dict, Tuple

from sensoryforge.filters.base import BaseFilter
from sensoryforge.neurons.base import BaseNeuron
from sensoryforge.stimuli.base import BaseStimulus


class TestBaseFilter:
    """Test suite for BaseFilter ABC."""

    def test_is_nn_module(self):
        """BaseFilter must inherit from nn.Module."""
        assert issubclass(BaseFilter, nn.Module)

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate abstract class directly."""
        with pytest.raises(TypeError):
            BaseFilter()

    def test_concrete_subclass_works(self):
        """A concrete subclass implementing all abstract methods works."""

        class ConcreteFilter(BaseFilter):
            def forward(self, x, dt=None):
                return x * 2

            def reset_state(self):
                pass

        f = ConcreteFilter(dt=0.01)
        assert f.dt == 0.01
        out = f(torch.ones(1, 3))
        assert torch.allclose(out, torch.full((1, 3), 2.0))

    def test_missing_forward_raises(self):
        """Subclass without forward() cannot be instantiated."""

        class BadFilter(BaseFilter):
            def reset_state(self):
                pass

        with pytest.raises(TypeError):
            BadFilter()

    def test_from_config(self):
        """from_config class method works on concrete subclass."""

        class ConfigFilter(BaseFilter):
            def __init__(self, dt=0.001, gain=1.0):
                super().__init__(dt=dt)
                self.gain = gain

            def forward(self, x, dt=None):
                return x * self.gain

            def reset_state(self):
                pass
            
            @classmethod
            def from_config(cls, config):
                # Override to handle custom parameters
                return cls(
                    dt=config.get('dt', 0.001),
                    gain=config.get('gain', 1.0)
                )

        f = ConfigFilter.from_config({"gain": 3.0})
        assert f.gain == 3.0

    def test_to_dict(self):
        """to_dict returns at least dt."""

        class SimpleFilter(BaseFilter):
            def forward(self, x, dt=None):
                return x

            def reset_state(self):
                pass

        f = SimpleFilter(dt=0.005)
        d = f.to_dict()
        assert d["dt"] == 0.005

    def test_from_config_with_dt_only(self):
        """Regression test for ReviewFinding#M1.
        
        Verifies that BaseFilter.from_config with {'dt': value} works
        correctly for subclasses that use the default implementation.
        
        Reference: reviews/REVIEW_AGENT_FINDINGS_20260211.md#M1
        """
        
        class MinimalFilter(BaseFilter):
            def forward(self, x, dt=None):
                return x
            
            def reset_state(self):
                pass
        
        # Should not raise TypeError
        f = MinimalFilter.from_config({'dt': 0.002})
        assert f.dt == 0.002
    
    def test_from_config_with_missing_dt_uses_default(self):
        """Test that from_config works with empty config (uses default dt)."""
        
        class MinimalFilter(BaseFilter):
            def forward(self, x, dt=None):
                return x
            
            def reset_state(self):
                pass
        
        f = MinimalFilter.from_config({})
        assert f.dt == 0.001  # Default value


class TestBaseNeuron:
    """Test suite for BaseNeuron ABC."""

    def test_is_nn_module(self):
        """BaseNeuron must inherit from nn.Module."""
        assert issubclass(BaseNeuron, nn.Module)

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate abstract class directly."""
        with pytest.raises(TypeError):
            BaseNeuron()

    def test_concrete_subclass_works(self):
        """A concrete subclass implementing all abstract methods works."""

        class ConcreteNeuron(BaseNeuron):
            def forward(self, input_current):
                b, s, f = input_current.shape
                v_trace = torch.zeros(b, s + 1, f)
                spikes = torch.zeros(b, s + 1, f, dtype=torch.bool)
                return v_trace, spikes

            def reset_state(self):
                pass

        n = ConcreteNeuron(dt=0.1)
        assert n.dt == 0.1
        v, s = n(torch.randn(2, 10, 5))
        assert v.shape == (2, 11, 5)
        assert s.shape == (2, 11, 5)

    def test_missing_forward_raises(self):
        """Subclass without forward() cannot be instantiated."""

        class BadNeuron(BaseNeuron):
            def reset_state(self):
                pass

        with pytest.raises(TypeError):
            BadNeuron()


class TestBaseStimulus:
    """Test suite for BaseStimulus ABC."""

    def test_is_nn_module(self):
        """BaseStimulus must inherit from nn.Module."""
        assert issubclass(BaseStimulus, nn.Module)

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate abstract class directly."""
        with pytest.raises(TypeError):
            BaseStimulus()

    def test_concrete_subclass_works(self):
        """A concrete subclass implementing all abstract methods works."""

        class ConcreteStimulus(BaseStimulus):
            def __init__(self, amplitude=1.0):
                super().__init__()
                self.amplitude = amplitude

            def forward(self, xx, yy, **kwargs):
                return self.amplitude * torch.exp(-(xx**2 + yy**2))

            def reset_state(self):
                pass

        s = ConcreteStimulus(amplitude=2.0)
        xx = torch.linspace(-1, 1, 5).unsqueeze(1).expand(5, 5)
        yy = torch.linspace(-1, 1, 5).unsqueeze(0).expand(5, 5)
        out = s(xx, yy)
        assert out.shape == (5, 5)
        # Center value should be 2.0 * exp(0) = 2.0
        assert abs(out[2, 2].item() - 2.0) < 0.01
