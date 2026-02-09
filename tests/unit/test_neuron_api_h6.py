"""Tests for normalized neuron forward() signatures (ReviewFinding#H6).

Verifies that all neuron models:
1. Accept ``input_current`` as the first positional argument
2. Have a ``reset_state()`` method
3. Return ``(v_trace, spikes)`` with correct shapes

Reference: reviews/REVIEW_AGENT_FINDINGS_20260209.md#H6
"""

import pytest
import torch

from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
from sensoryforge.neurons.adex import AdExNeuronTorch
from sensoryforge.neurons.mqif import MQIFNeuronTorch
from sensoryforge.neurons.fa import FANeuronTorch
from sensoryforge.neurons.sa import SANeuronTorch


BATCH, STEPS, FEATURES = 2, 50, 10

NEURON_CLASSES = [
    IzhikevichNeuronTorch,
    AdExNeuronTorch,
    MQIFNeuronTorch,
    FANeuronTorch,
    SANeuronTorch,
]


@pytest.fixture
def input_current():
    return torch.randn(BATCH, STEPS, FEATURES)


@pytest.mark.parametrize("NeuronClass", NEURON_CLASSES)
class TestNeuronAPIConsistency:
    """Verify that every neuron model exposes a consistent API."""

    def test_forward_accepts_input_current(self, NeuronClass, input_current):
        """All models accept ``input_current`` as positional arg."""
        neuron = NeuronClass()
        v_trace, spikes = neuron(input_current)
        assert v_trace.shape == (BATCH, STEPS + 1, FEATURES)
        assert spikes.shape == (BATCH, STEPS + 1, FEATURES)

    def test_has_reset_state(self, NeuronClass, input_current):
        """All models expose ``reset_state()``."""
        neuron = NeuronClass()
        assert hasattr(neuron, "reset_state")
        assert callable(neuron.reset_state)
        # Should not raise
        neuron.reset_state()

    def test_forward_returns_tuple(self, NeuronClass, input_current):
        """All models return a 2-tuple."""
        neuron = NeuronClass()
        result = neuron(input_current)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_spikes_dtype_is_bool(self, NeuronClass, input_current):
        """Spikes tensor should be boolean."""
        neuron = NeuronClass()
        _, spikes = neuron(input_current)
        assert spikes.dtype == torch.bool


class TestFANeuronRenamedParam:
    """Verify FA neuron specifically uses ``input_current`` parameter name."""

    def test_fa_forward_parameter_name(self):
        """FANeuronTorch.forward first param is ``input_current`` not ``x``."""
        import inspect

        sig = inspect.signature(FANeuronTorch.forward)
        params = list(sig.parameters.keys())
        assert params[0] == "self"
        assert params[1] == "input_current", (
            f"Expected 'input_current' as first param, got '{params[1]}'"
        )

    def test_fa_reset_state_clears_internals(self):
        """reset_state() clears refractory and EMA buffers."""
        neuron = FANeuronTorch()
        x = torch.randn(1, 20, 5)
        neuron(x)
        # After forward, internal state should be populated
        assert neuron._ref_count is not None
        neuron.reset_state()
        assert neuron._ref_count is None
        assert neuron._ema_mean is None
