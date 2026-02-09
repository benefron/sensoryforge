"""Regression test for Izhikevich u_init tuple-b bug (ReviewFinding#M1).

When ``b`` is a ``(mean, std)`` tuple, ``torch.full()`` was called with a
tensor fill value, causing a RuntimeError.

Reference: reviews/REVIEW_AGENT_FINDINGS_20260209.md#M1
"""

import pytest
import torch

from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch


class TestIzhikevichTupleB:
    """Verify tuple-b parameter variability works without crash."""

    def test_tuple_b_no_crash(self):
        """forward() should not crash when b=(mean, std)."""
        neuron = IzhikevichNeuronTorch(b=(0.2, 0.01))
        current = torch.ones(1, 100, 10) * 10.0
        v_trace, spikes = neuron(current)
        assert v_trace.shape == (1, 101, 10)
        assert spikes.shape == (1, 101, 10)

    def test_all_tuple_params(self):
        """forward() works when all params are (mean, std) tuples."""
        neuron = IzhikevichNeuronTorch(
            a=(0.02, 0.001),
            b=(0.2, 0.01),
            c=(-65.0, 1.0),
            d=(8.0, 0.5),
        )
        current = torch.ones(2, 50, 5) * 12.0
        v_trace, spikes = neuron(current)
        assert v_trace.shape == (2, 51, 5)
        assert spikes.dtype == torch.bool

    def test_scalar_b_still_works(self):
        """Scalar b should continue to work as before."""
        neuron = IzhikevichNeuronTorch(b=0.2)
        current = torch.ones(1, 50, 3) * 10.0
        v_trace, spikes = neuron(current)
        assert v_trace.shape == (1, 51, 3)
