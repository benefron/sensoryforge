"""Tests for DSL I-symbol consistency and compile() flexibility (ReviewFinding#H8).

Verifies that:
1. The 'I' symbol in equation parsing does not conflict with sympy's imaginary I
2. compile() accepts both string solver and BaseSolver instances
3. End-to-end DSL pipeline produces spikes

Reference: reviews/REVIEW_AGENT_FINDINGS_20260209.md#H8
"""

import pytest
import torch

try:
    from sensoryforge.neurons.model_dsl import NeuronModel
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from sensoryforge.solvers.euler import EulerSolver


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="sympy not installed")
class TestDSLISymbolConsistency:
    """Verify I is treated as input current, not imaginary unit."""

    @pytest.fixture
    def izh_model(self):
        return NeuronModel(
            equations="""
                dv/dt = 0.04*v**2 + 5*v + 140 - u + I
                du/dt = a*(b*v - u)
            """,
            threshold="v >= 30",
            reset="v = c\nu = u + d",
            parameters={"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
        )

    def test_compile_produces_real_outputs(self, izh_model):
        """Compiled module should produce real (not complex) outputs."""
        neuron = izh_model.compile(dt=0.05)
        current = torch.ones(1, 100, 5) * 10.0
        v_trace, spikes = neuron(current)
        assert not torch.is_complex(v_trace)
        assert v_trace.shape == (1, 101, 5)
        assert spikes.dtype == torch.bool

    def test_nonzero_current_produces_spikes(self, izh_model):
        """Strong input current should produce at least one spike."""
        neuron = izh_model.compile(dt=0.05)
        current = torch.ones(1, 200, 3) * 15.0
        _, spikes = neuron(current)
        assert spikes.any(), "Expected at least one spike with I=15 mA"

    def test_compile_accepts_solver_instance(self, izh_model):
        """compile() should accept a BaseSolver instance without error."""
        solver = EulerSolver(dt=0.05)
        neuron = izh_model.compile(solver=solver, dt=0.05)
        current = torch.ones(1, 50, 2) * 10.0
        v_trace, spikes = neuron(current)
        assert v_trace.shape == (1, 51, 2)

    def test_compile_rejects_unknown_solver_string(self, izh_model):
        """compile() should raise ValueError for unknown solver string."""
        with pytest.raises(ValueError, match="Unsupported solver"):
            izh_model.compile(solver="rk4")
