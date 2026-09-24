"""Unit tests for analog (non-spiking) DSL models (Phase 2, Wave N, N1).

``NeuronModel`` now accepts an optional ``threshold``/``reset``: with no
threshold, ``compile()`` produces a module that integrates the equations for
every step and returns ``(state_trace, None)`` instead of ``(v_trace,
spikes)``. A model *with* a threshold must still spike exactly as before.
"""

import numpy as np
import pytest
import torch

from sensoryforge.neurons.model_dsl import NeuronModel, SYMPY_AVAILABLE

pytestmark = pytest.mark.skipif(
    not SYMPY_AVAILABLE, reason="SymPy is required for DSL tests but is not installed"
)


def _hand_written_leaky_integrator(drive, dt, v_rest, R, tau_m, v0):
    """Reference Forward-Euler integration of dv/dt = (-(v-v_rest)+R*I)/tau_m.

    Args:
        drive: Input current array [steps, features].
        dt: Time step (ms).
        v_rest, R, tau_m: Model parameters.
        v0: Initial voltage.

    Returns:
        v_trace array [steps+1, features].
    """
    steps, features = drive.shape
    v = np.full(features, v0, dtype=np.float64)
    trace = np.zeros((steps + 1, features), dtype=np.float64)
    trace[0] = v
    for t in range(steps):
        dv = (-(v - v_rest) + R * drive[t]) / tau_m
        v = v + dt * dv
        trace[t + 1] = v
    return trace


class TestAnalogDSLModel:
    def test_no_threshold_returns_state_trace_and_none_spikes(self):
        model = NeuronModel(
            equations="dv/dt = (-(v - v_rest) + R*I) / tau_m",
            parameters={"v_rest": -65.0, "R": 1.0, "tau_m": 10.0},
            state_vars={"v": -65.0},
        )
        assert model.threshold_str is None
        assert model.reset_str is None

        neuron = model.compile(dt=0.5, device="cpu")
        drive = torch.randn(2, 20, 3, dtype=torch.float64) * 5.0
        state_trace, spikes = neuron(drive)

        assert spikes is None
        assert tuple(state_trace.shape) == (2, 21, 3)

    def test_analog_matches_hand_written_euler(self):
        v_rest, R, tau_m, v0 = -65.0, 1.0, 10.0, -65.0
        model = NeuronModel(
            equations="dv/dt = (-(v - v_rest) + R*I) / tau_m",
            parameters={"v_rest": v_rest, "R": R, "tau_m": tau_m},
            state_vars={"v": v0},
        )
        dt = 0.25
        neuron = model.compile(dt=dt, device="cpu")

        torch.manual_seed(0)
        drive = torch.randn(1, 50, 4, dtype=torch.float64) * 3.0
        state_trace, spikes = neuron(drive)

        assert spikes is None
        expected = _hand_written_leaky_integrator(
            drive[0].numpy(), dt, v_rest, R, tau_m, v0
        )
        actual = state_trace[0].numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_thresholded_model_still_spikes_exactly_as_before(self):
        # Simple LIF-with-threshold model, compared against a recorded array.
        model = NeuronModel(
            equations="dv/dt = (-(v - (-65.0)) + I) / 10.0",
            threshold="v >= -50.0",
            reset="v = -65.0",
            state_vars={"v": -65.0},
        )
        neuron = model.compile(dt=1.0, device="cpu")

        torch.manual_seed(42)
        drive = torch.ones(1, 30, 2, dtype=torch.float64) * 500.0
        v_trace, spikes = neuron(drive)

        assert spikes is not None
        assert spikes.dtype == torch.bool
        assert tuple(spikes.shape) == (1, 31, 2)

        # Recorded spike-count array for this exact config (seeded, no
        # noise, deterministic Euler integration).
        recorded_spike_counts = np.array([15, 15])
        np.testing.assert_array_equal(
            spikes.sum(dim=1)[0].numpy(), recorded_spike_counts
        )

    def test_to_dict_from_config_round_trip_analog(self):
        model = NeuronModel(
            equations="dv/dt = (-(v - v_rest)) / tau_m",
            parameters={"v_rest": -65.0, "tau_m": 10.0},
            state_vars={"v": -65.0},
        )
        d = model.to_dict()
        assert d["threshold"] is None
        assert d["reset"] is None

        restored = NeuronModel.from_config(d)
        assert restored.threshold_str is None
        assert restored.reset_str is None
        assert restored.to_dict() == d

    def test_to_dict_from_config_round_trip_spiking(self):
        model = NeuronModel(
            equations="dv/dt = -v + I",
            threshold="v >= 1.0",
            reset="v = 0.0",
            state_vars={"v": 0.0},
        )
        d = model.to_dict()
        assert d["threshold"] == "v >= 1.0"
        assert d["reset"] == "v = 0.0"

        restored = NeuronModel.from_config(d)
        assert restored.to_dict() == d

    def test_reset_without_threshold_raises(self):
        with pytest.raises(ValueError, match="reset"):
            NeuronModel(
                equations="dv/dt = -v + I",
                reset="v = 0.0",
                state_vars={"v": 0.0},
            )
