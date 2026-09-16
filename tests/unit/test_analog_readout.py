"""The shared backend kernel labels analog output (Phase 2, Wave N, N2).

Before N2, ``SimulationEngine._run_pop_from_drive`` unconditionally treated
``neuron_model``'s second return value as a spikes tensor
(``spikes_sub[:, 1:, :].float()``), which raises when it is ``None`` -- the
new analog-readout contract from N1 (a DSL model with no threshold returns
``(state_trace, None)``). Now, when ``spikes`` is ``None``, the result dict
carries ``"state"`` (bin-end samples, matching how ``"voltages"`` is already
reduced) and no ``"spikes"`` key; a spiking population is unaffected -- its
``"spikes"`` values are pinned here against a fixed-seed recorded array so a
future change cannot silently perturb them.
"""

import torch
import torch.nn as nn

from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch


class _AnalogStub(nn.Module):
    """A trivial neuron model with no spike condition (mimics a
    thresholdless compiled DSL model, N1): forward() returns
    ``(state_trace, None)``."""

    def __init__(self, dt: float = 0.05):
        super().__init__()
        self.dt = dt

    def forward(self, input_current: torch.Tensor):
        # state_trace[t] = cumulative sum of input up to step t (deterministic,
        # easy to hand-check), shape [batch, steps+1, features].
        batch, steps, features = input_current.shape
        cumulative = torch.cumsum(input_current, dim=1)
        zeros = torch.zeros(batch, 1, features, dtype=input_current.dtype)
        state_trace = torch.cat([zeros, cumulative], dim=1)
        return state_trace, None


class TestAnalogReadoutLabelling:
    def test_analog_population_returns_state_and_no_spikes(self):
        drive = torch.ones(1, 4, 2)  # [batch, time, neurons]
        neuron = _AnalogStub(dt=1.0)

        result = SimulationEngine._run_pop_from_drive(
            drive=drive,
            filter_module=None,
            neuron_model=neuron,
            dt_ms=1.0,
            integrate_dt_ms=1.0,
        )

        assert "spikes" not in result
        assert "state" in result
        assert tuple(result["state"].shape) == (1, 4, 2)

    def test_analog_shape_matches_spiking_shape_convention(self):
        # Both spiking and analog results are [1, T, N] for the same drive.
        drive = torch.ones(1, 6, 3)

        analog_result = SimulationEngine._run_pop_from_drive(
            drive=drive,
            filter_module=None,
            neuron_model=_AnalogStub(dt=1.0),
            dt_ms=1.0,
            integrate_dt_ms=1.0,
        )

        spiking_neuron = IzhikevichNeuronTorch(dt=1.0)
        spiking_result = SimulationEngine._run_pop_from_drive(
            drive=drive,
            filter_module=None,
            neuron_model=spiking_neuron,
            dt_ms=1.0,
            integrate_dt_ms=1.0,
        )

        assert tuple(analog_result["state"].shape) == tuple(
            spiking_result["spikes"].shape
        )
        assert tuple(analog_result["state"].shape) == (1, 6, 3)

    def test_analog_state_values_are_bin_end_samples(self):
        # Bin-end reduction: with n_substeps=1 (dt_ms == integrate_dt_ms),
        # state[:, t, :] must equal state_trace[:, t+1, :] (after dropping
        # the initial sample, matching how "voltages" is already reduced).
        drive = torch.tensor([[[1.0], [2.0], [3.0]]])  # [1, 3, 1]
        neuron = _AnalogStub(dt=1.0)
        result = SimulationEngine._run_pop_from_drive(
            drive=drive,
            filter_module=None,
            neuron_model=neuron,
            dt_ms=1.0,
            integrate_dt_ms=1.0,
        )
        # _AnalogStub's state_trace is [0, 1, 3, 6] (cumsum with a leading
        # zero); dropping the initial sample gives [1, 3, 6] as the state.
        expected = torch.tensor([[[1.0], [3.0], [6.0]]])
        torch.testing.assert_close(result["state"], expected)


class TestSpikingUnchanged:
    """Spiking populations are unaffected by the N2 change (behaviour
    preservation, Phase 2 guardrail 1): pinned against a recorded spike
    array for a fixed, deterministic (noise-free) Izhikevich RS neuron."""

    def test_spiking_population_matches_recorded_spike_counts(self):
        torch.manual_seed(0)
        drive = torch.full((1, 20, 2), 40.0)
        neuron = IzhikevichNeuronTorch(dt=0.5)

        result = SimulationEngine._run_pop_from_drive(
            drive=drive,
            filter_module=None,
            neuron_model=neuron,
            dt_ms=0.5,
            integrate_dt_ms=0.5,
        )

        assert "spikes" in result
        assert "state" not in result
        assert tuple(result["spikes"].shape) == (1, 20, 2)

        recorded_spike_counts = torch.tensor([[3.0, 3.0]])
        torch.testing.assert_close(
            result["spikes"].sum(dim=1), recorded_spike_counts
        )
