"""Tests for C3-Step4: SimulationEngine._run_pop_from_drive() shared backend.

Verifies that the extracted static method behaves correctly as a standalone
unit, ensuring the GUI (spiking_tab) and engine both call the same backend.
"""

import pytest
import torch

from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.filters.sa_ra import SAFilterTorch
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def drive():
    """Constant positive drive [1, 50, 4]."""
    return torch.ones(1, 50, 4) * 5.0


@pytest.fixture
def neuron():
    return IzhikevichNeuronTorch(dt=0.1).cpu()


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

def test_no_filter_returns_spikes(drive, neuron):
    result = SimulationEngine._run_pop_from_drive(
        drive=drive, filter_module=None, neuron_model=neuron
    )
    assert "spikes" in result
    assert isinstance(result["spikes"], torch.Tensor)


def test_with_sa_filter_returns_spikes(drive, neuron):
    filt = SAFilterTorch(dt=0.1).cpu()
    result = SimulationEngine._run_pop_from_drive(
        drive=drive, filter_module=filt, neuron_model=neuron
    )
    assert "spikes" in result


def test_intermediates_absent_by_default(drive, neuron):
    result = SimulationEngine._run_pop_from_drive(
        drive=drive, filter_module=None, neuron_model=neuron, return_intermediates=False
    )
    assert "drive" not in result
    assert "filtered" not in result


def test_intermediates_present_when_requested(drive, neuron):
    result = SimulationEngine._run_pop_from_drive(
        drive=drive, filter_module=None, neuron_model=neuron, return_intermediates=True
    )
    assert "drive" in result
    assert "filtered" in result


# ---------------------------------------------------------------------------
# Gain and noise
# ---------------------------------------------------------------------------

def test_gain_zero_silences_output(drive, neuron):
    result = SimulationEngine._run_pop_from_drive(
        drive=drive, filter_module=None, neuron_model=neuron,
        input_gain=0.0
    )
    assert result["spikes"].sum().item() == 0


def test_gain_applied_to_filtered(drive):
    """filtered == drive * gain when filter is None."""
    neuron = IzhikevichNeuronTorch(dt=0.1).cpu()
    result = SimulationEngine._run_pop_from_drive(
        drive=drive, filter_module=None, neuron_model=neuron,
        input_gain=3.0, return_intermediates=True
    )
    assert torch.allclose(result["filtered"], drive * 3.0, atol=1e-5)


def test_noise_std_zero_no_noise(drive):
    """With noise_std=0 and same drive, two calls produce identical filtered."""
    neuron1 = IzhikevichNeuronTorch(dt=0.1).cpu()
    neuron2 = IzhikevichNeuronTorch(dt=0.1).cpu()
    r1 = SimulationEngine._run_pop_from_drive(
        drive=drive, filter_module=None, neuron_model=neuron1,
        noise_std=0.0, return_intermediates=True
    )
    r2 = SimulationEngine._run_pop_from_drive(
        drive=drive, filter_module=None, neuron_model=neuron2,
        noise_std=0.0, return_intermediates=True
    )
    assert torch.allclose(r1["filtered"], r2["filtered"])


def test_noise_std_nonzero_adds_stochasticity(drive):
    """With noise_std > 0, two calls with different torch seeds differ."""
    neuron1 = IzhikevichNeuronTorch(dt=0.1).cpu()
    neuron2 = IzhikevichNeuronTorch(dt=0.1).cpu()
    torch.manual_seed(1)
    r1 = SimulationEngine._run_pop_from_drive(
        drive=drive, filter_module=None, neuron_model=neuron1,
        noise_std=5.0, return_intermediates=True
    )
    torch.manual_seed(2)
    r2 = SimulationEngine._run_pop_from_drive(
        drive=drive, filter_module=None, neuron_model=neuron2,
        noise_std=5.0, return_intermediates=True
    )
    # filtered tensors should differ (stochastic)
    assert not torch.equal(r1["filtered"], r2["filtered"])


# ---------------------------------------------------------------------------
# Filter state isolation
# ---------------------------------------------------------------------------

def test_filter_state_reset_between_calls():
    """Calling _run_pop_from_drive twice with the same filter and drive
    must produce identical filtered output (filter auto-resets on forward)."""
    filt = SAFilterTorch(dt=0.1).cpu()
    neuron1 = IzhikevichNeuronTorch(dt=0.1).cpu()
    neuron2 = IzhikevichNeuronTorch(dt=0.1).cpu()
    drive = torch.ones(1, 60, 4) * 5.0

    r1 = SimulationEngine._run_pop_from_drive(
        drive=drive, filter_module=filt, neuron_model=neuron1,
        return_intermediates=True
    )
    r2 = SimulationEngine._run_pop_from_drive(
        drive=drive, filter_module=filt, neuron_model=neuron2,
        return_intermediates=True
    )
    assert torch.allclose(r1["filtered"], r2["filtered"], atol=1e-5), (
        "Filter state leaked between _run_pop_from_drive calls."
    )
