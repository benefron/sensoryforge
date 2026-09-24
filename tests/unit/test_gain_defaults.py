"""Tests for the input_gain default (item 9 — units/gains fix).

Validates that:
- gain=50 produces spikes through the full SA-filter → Izhikevich chain
- gain=1  produces silence (confirming the original problem was real)
- PopulationConfig defaults to input_gain=50.0
- The GUI spinbox initial value is 50.0
"""

import pytest
import torch

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_small_drive(
    n_neurons: int = 4, n_steps: int = 2000, amplitude: float = 1.0
) -> torch.Tensor:
    """Build a constant [1, T, N] drive tensor simulating a 200ms Gaussian stimulus."""
    return torch.full((1, n_steps, n_neurons), amplitude, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Full-chain integration tests
# ---------------------------------------------------------------------------


def test_gain_50_produces_spikes():
    """SA filter → gain=50 → Izhikevich must fire at least one spike.

    Uses a 3 mA drive (typical for a Gaussian stimulus at amplitude=3).
    After the SA filter (k1≈0.05, steady state ≈0.15 mA) and gain=50, the
    effective input is ~7.5 mA, above the Izhikevich RS threshold (~3 mA).
    """
    from sensoryforge.filters.sa_ra import SAFilterTorch
    from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch

    filt = SAFilterTorch(dt=0.1)
    neuron = IzhikevichNeuronTorch(dt=0.1)

    drive = _build_small_drive(n_neurons=4, n_steps=2000, amplitude=3.0)
    filtered = filt(drive, reset_states=True)
    gained = filtered * 50.0
    _, spikes = neuron(gained)

    assert spikes.sum().item() > 0, (
        "With gain=50 and 3 mA drive, Izhikevich should fire at least one spike. "
        f"Total spikes: {spikes.sum().item()}"
    )


def test_gain_1_produces_silence():
    """SA filter → gain=1 → Izhikevich must produce no spikes even for a strong
    stimulus.

    This confirms the original bug: at default gain=1.0, the SA filter output is
    ~0.05 × amplitude mA, far below the Izhikevich threshold (~3 mA).
    Even with amplitude=5 mA the neuron receives only ~0.25 mA → silence.
    """
    from sensoryforge.filters.sa_ra import SAFilterTorch
    from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch

    filt = SAFilterTorch(dt=0.1)
    neuron = IzhikevichNeuronTorch(dt=0.1)

    drive = _build_small_drive(n_neurons=4, n_steps=2000, amplitude=5.0)
    filtered = filt(drive, reset_states=True)
    gained = filtered * 1.0
    _, spikes = neuron(gained)

    assert spikes.sum().item() == 0, (
        "With gain=1, Izhikevich should not fire even for a 5 mA drive "
        "(confirms the SA filter unit-scale mismatch). "
        f"Total spikes: {spikes.sum().item()}"
    )


# ---------------------------------------------------------------------------
# Schema default
# ---------------------------------------------------------------------------


def test_population_config_default_input_gain():
    """PopulationConfig must default input_gain to 50.0, not 1.0."""
    from sensoryforge.config.schema import PopulationConfig

    cfg = PopulationConfig(name="test", neuron_type="SA", target_grid="g")
    assert (
        cfg.input_gain == 50.0
    ), f"PopulationConfig.input_gain default should be 50.0, got {cfg.input_gain}"


def test_population_config_from_dict_default_input_gain():
    """PopulationConfig.from_dict must default input_gain to 50.0 when key is absent."""
    from sensoryforge.config.schema import PopulationConfig

    cfg = PopulationConfig.from_dict(
        {"name": "test", "neuron_type": "SA", "target_grid": "g"}
    )
    assert cfg.input_gain == 50.0, (
        f"PopulationConfig.from_dict default input_gain should be 50.0, got "
        f"{cfg.input_gain}"
    )


# ---------------------------------------------------------------------------
# GUI spinbox initial value
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_populations_screen_shows_the_default_gain_of_50(qtbot):
    """A new population's gain reads 50 on the Populations screen (the value
    that compensates the filter's N/mm^2 calibration, units_and_gains.md)."""
    from PyQt5 import QtWidgets

    from sensoryforge.config.schema import GridConfig, SensoryForgeConfig
    from sensoryforge.gui.screens.populations import PopulationsScreen
    from sensoryforge.gui.session import Session

    session = Session(SensoryForgeConfig(grids=[GridConfig(name="skin")]))
    screen = PopulationsScreen(session)
    qtbot.addWidget(screen)
    screen.btn_add.click()
    assert session.config.populations[0].input_gain == 50.0
    spins = screen.readout_card.findChildren(QtWidgets.QDoubleSpinBox)
    assert any(spin.value() == 50.0 for spin in spins)
