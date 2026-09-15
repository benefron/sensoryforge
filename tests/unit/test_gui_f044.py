"""GUI tests for F-044: invalid time steps must not crash the GUI.

Covers:
- The stimulus Δt spinbox snaps a typed value to a multiple of the
  integration step on ``editingFinished``.
- ``SpikingNeuronTab._run_simulation`` catches ``ValueError`` (not just
  ``RuntimeError``) from ``_simulate_population`` and reports it through the
  existing ``errors`` list/dialog instead of letting it escape the Qt slot.

Marked gui; run alone (this repo's Qt test suite is order-dependent, F-016).
"""

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

_APP = None


def _ensure_app():
    global _APP
    try:
        from PyQt5 import QtWidgets

        _APP = QtWidgets.QApplication.instance()
        if _APP is None:
            _APP = QtWidgets.QApplication(sys.argv[:1])
    except ImportError:
        pytest.skip("PyQt5 not available")


@pytest.fixture(scope="module")
def mech_tab():
    _ensure_app()
    try:
        from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab
    except ImportError:
        pytest.skip("PyQt5 not available")
    return MechanoreceptorTab()


@pytest.fixture(scope="module")
def stim_tab(mech_tab):
    _ensure_app()
    try:
        from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab
    except ImportError:
        pytest.skip("GUI tabs not available")
    return StimulusDesignerTab(mechanoreceptor_tab=mech_tab)


@pytest.fixture(scope="module")
def spiking_tab(mech_tab):
    try:
        from sensoryforge.gui.tabs.spiking_tab import SpikingNeuronTab
        from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab
    except ImportError:
        pytest.skip("GUI tabs not available")
    stim = StimulusDesignerTab(mechanoreceptor_tab=mech_tab)
    return SpikingNeuronTab(mechanoreceptor_tab=mech_tab, stimulus_tab=stim)


def test_dt_spinbox_snaps_to_integration_step(stim_tab):
    """Typing 0.12 into spin_dt and finishing editing must snap to 0.10."""
    stim_tab.spin_dt.setValue(0.12)
    stim_tab._on_dt_editing_finished()
    assert stim_tab.spin_dt.value() == pytest.approx(0.10)


def test_run_simulation_reports_value_error(spiking_tab, monkeypatch):
    """A ValueError from _simulate_population must not crash _run_simulation."""
    from sensoryforge.gui.tabs.spiking_tab import PopulationConfig
    from PyQt5 import QtWidgets
    import torch

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)

    cfg = PopulationConfig(name="Test Pop", neuron_type="SA", enabled=True)
    spiking_tab.population_configs["Test Pop"] = cfg

    monkeypatch.setattr(spiking_tab, "generator", object())
    monkeypatch.setattr(spiking_tab, "grid_manager", object())
    monkeypatch.setattr(spiking_tab, "_stimulus_frames", torch.zeros(1, 2, 2))
    monkeypatch.setattr(spiking_tab, "_stimulus_times", np.array([0.0]))
    monkeypatch.setattr(spiking_tab, "_stimulus_dt_ms", 0.1)
    monkeypatch.setattr(spiking_tab, "_find_population", lambda name: object())

    def _raise_value_error(*_args, **_kwargs):
        raise ValueError("dt_ms is not a whole multiple of integrate_dt_ms")

    monkeypatch.setattr(spiking_tab, "_simulate_population", _raise_value_error)

    spiking_tab._run_simulation()  # must return normally, not raise

    assert "Test Pop" not in spiking_tab.sim_results
