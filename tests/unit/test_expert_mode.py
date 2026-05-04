"""Smoke tests for Expert mode toggle on MechanoreceptorTab and SpikingNeuronTab (item 6).

Covers:
- MechanoreceptorTab has chk_expert_mode checkbox, default unchecked
- All _expert_only_widgets are hidden in Basic mode
- All _expert_only_widgets are visible (not hidden) in Expert mode
- SpikingNeuronTab has chk_expert_mode checkbox
- SpikingNeuronTab advanced sections hidden in Basic mode, visible in Expert mode

Design note: module-scoped fixtures avoid pyqtgraph GC-during-creation segfaults
that occur when multiple tabs with plot widgets are created and destroyed in the
same pytest session.
"""

import sys

import pytest

# Keep QApplication alive for the entire module — must be a module-level var
# so it is not GC'd between fixture calls.
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


# ---------------------------------------------------------------------------
# Module-scoped fixtures — each tab is created once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mech_tab():
    _ensure_app()
    try:
        from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab
    except ImportError:
        pytest.skip("PyQt5 not available")
    return MechanoreceptorTab()


@pytest.fixture(scope="module")
def spiking_tab(mech_tab):
    try:
        from sensoryforge.gui.tabs.spiking_tab import SpikingNeuronTab
        from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab
    except ImportError:
        pytest.skip("GUI tabs not available")
    stim = StimulusDesignerTab(mechanoreceptor_tab=mech_tab)
    return SpikingNeuronTab(mechanoreceptor_tab=mech_tab, stimulus_tab=stim)


# ---------------------------------------------------------------------------
# MechanoreceptorTab
# ---------------------------------------------------------------------------

def test_mechanoreceptor_tab_has_expert_mode_checkbox(mech_tab):
    """MechanoreceptorTab must expose chk_expert_mode."""
    assert hasattr(mech_tab, "chk_expert_mode"), "MechanoreceptorTab missing chk_expert_mode"


def test_mechanoreceptor_expert_only_widgets_non_empty(mech_tab):
    """_expert_only_widgets must contain at least one widget."""
    assert len(mech_tab._expert_only_widgets) > 0, "_expert_only_widgets must not be empty"


def test_mechanoreceptor_basic_mode_hides_advanced_widgets(mech_tab):
    """In Basic mode (unchecked), all _expert_only_widgets must be hidden."""
    mech_tab.chk_expert_mode.setChecked(False)
    for w in mech_tab._expert_only_widgets:
        assert w.isHidden(), f"Widget {w!r} should be hidden in Basic mode but is not"


def test_mechanoreceptor_expert_mode_shows_advanced_widgets(mech_tab):
    """In Expert mode (checked), all _expert_only_widgets must not be hidden."""
    mech_tab.chk_expert_mode.setChecked(True)
    for w in mech_tab._expert_only_widgets:
        assert not w.isHidden(), f"Widget {w!r} should not be hidden in Expert mode but is"


# ---------------------------------------------------------------------------
# SpikingNeuronTab
# ---------------------------------------------------------------------------

def test_spiking_tab_has_expert_mode_checkbox(spiking_tab):
    """SpikingNeuronTab must expose chk_expert_mode."""
    assert hasattr(spiking_tab, "chk_expert_mode"), "SpikingNeuronTab missing chk_expert_mode"


def test_spiking_expert_only_widgets_non_empty(spiking_tab):
    """_expert_only_widgets_spiking must contain at least one widget."""
    assert len(spiking_tab._expert_only_widgets_spiking) > 0, (
        "_expert_only_widgets_spiking must not be empty"
    )


def test_spiking_basic_mode_hides_advanced_widgets(spiking_tab):
    """In Basic mode, all _expert_only_widgets_spiking must be hidden."""
    spiking_tab.chk_expert_mode.setChecked(False)
    for w in spiking_tab._expert_only_widgets_spiking:
        assert w.isHidden(), f"Widget {w!r} should be hidden in Basic mode but is not"


def test_spiking_expert_mode_shows_advanced_widgets(spiking_tab):
    """In Expert mode, all _expert_only_widgets_spiking must not be hidden."""
    spiking_tab.chk_expert_mode.setChecked(True)
    for w in spiking_tab._expert_only_widgets_spiking:
        assert not w.isHidden(), f"Widget {w!r} should not be hidden in Expert mode but is"
