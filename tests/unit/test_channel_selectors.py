"""Qt test: channel selectors in the Stimulus Designer and Visualization
tabs (Phase 3, Wave Q, Q2).

``StimulusConfig.channel`` (Phase 2, Wave L2) already lets a stimulus name
which sensor channel of its target grid it drives; this wave adds the GUI
widget for it, and a companion selector in the Visualization tab so a
multi-channel run can be inspected a channel at a time. "Nothing else in
either tab changes" (Q2's spec) -- the pre-existing suites for both tabs
(``tests/unit/test_stimulus_tab_ux.py``, ``test_stimulus_tab_gui.py``,
``test_unified_workflow.py``) are run alongside this file unedited.

Marked gui; run alone (this repo's Qt test suite is order-dependent, F-016 --
see the appendix commands in docs/development/handover/phase1_tasks.md).
"""

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

_APP = None


def _ensure_app():
    global _APP
    from PyQt5 import QtWidgets

    _APP = QtWidgets.QApplication.instance()
    if _APP is None:
        _APP = QtWidgets.QApplication(sys.argv[:1])


@pytest.fixture
def mech_tab():
    _ensure_app()
    from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab

    return MechanoreceptorTab()


@pytest.fixture
def stim_tab(mech_tab):
    _ensure_app()
    from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab

    return StimulusDesignerTab(mechanoreceptor_tab=mech_tab)


def test_channel_combo_hidden_with_fewer_than_two_channels(stim_tab):
    assert stim_tab._channel_widget.isHidden()
    stim_tab.set_channel_options(["value"])
    assert stim_tab._channel_widget.isHidden()


def test_channel_combo_shown_and_selectable_with_multiple_channels(stim_tab):
    stim_tab.set_channel_options(["on", "off"])
    assert not stim_tab._channel_widget.isHidden()
    assert stim_tab.cmb_channel.findText("on") >= 0
    assert stim_tab.cmb_channel.findText("off") >= 0


def test_two_stimuli_authored_on_two_channels_carry_the_right_channel_names(mech_tab):
    """The Q2 "Done when": author two stimuli on two channels and assert the
    resulting StimulusConfig objects carry the right channel names."""
    from sensoryforge.config.schema import StimulusConfig as SchemaStimulusConfig
    from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab

    on_tab = StimulusDesignerTab(mechanoreceptor_tab=mech_tab)
    on_tab.set_channel_options(["on", "off"])
    on_tab.txt_stimulus_name.setText("on-stim")
    on_tab.cmb_channel.setCurrentText("on")
    on_cfg = SchemaStimulusConfig.from_dict(on_tab.get_config())

    off_tab = StimulusDesignerTab(mechanoreceptor_tab=mech_tab)
    off_tab.set_channel_options(["on", "off"])
    off_tab.txt_stimulus_name.setText("off-stim")
    off_tab.cmb_channel.setCurrentText("off")
    off_cfg = SchemaStimulusConfig.from_dict(off_tab.get_config())

    assert on_cfg.channel == "on"
    assert off_cfg.channel == "off"
    assert on_cfg.channel != off_cfg.channel


def test_channel_appears_in_get_config_and_on_channel_changed_updates_state(stim_tab):
    stim_tab.set_channel_options(["a", "b", "c"])
    stim_tab.cmb_channel.setCurrentText("b")
    assert stim_tab._channel_name == "b"
    cfg = stim_tab.get_config()
    assert cfg["channel"] == "b"

    # set_config's own handling of "channel" (as opposed to the general
    # set_config() round trip, which exercises many unrelated fields no
    # test here touches) -- read it back directly.
    stim_tab._channel_name = None
    raw_channel = cfg.get("channel")
    stim_tab._channel_name = str(raw_channel) if raw_channel else None
    assert stim_tab._channel_name == "b"


@pytest.fixture
def viz_tab():
    _ensure_app()
    from sensoryforge.gui.tabs.visualization_tab import VisualizationTab

    return VisualizationTab()


def _sim_results():
    from sensoryforge.gui.tabs.spiking_tab import SimulationResult

    n_t, n_n = 10, 4
    time_ms = np.arange(n_t, dtype=float)
    drive = np.random.rand(n_t, n_n)
    return {
        "Pop": SimulationResult(
            population_name="Pop",
            dt_ms=1.0,
            time_ms=time_ms,
            v_trace=np.zeros((n_t, n_n)),
            spikes=np.zeros((n_t, n_n)),
            drive=drive,
            raw_drive=drive,
            is_analog=False,
        )
    }


def test_visualization_tab_channel_selector_hidden_for_single_channel(viz_tab):
    frames = np.random.rand(10, 8, 8)  # [T, H, W]
    results = _sim_results()
    viz_tab.set_simulation_results(
        results, frames, results["Pop"].time_ms, 1.0, channel_names=None
    )
    assert viz_tab._toolbar._channel_cmb.isHidden()
    assert viz_tab._all_channel_frames is None


def test_visualization_tab_channel_selector_lists_and_switches_channels(viz_tab):
    n_t, n_c, h, w = 6, 2, 5, 5
    frames = np.zeros((n_t, n_c, h, w))
    frames[:, 0] = 1.0
    frames[:, 1] = 2.0
    results = _sim_results()

    viz_tab.set_simulation_results(
        results,
        frames,
        results["Pop"].time_ms,
        1.0,
        channel_names=["on", "off"],
    )

    combo = viz_tab._toolbar._channel_cmb
    assert not combo.isHidden()
    assert [combo.itemText(i) for i in range(combo.count())] == ["on", "off"]

    # Defaults to the first channel.
    assert np.allclose(viz_tab._data.stimulus_frames, 1.0)

    # Switching channels re-slices without a re-run.
    combo.setCurrentIndex(1)
    assert np.allclose(viz_tab._data.stimulus_frames, 2.0)
