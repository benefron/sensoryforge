"""Build an experiment from nothing through the widgets alone, then run it (Task 2.8).

Every step is a click or an edit a user could make; the assertions read
``session.config`` and the run result. It re-derives the intent of the old
tab-bound ``test_unified_workflow.py`` against the GUI v2 screens.
"""

import pytest

pytestmark = pytest.mark.gui

from PyQt5 import QtWidgets  # noqa: E402

from sensoryforge.config.schema import SensoryForgeConfig  # noqa: E402
from sensoryforge.gui.app import STAGE_ORDER, SensoryForgeApp  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402


def _show(window, stage):
    window.stage_list.setCurrentRow(STAGE_ORDER.index(stage))
    return window._screens[stage]


def _set_spin(widget, value):
    widget.setValue(value)
    widget.editingFinished.emit()


def test_a_blank_experiment_built_through_the_screens_runs(qtbot):
    session = Session(SensoryForgeConfig())
    window = SensoryForgeApp(session)
    qtbot.addWidget(window)
    window.run_bar.show_dialogs = False

    # Nothing to run yet, and the run bar says why.
    assert not window.run_bar.run_button.isEnabled()
    assert "grid" in window.run_bar.problem_label.text()

    # Sensors: add a small grid.
    sensors = _show(window, "sensors")
    sensors.btn_add.click()
    assert len(session.config.grids) == 1
    _set_spin(sensors._param_form.widget_for("rows"), 8)
    _set_spin(sensors._param_form.widget_for("cols"), 8)
    assert (session.config.grids[0].rows, session.config.grids[0].cols) == (8, 8)

    # Populations: add one, reading the grid; give it an SA filter.
    populations = _show(window, "populations")
    populations.btn_add.click()
    assert len(session.config.populations) == 1
    assert session.config.populations[0].target_grid == session.config.grids[0].name
    filter_combo = populations.filter_card.findChild(QtWidgets.QComboBox)
    filter_combo.setCurrentText("SA")
    assert session.config.populations[0].filter_method == "SA"

    # Stimulus: a Gaussian, amplitude set in the form.
    stimulus = _show(window, "stimulus")
    stimulus.type_combo.setCurrentIndex(stimulus.type_combo.findData("gaussian"))
    amplitude = stimulus.param_form.widget_for("amplitude")
    _set_spin(amplitude, 25.0)
    assert session.config.stimulus.type == "gaussian"
    assert session.config.stimulus.amplitude == 25.0

    # Run bar: a short run.
    assert session.errors == {}, session.errors
    assert window.run_bar.run_button.isEnabled()
    _set_spin(window.run_bar.duration_spin, 50.0)
    assert session.config.simulation.duration_ms == 50.0

    with qtbot.waitSignal(window.run_controller.finished, timeout=60000) as blocker:
        window.run_bar.run_button.click()
    result = blocker.args[0]
    name = session.config.populations[0].name
    spikes = result.results[name]["spikes"]
    assert spikes.shape[1] == 50
    assert int(spikes.sum()) > 0

    # Run & Results shows the run.
    results = _show(window, "results")
    assert results.raster_panel.plot.getPlotItem().listDataItems()


def test_population_first_then_grid_and_a_rename_keep_everything_connected(qtbot):
    session = Session(SensoryForgeConfig())
    window = SensoryForgeApp(session)
    qtbot.addWidget(window)

    populations = _show(window, "populations")
    populations.btn_add.click()
    assert "reads no sensor array" in window.run_bar.problem_label.toolTip()

    sensors = _show(window, "sensors")
    sensors.btn_add.click()
    grid_name = session.config.grids[0].name
    assert session.config.populations[0].target_grid == grid_name
    grid_combo = populations.inputs_card.findChild(QtWidgets.QComboBox)
    assert grid_combo.findText(grid_name) >= 0

    session.config.stimulus.target_layer = grid_name
    sensors.name_edit.setText("skin")
    sensors.name_edit.editingFinished.emit()
    assert session.config.grids[0].name == "skin"
    assert session.config.populations[0].target_grid == "skin"
    assert session.config.stimulus.target_layer == "skin"
    grid_combo = populations.inputs_card.findChild(QtWidgets.QComboBox)
    assert grid_combo.findText("skin") >= 0
    assert "populations.0.target_grid" not in session.errors
