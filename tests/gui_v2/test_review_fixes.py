"""Regressions for the defects the Phase 3.5 review reproduced (2026-09-21).

Each test drives the real widgets the way a user would and checks the config
or the run, not the widget code.
"""

import pytest

pytestmark = pytest.mark.gui

from PyQt5 import QtTest, QtWidgets  # noqa: E402

from sensoryforge.config.schema import SensoryForgeConfig  # noqa: E402
from sensoryforge.gui.app import STAGE_ORDER, SensoryForgeApp  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402

PRESET = "sensoryforge/presets/tactile_sa1_ra1.yml"


def _config(rows=16):
    config = SensoryForgeConfig.from_yaml_file(PRESET)
    config.grids[0].rows = rows
    config.grids[0].cols = rows
    return config


def _app(qtbot, config=None):
    session = Session(config if config is not None else _config())
    window = SensoryForgeApp(session)
    window.show_dialogs = False
    window.run_bar.show_dialogs = False
    qtbot.addWidget(window)
    window.show()
    return session, window


def _screen(window, stage):
    window.stage_list.setCurrentRow(STAGE_ORDER.index(stage))
    return window._screens[stage]


def _row_widget(card, label_text):
    form = card._content.layout()
    for row in range(form.rowCount()):
        label = form.itemAt(row, QtWidgets.QFormLayout.LabelRole)
        field = form.itemAt(row, QtWidgets.QFormLayout.FieldRole)
        if label and label.widget() and label.widget().text() == label_text:
            return field.widget()
    raise KeyError(label_text)


# 1 -------------------------------------------------------------------------
def test_strip_follows_population_add_duplicate_and_remove(qtbot):
    session, window = _app(qtbot)
    populations = _screen(window, "populations")
    populations.list_widget.setCurrentRow(0)
    populations.btn_duplicate.click()
    populations.btn_add.click()
    populations.list_widget.setCurrentRow(0)
    populations.btn_remove.click()
    names = [p.name for p in session.config.populations]
    strip = window.pipeline_strip
    assert len(strip._pop_rows) == len(names)


# 2 -------------------------------------------------------------------------
def test_typing_a_gain_keeps_every_digit(qtbot):
    session, window = _app(qtbot)
    populations = _screen(window, "populations")
    populations.list_widget.setCurrentRow(0)
    spin = _row_widget(populations.readout_card, "Input gain:")
    from PyQt5 import sip

    spin.lineEdit().selectAll()
    for key in "125":
        # Each keystroke writes the config; the card used to rebuild and
        # delete this very spin box, keeping only the first digit.
        assert not sip.isdeleted(spin)
        QtTest.QTest.keyClick(spin.lineEdit(), key)
        QtTest.QTest.qWait(20)
    assert session.config.populations[0].input_gain == 125.0


# 3 -------------------------------------------------------------------------
def test_neuron_form_does_not_offer_parameters_the_engine_overrides(qtbot):
    from sensoryforge.gui.widgets.param_form import ParamForm

    session, window = _app(qtbot)
    populations = _screen(window, "populations")
    populations.list_widget.setCurrentRow(0)
    form = populations.neuron_card.findChild(ParamForm)
    names = {spec.name for spec in form._specs}
    assert "dt" not in names and "noise_std" not in names
    assert "a" in names


# 4 -------------------------------------------------------------------------
def test_quick_run_does_not_replace_the_full_results(qtbot):
    session, window = _app(qtbot)
    window.run_bar.duration_spin.setValue(40.0)
    with qtbot.waitSignal(window.run_controller.finished, timeout=60000):
        window.run_bar.run_button.click()
    full = session.last_results
    populations = _screen(window, "populations")
    populations.list_widget.setCurrentRow(0)
    with qtbot.waitSignal(populations._run_controller.finished, timeout=60000):
        populations._on_quick_run()
    assert session.last_results is full


def test_a_second_run_cannot_start_while_one_is_running(qtbot):
    session, window = _app(qtbot, _config(rows=40))
    window.run_bar.duration_spin.setValue(400.0)
    window.run_bar.run_button.click()
    assert session.run_in_progress
    populations = _screen(window, "populations")
    populations.list_widget.setCurrentRow(0)
    populations._on_quick_run()
    assert not populations._run_controller.running
    assert populations.quick_error.isVisibleTo(populations)
    window.run_controller.cancel()
    qtbot.waitUntil(lambda: not session.run_in_progress, timeout=60000)


# 6 -------------------------------------------------------------------------
def test_rf_bench_and_export_follow_a_grid_edit(qtbot, tmp_path):
    import json

    session, window = _app(qtbot)
    populations = _screen(window, "populations")
    populations.list_widget.setCurrentRow(0)
    session.set_by_path("grids.0.rows", 10)
    session.set_by_path("grids.0.cols", 10)
    folder = populations.rf_bench.export_receptive_fields(tmp_path / "rf")
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["num_receptors"] == 100


# 8 -------------------------------------------------------------------------
def test_the_stimulus_grid_cannot_be_removed_and_a_dangling_one_is_an_error(qtbot):
    from sensoryforge.config.schema import GridConfig
    from sensoryforge.gui.validation import validate

    config = _config()
    config.grids.append(GridConfig(name="other", rows=8, cols=8))
    config.stimulus.target_layer = "other"
    session, window = _app(qtbot, config)
    sensors = _screen(window, "sensors")
    sensors._refresh_list(select_index=1)
    sensors.btn_remove.click()
    assert [g.name for g in session.config.grids] == ["Main Grid", "other"]
    assert "stimulus" in sensors.list_message.text()

    config = _config()
    config.stimulus.target_layer = "gone"
    assert "stimulus" in validate(config)


def test_a_failed_start_leaves_the_run_bar_usable(qtbot, monkeypatch):
    session, window = _app(qtbot)

    def boom(**_kwargs):
        raise ValueError("cannot render")

    monkeypatch.setattr(window.run_controller, "run", boom)
    window.run_bar.run_button.click()
    assert window.run_bar.run_button.isEnabled()
    assert not window.run_bar.cancel_button.isEnabled()
    assert "cannot render" in window.run_bar.problem_label.text()


# 9 -------------------------------------------------------------------------
def test_switching_filter_or_neuron_drops_the_old_parameters(qtbot):
    session, window = _app(qtbot)
    populations = _screen(window, "populations")
    populations.list_widget.setCurrentRow(0)
    session.set_by_path("populations.0.filter_params", {"tau_r": 7.0})
    populations.filter_card.findChild(QtWidgets.QComboBox).setCurrentText("RA")
    assert session.config.populations[0].filter_params == {}
    assert "populations.0.filter" not in session.errors

    session.set_by_path("populations.0.model_params", {"a": 0.05})
    populations.neuron_card.findChild(QtWidgets.QComboBox).setCurrentText("AdEx")
    assert session.config.populations[0].model_params == {}


# 10 ------------------------------------------------------------------------
def test_advanced_reaches_the_population_forms(qtbot):
    from sensoryforge.gui.widgets.param_form import ParamForm

    session, window = _app(qtbot)
    populations = _screen(window, "populations")
    populations.list_widget.setCurrentRow(0)
    window.advanced_check.setChecked(False)  # the preference may persist
    form = populations.inputs_card.findChild(ParamForm)
    normalize = form.widget_for("normalize")
    assert normalize.isHidden()
    window.advanced_check.setChecked(True)
    assert not normalize.isHidden()


# 11 ------------------------------------------------------------------------
def test_sub_stimulus_values_are_neither_rounded_nor_clamped(qtbot):
    session, window = _app(qtbot)
    stimulus = _screen(window, "stimulus")
    stimulus.type_combo.setCurrentIndex(stimulus.type_combo.findData("composite"))
    entries = [
        {
            "class": "StaticStimulus",
            "stim_type": "gaussian",
            "params": {
                "amplitude": -15.0,
                "sigma": 0.125,
                "center_x": 0.0,
                "center_y": 0.0,
            },
        },
    ]
    session.set_by_path("stimulus.stimuli", entries)
    table = stimulus.sub_stimuli.table
    for column in range(1, 5):
        table.cellWidget(0, column).editingFinished.emit()
    assert session.config.stimulus.stimuli[0]["params"]["sigma"] == 0.125
    assert session.config.stimulus.stimuli[0]["params"]["amplitude"] == -15.0


# 12 ------------------------------------------------------------------------
def test_dsl_ok_is_disabled_by_an_edit_after_compiling(qtbot):
    from sensoryforge.gui.screens.dsl_editor import DslEditorDialog

    config = _config()
    config.populations[0].neuron_model = "DSL"
    config.populations[0].dsl_config = {
        "equations": "dv/dt = (-v + I) / 10.0",
        "threshold": "v >= 30",
        "reset": "v = 0",
        "state_vars": {"v": 0.0},
        "parameters": {},
    }
    session = Session(config)
    dialog = DslEditorDialog(session, config.populations[0].name)
    qtbot.addWidget(dialog)
    ok = dialog.button_box.button(QtWidgets.QDialogButtonBox.Ok)
    dialog._on_compile()
    assert ok.isEnabled()
    dialog.threshold_edit.setText("")
    assert not ok.isEnabled()
    dialog._on_ok()
    assert session.config.populations[0].dsl_config["threshold"] == "v >= 30"


# 13 ------------------------------------------------------------------------
def test_a_malformed_config_is_reported_not_raised(qtbot, tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("grids: [\n  - name: x\n", encoding="utf-8")
    session, window = _app(qtbot)
    before = session.config
    window._load_config_file(str(bad))
    assert session.config is before


# 14 ------------------------------------------------------------------------
def test_repetitions_differ_even_with_a_population_noise_seed(qtbot):
    from sensoryforge.gui.screens.batch import BatchScreen

    config = _config()
    config.populations[0].noise_seed = 11
    session = Session(config)
    batch = BatchScreen(session)
    qtbot.addWidget(batch)
    row = batch.add_field("populations.0.input_gain")
    batch._rows[row]["values_edit"].setText("50")
    batch.reps_spin.setValue(3)
    jobs, error = batch.jobs()
    assert error is None
    assert [job["populations.0.noise_seed"] for job in jobs] == [11, 12, 13]


def test_batch_follows_a_loaded_duration_and_edits(qtbot):
    from sensoryforge.gui.screens.batch import BatchScreen

    session = Session(_config())
    batch = BatchScreen(session)
    qtbot.addWidget(batch)
    other = _config()
    other.simulation.duration_ms = 250.0
    session.replace_config(other)
    assert batch.duration_spin.value() == 250.0


# 15 ------------------------------------------------------------------------
def test_saving_clears_the_edited_mark(qtbot, tmp_path):
    session, window = _app(qtbot)
    window._new_project(str(tmp_path / "project"))
    session.set_by_path("populations.0.input_gain", 40.0)
    assert "edited" in window.pipeline_strip._status_label.text()
    window._on_save_config()
    assert "edited" not in window.pipeline_strip._status_label.text()
