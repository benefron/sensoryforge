"""Tests for :class:`sensoryforge.gui.screens.dsl_editor.DslEditorDialog` (Task 2.4).

Drives the real widgets and asserts on ``session.config`` and whether the
compiled model actually runs through ``SimulationEngine`` -- not just that
the dialog constructs.
"""

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets  # noqa: E402

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine  # noqa: E402
from sensoryforge.gui.screens.dsl_editor import DslEditorDialog  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402

pytestmark = pytest.mark.gui


def _dsl_config() -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[GridConfig(name="g", rows=4, cols=4)],
        populations=[
            PopulationConfig(name="P", target_grid="g", neuron_model="DSL (Custom)")
        ],
    )


class TestCompile:
    def test_syntax_error_refuses_ok(self, qtbot):
        session = Session(_dsl_config())
        dialog = DslEditorDialog(session, "P")
        qtbot.addWidget(dialog)

        dialog.eq_edit.setPlainText("this is not an equation")
        dialog._on_compile()

        assert dialog._compiled_ok is False
        ok_button = dialog.button_box.button(QtWidgets.QDialogButtonBox.Ok)
        assert ok_button.isEnabled() is False
        assert dialog.error_label.isHidden() is False

        # OK must not write anything -- even called directly, bypassing the
        # (disabled) button.
        dialog._on_ok()
        assert session.config.populations[0].dsl_config is None

    def test_leaky_integrator_no_threshold_is_analog(self, qtbot):
        session = Session(_dsl_config())
        dialog = DslEditorDialog(session, "P")
        qtbot.addWidget(dialog)

        dialog.eq_edit.setPlainText("dv/dt = -v + I")
        dialog.threshold_edit.setText("")
        dialog.reset_edit.setText("")
        dialog._on_compile()

        assert dialog._compiled_ok is True
        assert dialog.badge.text() == "analog"

    def test_threshold_and_reset_is_spiking(self, qtbot):
        session = Session(_dsl_config())
        dialog = DslEditorDialog(session, "P")
        qtbot.addWidget(dialog)

        dialog.eq_edit.setPlainText("dv/dt = -v + I")
        dialog.threshold_edit.setText("v >= 1.0")
        dialog.reset_edit.setText("v = 0.0")
        dialog._on_compile()

        assert dialog._compiled_ok is True
        assert dialog.badge.text() == "spiking"

    def test_compile_plots_a_trace(self, qtbot):
        session = Session(_dsl_config())
        dialog = DslEditorDialog(session, "P")
        qtbot.addWidget(dialog)

        dialog.eq_edit.setPlainText("dv/dt = -v + I")
        dialog._on_compile()

        x, y = dialog.trace_curve.getData()
        assert len(x) > 1
        assert len(y) == len(x)


class TestOk:
    def test_ok_writes_dsl_config_and_engine_runs(self, qtbot):
        session = Session(_dsl_config())
        dialog = DslEditorDialog(session, "P")
        qtbot.addWidget(dialog)

        dialog.eq_edit.setPlainText("dv/dt = -v + I")
        dialog.threshold_edit.setText("v >= 1.0")
        dialog.reset_edit.setText("v = 0.0")
        dialog._on_compile()
        dialog._on_ok()

        dsl_config = session.config.populations[0].dsl_config
        assert dsl_config is not None
        assert dsl_config["equations"] == "dv/dt = -v + I"
        assert dsl_config["threshold"] == "v >= 1.0"

        engine = SimulationEngine(session.config, device="cpu")
        assert len(engine.populations) == 1

    def test_ok_resets_incompatible_readout_to_auto(self, qtbot):
        config = _dsl_config()
        config.populations[0].readout = "spiking"
        session = Session(config)
        dialog = DslEditorDialog(session, "P")
        qtbot.addWidget(dialog)

        # Analog (no threshold) is incompatible with the population's
        # current readout='spiking'.
        dialog.eq_edit.setPlainText("dv/dt = -v + I")
        dialog.threshold_edit.setText("")
        dialog.reset_edit.setText("")
        dialog._on_compile()
        dialog._on_ok()

        assert session.config.populations[0].readout == "auto"
        engine = SimulationEngine(session.config, device="cpu")
        assert len(engine.populations) == 1

    def test_ok_disabled_before_compile(self, qtbot):
        session = Session(_dsl_config())
        dialog = DslEditorDialog(session, "P")
        qtbot.addWidget(dialog)

        ok_button = dialog.button_box.button(QtWidgets.QDialogButtonBox.Ok)
        assert ok_button.isEnabled() is False


class TestKeyValueTable:
    def test_duplicate_name_raises(self, qtbot):
        session = Session(_dsl_config())
        dialog = DslEditorDialog(session, "P")
        qtbot.addWidget(dialog)

        dialog.param_table._add_row()
        dialog.param_table.table.item(0, 0).setText("a")
        dialog.param_table.table.item(0, 1).setText("1.0")
        dialog.param_table._add_row()
        dialog.param_table.table.item(1, 0).setText("a")
        dialog.param_table.table.item(1, 1).setText("2.0")

        with pytest.raises(ValueError):
            dialog.param_table.values()
