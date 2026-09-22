"""Tests for :class:`sensoryforge.gui.widgets.run_bar.RunBar`.

``RunController`` is being written by a concurrent lane; this exercises
``RunBar`` against the contract's signal/method names (section 7 of the
phase 1 interface contract) using a small stub ``QObject``.
"""

import pytest
from PyQt5 import QtCore

pytestmark = pytest.mark.gui

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
)
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.gui.widgets.run_bar import RunBar  # noqa: E402


class ControllerStub(QtCore.QObject):
    """Stands in for ``RunController`` (contract section 7)."""

    started = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.run_calls = []
        self.cancel_calls = 0
        self._running = False

    def run(self, *, duration_ms, bundle=True, quick_population=None):
        self._running = True
        self.run_calls.append(
            {
                "duration_ms": duration_ms,
                "bundle": bundle,
                "quick_population": quick_population,
            }
        )

    def cancel(self):
        self.cancel_calls += 1
        self._running = False

    @property
    def running(self):
        return self._running


class _Result:
    def __init__(self, bundle_dir=None):
        self.bundle_dir = bundle_dir


def _config() -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[GridConfig(name="skin", rows=4, cols=4)],
        populations=[PopulationConfig(name="SA", target_grid="skin")],
    )


def test_device_combo_reflects_available_devices(qtbot):
    session = Session(_config())
    bar = RunBar(session)
    qtbot.addWidget(bar)
    items = [bar.device_combo.itemText(i) for i in range(bar.device_combo.count())]
    assert items == session.available_devices
    assert bar.device_combo.currentText() == session.device


def test_run_calls_controller_with_spinbox_duration(qtbot):
    session = Session(_config())
    bar = RunBar(session)
    qtbot.addWidget(bar)
    controller = ControllerStub()
    bar.set_controller(controller)
    bar.duration_spin.setValue(250.0)

    bar.run_button.click()

    assert len(controller.run_calls) == 1
    assert controller.run_calls[0]["duration_ms"] == 250.0


def test_progress_and_finished_update_bar_and_label(qtbot):
    session = Session(_config())
    bar = RunBar(session)
    bar.show_dialogs = False
    qtbot.addWidget(bar)
    controller = ControllerStub()
    bar.set_controller(controller)

    controller.started.emit()
    controller.progress.emit(0, 2, "SA")
    assert "SA" in bar.progress_bar.format() or "SA" in bar.progress_bar.text()

    controller.finished.emit(_Result(bundle_dir=None))
    assert bar.progress_bar.value() == 100
    assert "not saved" in bar.bundle_label.text()


def test_cancel_calls_controller_cancel(qtbot):
    session = Session(_config())
    bar = RunBar(session)
    qtbot.addWidget(bar)
    controller = ControllerStub()
    bar.set_controller(controller)

    bar.run_button.click()
    bar.cancel_button.click()
    assert controller.cancel_calls == 1


def test_run_disabled_while_running(qtbot):
    session = Session(_config())
    bar = RunBar(session)
    qtbot.addWidget(bar)
    controller = ControllerStub()
    bar.set_controller(controller)

    assert bar.run_button.isEnabled()
    controller.started.emit()
    assert not bar.run_button.isEnabled()
    controller.finished.emit(_Result(bundle_dir=None))
    assert bar.run_button.isEnabled()


def test_duration_is_written_to_the_config_and_follows_a_loaded_one(qtbot):
    from sensoryforge.config.schema import SensoryForgeConfig
    from sensoryforge.gui.session import Session
    from sensoryforge.gui.widgets.run_bar import RunBar

    session = Session(SensoryForgeConfig())
    bar = RunBar(session)
    qtbot.addWidget(bar)
    assert session.config.simulation.duration_ms is None
    assert bar.duration_spin.value() == 1000.0

    bar.duration_spin.setValue(250.0)
    bar.duration_spin.editingFinished.emit()
    assert session.config.simulation.duration_ms == 250.0

    other = SensoryForgeConfig()
    other.simulation.duration_ms = 40.0
    other.simulation.dt_ms = 0.5
    session.replace_config(other)
    assert bar.duration_spin.value() == 40.0
    assert bar.dt_spin.value() == 0.5


def test_run_is_refused_with_the_reason_while_the_config_would_not_build(qtbot):
    from sensoryforge.config.schema import SensoryForgeConfig
    from sensoryforge.gui.session import Session
    from sensoryforge.gui.widgets.run_bar import RunBar

    config = SensoryForgeConfig.from_yaml_file(
        "sensoryforge/presets/tactile_sa1_ra1.yml"
    )
    config.grids[0].rows = 8
    config.grids[0].cols = 8
    session = Session(config)
    bar = RunBar(session)
    qtbot.addWidget(bar)
    assert bar.run_button.isEnabled()
    assert not bar.problem_label.isVisibleTo(bar)

    session.set_by_path("populations.0.filter_params", {"bogus": 1.0})
    assert not bar.run_button.isEnabled()
    assert bar.problem_label.isVisibleTo(bar)
    assert "bogus" in bar.problem_label.text()

    session.set_by_path("populations.0.filter_params", {})
    assert bar.run_button.isEnabled()
    assert not bar.problem_label.isVisibleTo(bar)
