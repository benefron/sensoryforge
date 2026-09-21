"""Tests for the GUI v2 app shell, :mod:`sensoryforge.gui.app`."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.gui

from sensoryforge.config.schema import SensoryForgeConfig  # noqa: E402
from sensoryforge.gui.app import SensoryForgeApp, STAGE_ORDER  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.gui.settings import gui_settings  # noqa: E402


def _preset_config() -> SensoryForgeConfig:
    return SensoryForgeConfig.from_yaml_file("sensoryforge/presets/tactile_sa1_ra1.yml")


def _make_app(qtbot):
    session = Session(_preset_config())
    window = SensoryForgeApp(session)
    window.show_dialogs = False
    qtbot.addWidget(window)
    return window, session


def test_window_builds_with_preset(qtbot):
    window, session = _make_app(qtbot)
    assert window.pipeline_strip is not None
    assert set(window.pipeline_strip._pop_rows.keys()) == {0, 1}
    assert window.stage_list.count() == len(STAGE_ORDER)


def test_stage_nav_switches_stack(qtbot):
    window, session = _make_app(qtbot)
    window.stage_list.setCurrentRow(2)
    assert window.stack.currentIndex() == 2
    assert window.stack.currentWidget() is window._screens["populations"]


def test_open_preset_replaces_config_and_rebuilds_strip(qtbot, tmp_path):
    window, session = _make_app(qtbot)
    other = SensoryForgeConfig.from_yaml_file(
        "sensoryforge/presets/tactile_stochastic_control.yml"
    )
    path = tmp_path / "other.yml"
    path.write_text(other.to_yaml(), encoding="utf-8")

    window._load_config_file(str(path))

    assert session.config.populations[0].name == other.populations[0].name
    assert set(window.pipeline_strip._pop_rows.keys()) == set(
        range(len(other.populations))
    )


def test_save_config_via_new_project_handler(qtbot, tmp_path):
    window, session = _make_app(qtbot)
    project_root = tmp_path / "myproject"

    window._new_project(str(project_root))
    assert session.project is not None
    assert (project_root / "config.yml").is_file()

    session.set_by_path("populations.0.filter_method", "none")
    window._on_save_config()

    saved = SensoryForgeConfig.from_yaml_file(project_root / "config.yml")
    assert saved.populations[0].filter_method == "none"


def test_export_yaml_round_trips(qtbot, tmp_path):
    window, session = _make_app(qtbot)
    out_path = tmp_path / "exported.yml"

    window._export_yaml(str(out_path))

    reloaded = SensoryForgeConfig.from_yaml_file(out_path)
    assert reloaded == session.config


def test_advanced_toggle_persists_through_gui_settings(qtbot):
    window, session = _make_app(qtbot)
    assert window.advanced_check.isChecked() is False

    window.advanced_check.setChecked(True)

    assert gui_settings().value("gui/advanced", False, type=bool) is True


def test_pressing_run_executes_the_sessions_config(qtbot, tmp_path):
    """The window's Run button really runs the experiment, end to end.

    The shell owns one ``RunController`` and the run bar drives it; this is
    the wiring that makes the Run button do anything at all, so it is checked
    against the real engine rather than a stub.
    """
    window, session = _make_app(qtbot)
    window.run_bar.show_dialogs = False
    # The 80x80 preset takes seconds; 12x12 for 10 ms takes milliseconds.
    session.config.grids[0].rows = 12
    session.config.grids[0].cols = 12
    window._new_project(str(tmp_path / "project"))
    window.run_bar.duration_spin.setValue(10.0)

    with qtbot.waitSignal(window.run_controller.finished, timeout=60000) as blocker:
        window.run_bar.run_button.click()

    result = blocker.args[0]
    assert session.last_results is result
    assert set(result.results) == {p.name for p in session.config.populations}
    assert result.bundle_dir is not None
    assert result.bundle_dir.parent == session.project.runs_dir
    assert window.run_bar.bundle_label.text() == str(result.bundle_dir)
