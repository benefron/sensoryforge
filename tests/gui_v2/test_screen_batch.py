"""Tests for :mod:`sensoryforge.gui.screens.batch` (the Batch screen).

Drives the real widgets (table rows, combo boxes, spin boxes, buttons) and
asserts on written files, the ``SweepController`` it wires up, and
``session.config`` -- never merely that a widget was constructed.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gui

from sensoryforge.config.schema import SensoryForgeConfig  # noqa: E402
from sensoryforge.gui.project import ProjectHandle  # noqa: E402
from sensoryforge.gui.screens.batch import BatchScreen  # noqa: E402
from sensoryforge.gui.screens.batch_combos import (  # noqa: E402
    build_combinations,
    expand_repetitions,
)
from sensoryforge.gui.screens.batch_fields import parse_field_values  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.io.bundle import load_bundle  # noqa: E402

PRESET = "sensoryforge/presets/tactile_sa1_ra1.yml"

RUN_TIMEOUT_MS = 180000


def _config() -> SensoryForgeConfig:
    config = SensoryForgeConfig.from_yaml_file(PRESET)
    config.grids[0].rows = 6
    config.grids[0].cols = 6
    return config


def _screen(qtbot, config=None) -> BatchScreen:
    session = Session(config if config is not None else _config())
    screen = BatchScreen(session)
    screen.show_dialogs = False
    qtbot.addWidget(screen)
    return screen


def _can_run_subprocess() -> bool:
    """Whether a child interpreter can import sensoryforge from this worktree."""
    import os

    import sensoryforge

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(sensoryforge.__file__).resolve().parent.parent)
    completed = subprocess.run(
        [sys.executable, "-c", "import sensoryforge"], env=env, capture_output=True
    )
    return completed.returncode == 0


# ------------------------------------------------------------- value parsing


def test_parse_field_values_list_mode():
    values, error = parse_field_values("10, 20, 40", "list", is_int=True)
    assert error is None
    assert values == [10, 20, 40]


def test_parse_field_values_linear_range():
    values, error = parse_field_values("0.0, 1.0, 5", "linear", is_int=False)
    assert error is None
    assert values == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_parse_field_values_log_range():
    values, error = parse_field_values("1, 100, 3", "log", is_int=False)
    assert error is None
    assert values == pytest.approx([1.0, 10.0, 100.0])


def test_parse_field_values_bad_text_reports_error_not_raise():
    values, error = parse_field_values("ten, twenty", "list", is_int=False)
    assert values is None
    assert "ten" in error

    values, error = parse_field_values("1, 2", "linear", is_int=False)
    assert values is None
    assert "start, stop, count" in error

    values, error = parse_field_values("", "list", is_int=False)
    assert values is None
    assert error is not None


# ------------------------------------------------------------------- combos


def test_full_grid_combinations_are_a_cartesian_product():
    combos = build_combinations([("a", [1, 2]), ("b", [10, 20, 30])], mode="full_grid")
    assert len(combos) == 6
    assert combos[0] == {"a": 1, "b": 10}
    assert combos[-1] == {"a": 2, "b": 30}


def test_zipped_combinations_require_equal_lengths():
    with pytest.raises(ValueError, match="zipped"):
        build_combinations([("a", [1, 2]), ("b", [10, 20, 30])], mode="zipped")

    combos = build_combinations([("a", [1, 2]), ("b", [10, 20])], mode="zipped")
    assert combos == [{"a": 1, "b": 10}, {"a": 2, "b": 20}]


def test_repetitions_give_each_repeat_a_distinct_seed():
    combos = [{"a": 1}, {"a": 2}]
    expanded = expand_repetitions(combos, repetitions=3, base_seed=5)
    assert len(expanded) == 6
    seeds = [c["simulation.seed"] for c in expanded]
    assert seeds == [5, 6, 7, 5, 6, 7]


# --------------------------------------------------------------------- table


def test_add_field_and_full_grid_job_count_reports_correctly(qtbot):
    screen = _screen(qtbot)

    row0 = screen.add_field("populations.0.input_gain")
    screen._rows[row0]["values_edit"].setText("10, 20, 40")
    row1 = screen.add_field("simulation.dt_ms")
    screen._rows[row1]["values_edit"].setText("0.5, 1.0")

    combos, error = screen.base_combinations()
    assert error is None
    assert len(combos) == 6
    assert screen.job_count() == 6
    assert "6 combination(s)" in screen.summary_label.text()
    assert "1 repetition(s)" in screen.summary_label.text()

    screen.reps_spin.setValue(2)
    assert screen.job_count() == 12
    assert "2 repetition(s)" in screen.summary_label.text()


def test_zipped_mode_with_unequal_lengths_is_refused_inline(qtbot):
    screen = _screen(qtbot)
    row0 = screen.add_field("populations.0.input_gain")
    screen._rows[row0]["values_edit"].setText("10, 20, 40")
    row1 = screen.add_field("simulation.dt_ms")
    screen._rows[row1]["values_edit"].setText("0.5, 1.0")

    index = screen.mode_combo.findData("zipped")
    screen.mode_combo.setCurrentIndex(index)

    combos, error = screen.base_combinations()
    assert combos == []
    assert error is not None and "zipped" in error
    assert not screen.write_button.isEnabled()
    assert not screen.run_button.isEnabled()

    # Never raises -- the label just shows the error.
    assert "zipped" in screen.summary_label.text()


def test_field_row_shows_inline_parse_error_not_a_raise(qtbot):
    screen = _screen(qtbot)
    row0 = screen.add_field("populations.0.input_gain")
    screen._rows[row0]["values_edit"].setText("ten, twenty")

    combos, error = screen.base_combinations()
    assert combos == []
    assert "not a number" in error
    assert "not a number" in screen._rows[row0]["status_label"].text()


# ---------------------------------------------------------------- write_sweep


def test_write_sweep_writes_expected_configs_and_leaves_others_unchanged(
    qtbot, tmp_path
):
    config = _config()
    screen = _screen(qtbot, config)
    row0 = screen.add_field("populations.0.input_gain")
    screen._rows[row0]["values_edit"].setText("10, 20, 40")
    row1 = screen.add_field("simulation.dt_ms")
    screen._rows[row1]["values_edit"].setText("0.5, 1.0")
    screen.duration_spin.setValue(25.0)

    target_dir = tmp_path / "chosen"
    screen.choose_directory = lambda parent, caption: str(target_dir)

    before = copy.deepcopy(screen._session.config)
    manifest = screen.write_sweep()
    after = screen._session.config

    assert manifest is not None
    assert len(manifest.combos) == 6
    assert (
        after.to_yaml() == before.to_yaml()
    ), "write_sweep must not touch session.config"

    seen = set()
    for index, combo_entry in enumerate(manifest.combos):
        written = SensoryForgeConfig.from_yaml_file(manifest.config_path(index))
        expected_gain = combo_entry["values"]["populations.0.input_gain"]
        expected_dt = combo_entry["values"]["simulation.dt_ms"]
        assert written.populations[0].input_gain == expected_gain
        assert written.simulation.dt_ms == expected_dt
        assert written.simulation.duration_ms == 25.0
        seen.add((expected_gain, expected_dt))

        # Every other field equals the session config: rebuild the expected
        # config the same way write_combo_sweep does (duration + the two
        # swept paths on top of a copy of the original) and compare whole.
        expected = copy.deepcopy(before)
        expected.simulation.duration_ms = 25.0
        expected.populations[0].input_gain = expected_gain
        expected.simulation.dt_ms = expected_dt
        assert written.to_yaml() == expected.to_yaml()

    assert seen == {(gain, dt) for gain in (10.0, 20.0, 40.0) for dt in (0.5, 1.0)}
    assert (manifest.root / "manifest.json").is_file()


def test_write_sweep_uses_project_sweeps_directory_when_project_open(qtbot, tmp_path):
    config = _config()
    screen = _screen(qtbot, config)
    project = ProjectHandle.create(tmp_path / "proj", config)
    screen._session.set_project(project)

    row0 = screen.add_field("populations.0.input_gain")
    screen._rows[row0]["values_edit"].setText("10, 20")

    manifest = screen.write_sweep()
    assert manifest is not None
    assert manifest.root.parent == project.root / "sweeps"


# -------------------------------------------------------------------- preview


def test_preview_shows_first_combination_yaml_and_command(qtbot):
    screen = _screen(qtbot)
    row0 = screen.add_field("populations.0.input_gain")
    screen._rows[row0]["values_edit"].setText("10, 20")

    text = screen.preview.sweep_yaml.toPlainText()
    assert "input_gain: 10" in text
    command = screen.preview.sweep_command.text()
    assert command.endswith("config.yml") or "combo_000" in command
    assert "sensoryforge.cli" in command
    assert "run" in command


def test_export_pane_shows_session_config_yaml_and_single_run_command(qtbot):
    screen = _screen(qtbot)
    yaml_text = screen.preview.export_yaml.toPlainText()
    assert yaml_text == screen._session.config.to_yaml()
    command = screen.preview.export_command.text()
    assert "sensoryforge.cli run" in command
    assert "--duration" in command
    assert "--bundle" in command


# ---------------------------------------------------------------- local runs


@pytest.mark.slow
def test_two_job_sweep_runs_locally_with_differing_results(qtbot, tmp_path):
    if not _can_run_subprocess():
        pytest.skip("a child interpreter cannot import sensoryforge here")

    config = _config()
    config.populations = config.populations[:1]  # one population: faster
    screen = _screen(qtbot, config)
    project = ProjectHandle.create(tmp_path / "proj", config)
    screen._session.set_project(project)

    row0 = screen.add_field("populations.0.input_gain")
    screen._rows[row0]["values_edit"].setText("10, 80")
    screen.duration_spin.setValue(10.0)

    before = screen._session.config.to_yaml()

    with qtbot.waitSignal(
        screen._controller.finished, timeout=RUN_TIMEOUT_MS
    ) as blocker:
        screen.run_button.click()

    log_text = screen.run_panel.log_view.toPlainText()
    assert blocker.args[0] == 0, log_text
    assert screen._session.config.to_yaml() == before

    manifest = screen._last_manifest
    assert manifest is not None and len(manifest.combos) == 2

    bundles = [load_bundle(manifest.bundle_path(i)) for i in range(2)]
    gains = [b.config.populations[0].input_gain for b in bundles]
    assert sorted(gains) == [10.0, 80.0]

    pop_name = bundles[0].config.populations[0].name
    filtered0 = bundles[0].populations[pop_name]["filtered"]
    filtered1 = bundles[1].populations[pop_name]["filtered"]
    assert not (filtered0.shape == filtered1.shape and (filtered0 == filtered1).all())

    for combo_dir in ("combo_000", "combo_001"):
        assert screen.run_panel.status_of(combo_dir).endswith(": done")


@pytest.mark.slow
def test_a_job_with_an_invalid_value_is_shown_failed_while_the_other_completes(
    qtbot, tmp_path
):
    if not _can_run_subprocess():
        pytest.skip("a child interpreter cannot import sensoryforge here")

    config = _config()
    config.populations = config.populations[:1]
    screen = _screen(qtbot, config)
    project = ProjectHandle.create(tmp_path / "proj", config)
    screen._session.set_project(project)

    row0 = screen.add_field("simulation.dt_ms")
    # 1.0 is a whole multiple of the default integrate_dt_ms (0.05) and
    # runs; 0.33 is not, so SensoryForgeConfig.from_yaml_file raises inside
    # the subprocess (F-042) and the CLI exits non-zero for that job only.
    screen._rows[row0]["values_edit"].setText("1.0, 0.33")
    screen.duration_spin.setValue(10.0)

    with qtbot.waitSignal(
        screen._controller.finished, timeout=RUN_TIMEOUT_MS
    ) as blocker:
        screen.run_button.click()

    assert blocker.args[0] == 1, screen.run_panel.log_view.toPlainText()

    manifest = screen._last_manifest
    good_index = [
        i
        for i, c in enumerate(manifest.combos)
        if c["values"]["simulation.dt_ms"] == 1.0
    ][0]
    bad_index = 1 - good_index
    good_dir = manifest.combos[good_index]["dir"]
    bad_dir = manifest.combos[bad_index]["dir"]

    assert screen.run_panel.status_of(good_dir).endswith(": done")
    assert "failed" in screen.run_panel.status_of(bad_dir)
    assert (manifest.bundle_path(good_index) / "config.json").is_file()
    assert not (manifest.bundle_path(bad_index) / "config.json").is_file()


# --------------------------------------------------------------- config unity


def test_session_config_equal_before_and_after_the_whole_flow(qtbot, tmp_path):
    config = _config()
    screen = _screen(qtbot, config)
    before = screen._session.config.to_yaml()

    row0 = screen.add_field("populations.0.input_gain")
    screen._rows[row0]["values_edit"].setText("10, 20")
    screen.reps_spin.setValue(2)
    screen.base_seed_spin.setValue(7)

    target_dir = tmp_path / "sweep_out"
    screen.choose_directory = lambda parent, caption: str(target_dir)
    manifest = screen.write_sweep()
    assert manifest is not None

    after = screen._session.config.to_yaml()
    assert after == before


def test_slurm_export_covers_the_sweep_shown_not_an_earlier_one(qtbot, tmp_path):
    """Editing the table after a write must not leave the script on the old sweep."""
    screen = _screen(qtbot)
    screen.choose_directory = lambda *a: str(tmp_path)
    row = screen.add_field("populations.0.input_gain")
    screen._rows[row]["values_edit"].setText("10, 50")
    assert screen.write_sweep() is not None

    screen._rows[row]["values_edit"].setText("10, 50, 90")
    assert screen.job_count() == 3
    script = screen.export_slurm(
        {
            "job_name": "x",
            "partition": "p",
            "time": "01:00:00",
            "mem_gb": 4,
            "cpus_per_task": 1,
            "gpus": 0,
            "conda_env": "sensoryforge",
        }
    )
    array_lines = [ln for ln in script.read_text().splitlines() if "--array" in ln]
    assert array_lines == ["#SBATCH --array=0-2"]
