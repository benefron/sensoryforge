"""A GUI v2 project is a directory: ``config.yml`` plus ``runs/<bundle dirs>``.

Covers :class:`sensoryforge.gui.project.ProjectHandle` -- creation refusing to
overwrite, opening refusing a root without a config, the YAML round trip, run
directory naming, and which run directories ``list_runs`` reports.
"""

import json
import os
import re
import time

import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
)
from sensoryforge.gui.project import ProjectHandle  # noqa: E402


def _config() -> SensoryForgeConfig:
    """A small but non-default config, so a round trip can actually fail."""
    return SensoryForgeConfig(
        grids=[GridConfig(name="skin", rows=8, cols=8, spacing=0.4)],
        populations=[PopulationConfig(name="SA", target_grid="skin")],
        metadata={"name": "demo project"},
    )


def _write_bundle(directory) -> None:
    """Make ``directory`` look like a bundle: a directory with config.json."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"schema_version": "2.0.0"}))


# ---------------------------------------------------------------- create/open


def test_create_writes_config_yml_and_a_runs_directory(tmp_path):
    project = ProjectHandle.create(tmp_path / "proj", _config())

    assert project.root == tmp_path / "proj"
    assert project.config_path == tmp_path / "proj" / "config.yml"
    assert project.runs_dir == tmp_path / "proj" / "runs"
    assert project.layout_path == tmp_path / "proj" / "layout.json"
    assert project.config_path.is_file()
    assert project.runs_dir.is_dir()


def test_create_refuses_a_root_that_already_holds_a_project(tmp_path):
    ProjectHandle.create(tmp_path / "proj", _config())

    with pytest.raises(ValueError, match="config.yml"):
        ProjectHandle.create(tmp_path / "proj", _config())


def test_open_requires_a_config_yml(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match="config.yml"):
        ProjectHandle.open(tmp_path / "empty")


def test_open_returns_a_handle_on_an_existing_project(tmp_path):
    ProjectHandle.create(tmp_path / "proj", _config())

    project = ProjectHandle.open(tmp_path / "proj")

    assert project.root == tmp_path / "proj"


def test_a_str_root_becomes_a_path(tmp_path):
    project = ProjectHandle(str(tmp_path / "proj"))

    assert project.config_path == tmp_path / "proj" / "config.yml"


# ------------------------------------------------------------------ save/load


def test_config_survives_a_save_load_round_trip(tmp_path):
    project = ProjectHandle.create(tmp_path / "proj", _config())

    loaded = project.load_config()

    assert loaded == _config()


def test_save_config_overwrites_the_previous_one(tmp_path):
    project = ProjectHandle.create(tmp_path / "proj", _config())
    edited = _config()
    edited.grids[0].spacing = 0.9

    project.save_config(edited)

    assert project.load_config().grids[0].spacing == pytest.approx(0.9)


def test_load_config_on_a_root_without_one_raises(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match="config.yml"):
        ProjectHandle(tmp_path / "empty").load_config()


# --------------------------------------------------------------- run handling


def test_new_run_dir_is_a_timestamped_child_of_runs_and_is_not_created(tmp_path):
    project = ProjectHandle.create(tmp_path / "proj", _config())

    run_dir = project.new_run_dir("SA #6 sweep")

    assert run_dir.parent == project.runs_dir
    assert re.fullmatch(r"\d{8}-\d{6}_SA_6_sweep", run_dir.name), run_dir.name
    assert not run_dir.exists()


def test_new_run_dir_falls_back_to_a_placeholder_for_an_unusable_name(tmp_path):
    project = ProjectHandle.create(tmp_path / "proj", _config())

    run_dir = project.new_run_dir("///")

    assert re.fullmatch(r"\d{8}-\d{6}_run", run_dir.name), run_dir.name


def test_new_run_dir_does_not_collide_with_an_existing_directory(tmp_path):
    project = ProjectHandle.create(tmp_path / "proj", _config())
    first = project.new_run_dir("run")
    first.mkdir(parents=True)

    second = project.new_run_dir("run")

    assert second != first
    assert not second.exists()


def test_list_runs_reports_only_directories_holding_a_config_json(tmp_path):
    project = ProjectHandle.create(tmp_path / "proj", _config())
    _write_bundle(project.runs_dir / "20260101-000000_a")
    (project.runs_dir / "20260101-000001_not_a_bundle").mkdir()
    (project.runs_dir / "stray.txt").write_text("not a run")

    runs = project.list_runs()

    assert [p.name for p in runs] == ["20260101-000000_a"]


def test_list_runs_is_newest_first(tmp_path):
    project = ProjectHandle.create(tmp_path / "proj", _config())
    for name in ("20260101-000000_old", "20260101-000001_mid", "20260101-000002_new"):
        _write_bundle(project.runs_dir / name)
        time.sleep(0.01)
    # Pin the mtimes so the assertion cannot ride on filesystem granularity.
    for offset, name in enumerate(
        ("20260101-000000_old", "20260101-000001_mid", "20260101-000002_new")
    ):
        stamp = 1_700_000_000 + offset * 60
        os.utime(project.runs_dir / name, (stamp, stamp))

    runs = project.list_runs()

    assert [p.name for p in runs] == [
        "20260101-000002_new",
        "20260101-000001_mid",
        "20260101-000000_old",
    ]


def test_list_runs_is_empty_when_there_is_no_runs_directory(tmp_path):
    project = ProjectHandle(tmp_path / "never-created")

    assert project.list_runs() == []
