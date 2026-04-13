"""Tests for ExperimentManager — project directory lifecycle."""

import json
import pytest
from pathlib import Path

from sensoryforge.core.experiment_manager import (
    ExperimentManager,
    STIMULI_SUBDIR,
    RESULTS_SUBDIR,
    FIGURES_SUBDIR,
    _MANIFEST_FILENAME,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def em():
    return ExperimentManager()


@pytest.fixture
def project_path(tmp_path):
    return tmp_path / "my_experiment"


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initially_closed(em):
    assert em.is_open is False
    assert em.project_dir is None
    assert em.config_path is None
    assert em.stimuli_dir is None
    assert em.results_dir is None
    assert em.figures_dir is None


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------

def test_create_makes_directory(em, project_path):
    em.create(project_path)
    assert project_path.is_dir()
    assert em.is_open
    assert em.project_dir == project_path


def test_create_makes_subdirs(em, project_path):
    em.create(project_path)
    assert (project_path / STIMULI_SUBDIR).is_dir()
    assert (project_path / RESULTS_SUBDIR).is_dir()
    assert (project_path / FIGURES_SUBDIR).is_dir()


def test_create_writes_manifest(em, project_path):
    em.create(project_path, name="MyExp")
    manifest = json.loads((project_path / _MANIFEST_FILENAME).read_text())
    assert manifest["kind"] == "sensoryforge_experiment"
    assert manifest["name"] == "MyExp"


def test_create_raises_if_non_empty(em, project_path):
    project_path.mkdir()
    (project_path / "existing_file.txt").write_text("data")
    with pytest.raises(FileExistsError):
        em.create(project_path)


# ---------------------------------------------------------------------------
# open()
# ---------------------------------------------------------------------------

def test_open_existing_project(em, project_path):
    em.create(project_path)
    em.close()
    em.open(project_path)
    assert em.is_open
    assert em.project_dir == project_path


def test_open_creates_missing_subdirs(em, project_path):
    """Tolerant open: adopts any directory and creates missing sub-dirs."""
    project_path.mkdir()
    em.open(project_path)
    assert (project_path / STIMULI_SUBDIR).is_dir()
    assert (project_path / RESULTS_SUBDIR).is_dir()


def test_open_raises_for_file(em, tmp_path):
    f = tmp_path / "not_a_dir.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        em.open(f)


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

def test_close_clears_state(em, project_path):
    em.create(project_path)
    em.close()
    assert em.is_open is False
    assert em.project_dir is None


# ---------------------------------------------------------------------------
# Property paths
# ---------------------------------------------------------------------------

def test_property_paths(em, project_path):
    em.create(project_path)
    assert em.config_path == project_path / "config.yaml"
    assert em.stimuli_dir == project_path / STIMULI_SUBDIR
    assert em.results_dir == project_path / RESULTS_SUBDIR
    assert em.figures_dir == project_path / FIGURES_SUBDIR


# ---------------------------------------------------------------------------
# next_results_path / next_stimulus_path
# ---------------------------------------------------------------------------

def test_next_results_path_numbered(em, project_path):
    em.create(project_path)
    p1 = em.next_results_path("run", ".h5")
    assert p1.name == "run_001.h5"
    p1.touch()
    p2 = em.next_results_path("run", ".h5")
    assert p2.name == "run_002.h5"


def test_next_stimulus_path_numbered(em, project_path):
    em.create(project_path)
    p1 = em.next_stimulus_path("stim", ".pt")
    assert p1.name == "stim_001.pt"
    p1.touch()
    p2 = em.next_stimulus_path("stim", ".pt")
    assert p2.name == "stim_002.pt"


def test_next_results_path_raises_when_closed(em):
    with pytest.raises(RuntimeError, match="No project open"):
        em.next_results_path()


# ---------------------------------------------------------------------------
# list_stimuli / list_results
# ---------------------------------------------------------------------------

def test_list_stimuli_empty_when_closed(em):
    assert em.list_stimuli() == []


def test_list_stimuli_returns_files(em, project_path):
    em.create(project_path)
    (project_path / STIMULI_SUBDIR / "a.pt").touch()
    (project_path / STIMULI_SUBDIR / "b.pt").touch()
    files = em.list_stimuli()
    assert len(files) == 2
    assert all(f.suffix == ".pt" for f in files)


def test_list_results_returns_files(em, project_path):
    em.create(project_path)
    (project_path / RESULTS_SUBDIR / "run_001.h5").touch()
    files = em.list_results()
    assert len(files) == 1
