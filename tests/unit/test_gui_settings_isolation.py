"""GUI preferences are hermetic under test (F-072).

The GUI tests used to read and write the real per-user Qt settings of whoever
ran them. One test passed on a machine where the "Population Settings" section
had once been saved expanded and failed on a fresh CI runner where it starts
collapsed, and a test run could overwrite a developer's saved GUI state.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")

from sensoryforge.gui import settings as gui_settings_module  # noqa: E402
from sensoryforge.gui.settings import SETTINGS_DIR_ENV, gui_settings  # noqa: E402


def test_the_test_session_redirects_settings_to_a_private_directory():
    """tests/conftest.py must set this before any GUI code runs."""
    directory = os.environ.get(SETTINGS_DIR_ENV)
    assert directory, f"{SETTINGS_DIR_ENV} is not set for the test session"
    assert Path(directory).is_dir()
    assert "sensoryforge-test-settings-" in Path(directory).name


def test_settings_are_an_ini_file_inside_that_directory():
    store = gui_settings()
    assert Path(store.fileName()).parent == Path(os.environ[SETTINGS_DIR_ENV])
    assert store.fileName().endswith(".ini")


def test_without_the_override_the_native_store_is_used(monkeypatch):
    monkeypatch.delenv(SETTINGS_DIR_ENV, raising=False)
    store = gui_settings()
    assert store.organizationName() == gui_settings_module.ORGANIZATION
    assert store.applicationName() == gui_settings_module.APPLICATION


def test_a_value_written_under_test_does_not_reach_the_native_store(monkeypatch):
    key = "gui/test_isolation/probe"
    gui_settings().setValue(key, "written-under-test")
    gui_settings().sync()
    monkeypatch.delenv(SETTINGS_DIR_ENV, raising=False)
    assert gui_settings().value(key) is None


def test_no_gui_module_constructs_qsettings_directly():
    """Everything must go through gui_settings(), or the redirect leaks."""
    root = Path(__file__).resolve().parents[2] / "sensoryforge" / "gui"
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.name != "settings.py" and "QSettings(" in path.read_text()
    ]
    assert not offenders, f"direct QSettings construction in: {offenders}"
