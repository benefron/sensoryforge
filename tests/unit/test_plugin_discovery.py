"""Tests for G2: third-party plugin discovery.

Covers:
- ``load_plugin_import_paths`` imports a ``module:attr`` path and calls the
  callable attribute.
- A broken import path produces a ``UserWarning`` instead of raising.
- ``discover_entry_point_plugins`` loads and calls entry points registered
  under the "sensoryforge.components" group, warning (not raising) on a
  broken one.
- ``cli.load_config_file`` triggers plugin loading for a config with a
  ``plugins:`` list.
"""

import sys
import types
import warnings
from importlib.metadata import EntryPoint

import pytest

from sensoryforge.plugins import (
    discover_entry_point_plugins,
    load_plugin_import_paths,
)


@pytest.fixture
def fake_plugin_module():
    """A throwaway module with a `register()` side-effect function."""
    mod = types.ModuleType("_sf_test_fake_plugin")
    mod.calls = []

    def register():
        mod.calls.append("registered")

    mod.register = register
    sys.modules["_sf_test_fake_plugin"] = mod
    yield mod
    del sys.modules["_sf_test_fake_plugin"]


def test_load_plugin_import_paths_calls_named_attr(fake_plugin_module):
    loaded = load_plugin_import_paths(["_sf_test_fake_plugin:register"])
    assert loaded == ["_sf_test_fake_plugin:register"]
    assert fake_plugin_module.calls == ["registered"]


def test_load_plugin_import_paths_warns_on_missing_module():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_plugin_import_paths(["_sf_test_nonexistent_module_xyz"])
    assert loaded == []
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_discover_entry_point_plugins_calls_registered_callable(
    monkeypatch, fake_plugin_module
):
    ep = EntryPoint(
        name="fake",
        value="_sf_test_fake_plugin:register",
        group="sensoryforge.components",
    )

    def fake_entry_points(*, group):
        assert group == "sensoryforge.components"
        return [ep]

    monkeypatch.setattr(
        "importlib.metadata.entry_points", fake_entry_points, raising=False
    )
    loaded = discover_entry_point_plugins()
    assert loaded == ["fake"]
    assert fake_plugin_module.calls == ["registered"]


def test_discover_entry_point_plugins_warns_on_broken_entry_point(monkeypatch):
    ep = EntryPoint(
        name="broken",
        value="_sf_test_nonexistent_module_xyz:register",
        group="sensoryforge.components",
    )

    def fake_entry_points(*, group):
        return [ep]

    monkeypatch.setattr(
        "importlib.metadata.entry_points", fake_entry_points, raising=False
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = discover_entry_point_plugins()
    assert loaded == []
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_cli_load_config_file_loads_plugins(tmp_path, fake_plugin_module):
    from sensoryforge.cli import load_config_file

    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "plugins:\n  - _sf_test_fake_plugin:register\n" "grids: []\npopulations: []\n"
    )
    config = load_config_file(str(config_file))
    assert config["plugins"] == ["_sf_test_fake_plugin:register"]
    assert fake_plugin_module.calls == ["registered"]
