"""Tests for F-048: every YAML config loader honours ``plugins:`` identically.

Before this fix, only ``sensoryforge.cli.load_config_file`` imported a
config's ``plugins:`` list. ``BatchExecutor.from_yaml``,
``SensoryForgeConfig.from_yaml_file``, and the GUI's "Load YAML
Configuration" action all called ``yaml.safe_load``/``load_yaml`` directly
and silently dropped ``plugins:``. This module writes a throwaway plugin
module that registers a uniquely-named component (a trivial filter class)
into ``FILTER_REGISTRY``, then exercises all four loaders and checks the
component ends up registered every time.

Registration is idempotent for the same class under the same name (H1), so
re-loading the same plugin module across all four loaders in one test
session is safe -- there is no need to unregister between legs.
"""

import sys
import types

import pytest

from sensoryforge.registry import FILTER_REGISTRY

PLUGIN_MODULE_NAME = "_sf_test_h4_fake_plugin"
COMPONENT_NAME = "_sf_test_h4_component"


class _FakeFilter:
    """Trivial stand-in filter component used only to prove registration."""

    @classmethod
    def from_config(cls, config):
        return cls()

    def to_dict(self):
        return {}


@pytest.fixture
def fake_plugin_module():
    """A throwaway module with a `register()` side-effect function."""
    mod = types.ModuleType(PLUGIN_MODULE_NAME)

    def register():
        FILTER_REGISTRY.register(COMPONENT_NAME, _FakeFilter)

    mod.register = register
    sys.modules[PLUGIN_MODULE_NAME] = mod
    yield mod
    del sys.modules[PLUGIN_MODULE_NAME]
    FILTER_REGISTRY._registry.pop(COMPONENT_NAME.casefold(), None)


@pytest.fixture
def plugin_config_path(tmp_path, fake_plugin_module):
    """A canonical-format config file whose plugins: list registers the fake filter."""
    config_file = tmp_path / "plugin_config.yml"
    config_file.write_text(
        f"plugins:\n  - {PLUGIN_MODULE_NAME}:register\n" "grids: []\npopulations: []\n"
    )
    return config_file


@pytest.fixture
def batch_plugin_config_path(tmp_path, fake_plugin_module):
    """A legacy batch config file (base_config/batch) whose plugins: registers the fake filter."""
    config_file = tmp_path / "batch_plugin_config.yml"
    output_dir = tmp_path / "batch_output"
    config_file.write_text(f"""\
plugins:
  - {PLUGIN_MODULE_NAME}:register
metadata:
  batch_name: h4_test_batch
base_config:
  pipeline:
    device: cpu
    seed: 42
    grid_size: 5
  neurons:
    sa_neurons: 2
    ra_neurons: 2
    sa2_neurons: 2
    dt: 1.0
  temporal:
    t_pre: 5
    t_ramp: 2
    t_plateau: 10
    t_post: 5
    dt: 1.0
batch:
  output_dir: {output_dir.as_posix()!r}
  save_format: pytorch
  stimuli:
    - type: gaussian_sweep
      parameters:
        amplitude: [10.0]
        sigma: [0.5]
      repetitions: 1
""")
    return config_file


def test_cli_loader_registers_plugin_component(plugin_config_path):
    """sensoryforge.cli.load_config_file honours plugins: (already worked pre-fix)."""
    from sensoryforge.cli import load_config_file

    config = load_config_file(str(plugin_config_path))

    assert FILTER_REGISTRY.is_registered(COMPONENT_NAME)
    # plugins: key is preserved in the returned dict for downstream consumers.
    assert config["plugins"] == [f"{PLUGIN_MODULE_NAME}:register"]


def test_batch_executor_loader_registers_plugin_component(batch_plugin_config_path):
    """BatchExecutor.from_yaml must honour plugins: too (F-048)."""
    from sensoryforge.core.batch_executor import BatchExecutor

    BatchExecutor.from_yaml(str(batch_plugin_config_path))

    assert FILTER_REGISTRY.is_registered(COMPONENT_NAME)


def test_schema_from_yaml_file_registers_plugin_component(plugin_config_path):
    """SensoryForgeConfig.from_yaml_file must honour plugins: too (F-048)."""
    from sensoryforge.config.schema import SensoryForgeConfig

    SensoryForgeConfig.from_yaml_file(str(plugin_config_path))

    assert FILTER_REGISTRY.is_registered(COMPONENT_NAME)


def test_schema_from_yaml_path_branch_registers_plugin_component(plugin_config_path):
    """SensoryForgeConfig.from_yaml's path-detection branch also honours plugins:."""
    from sensoryforge.config.schema import SensoryForgeConfig

    SensoryForgeConfig.from_yaml(str(plugin_config_path))

    assert FILTER_REGISTRY.is_registered(COMPONENT_NAME)


# --- GUI leg -----------------------------------------------------------

_APP = None


def _ensure_app():
    global _APP
    try:
        from PyQt5 import QtWidgets

        _APP = QtWidgets.QApplication.instance()
        if _APP is None:
            _APP = QtWidgets.QApplication(sys.argv[:1])
    except ImportError:
        pytest.skip("PyQt5 not available")


@pytest.mark.gui
def test_gui_load_config_registers_plugin_component(plugin_config_path, qtbot):
    """Opening a config in the GUI must honour its plugins: list (F-048)."""
    from sensoryforge.config.schema import SensoryForgeConfig
    from sensoryforge.gui.app import SensoryForgeApp
    from sensoryforge.gui.session import Session

    window = SensoryForgeApp(Session(SensoryForgeConfig()))
    qtbot.addWidget(window)
    window._load_config_file(str(plugin_config_path))

    assert FILTER_REGISTRY.is_registered(COMPONENT_NAME)


@pytest.mark.gui
def test_gui_open_project_registers_plugin_component(plugin_config_path, tmp_path):
    """A project whose config.yml lists plugins: registers them when opened."""
    import shutil

    from sensoryforge.gui.project import ProjectHandle

    root = tmp_path / "project"
    root.mkdir()
    shutil.copy(plugin_config_path, root / "config.yml")
    ProjectHandle.open(root).load_config()

    assert FILTER_REGISTRY.is_registered(COMPONENT_NAME)
