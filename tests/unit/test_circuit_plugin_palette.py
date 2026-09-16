"""Wave P, P3: a plugin package this checkout has never imported is
discovered through the real entry-point mechanism, appears in the Circuit
tab's registry-driven palette, has its parameters rendered by the
inspector from ``get_param_spec()`` alone, and takes part in a
``SimulationEngine`` run.

This deliberately does NOT call ``FILTER_REGISTRY.register()`` in-process
(the Wave P spec calls that "proves almost nothing" -- it bypasses entry-
point discovery, packaging and the inspector). Instead it builds a real
``<dist>-<version>.dist-info/entry_points.txt`` next to a real importable
package on disk, in ``tmp_path``, puts that directory on ``sys.path``, and
lets the *real* ``importlib.metadata.entry_points(group=...)`` -- the same
call ``sensoryforge.plugins.discover_entry_point_plugins`` makes -- find
it. This is genuine entry-point discovery without a ``pip install`` into
the shared conda environment (forbidden, F-053): ``importlib.metadata``
finds distributions by their ``*.dist-info`` metadata directory being on
``sys.path``, which is exactly what an editable/normal ``pip install``
also arranges, just without the installer.

Marked gui (imports PyQt5 for the Circuit tab/inspector); run alone
(F-016).
"""

import importlib
import importlib.metadata
import sys

import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

_APP = None
_PLUGIN_PACKAGE = "sensoryforge_wave_p_pytest_plugin"
_FILTER_NAME = "wave_p_pytest_demo_filter"

_COMPONENT_PY = '''
"""Real, on-disk plugin module -- not registered in-process."""
import torch

from sensoryforge.filters.base import BaseFilter
from sensoryforge.stimuli.base import ParamSpec


class WavePPytestDemoFilterTorch(BaseFilter):
    def __init__(self, *, gain=1.0, rectify=False, mode="linear", dt=1.0):
        super().__init__(dt=dt)
        self.gain = gain
        self.rectify = rectify
        self.mode = mode

    def forward(self, x, dt=None):
        y = x * self.gain
        if self.rectify:
            y = torch.relu(y)
        return y

    def reset_state(self):
        pass

    def to_dict(self):
        return {"gain": self.gain, "rectify": self.rectify, "mode": self.mode, "dt": self.dt}

    @classmethod
    def from_config(cls, config):
        return cls(**{k: v for k, v in config.items() if k in ("gain", "rectify", "mode", "dt")})

    @classmethod
    def get_param_spec(cls):
        return [
            ParamSpec("gain", dtype="float", default=1.0, min_val=0.0, max_val=10.0, group="Filter"),
            ParamSpec("rectify", dtype="bool", default=False, group="Filter"),
            ParamSpec("mode", dtype="str", default="linear",
                       choices=["linear", "log", "exp"], group="Filter", advanced=True),
        ]


def register():
    from sensoryforge.registry import FILTER_REGISTRY
    FILTER_REGISTRY.register(%(filter_name)r, WavePPytestDemoFilterTorch)
''' % {"filter_name": _FILTER_NAME}


def _ensure_app():
    global _APP
    from PyQt5 import QtWidgets

    _APP = QtWidgets.QApplication.instance()
    if _APP is None:
        _APP = QtWidgets.QApplication(sys.argv[:1])


@pytest.fixture
def real_entry_point_plugin(tmp_path, monkeypatch):
    """Build a real, importable plugin package + dist-info on disk and put
    it on sys.path, so importlib.metadata's real entry-point scan finds it."""
    pkg_dir = tmp_path / _PLUGIN_PACKAGE
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "component.py").write_text(_COMPONENT_PY)

    dist_info = tmp_path / f"{_PLUGIN_PACKAGE.replace('_', '-')}-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        f"Name: {_PLUGIN_PACKAGE.replace('_', '-')}\n"
        "Version: 0.1.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        "[sensoryforge.components]\n"
        f"{_FILTER_NAME} = {_PLUGIN_PACKAGE}.component:register\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield
    sys.modules.pop(_PLUGIN_PACKAGE, None)
    sys.modules.pop(f"{_PLUGIN_PACKAGE}.component", None)


def test_real_entry_points_scan_sees_the_plugin(real_entry_point_plugin):
    """Step 1 (discovered): the real importlib.metadata call, unmodified,
    finds the plugin's entry point -- not a monkeypatched fake."""
    eps = importlib.metadata.entry_points(group="sensoryforge.components")
    names = [ep.name for ep in eps]
    assert _FILTER_NAME in names


def test_discover_entry_point_plugins_registers_it_for_real(real_entry_point_plugin):
    from sensoryforge.registry import FILTER_REGISTRY
    from sensoryforge.register_components import register_all

    register_all()  # runs discover_entry_point_plugins() internally
    assert FILTER_REGISTRY.is_registered(_FILTER_NAME)


def test_plugin_appears_in_the_component_palette(real_entry_point_plugin):
    """Step 2 (appears in the palette)."""
    _ensure_app()
    from sensoryforge.register_components import register_all
    from sensoryforge.gui.tabs.circuit_tab import CircuitTab

    register_all()
    tab = CircuitTab()
    filter_category = None
    for i in range(tab.component_palette.topLevelItemCount()):
        item = tab.component_palette.topLevelItem(i)
        if item.text(0) == "Filter":
            filter_category = item
            break
    assert filter_category is not None
    leaf_names = {
        filter_category.child(j).text(0) for j in range(filter_category.childCount())
    }
    assert _FILTER_NAME in leaf_names


def test_plugin_is_configurable_through_the_inspector(real_entry_point_plugin):
    """Step 3 (configurable through the inspector, from get_param_spec()
    alone -- no GUI code written for this component)."""
    _ensure_app()
    from PyQt5 import QtWidgets

    from sensoryforge.register_components import register_all
    from sensoryforge.gui.tabs.circuit_tab import CircuitTab

    register_all()
    tab = CircuitTab()
    node = tab.add_node("Filter")
    tab._apply_component_selection(node, "Filter", _FILTER_NAME)
    tab.select_node(node)

    gain = tab.inspector_panel.findChild(QtWidgets.QWidget, "param_gain")
    rectify = tab.inspector_panel.findChild(QtWidgets.QWidget, "param_rectify")
    mode = tab.inspector_panel.findChild(QtWidgets.QWidget, "param_mode")
    assert isinstance(gain, QtWidgets.QDoubleSpinBox)
    assert isinstance(rectify, QtWidgets.QCheckBox)
    assert isinstance(mode, QtWidgets.QComboBox)

    gain.setValue(2.5)
    rectify.setChecked(True)
    assert node.filter_params["gain"] == 2.5
    assert node.filter_params["rectify"] is True


def test_plugin_runs_through_simulation_engine(real_entry_point_plugin):
    """Step 4 (runs): a full population, using the plugin as its
    filter_method, through SimulationEngine."""
    import torch

    from sensoryforge.config.schema import (
        GridConfig,
        PopulationConfig,
        SensoryForgeConfig,
        SimulationConfig,
        StimulusConfig,
    )
    from sensoryforge.core.simulation_engine import SimulationEngine
    from sensoryforge.register_components import register_all

    register_all()

    config = SensoryForgeConfig(
        grids=[GridConfig(name="g", arrangement="grid", rows=4, cols=4, spacing=0.5)],
        populations=[
            PopulationConfig(
                name="pop",
                neuron_type="SA",
                neuron_model="izhikevich",
                filter_method=_FILTER_NAME,
                filter_params={"gain": 3.0, "rectify": True},
                innervation_method="gaussian",
                neurons_per_row=2,
                target_grid="g",
            )
        ],
        stimulus=StimulusConfig(type="gaussian", amplitude=30.0, sigma=0.5),
        simulation=SimulationConfig(device="cpu", dt=1.0),
    )
    engine = SimulationEngine(config)
    stim = torch.rand(1, 5, 4, 4)
    results = engine.run(stim, return_intermediates=True)
    assert "pop" in results
    assert results["pop"]["filtered"].shape[0] == 1
