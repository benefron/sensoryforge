"""Node positions persist beside a config, advisorily (F-065).

Wave O's specification promised that view state -- where the user dragged
each node -- is saved through ``Flowchart.saveState()`` into a sibling
``<config>.layout.json``, and that a missing or stale layout file never
prevents a config from loading. Wave R found by grepping that none of it
had been wired, rather than assuming the earlier report was complete.

Positions are deliberately not configuration: two people can arrange the
same experiment differently and it is still the same experiment, and the
config has to stay exactly what the command line would run. So they live
in their own file, and every failure to read that file is a no-op. Losing
your arrangement is an annoyance; failing to open your experiment is not.
"""

import json
import sys

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5 import QtWidgets  # noqa: E402
from pyqtgraph.flowchart import Flowchart  # noqa: E402

from sensoryforge.gui.circuit.nodes import build_node_library  # noqa: E402
from sensoryforge.gui.circuit.serialise import (  # noqa: E402
    apply_layout,
    layout_path_for,
    save_layout,
)

pytestmark = pytest.mark.gui


def _ensure_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    return app


@pytest.fixture
def qapp():
    return _ensure_app()


def _flowchart():
    flowchart = Flowchart(terminals={})
    flowchart.setLibrary(build_node_library())
    return flowchart


def _placed(flowchart, name, x, y, node_type="SensorArray"):
    node = flowchart.createNode(node_type, name=name)
    node.graphicsItem().setPos(x, y)
    return node


def _pos(node):
    item = node.graphicsItem().pos()
    return (item.x(), item.y())


class TestRoundTrip:
    def test_positions_survive_save_and_apply(self, qapp, tmp_path):
        config = tmp_path / "experiment.yml"
        source = _flowchart()
        _placed(source, "Skin", 12.5, -34.0)
        _placed(source, "Probe", -8.0, 61.25, node_type="Stimulus")
        save_layout(source, config)

        target = _flowchart()
        skin = _placed(target, "Skin", 0.0, 0.0)
        probe = _placed(target, "Probe", 0.0, 0.0, node_type="Stimulus")
        assert apply_layout(target, config) == 2
        assert _pos(skin) == (12.5, -34.0)
        assert _pos(probe) == (-8.0, 61.25)

    def test_the_layout_file_sits_beside_the_config(self, qapp, tmp_path):
        config = tmp_path / "experiment.yml"
        save_layout(_flowchart(), config)
        assert layout_path_for(config) == tmp_path / "experiment.yml.layout.json"
        assert layout_path_for(config).exists()

    def test_the_file_is_readable_json_not_an_opaque_blob(self, qapp, tmp_path):
        """A user should be able to look at it, and a diff should be legible."""
        config = tmp_path / "experiment.yml"
        source = _flowchart()
        _placed(source, "Skin", 1.0, 2.0)
        payload = json.loads(save_layout(source, config).read_text())
        assert payload["positions"]["Skin"] == [1.0, 2.0]


class TestAdvisory:
    """Every way of failing to read a layout is a no-op, never an error."""

    def test_a_missing_file_is_a_no_op(self, qapp, tmp_path):
        assert apply_layout(_flowchart(), tmp_path / "never_saved.yml") == 0

    def test_malformed_json_is_a_no_op(self, qapp, tmp_path):
        config = tmp_path / "experiment.yml"
        layout_path_for(config).write_text("{ not json")
        assert apply_layout(_flowchart(), config) == 0

    @pytest.mark.parametrize("payload", ["[]", '"text"', "{}", '{"positions": 5}'])
    def test_an_unexpected_shape_is_a_no_op(self, qapp, tmp_path, payload):
        config = tmp_path / "experiment.yml"
        layout_path_for(config).write_text(payload)
        assert apply_layout(_flowchart(), config) == 0

    def test_a_stale_node_name_is_skipped_and_the_rest_applied(self, qapp, tmp_path):
        """A layout saved before a node was renamed must still be useful."""
        config = tmp_path / "experiment.yml"
        layout_path_for(config).write_text(
            json.dumps(
                {"version": 1, "positions": {"Skin": [5.0, 6.0], "Gone": [1.0, 1.0]}}
            )
        )
        target = _flowchart()
        skin = _placed(target, "Skin", 0.0, 0.0)
        assert apply_layout(target, config) == 1
        assert _pos(skin) == (5.0, 6.0)

    def test_a_node_with_no_recorded_position_keeps_its_placement(self, qapp, tmp_path):
        config = tmp_path / "experiment.yml"
        layout_path_for(config).write_text(
            json.dumps({"version": 1, "positions": {"Skin": [5.0, 6.0]}})
        )
        target = _flowchart()
        _placed(target, "Skin", 0.0, 0.0)
        untouched = _placed(target, "Probe", 77.0, 88.0, node_type="Stimulus")
        apply_layout(target, config)
        assert _pos(untouched) == (77.0, 88.0)

    def test_a_bad_position_value_is_skipped_not_fatal(self, qapp, tmp_path):
        config = tmp_path / "experiment.yml"
        layout_path_for(config).write_text(
            json.dumps(
                {"version": 1, "positions": {"Skin": ["x", "y"], "Probe": [3.0, 4.0]}}
            )
        )
        target = _flowchart()
        _placed(target, "Skin", 0.0, 0.0)
        probe = _placed(target, "Probe", 0.0, 0.0, node_type="Stimulus")
        assert apply_layout(target, config) == 1
        assert _pos(probe) == (3.0, 4.0)
