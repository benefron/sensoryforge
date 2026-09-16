"""Qt test: the Circuit tab canvas -- add, connect, move, delete nodes (Wave O, O2).

Marked gui; run alone (this repo's Qt test suite is order-dependent, F-016 --
see the appendix commands in docs/development/handover/phase1_tasks.md).
"""

import sys

import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

_APP = None


def _ensure_app():
    global _APP
    from PyQt5 import QtWidgets

    _APP = QtWidgets.QApplication.instance()
    if _APP is None:
        _APP = QtWidgets.QApplication(sys.argv[:1])


@pytest.fixture
def circuit_tab():
    _ensure_app()
    from sensoryforge.gui.tabs.circuit_tab import CircuitTab

    return CircuitTab()


def test_circuit_tab_instantiates_with_palette_canvas_and_inspector(circuit_tab):
    from PyQt5 import QtWidgets

    assert isinstance(circuit_tab.node_palette, QtWidgets.QListWidget)
    # Every node type appears in the palette.
    from sensoryforge.gui.circuit.nodes import NODE_CLASSES

    palette_items = {
        circuit_tab.node_palette.item(i).text()
        for i in range(circuit_tab.node_palette.count())
    }
    assert palette_items == set(NODE_CLASSES)
    assert isinstance(circuit_tab.inspector_panel, QtWidgets.QWidget)


def test_build_three_node_graph_connect_and_read_back(circuit_tab):
    """Build SensorArray -> RFBank -> Filter, connect it, and read the
    connections back -- adding, connecting and reading is the O2 'Done when'."""
    grid = circuit_tab.add_node("SensorArray", "grid1")
    rf = circuit_tab.add_node("RFBank", "rf1")
    filt = circuit_tab.add_node("Filter", "filter1")

    assert set(circuit_tab.nodes()) >= {"grid1", "rf1", "filter1"}

    circuit_tab.connect_nodes(grid, "value", rf, "Channel")
    circuit_tab.connect_nodes(rf, "Drive", filt, "Drive")

    conns = circuit_tab.connections()
    pairs = {
        (out.node().name(), out.name(), in_.node().name(), in_.name())
        for out, in_ in conns
    }
    assert ("grid1", "value", "rf1", "Channel") in pairs
    assert ("rf1", "Drive", "filter1", "Drive") in pairs


def test_moving_and_deleting_a_node(circuit_tab):
    node = circuit_tab.add_node("SensorArray", "grid1")
    item = node.graphicsItem()
    item.setPos(50, 60)
    assert (item.pos().x(), item.pos().y()) == (50, 60)

    circuit_tab.flowchart.removeNode(node)
    assert "grid1" not in circuit_tab.nodes()


def test_circuit_tab_is_the_first_tab_in_the_main_window():
    _ensure_app()
    from sensoryforge.gui.main import SensoryForgeWindow
    from sensoryforge.gui.tabs.circuit_tab import CircuitTab

    window = SensoryForgeWindow()
    tabs = window.centralWidget()
    assert isinstance(tabs.widget(0), CircuitTab)
    assert tabs.tabText(0) == "Circuit"
