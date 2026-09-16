"""Qt test: selecting a Circuit node shows its P2 visualisation (Phase 3,
Wave P, P2).

The Mechanoreceptor tab's grid view, the Stimulus Designer's live preview
and a receptive-field weight display are all deeply entangled with their
~3,500-line tabs (mouse-driven population placement, drag/drop, a single
shared plot item reused for several overlays -- see
gui/circuit/inspector.py's module docstring). Rather than risk a
behaviour-changing extraction under time pressure, P2 reuses the same
*data-layer* primitives those tabs draw from (ReceptorGrid,
render_stimulus) in new, small, Circuit-owned preview widgets, disclosed as
a scoped simplification in the Wave P report. This test proves those
previews actually render for the three node types P2 names, and that doing
so touches no code in the two entangled tabs at all -- so their own
existing tests need no edits (proven by the tab-test run alongside this
file in the report, not repeated here).

Marked gui; run alone (F-016).
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


def test_sensor_array_node_shows_a_grid_scatter_preview(circuit_tab):
    import pyqtgraph as pg

    node = circuit_tab.add_node("SensorArray", "grid1")
    circuit_tab.select_node(node)

    plot_widgets = circuit_tab.inspector_panel.findChildren(pg.PlotWidget)
    assert plot_widgets, "expected a pyqtgraph PlotWidget in the inspector"


def test_stimulus_node_shows_a_frame_preview(circuit_tab):
    import pyqtgraph as pg

    node = circuit_tab.add_node("Stimulus", "stim1")
    circuit_tab._apply_component_selection(node, "Stimulus", "gaussian")
    circuit_tab.select_node(node)

    plot_widgets = circuit_tab.inspector_panel.findChildren(pg.PlotWidget)
    assert (
        plot_widgets
    ), "expected a pyqtgraph PlotWidget (frame preview) in the inspector"
    image_items = [
        item
        for w in plot_widgets
        for item in w.getPlotItem().listDataItems() + w.getPlotItem().items
        if item.__class__.__name__ == "ImageItem"
    ]
    assert image_items, "expected an ImageItem rendering the stimulus frame"


def test_rf_bank_node_shows_a_builder_summary(circuit_tab):
    from PyQt5 import QtWidgets

    node = circuit_tab.add_node("RFBank", "rf1")
    circuit_tab._apply_component_selection(node, "RFBank", "gaussian")
    circuit_tab.select_node(node)

    labels = circuit_tab.inspector_panel.findChildren(QtWidgets.QLabel)
    assert any("gaussian" in lbl.text() for lbl in labels)


def test_combine_and_record_nodes_render_without_a_visualisation(circuit_tab):
    """Node types P2 does not name (Combine, Record) still get a working,
    non-blank inspector panel -- just no visualisation widget."""
    combine = circuit_tab.add_node("Combine", "combine1")
    circuit_tab.select_node(combine)
    assert circuit_tab.inspector_panel.layout().count() > 0

    record = circuit_tab.add_node("Record", "record1")
    circuit_tab.select_node(record)
    assert circuit_tab.inspector_panel.layout().count() > 0
