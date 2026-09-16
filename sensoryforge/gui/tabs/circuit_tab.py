"""The Circuit tab (Phase 3, Wave O): a node-graph editor for a SensoryForge experiment.

A ``pyqtgraph.flowchart.Flowchart`` is the graph model; ``sensoryforge.gui.circuit.nodes``
supplies the node types (one per config dataclass / fragment) and
``sensoryforge.gui.circuit.serialise`` converts the graph to and from a
``SensoryForgeConfig`` (O3). This tab is the entry point to building an
experiment and is added to ``SensoryForgeWindow`` as the first tab.

The inspector dock on the right is filled in by Wave P; here it is an empty
placeholder so the layout is stable across waves.
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtWidgets

from pyqtgraph.flowchart import Flowchart

from sensoryforge.gui.circuit.nodes import NODE_CLASSES, build_node_library


class CircuitTab(QtWidgets.QWidget):
    """Node-graph editor for building a :class:`SensoryForgeConfig` visually."""

    #: Same payload shape as SpikingNeuronTab.simulation_finished
    #: (results dict, stimulus_frames, time_ms, dt_ms, xlim, ylim), so
    #: VisualizationTab.set_simulation_results needs no changes to receive
    #: graph runs (O4).
    simulation_finished = QtCore.pyqtSignal(
        object, object, object, object, object, object
    )

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        experiment_manager=None,
    ) -> None:
        super().__init__(parent)
        self.experiment_manager = experiment_manager

        self.flowchart = Flowchart(terminals={}, name="Circuit")
        self.flowchart.library = build_node_library()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter)

        # --- Left: node palette -------------------------------------------------
        palette_panel = QtWidgets.QWidget()
        palette_layout = QtWidgets.QVBoxLayout(palette_panel)
        palette_layout.addWidget(QtWidgets.QLabel("Node palette"))
        self.node_palette = QtWidgets.QListWidget()
        for node_name in sorted(NODE_CLASSES):
            self.node_palette.addItem(node_name)
        self.node_palette.itemDoubleClicked.connect(self._on_palette_double_click)
        palette_layout.addWidget(self.node_palette)

        self.btn_add_node = QtWidgets.QPushButton("Add to canvas")
        self.btn_add_node.clicked.connect(self._on_add_node_clicked)
        palette_layout.addWidget(self.btn_add_node)

        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.clicked.connect(self.run_graph)
        palette_layout.addWidget(self.btn_run)

        splitter.addWidget(palette_panel)

        # --- Middle: canvas -------------------------------------------------
        ctrl_widget = self.flowchart.widget()
        splitter.addWidget(ctrl_widget.chartWidget)

        # --- Right: inspector (filled in Wave P) -----------------------------
        self.inspector_panel = QtWidgets.QWidget()
        inspector_layout = QtWidgets.QVBoxLayout(self.inspector_panel)
        inspector_layout.addWidget(QtWidgets.QLabel("Inspector"))
        inspector_layout.addStretch(1)
        splitter.addWidget(self.inspector_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------
    def add_node(self, node_type: str, name: Optional[str] = None):
        """Add a node of ``node_type`` (a key of ``NODE_CLASSES``) to the graph."""
        if node_type not in NODE_CLASSES:
            raise ValueError(f"Unknown Circuit node type {node_type!r}")
        return self.flowchart.createNode(node_type, name=name or node_type)

    def _on_palette_double_click(self, item: QtWidgets.QListWidgetItem) -> None:
        self.add_node(item.text())

    def _on_add_node_clicked(self) -> None:
        item = self.node_palette.currentItem()
        if item is not None:
            self.add_node(item.text())

    def connect_nodes(
        self, out_node, out_terminal: str, in_node, in_terminal: str
    ) -> None:
        """Connect one node's output terminal to another node's input terminal."""
        self.flowchart.connectTerminals(out_node[out_terminal], in_node[in_terminal])

    def connections(self):
        """Return the current set of ``(out terminal, in terminal)`` connections."""
        return self.flowchart.listConnections()

    def nodes(self):
        """Return the graph's ``{name: Node}`` mapping (includes the two I/O nodes)."""
        return self.flowchart.nodes()

    # ------------------------------------------------------------------
    # Run (O4 -- wired up in the O4 commit, once serialise.graph_to_config
    # exists)
    # ------------------------------------------------------------------
    def run_graph(self, *, duration_ms: float = 200.0) -> dict:
        """Build a config from the current graph, run it, and emit ``simulation_finished``."""
        raise NotImplementedError("run_graph is implemented in Wave O's O4 task")
