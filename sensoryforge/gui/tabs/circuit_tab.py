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

from sensoryforge.gui.circuit.inspector import build_node_inspector
from sensoryforge.gui.circuit.nodes import NODE_CLASSES, build_node_library

# Re-exported for backward compatibility: these were defined in this module
# until Wave Q (Q1) pulled the whole config-build/render/run sequence out
# into sensoryforge.gui.circuit.run (run_graph_once) so the batch sweep
# (gui/circuit/sweep.py) could reuse it. tests/unit/test_circuit_dropped_params.py
# imports both names from here.
from sensoryforge.gui.circuit.run import (  # noqa: F401
    _dropped_params_warning,
    _is_schema_default,
)


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

        # --- Left (cont'd): registry-driven component palette (P3) --------
        # A second, separate widget from node_palette above -- node_palette
        # stays exactly as Wave O built it (its own existing test asserts
        # its contents are exactly NODE_CLASSES) and this tree adds the
        # per-registry component listing P3 asks for alongside it, so
        # nothing hard-codes which builders/filters/neurons/stimuli/
        # processing layers/grid arrangements exist -- a plugin that
        # registers a new one appears here automatically.
        palette_layout.addWidget(QtWidgets.QLabel("Registry components"))
        self.component_palette = QtWidgets.QTreeWidget()
        self.component_palette.setHeaderHidden(True)
        self.component_palette.itemDoubleClicked.connect(
            self._on_component_double_click
        )
        palette_layout.addWidget(self.component_palette)
        self.refresh_component_palette()

        self.chk_expert_mode = QtWidgets.QCheckBox("Expert mode")
        self.chk_expert_mode.toggled.connect(self._on_expert_mode_toggled)
        palette_layout.addWidget(self.chk_expert_mode)

        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.clicked.connect(self.run_graph)
        palette_layout.addWidget(self.btn_run)

        splitter.addWidget(palette_panel)

        # --- Middle: canvas -------------------------------------------------
        ctrl_widget = self.flowchart.widget()
        splitter.addWidget(ctrl_widget.chartWidget)
        ctrl_widget.chartWidget.scene().selectionChanged.connect(
            self._on_scene_selection_changed
        )

        # --- Right: inspector (Wave P) ---------------------------------------
        self.inspector_panel = QtWidgets.QWidget()
        inspector_layout = QtWidgets.QVBoxLayout(self.inspector_panel)
        inspector_layout.addWidget(QtWidgets.QLabel("Inspector"))
        inspector_layout.addStretch(1)
        splitter.addWidget(self.inspector_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self._selected_node = None

    # ------------------------------------------------------------------
    # Registry-driven component palette (P3)
    # ------------------------------------------------------------------

    #: Structural node type -> the registry it selects a component from.
    #: ``Combine`` and ``Record`` have no registered component and are
    #: intentionally absent (they are still placeable via node_palette).
    REGISTRY_NODE_TYPES = (
        "SensorArray",
        "Stimulus",
        "RFBank",
        "Processing",
        "Filter",
        "Readout",
    )

    @staticmethod
    def _registry_for_node_type(node_type: str):
        from sensoryforge.registry import (
            FILTER_REGISTRY,
            GRID_REGISTRY,
            INNERVATION_REGISTRY,
            NEURON_REGISTRY,
            PROCESSING_REGISTRY,
            STIMULUS_REGISTRY,
        )

        return {
            "SensorArray": GRID_REGISTRY,
            "Stimulus": STIMULUS_REGISTRY,
            "RFBank": INNERVATION_REGISTRY,
            "Processing": PROCESSING_REGISTRY,
            "Filter": FILTER_REGISTRY,
            "Readout": NEURON_REGISTRY,
        }[node_type]

    def refresh_component_palette(self) -> None:
        """Rebuild the registry-component tree from the live registries.

        Called at construction and safe to call again (e.g. after a plugin
        registers a new component at runtime) -- nothing here is cached
        beyond the tree widget's own items.
        """
        self.component_palette.clear()
        for node_type in self.REGISTRY_NODE_TYPES:
            registry = self._registry_for_node_type(node_type)
            category = QtWidgets.QTreeWidgetItem([node_type])
            for component_name in registry.list_registered():
                leaf = QtWidgets.QTreeWidgetItem([component_name])
                leaf.setData(0, QtCore.Qt.UserRole, (node_type, component_name))
                category.addChild(leaf)
            self.component_palette.addTopLevelItem(category)

    def _on_component_double_click(
        self, item: QtWidgets.QTreeWidgetItem, _column: int
    ) -> None:
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload is None:
            return  # a category header, not a leaf
        node_type, component_name = payload
        node = self.add_node(node_type)
        self._apply_component_selection(node, node_type, component_name)
        self.select_node(node)

    @staticmethod
    def _apply_component_selection(node, node_type: str, component_name: str) -> None:
        """Set the newly-placed node's registry selection to ``component_name``."""
        if node_type == "SensorArray":
            node.grid.arrangement = component_name
        elif node_type == "Stimulus":
            node.stimulus.type = component_name
        elif node_type == "RFBank":
            node.pop_input.rf.method = component_name
        elif node_type == "Processing":
            node.spec["method"] = component_name
        elif node_type == "Filter":
            node.filter_method = component_name
        elif node_type == "Readout":
            node.fields["neuron_model"] = component_name

    # ------------------------------------------------------------------
    # Inspector (Wave P: P1/P2)
    # ------------------------------------------------------------------
    def select_node(self, node) -> None:
        """Show ``node``'s inspector panel (params rendered from its
        component's ``get_param_spec()``, plus its P2 visualisation)."""
        self._selected_node = node
        self._rebuild_inspector()

    def _rebuild_inspector(self) -> None:
        old_layout = self.inspector_panel.layout()
        if old_layout is not None:
            while old_layout.count():
                child = old_layout.takeAt(0)
                widget = child.widget()
                if widget is not None:
                    widget.setParent(None)
            QtWidgets.QWidget().setLayout(old_layout)  # detach old layout
        layout = QtWidgets.QVBoxLayout(self.inspector_panel)
        if self._selected_node is None:
            layout.addWidget(QtWidgets.QLabel("Inspector"))
            layout.addStretch(1)
            return
        content = build_node_inspector(
            self._selected_node, expert_mode=self.chk_expert_mode.isChecked()
        )
        layout.addWidget(content)

    def _on_expert_mode_toggled(self, _checked: bool) -> None:
        if self._selected_node is not None:
            self._rebuild_inspector()

    def _on_scene_selection_changed(self) -> None:
        scene = self.flowchart.widget().chartWidget.scene()
        selected = [item for item in scene.selectedItems() if hasattr(item, "node")]
        if selected:
            self.select_node(selected[0].node)

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
    # Run (O4)
    # ------------------------------------------------------------------
    def run_graph(self, *, duration_ms: float = 200.0) -> dict:
        """Build a config from the current graph, run it, and emit ``simulation_finished``.

        Mirrors the CLI's ``run`` command (``sensoryforge/cli.py``): a stimulus
        tensor is rendered on the first grid's receptor coordinates via
        :func:`sensoryforge.stimuli.render.render_stimulus`, then
        :class:`~sensoryforge.core.simulation_engine.SimulationEngine` runs it.
        A ``RecordNode`` in the graph (``output_dir``, via
        ``config.metadata["record_output_dir"]``) makes the run also write a
        bundle via :func:`sensoryforge.io.bundle.write_bundle` (through
        ``SimulationEngine.run(bundle_dir=...)``), exactly as ``--bundle``
        does for the CLI (the Wave J writer).

        Returns:
            The ``sim_results`` dict (population name -> ``SimulationResult``)
            emitted on ``simulation_finished``.
        """
        import numpy as np

        from sensoryforge.gui.circuit.run import run_graph_once
        from sensoryforge.gui.tabs.spiking_tab import SimulationResult

        config, raw_results, frames, dt_ms = run_graph_once(
            self.flowchart, duration_ms=duration_ms
        )

        sim_results: dict = {}
        for pop_name, pop_results in raw_results.items():
            is_analog = "state" in pop_results
            spikes_or_state = pop_results.get("spikes", pop_results.get("state"))
            spikes_np = spikes_or_state[0].detach().cpu().numpy()
            filtered = pop_results.get("filtered")
            drive = pop_results.get("drive")
            filtered_np = (
                filtered[0].detach().cpu().numpy()
                if filtered is not None
                else spikes_np
            )
            raw_drive_np = (
                drive[0].detach().cpu().numpy() if drive is not None else None
            )
            voltages = pop_results.get("voltages")
            v_trace_np = (
                voltages[0].detach().cpu().numpy()
                if voltages is not None
                else np.zeros_like(spikes_np, dtype=float)
            )
            time_ms = np.arange(spikes_np.shape[0], dtype=float) * dt_ms
            sim_results[pop_name] = SimulationResult(
                population_name=pop_name,
                dt_ms=dt_ms,
                time_ms=time_ms,
                v_trace=v_trace_np,
                spikes=(np.zeros_like(spikes_np) if is_analog else spikes_np),
                drive=filtered_np,
                raw_drive=raw_drive_np,
                is_analog=is_analog,
            )

        frames_np = frames.detach().cpu().float().numpy()
        time_ms_axis = (
            next(iter(sim_results.values())).time_ms if sim_results else np.zeros(0)
        )
        xlim = (-5.0, 5.0)
        ylim = (-5.0, 5.0)

        self.simulation_finished.emit(
            sim_results, frames_np, time_ms_axis, dt_ms, xlim, ylim
        )
        return sim_results
