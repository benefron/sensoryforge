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

import warnings

from typing import Optional

from PyQt5 import QtCore, QtWidgets

from pyqtgraph.flowchart import Flowchart

from sensoryforge.gui.circuit.nodes import NODE_CLASSES, build_node_library


def _dropped_params_warning(stimulus_type: str, dropped) -> Optional[str]:
    """The warning text for discarded stimulus settings, or ``None``.

    Args:
        stimulus_type: The stimulus's registered name, for the message.
        dropped: ``(field_name, value)`` pairs the constructor rejected.

    Returns:
        A message naming only the fields whose value the user had changed
        from the schema default, or ``None`` when every discarded field was
        untouched and there is nothing worth saying.
    """
    deliberate = [
        f"{key}={value!r}"
        for key, value in dropped
        if not _is_schema_default(key, value)
    ]
    if not deliberate:
        return None
    return (
        f"Stimulus {stimulus_type!r} does not accept {', '.join(deliberate)}; "
        "the value(s) you set were ignored and the stimulus ran without them."
    )


def _is_schema_default(field_name: str, value) -> bool:
    """Whether *value* is what ``StimulusConfig`` would hold untouched.

    ``StimulusConfig.to_dict()`` carries every field the schema defines,
    most of which a given stimulus class knows nothing about. Discarding
    those is housekeeping. Discarding one the user actually set is a
    changed stimulus, so the two cases are told apart here rather than
    warning about all of them and training the reader to ignore it.

    Args:
        field_name: The dropped keyword.
        value: The value it held.

    Returns:
        ``True`` when the field is unknown to the schema or still at its
        declared default.
    """
    import dataclasses

    from sensoryforge.config.schema import StimulusConfig

    for field in dataclasses.fields(StimulusConfig):
        if field.name != field_name:
            continue
        if field.default is not dataclasses.MISSING:
            return value == field.default
        if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            return value == field.default_factory()  # type: ignore[misc]
        return False
    return True


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
        import re

        import numpy as np
        import torch

        from sensoryforge.core.grid import ReceptorGrid
        from sensoryforge.core.simulation_engine import SimulationEngine
        from sensoryforge.gui.circuit.serialise import graph_to_config
        from sensoryforge.gui.tabs.spiking_tab import SimulationResult
        from sensoryforge.stimuli.render import render_stimulus

        config = graph_to_config(self.flowchart)

        if config.grids:
            grid_cfg = config.grids[0]
            stim_grid = ReceptorGrid(
                grid_size=(grid_cfg.rows or 40, grid_cfg.cols or 40),
                spacing=grid_cfg.spacing,
                arrangement=grid_cfg.arrangement,
                center=(grid_cfg.center_x, grid_cfg.center_y),
                density=grid_cfg.density,
                device=config.simulation.device,
                seed=grid_cfg.seed,
            )
            xx, yy = stim_grid.get_coordinates()
        else:
            xx, yy = torch.meshgrid(
                torch.linspace(-1, 1, 40),
                torch.linspace(-1, 1, 40),
                indexing="ij",
            )

        stim = config.stimulus
        # StimulusConfig.to_dict() carries every field the schema has
        # (administrative ones like motion/composition_mode/channel
        # included); a given registered stimulus class's constructor only
        # accepts its own subset. Retry dropping whichever keyword the
        # constructor just rejected, the same way render.py's own envelope-
        # key retry works, rather than hard-coding a per-type field list here.
        stimulus_params = {
            k: v for k, v in stim.to_dict().items() if k not in ("name", "type")
        }
        dropped: list = []
        while True:
            try:
                frames, _ = render_stimulus(
                    stim.type,
                    stimulus_params,
                    xx,
                    yy,
                    dt_ms=config.simulation.dt_ms,
                    duration_ms=duration_ms,
                    device=config.simulation.device,
                )
                break
            except TypeError as exc:
                match = re.search(r"unexpected keyword argument '(\w+)'", str(exc))
                if match is None or match.group(1) not in stimulus_params:
                    raise
                key = match.group(1)
                dropped.append((key, stimulus_params.pop(key)))

        # Dropping a field the user never set is housekeeping; dropping one
        # they did set changes the stimulus they asked for, and doing that
        # silently is how a graph ends up describing a run that did not
        # happen. Warn for the second case only, so the message means
        # something when it appears.
        message = _dropped_params_warning(stim.type, dropped)
        if message is not None:
            warnings.warn(message, UserWarning, stacklevel=2)
        stimulus_tensor = frames.unsqueeze(0)

        engine = SimulationEngine(config)
        bundle_dir = config.metadata.get("record_output_dir")
        raw_results = engine.run(
            stimulus_tensor,
            return_intermediates=True,
            bundle_dir=bundle_dir,
            stimulus_config=stim.to_dict(),
        )

        dt_ms = config.simulation.dt_ms
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
