"""Batch execution tab — run parameter sweeps and export SLURM scripts.

Provides a GUI front-end for :class:`~sensoryforge.core.batch_executor.BatchExecutor`:

* Load a batch YAML configuration file
* Preview the expanded stimulus sweep (N runs)
* Run the batch locally with a progress bar
* Export a SLURM job array script for cluster execution
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtWidgets

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class _BatchWorker(QtCore.QThread):
    """Runs BatchExecutor.execute() in a background thread.

    Signals:
        progress: (current, total) ints as each stimulus completes.
        log_line: A single log message string.
        finished: Emitted on successful completion with result dict.
        errored: Emitted on failure with an exception string.
    """

    progress = QtCore.pyqtSignal(int, int)
    log_line = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(dict)
    errored = QtCore.pyqtSignal(str)

    def __init__(self, executor, save_format: str, save_intermediates: bool) -> None:
        super().__init__()
        self._executor = executor
        self._save_format = save_format
        self._save_intermediates = save_intermediates

    def run(self) -> None:
        try:
            # Monkey-patch print so output goes to the log signal
            import builtins

            _orig_print = builtins.print

            def _patched_print(*args, **kwargs):
                line = " ".join(str(a) for a in args)
                self.log_line.emit(line)
                _orig_print(*args, **kwargs)

            builtins.print = _patched_print
            try:
                result = self._executor.execute(
                    save_format=self._save_format,
                    save_intermediates=self._save_intermediates,
                )
            finally:
                builtins.print = _orig_print

            self.finished.emit(result)
        except Exception as exc:
            import traceback

            self.errored.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Graph-sweep helpers (Q1)
# ---------------------------------------------------------------------------


def _param_names_for_node(node) -> list:
    """The parameter names offered for ``node`` in the sweep dialog: the
    registered component's ``get_param_spec()`` names, per P3's
    registry-driven convention (``CircuitTab.REGISTRY_NODE_TYPES``), or an
    empty list when the node type has no registered component
    (``CombineNode``, ``RecordNode``) or the component cannot be resolved.
    """
    from sensoryforge.gui.circuit.nodes import (
        FilterNode,
        ProcessingNode,
        RFBankNode,
        ReadoutNode,
        SensorArrayNode,
        StimulusNode,
    )
    from sensoryforge.registry import (
        FILTER_REGISTRY,
        GRID_REGISTRY,
        INNERVATION_REGISTRY,
        NEURON_REGISTRY,
        PROCESSING_REGISTRY,
        STIMULUS_REGISTRY,
    )

    lookup = None
    component_name = None
    if isinstance(node, SensorArrayNode):
        lookup, component_name = GRID_REGISTRY, node.grid.arrangement
    elif isinstance(node, StimulusNode):
        lookup, component_name = STIMULUS_REGISTRY, node.stimulus.type
    elif isinstance(node, RFBankNode):
        lookup, component_name = INNERVATION_REGISTRY, node.pop_input.rf.method
    elif isinstance(node, ProcessingNode):
        lookup, component_name = PROCESSING_REGISTRY, node.spec.get("method")
    elif isinstance(node, FilterNode):
        lookup, component_name = FILTER_REGISTRY, node.filter_method
    elif isinstance(node, ReadoutNode):
        lookup, component_name = NEURON_REGISTRY, node.fields.get("neuron_model")

    if lookup is None or not component_name:
        return []
    try:
        specs = lookup.get_param_spec(component_name)
    except (KeyError, ValueError):
        return []
    return [spec.name for spec in specs]


def _parse_sweep_value(text: str):
    """Parse one comma-separated sweep value: int, then float, then the
    literal string (so e.g. an ``arrangement`` sweep over ``"grid, hex"``
    keeps its string values)."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


# ---------------------------------------------------------------------------
# BatchTab
# ---------------------------------------------------------------------------


class BatchTab(QtWidgets.QWidget):
    """Batch execution and SLURM export tab.

    Wire-up in main.py::

        batch_tab = BatchTab()
        tabs.addTab(batch_tab, "Batch")
        batch_tab.set_experiment_manager(em)
    """

    #: Emitted once per run during a graph sweep (Q1): (index, total).
    graph_sweep_progress = QtCore.pyqtSignal(int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._em = None
        self._executor = None
        self._worker: Optional[_BatchWorker] = None
        self._config_path: Optional[Path] = None
        self._circuit_tab = None
        self._build_ui()
        self._build_graph_sweep_ui()

    def set_experiment_manager(self, em) -> None:
        """Inject shared ExperimentManager from main window."""
        self._em = em

    def set_circuit_tab(self, circuit_tab) -> None:
        """Inject the Circuit tab (Q1) so its live graph can be swept.

        Args:
            circuit_tab: The :class:`~sensoryforge.gui.tabs.circuit_tab.CircuitTab`
                instance sharing this window, or ``None``.
        """
        self._circuit_tab = circuit_tab
        self._refresh_sweep_node_combo()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # ── Config file row ──────────────────────────────────────────
        cfg_group = QtWidgets.QGroupBox("Batch Configuration")
        cfg_layout = QtWidgets.QVBoxLayout(cfg_group)

        path_row = QtWidgets.QHBoxLayout()
        self._lbl_config = QtWidgets.QLabel("No file loaded")
        self._lbl_config.setStyleSheet("color: #888; font-style: italic;")
        path_row.addWidget(self._lbl_config, stretch=1)
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._on_browse)
        path_row.addWidget(btn_browse)
        cfg_layout.addLayout(path_row)

        # Sweep summary label
        self._lbl_sweep = QtWidgets.QLabel("")
        self._lbl_sweep.setStyleSheet("font-size: 11px; color: #666;")
        cfg_layout.addWidget(self._lbl_sweep)

        outer.addWidget(cfg_group)

        # ── Run settings ─────────────────────────────────────────────
        run_group = QtWidgets.QGroupBox("Run Settings")
        run_form = QtWidgets.QFormLayout(run_group)
        run_form.setLabelAlignment(QtCore.Qt.AlignRight)

        self._cmb_format = QtWidgets.QComboBox()
        self._cmb_format.addItems(["pytorch", "hdf5"])
        run_form.addRow("Save format:", self._cmb_format)

        self._chk_intermediates = QtWidgets.QCheckBox(
            "Save filtered drive and voltages"
        )
        run_form.addRow("", self._chk_intermediates)

        self._lbl_output = QtWidgets.QLabel("—")
        self._lbl_output.setStyleSheet("font-size: 11px; color: #666;")
        run_form.addRow("Output dir:", self._lbl_output)

        outer.addWidget(run_group)

        # ── Action buttons ───────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()

        self._btn_run = QtWidgets.QPushButton("▶  Run Batch")
        self._btn_run.setEnabled(False)
        self._btn_run.setFixedHeight(34)
        self._btn_run.clicked.connect(self._on_run)
        btn_row.addWidget(self._btn_run)

        self._btn_stop = QtWidgets.QPushButton("■  Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setFixedHeight(34)
        self._btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self._btn_stop)

        btn_row.addStretch()

        self._btn_slurm = QtWidgets.QPushButton("Export SLURM Script…")
        self._btn_slurm.setEnabled(False)
        self._btn_slurm.setFixedHeight(34)
        self._btn_slurm.clicked.connect(self._on_export_slurm)
        btn_row.addWidget(self._btn_slurm)

        outer.addLayout(btn_row)

        # ── Progress bar ─────────────────────────────────────────────
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        outer.addWidget(self._progress)

        # ── Log output ───────────────────────────────────────────────
        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setStyleSheet(
            "QPlainTextEdit { font-family: 'Courier New'; font-size: 11px; }"
        )
        log_layout.addWidget(self._log)
        outer.addWidget(log_group, stretch=1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        initial = ""
        if self._em is not None and self._em.is_open:
            initial = str(self._em.project_dir)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open batch YAML configuration",
            initial,
            "YAML Files (*.yml *.yaml);;All Files (*)",
        )
        if not path:
            return
        self._load_config(Path(path))

    def _load_config(self, path: Path) -> None:
        self._config_path = path
        self._lbl_config.setText(str(path))
        self._lbl_config.setStyleSheet("")
        self._log.clear()

        try:
            from sensoryforge.core.batch_executor import BatchExecutor

            self._executor = BatchExecutor.from_yaml(str(path))
            n = len(self._executor.stimulus_configs)
            self._lbl_sweep.setText(
                f"{n} stimulus configuration{'s' if n != 1 else ''} in sweep"
            )
            self._lbl_output.setText(str(self._executor.output_dir))
            self._btn_run.setEnabled(True)
            self._btn_slurm.setEnabled(True)
            self._log_line(f"Loaded: {path}")
            self._log_line(f"Batch ID: {self._executor.batch_id}")
            self._log_line(f"Sweep size: {n}")
        except Exception as exc:
            self._executor = None
            self._btn_run.setEnabled(False)
            self._btn_slurm.setEnabled(False)
            import traceback

            self._log_line(f"ERROR loading config:\n{traceback.format_exc()}")

    def _on_run(self) -> None:
        if self._executor is None:
            return
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_slurm.setEnabled(False)
        self._progress.setValue(0)
        n = len(self._executor.stimulus_configs)
        self._progress.setRange(0, n)

        self._worker = _BatchWorker(
            self._executor,
            save_format=self._cmb_format.currentText(),
            save_intermediates=self._chk_intermediates.isChecked(),
        )
        self._worker.log_line.connect(self._log_line)
        self._worker.progress.connect(lambda cur, total: self._progress.setValue(cur))
        self._worker.finished.connect(self._on_batch_finished)
        self._worker.errored.connect(self._on_batch_errored)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._log_line("Batch execution stopped by user.")
            self._btn_run.setEnabled(True)
            self._btn_stop.setEnabled(False)

    def _on_batch_finished(self, result: dict) -> None:
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_slurm.setEnabled(True)
        total = result.get("num_stimuli", 0)
        self._progress.setValue(self._progress.maximum())
        self._log_line(
            f"\n✓ Batch complete — {total} stimuli, "
            f"{result.get('duration_seconds', 0):.1f}s. "
            f"Output: {result.get('output_path', '—')}"
        )

    def _on_batch_errored(self, msg: str) -> None:
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._log_line(f"\n✗ Batch failed:\n{msg}")

    def _on_export_slurm(self) -> None:
        if self._executor is None or self._config_path is None:
            return

        # Collect SLURM settings from a simple dialog
        dialog = _SlurmSettingsDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        settings = dialog.get_settings()

        script = self._executor.generate_slurm_script(
            config_yaml_path=str(self._config_path.resolve()),
            **settings,
        )

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save SLURM script",
            str(self._config_path.parent / "run_batch.sh"),
            "Shell Scripts (*.sh);;All Files (*)",
        )
        if not save_path:
            return

        Path(save_path).write_text(script, encoding="utf-8")
        self._log_line(f"SLURM script written to: {save_path}")
        QtWidgets.QMessageBox.information(
            self,
            "SLURM Script Saved",
            f"Submit with:\n  sbatch {save_path}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_line(self, msg: str) -> None:
        self._log.appendPlainText(msg)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    # ------------------------------------------------------------------
    # Sweep this graph (Q1)
    # ------------------------------------------------------------------

    def _build_graph_sweep_ui(self) -> None:
        """Build the "sweep this graph" section: pick a node + one of its
        ``ParamSpec`` settings, enter a comma-separated list of values, and
        run one bundle per value via
        :func:`sensoryforge.gui.circuit.sweep.sweep_graph`.
        """
        outer = self.layout()

        group = QtWidgets.QGroupBox("Sweep This Graph")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        self._sweep_node_combo = QtWidgets.QComboBox()
        self._sweep_node_combo.currentTextChanged.connect(
            self._refresh_sweep_param_combo
        )
        form.addRow("Node:", self._sweep_node_combo)

        self._sweep_param_combo = QtWidgets.QComboBox()
        self._sweep_param_combo.setEditable(True)
        form.addRow("Parameter:", self._sweep_param_combo)

        self._sweep_values_edit = QtWidgets.QLineEdit()
        self._sweep_values_edit.setPlaceholderText("e.g. 10, 20, 30")
        form.addRow("Values (comma-separated):", self._sweep_values_edit)

        out_row = QtWidgets.QHBoxLayout()
        self._sweep_output_edit = QtWidgets.QLineEdit()
        out_row.addWidget(self._sweep_output_edit, stretch=1)
        btn_browse_out = QtWidgets.QPushButton("Browse…")
        btn_browse_out.clicked.connect(self._on_browse_sweep_output)
        out_row.addWidget(btn_browse_out)
        form.addRow("Output dir:", out_row)

        btn_row = QtWidgets.QHBoxLayout()
        btn_refresh = QtWidgets.QPushButton("Refresh from graph")
        btn_refresh.clicked.connect(self._refresh_sweep_node_combo)
        btn_row.addWidget(btn_refresh)
        self._btn_run_sweep = QtWidgets.QPushButton("Run Sweep")
        self._btn_run_sweep.clicked.connect(self._on_run_graph_sweep)
        btn_row.addWidget(self._btn_run_sweep)
        form.addRow(btn_row)

        outer.addWidget(group)
        self._refresh_sweep_node_combo()

    def _refresh_sweep_node_combo(self) -> None:
        self._sweep_node_combo.clear()
        if self._circuit_tab is None:
            return
        for name in sorted(self._circuit_tab.nodes()):
            self._sweep_node_combo.addItem(name)
        self._refresh_sweep_param_combo()

    def _refresh_sweep_param_combo(self, *_args) -> None:
        """Populate the parameter combo from the selected node's registered
        component ``get_param_spec()`` (P3's registry-driven convention),
        falling back to an empty (freely-editable) box when the node's
        current component has no param spec or is not selected."""
        self._sweep_param_combo.clear()
        if self._circuit_tab is None:
            return
        node_name = self._sweep_node_combo.currentText()
        nodes = self._circuit_tab.nodes()
        node = nodes.get(node_name)
        if node is None:
            return
        for name in _param_names_for_node(node):
            self._sweep_param_combo.addItem(name)

    def _on_browse_sweep_output(self) -> None:
        initial = ""
        if self._em is not None and self._em.is_open:
            initial = str(self._em.project_dir)
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select sweep output directory", initial
        )
        if path:
            self._sweep_output_edit.setText(path)

    def _on_run_graph_sweep(self) -> None:
        if self._circuit_tab is None:
            QtWidgets.QMessageBox.warning(
                self, "Sweep This Graph", "No Circuit tab is connected."
            )
            return
        node_name = self._sweep_node_combo.currentText()
        param_name = self._sweep_param_combo.currentText().strip()
        output_dir = self._sweep_output_edit.text().strip()
        if not node_name or not param_name or not output_dir:
            QtWidgets.QMessageBox.warning(
                self,
                "Sweep This Graph",
                "Select a node, a parameter, and an output directory.",
            )
            return
        raw_values = [v.strip() for v in self._sweep_values_edit.text().split(",")]
        raw_values = [v for v in raw_values if v]
        values = [_parse_sweep_value(v) for v in raw_values]

        try:
            bundle_dirs = self.run_graph_sweep(
                node_name, param_name, values, output_dir
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            import traceback

            self._log_line(f"\n✗ Sweep failed:\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(self, "Sweep This Graph", str(exc))
            return

        for i, bundle_dir in enumerate(bundle_dirs, start=1):
            self.graph_sweep_progress.emit(i, len(bundle_dirs))
            self._log_line(f"Sweep run {i}/{len(bundle_dirs)}: {bundle_dir}")
        self._log_line(
            f"\n✓ Sweep complete — {len(bundle_dirs)} bundles in {output_dir}"
        )

    def run_graph_sweep(
        self, node_name: str, param_name: str, values, output_dir
    ) -> list:
        """Run :func:`sensoryforge.gui.circuit.sweep.sweep_graph` against the
        connected Circuit tab's live flowchart.

        Kept as its own method (rather than inlined in the button handler)
        so GUI tests can call it directly without going through the combo
        boxes/line edits.

        Returns:
            The list of bundle directories written, one per value.
        """
        from sensoryforge.gui.circuit.sweep import sweep_graph

        return sweep_graph(
            self._circuit_tab.flowchart, node_name, param_name, values, output_dir
        )


# ---------------------------------------------------------------------------
# SLURM settings dialog
# ---------------------------------------------------------------------------


class _SlurmSettingsDialog(QtWidgets.QDialog):
    """Small dialog for SLURM job parameters."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SLURM Job Settings")
        self.setMinimumWidth(320)

        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        self._job_name = QtWidgets.QLineEdit("sensoryforge_batch")
        form.addRow("Job name:", self._job_name)

        self._partition = QtWidgets.QLineEdit("gpu")
        form.addRow("Partition:", self._partition)

        self._time = QtWidgets.QLineEdit("04:00:00")
        form.addRow("Wall time (HH:MM:SS):", self._time)

        self._mem = QtWidgets.QSpinBox()
        self._mem.setRange(1, 1024)
        self._mem.setValue(32)
        self._mem.setSuffix(" GB")
        form.addRow("Memory:", self._mem)

        self._cpus = QtWidgets.QSpinBox()
        self._cpus.setRange(1, 128)
        self._cpus.setValue(4)
        form.addRow("CPUs per task:", self._cpus)

        self._gpus = QtWidgets.QSpinBox()
        self._gpus.setRange(0, 8)
        self._gpus.setValue(1)
        form.addRow("GPUs:", self._gpus)

        self._conda_env = QtWidgets.QLineEdit("sensoryforge")
        form.addRow("Conda env:", self._conda_env)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        form.addRow(btn_box)

    def get_settings(self) -> dict:
        return {
            "job_name": self._job_name.text().strip() or "sensoryforge_batch",
            "partition": self._partition.text().strip() or "gpu",
            "time": self._time.text().strip() or "04:00:00",
            "mem_gb": self._mem.value(),
            "cpus_per_task": self._cpus.value(),
            "gpus": self._gpus.value(),
            "conda_env": self._conda_env.text().strip() or "sensoryforge",
        }
