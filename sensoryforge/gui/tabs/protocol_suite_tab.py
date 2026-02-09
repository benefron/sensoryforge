"""Protocol Suite tab for managing STA-oriented stimulus executions."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from PyQt5 import QtCore, QtWidgets

from sensoryforge.gui.protocol_backend import ProtocolSpec
from sensoryforge.utils.project_registry import ProjectRegistry, ProtocolRunRecord


@dataclass
class ProtocolSummary:
    """Lightweight description of a protocol entry in the library."""

    key: str
    title: str
    description: str
    stimulus_kind: str
    duration_ms: float
    repetitions: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class QueuedRun:
    """Container describing a queued protocol execution."""

    summary: ProtocolSummary
    overrides: Dict[str, Any]
    row: int


class ProtocolSuiteTab(QtWidgets.QWidget):
    """UI workspace for configuring and executing STA protocol batches."""

    run_requested = QtCore.pyqtSignal(list)
    load_run_requested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        mechanoreceptor_tab: QtWidgets.QWidget,
        stimulus_tab: QtWidgets.QWidget,
        spiking_tab: QtWidgets.QWidget,
        project_registry: Optional[ProjectRegistry] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._mech_tab = mechanoreceptor_tab
        self._stimulus_tab = stimulus_tab
        self._spiking_tab = spiking_tab
        if project_registry is None:
            default_root = Path.cwd() / "project_registry"
            self._registry = ProjectRegistry(default_root)
        else:
            self._registry = project_registry
        self._protocols: Dict[str, ProtocolSummary] = {}
        self._protocol_specs: Dict[str, ProtocolSpec] = {}
        self._queued_runs: List[QueuedRun] = []
        self._running = False
        self._setup_ui()
        self._load_builtin_protocols()
        try:
            self.refresh_run_records(self._registry.list_runs())
        except Exception:
            # Registry directory may not be initialized yet; ignore on startup.
            pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        left_pane = QtWidgets.QVBoxLayout()
        layout.addLayout(left_pane, 1)

        self.protocol_list = QtWidgets.QListWidget()
        self.protocol_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.protocol_list.itemSelectionChanged.connect(self._on_protocol_selected)
        left_pane.addWidget(
            self._wrap_with_label(self.protocol_list, "Available Protocols")
        )

        buttons_row = QtWidgets.QHBoxLayout()
        self.btn_manage_library = QtWidgets.QToolButton()
        self.btn_manage_library.setText("Manage Library…")
        self.btn_manage_library.clicked.connect(self._on_manage_library)
        buttons_row.addWidget(self.btn_manage_library)
        self.btn_clone = QtWidgets.QToolButton()
        self.btn_clone.setText("Clone")
        self.btn_clone.clicked.connect(self._on_clone_protocol)
        buttons_row.addWidget(self.btn_clone)
        buttons_row.addStretch(1)
        left_pane.addLayout(buttons_row)

        self.saved_runs = QtWidgets.QListWidget()
        self.saved_runs.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.saved_runs.itemDoubleClicked.connect(self._on_saved_run_double_clicked)
        left_pane.addWidget(
            self._wrap_with_label(self.saved_runs, "Saved Protocol Runs")
        )

        right_pane = QtWidgets.QVBoxLayout()
        layout.addLayout(right_pane, 2)

        self.editor_stack = QtWidgets.QStackedWidget()
        self.editor_placeholder = QtWidgets.QTextEdit()
        self.editor_placeholder.setReadOnly(True)
        self.editor_placeholder.setPlaceholderText(
            "Select a protocol to review its parameters and editing options."
        )
        self.editor_stack.addWidget(self.editor_placeholder)
        right_pane.addWidget(
            self._wrap_with_label(self.editor_stack, "Protocol Editor")
        )

        queue_box = QtWidgets.QGroupBox("Run Queue")
        queue_layout = QtWidgets.QVBoxLayout(queue_box)
        self.run_table = QtWidgets.QTableWidget(0, 4)
        self.run_table.setHorizontalHeaderLabels(
            ["Protocol", "Overrides", "Assigned Population", "Status"]
        )
        self.run_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        queue_layout.addWidget(self.run_table)

        controls = QtWidgets.QHBoxLayout()
        self.btn_queue_selected = QtWidgets.QPushButton("Queue Selected")
        self.btn_queue_selected.clicked.connect(self._on_queue_selected)
        controls.addWidget(self.btn_queue_selected)

        self.btn_run_queue = QtWidgets.QPushButton("Run Queue")
        self.btn_run_queue.clicked.connect(self._on_run_queue)
        controls.addWidget(self.btn_run_queue)

        self.btn_clear_queue = QtWidgets.QPushButton("Clear")
        self.btn_clear_queue.clicked.connect(self._on_clear_queue)
        controls.addWidget(self.btn_clear_queue)
        controls.addStretch(1)
        queue_layout.addLayout(controls)

        right_pane.addWidget(queue_box)

        self.log_console = QtWidgets.QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setPlaceholderText(
            "Execution logs and progress messages will appear here."
        )
        right_pane.addWidget(self._wrap_with_label(self.log_console, "Run Log"))

    def _wrap_with_label(
        self, widget: QtWidgets.QWidget, title: str
    ) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(widget)
        return box

    def _load_builtin_protocols(self) -> None:
        summaries = [
            ProtocolSummary(
                key="slow_gaussian",
                title="Slow Gaussian Ramps",
                description="Large spatial ramps probing SA tuning.",
                stimulus_kind="gaussian",
                duration_ms=520.0,
                repetitions=3,
            ),
            ProtocolSummary(
                key="fast_gaussian",
                title="Rapid Gaussian Pulses",
                description="Short, high-slope Gaussian bursts for RA cells.",
                stimulus_kind="gaussian",
                duration_ms=160.0,
                repetitions=12,
            ),
            ProtocolSummary(
                key="moving_gaussian",
                title="Moving Gaussian Sweeps",
                description="Translating bumps that traverse canonical paths.",
                stimulus_kind="gaussian_motion",
                duration_ms=340.0,
                repetitions=4,
            ),
            ProtocolSummary(
                key="macro_gaussian",
                title="Macro Gaussian Presses",
                description=("Broad high-amplitude presses activating many units."),
                stimulus_kind="gaussian",
                duration_ms=720.0,
                repetitions=3,
            ),
            ProtocolSummary(
                key="edge_steps",
                title="Edge Steps",
                description="Directional edge onsets for transient encoding.",
                stimulus_kind="edge",
                duration_ms=280.0,
                repetitions=5,
            ),
            ProtocolSummary(
                key="point_pulses",
                title="Point Pulses",
                description="Threshold-level taps producing sparse spikes.",
                stimulus_kind="point",
                duration_ms=140.0,
                repetitions=12,
            ),
            ProtocolSummary(
                key="large_disk_pulses",
                title="Large Disk Pulses",
                description=("Wide-diameter indentations for global activation."),
                stimulus_kind="disk",
                duration_ms=220.0,
                repetitions=6,
            ),
            ProtocolSummary(
                key="texture_noise",
                title="Binary Texture Noise",
                description="Frozen random textures tiling the grid.",
                stimulus_kind="texture",
                duration_ms=180.0,
                repetitions=6,
            ),
            ProtocolSummary(
                key="shear_wave",
                title="Shear Wave Gratings",
                description=("Sinusoidal shear with diverse spatial/temporal freq."),
                stimulus_kind="shear",
                duration_ms=420.0,
                repetitions=6,
            ),
            ProtocolSummary(
                key="rotating_edge",
                title="Rotating Edge Sweep",
                description=(
                    "Continuously rotating edge stimulus for " "orientation tuning."
                ),
                stimulus_kind="edge_motion",
                duration_ms=360.0,
                repetitions=2,
            ),
            ProtocolSummary(
                key="center_surround",
                title="Center-Surround Pulses",
                description=("Difference-of-Gaussian presses probing antagonism."),
                stimulus_kind="dog",
                duration_ms=320.0,
                repetitions=6,
            ),
        ]
        for summary in summaries:
            self._protocols[summary.key] = summary
            self._protocol_specs[summary.key] = ProtocolSpec(
                key=summary.key,
                title=summary.title,
                description=summary.description,
            )
            item = QtWidgets.QListWidgetItem(summary.title)
            item.setData(QtCore.Qt.UserRole, summary.key)
            item.setToolTip(summary.description)
            self.protocol_list.addItem(item)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_protocol_selected(self) -> None:
        selected = self.protocol_list.selectedItems()
        if not selected:
            self.editor_placeholder.setPlainText(
                ("Select a protocol to review its parameters and editing " "options.")
            )
            return
        summaries = [
            self._protocols[item.data(QtCore.Qt.UserRole)] for item in selected
        ]
        text = "\n\n".join(summary.to_json() for summary in summaries)
        self.editor_placeholder.setPlainText(text)

    def _on_manage_library(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Library Management",
            "Library editing is tracked as a future milestone.",
        )

    def _on_clone_protocol(self) -> None:
        selected = self.protocol_list.selectedItems()
        if not selected:
            return
        source_item = selected[0]
        key = str(source_item.data(QtCore.Qt.UserRole))
        summary = self._protocols[key]
        clone_key = f"{summary.key}_copy"
        index = 1
        while clone_key in self._protocols:
            clone_key = f"{summary.key}_copy{index}"
            index += 1
        clone = ProtocolSummary(
            key=clone_key,
            title=f"{summary.title} (Copy)",
            description=summary.description,
            stimulus_kind=summary.stimulus_kind,
            duration_ms=summary.duration_ms,
            repetitions=summary.repetitions,
        )
        self._protocols[clone_key] = clone
        self._protocol_specs[clone_key] = ProtocolSpec(
            key=clone.key,
            title=clone.title,
            description=clone.description,
        )
        clone_item = QtWidgets.QListWidgetItem(clone.title)
        clone_item.setData(QtCore.Qt.UserRole, clone.key)
        clone_item.setToolTip(clone.description)
        self.protocol_list.addItem(clone_item)
        self.protocol_list.setCurrentItem(clone_item)

    def _on_queue_selected(self) -> None:
        if self._running:
            QtWidgets.QMessageBox.information(
                self,
                "Protocols Running",
                ("Wait for the current batch to finish before queuing " "more runs."),
            )
            return
        selected = self.protocol_list.selectedItems()
        if not selected:
            return
        for item in selected:
            key = str(item.data(QtCore.Qt.UserRole))
            summary = self._protocols[key]
            overrides = {
                "dt_ms": None,
                "repetitions": summary.repetitions,
            }
            row = self.run_table.rowCount()
            self.run_table.insertRow(row)
            self.run_table.setItem(row, 0, QtWidgets.QTableWidgetItem(summary.title))
            self.run_table.setItem(
                row,
                1,
                QtWidgets.QTableWidgetItem(json.dumps(overrides)),
            )
            self.run_table.setItem(
                row,
                2,
                QtWidgets.QTableWidgetItem("Auto"),
            )
            self.run_table.setItem(
                row,
                3,
                QtWidgets.QTableWidgetItem("Pending"),
            )
            self._queued_runs.append(QueuedRun(summary, overrides, row))

    def _on_run_queue(self) -> None:
        if self._running:
            QtWidgets.QMessageBox.information(
                self, "Protocols Running", "A protocol batch is already running."
            )
            return
        if not self._queued_runs:
            QtWidgets.QMessageBox.information(
                self, "Run Queue", "Please queue at least one protocol."
            )
            return
        payload = [
            {
                "protocol_key": queued.summary.key,
                "overrides": dict(queued.overrides),
                "row": queued.row,
            }
            for queued in self._queued_runs
        ]
        keys = ", ".join(entry["protocol_key"] for entry in payload)
        self.append_log(f"[{datetime.now():%H:%M:%S}] Launching protocol batch: {keys}")
        self.set_running(True)
        self.run_requested.emit(payload)

    def _on_clear_queue(self) -> None:
        if self._running:
            QtWidgets.QMessageBox.information(
                self, "Protocols Running", "Unable to clear while a batch is running."
            )
            return
        self.run_table.setRowCount(0)
        self._queued_runs.clear()

    def _on_saved_run_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        run_id = item.data(QtCore.Qt.UserRole)
        if isinstance(run_id, str) and run_id:
            self.load_run_requested.emit(run_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def protocol_spec(self, key: str) -> ProtocolSpec:
        return self._protocol_specs[key]

    def protocol_summaries(self) -> Iterable[ProtocolSummary]:
        return self._protocols.values()

    def set_running(self, running: bool) -> None:
        self._running = running
        self.btn_queue_selected.setEnabled(not running)
        self.btn_run_queue.setEnabled(not running)
        self.btn_clear_queue.setEnabled(not running)
        self.protocol_list.setEnabled(not running)

    def set_row_status(self, row: int, status: str) -> None:
        if 0 <= row < self.run_table.rowCount():
            self.run_table.setItem(row, 3, QtWidgets.QTableWidgetItem(status))

    def set_row_population(self, row: int, population: str) -> None:
        if 0 <= row < self.run_table.rowCount():
            self.run_table.setItem(row, 2, QtWidgets.QTableWidgetItem(population))

    def clear_queue_after_run(self) -> None:
        self.run_table.setRowCount(0)
        self._queued_runs.clear()
        self.set_running(False)

    def append_log(self, message: str) -> None:
        self.log_console.appendPlainText(message)
        try:
            print(f"[ProtocolSuite] {message}", flush=True)
        except Exception:
            pass

    def refresh_run_records(self, records: Sequence[ProtocolRunRecord]) -> None:
        self.saved_runs.clear()
        for record in records:
            label = f"{record.run_id} ({record.protocol_id})"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, record.run_id)
            tooltip_lines = [
                f"Protocol: {record.protocol_id}",
                f"Created: {record.created_at}",
                f"Bundles: {len(record.neuron_modules)}",
            ]
            item.setToolTip("\n".join(tooltip_lines))
            self.saved_runs.addItem(item)

    def queued_protocol_keys(self) -> List[str]:
        return [queued.summary.key for queued in self._queued_runs]

    def queued_specs(self) -> List[ProtocolSpec]:
        return [
            self._protocol_specs[queued.summary.key] for queued in self._queued_runs
        ]

    def on_run_completed(self, run_id: str) -> None:
        """Handle successful completion of a protocol run."""
        self.append_log(f"[{datetime.now():%H:%M:%S}] Run completed: {run_id}")

    def on_run_failed(self, message: str) -> None:
        """Handle failed protocol run."""
        self.append_log(f"[{datetime.now():%H:%M:%S}] Run failed: {message}")

    def on_batch_finished(self) -> None:
        """Handle completion of entire batch."""
        self.append_log(f"[{datetime.now():%H:%M:%S}] Batch execution finished")
        self.clear_queue_after_run()
