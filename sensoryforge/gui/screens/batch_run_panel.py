"""The Batch screen's run panel: progress bar, per-job status, log.

Wraps what :class:`~sensoryforge.gui.execution.sweep_controller.SweepController`
reports (``log(str)``, ``progress(int, int)``, ``finished(int)``,
``failed(str)``) into a progress bar, a ``queued``/``running``/``done``/
``failed`` status per combination (parsed from the controller's log lines,
each prefixed ``[combo_NNN]``, since the controller itself only reports
aggregate progress), and a streaming log pane.
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt5 import QtGui, QtWidgets

from sensoryforge.gui.execution.sweep_controller import SweepManifest


class SweepRunPanel(QtWidgets.QWidget):
    """Progress bar, per-job status list and log pane for a running sweep.

    Args:
        parent: Qt parent.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._items: Dict[str, QtWidgets.QListWidgetItem] = {}

        layout = QtWidgets.QVBoxLayout(self)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        split = QtWidgets.QHBoxLayout()
        jobs_column = QtWidgets.QVBoxLayout()
        jobs_column.addWidget(QtWidgets.QLabel("Jobs"))
        self.job_list = QtWidgets.QListWidget()
        jobs_column.addWidget(self.job_list, 1)
        split.addLayout(jobs_column, 1)
        log_column = QtWidgets.QVBoxLayout()
        log_column.addWidget(QtWidgets.QLabel("Log"))
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        # A family that exists everywhere; "Monospace" does not on macOS and
        # costs a font-alias scan at start-up.
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.log_view.setFont(mono)
        log_column.addWidget(self.log_view, 1)
        split.addLayout(log_column, 2)
        layout.addLayout(split, 1)

    # --------------------------------------------------------------- driving

    def reset(self, manifest: SweepManifest) -> None:
        """Show one ``queued`` row per combination of a freshly written sweep."""
        self.job_list.clear()
        self._items = {}
        for combo in manifest.combos:
            item = QtWidgets.QListWidgetItem(f"{combo['dir']}: queued")
            self.job_list.addItem(item)
            self._items[combo["dir"]] = item
        self.progress_bar.setRange(0, max(1, len(manifest.combos)))
        self.progress_bar.setValue(0)

    def append(self, text: str) -> None:
        """Add one line to the log pane without touching any job status."""
        self.log_view.appendPlainText(text)

    def on_log(self, line: str) -> None:
        """Append a controller log line, updating that job's status."""
        self.log_view.appendPlainText(line)
        combo_dir = self._combo_dir(line)
        item = self._items.get(combo_dir) if combo_dir else None
        if item is None:
            return
        detail = line.split("]", 1)[1].strip()
        if detail.startswith("$ "):
            item.setText(f"{combo_dir}: running")
        elif "could not start" in detail or detail.endswith("(failed)"):
            item.setText(f"{combo_dir}: failed - {detail}")
        elif detail.startswith("exit 0"):
            item.setText(f"{combo_dir}: done")
        elif detail.startswith("exit"):
            item.setText(f"{combo_dir}: failed - {detail}")

    def on_progress(self, done: int, total: int) -> None:
        """Advance the progress bar to ``done`` of ``total``."""
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(done)

    def on_finished(self, n_failed: int) -> None:
        """Log the sweep's outcome."""
        self.append(
            f"sweep finished: {n_failed} failed of {self.progress_bar.maximum()}"
        )

    def on_failed(self, message: str) -> None:
        """Log a subprocess that could not be started at all."""
        self.append(f"error: {message}")

    def status_of(self, combo_dir: str) -> str:
        """The status text currently shown for ``combo_dir`` (tests)."""
        item = self._items.get(combo_dir)
        return item.text() if item is not None else ""

    @staticmethod
    def _combo_dir(line: str) -> Optional[str]:
        if not line.startswith("["):
            return None
        end = line.find("]")
        return None if end < 0 else line[1:end]
