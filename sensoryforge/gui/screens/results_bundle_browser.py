"""Open bundle... dialog: pick one of the project's runs, or Browse any dir.

:func:`describe_run` reads a run directory's ``config.json`` only (never
``data.h5``), so listing a project's runs stays cheap even with many of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt5 import QtWidgets


def describe_run(run_dir: Path) -> Tuple[str, List[str], Optional[float]]:
    """A run's name, population names and duration, from ``config.json`` alone.

    Args:
        run_dir: A bundle directory (holds ``config.json``).

    Returns:
        ``(name, population_names, duration_ms)`` -- ``duration_ms`` is
        ``None`` if the stored config does not carry ``simulation.duration_ms``.
    """
    config_path = run_dir / "config.json"
    try:
        with open(config_path, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return run_dir.name, [], None
    config = payload.get("config", {})
    populations = [pop.get("name", "") for pop in config.get("populations", [])]
    duration_ms = config.get("simulation", {}).get("duration_ms")
    return run_dir.name, populations, duration_ms


class OpenBundleDialog(QtWidgets.QDialog):
    """Lists a project's runs (newest first) plus a Browse... option."""

    def __init__(
        self,
        run_dirs: List[Path],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open bundle")
        self.selected_path: Optional[Path] = None

        layout = QtWidgets.QVBoxLayout(self)
        self.list_widget = QtWidgets.QListWidget()
        for run_dir in run_dirs:
            name, populations, duration_ms = describe_run(run_dir)
            duration_text = (
                f"{duration_ms:.0f} ms" if duration_ms else "duration unknown"
            )
            label = (
                f"{name}  ({', '.join(populations) or 'no populations'}, "
                f"{duration_text})"
            )
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtWidgets.QListWidgetItem.UserType, str(run_dir))
            self.list_widget.addItem(item)
        self.list_widget.itemDoubleClicked.connect(self._accept_selected)
        layout.addWidget(self.list_widget)

        buttons = QtWidgets.QHBoxLayout()
        self.browse_button = QtWidgets.QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse)
        buttons.addWidget(self.browse_button)
        buttons.addStretch(1)
        self.open_button = QtWidgets.QPushButton("Open")
        self.open_button.clicked.connect(self._accept_current)
        buttons.addWidget(self.open_button)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

        self._run_dirs = run_dirs

    def _accept_current(self) -> None:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self._run_dirs):
            self.selected_path = self._run_dirs[row]
            self.accept()

    def _accept_selected(self, item: QtWidgets.QListWidgetItem) -> None:
        row = self.list_widget.row(item)
        if 0 <= row < len(self._run_dirs):
            self.selected_path = self._run_dirs[row]
            self.accept()

    def _browse(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open bundle directory"
        )
        if directory:
            self.selected_path = Path(directory)
            self.accept()
