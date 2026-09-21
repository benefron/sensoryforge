"""The Batch screen: sweep builder, local runner, SLURM export, YAML export.

Config-field sweeps here are one ``config.yml`` per combination (the ruling
recorded for this project -- the legacy
:class:`~sensoryforge.core.batch_executor.BatchExecutor` stimulus-sweep YAML
stays for old files and is not what this screen builds). Everything below
``config`` is deep-copied per combination;
:attr:`~sensoryforge.gui.session.Session.config` itself is never written to
or mutated by anything on this screen -- writing and running a sweep must
leave the session exactly as it found it (the brief).

Layout: sweep builder (left, ~60%) -- a table of swept fields, mode
(full grid / zipped), repetitions with a base seed, duration, a live job-count
summary, Write sweep / Run locally / Export SLURM script; preview (right,
~40%, :class:`~sensoryforge.gui.screens.batch_preview.SweepPreviewPane`) --
the literal YAML and command of the sweep's first combination, and the same
for a single run of the session's own config (this doubles as the plan's
"Export" view).
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtWidgets

from sensoryforge.gui.execution.sweep_controller import (
    SweepController,
    SweepManifest,
    sweep_command,
    write_slurm_script,
)
from sensoryforge.gui.screens.batch_combos import (
    build_combinations,
    build_preview_manifest,
    expand_repetitions,
    format_combo,
    preview_config,
    write_combo_sweep,
)
from sensoryforge.gui.screens.batch_fields import (
    FIELD_MODES,
    infer_is_int,
    parse_field_values,
)
from sensoryforge.gui.screens.batch_picker import ParameterPickerDialog
from sensoryforge.gui.screens.batch_preview import SweepPreviewPane
from sensoryforge.gui.screens.batch_run_panel import SweepRunPanel
from sensoryforge.gui.screens.batch_slurm import SlurmSettingsDialog
from sensoryforge.gui.session import Session
from sensoryforge.stimuli.base import ParamSpec

#: Column index of each field in the sweep table.
_COL_PATH, _COL_MODE, _COL_VALUES, _COL_STATUS, _COL_REMOVE = range(5)

#: How many combinations the live summary spells out.
_SUMMARY_PREVIEW_COUNT = 3

_MODE_LABELS = {"list": "List", "linear": "Linear range", "log": "Log range"}
_COMBINE_LABELS = {"full_grid": "Full grid", "zipped": "Zipped"}


def _safe_name(name: str) -> str:
    """Filesystem-safe sweep name (mirrors ``project.py::_safe_name``)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "sweep"


def _timestamped_root(parent: Path, name: str) -> Path:
    """``parent/<timestamp>_<safe name>``, matching a project's run naming."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return parent / f"{stamp}_{_safe_name(name)}"


def _single_run_command(config_path: str, duration_ms: float, bundle_dir: str) -> str:
    """The exact CLI command that runs ``config_path`` as a single run."""
    return (
        f"{sys.executable} -m sensoryforge.cli run {config_path} "
        f"--duration {float(duration_ms)} --bundle {bundle_dir}"
    )


class BatchScreen(QtWidgets.QWidget):
    """Sweep builder, local runner and SLURM/YAML export, bound to a ``Session``.

    Args:
        session: The experiment this screen sweeps and runs.
        parent: Qt parent.
    """

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._rows: List[Dict] = []
        self._last_manifest: Optional[SweepManifest] = None
        #: Guard so tests can suppress modal dialogs (mirrors RunBar).
        self.show_dialogs = True
        #: Swapped out in tests instead of a real file-chooser dialog.
        self.choose_directory = (
            lambda parent, caption: QtWidgets.QFileDialog.getExistingDirectory(
                parent, caption
            )
        )

        self._controller = SweepController(self)
        self._controller.log.connect(self._on_log)
        self._controller.progress.connect(self._on_progress)
        self._controller.finished.connect(self._on_finished)
        self._controller.failed.connect(self._on_failed)

        outer = QtWidgets.QHBoxLayout(self)
        outer.addWidget(self._build_editor(), 6)
        self.preview = SweepPreviewPane()
        outer.addWidget(self.preview, 4)

        self._session.configChanged.connect(lambda _p: self._refresh_export())
        self._session.configReplaced.connect(self._refresh_export)
        self._refresh_export()
        self._update_summary()

    # ------------------------------------------------------------ building UI

    def _build_editor(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        add_row = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("Add parameter...")
        self.add_button.clicked.connect(self._on_add_parameter_clicked)
        add_row.addWidget(self.add_button)
        add_row.addStretch(1)
        layout.addLayout(add_row)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Path", "Entry", "Values", "Status", ""])
        self.table.horizontalHeader().setSectionResizeMode(
            _COL_VALUES, QtWidgets.QHeaderView.Stretch
        )
        layout.addWidget(self.table, 1)

        layout.addLayout(self._build_options())

        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        layout.addLayout(self._build_action_buttons())

        self.written_path_label = QtWidgets.QLabel("")
        self.written_path_label.setWordWrap(True)
        layout.addWidget(self.written_path_label)

        self.run_panel = SweepRunPanel()
        layout.addWidget(self.run_panel, 1)

        return panel

    def _build_options(self) -> QtWidgets.QFormLayout:
        options = QtWidgets.QFormLayout()

        self.mode_combo = QtWidgets.QComboBox()
        for key, label in _COMBINE_LABELS.items():
            self.mode_combo.addItem(label, key)
        self.mode_combo.currentIndexChanged.connect(self._update_summary)
        options.addRow("Combine mode:", self.mode_combo)

        self.reps_spin = QtWidgets.QSpinBox()
        self.reps_spin.setRange(1, 1000)
        self.reps_spin.setValue(1)
        self.reps_spin.valueChanged.connect(self._update_summary)
        options.addRow("Repetitions:", self.reps_spin)

        self.base_seed_spin = QtWidgets.QSpinBox()
        self.base_seed_spin.setRange(0, 2**31 - 1)
        self.base_seed_spin.setValue(0)
        self.base_seed_spin.valueChanged.connect(self._update_summary)
        options.addRow("Base seed:", self.base_seed_spin)

        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 600000.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setValue(
            self._session.config.simulation.duration_ms or 1000.0
        )
        self.duration_spin.valueChanged.connect(self._update_summary)
        options.addRow("Duration (ms):", self.duration_spin)

        self.sweep_name_edit = QtWidgets.QLineEdit("sweep")
        self.sweep_name_edit.textChanged.connect(self._update_summary)
        options.addRow("Sweep name:", self.sweep_name_edit)

        self.concurrency_spin = QtWidgets.QSpinBox()
        self.concurrency_spin.setRange(1, 64)
        self.concurrency_spin.setValue(1)
        options.addRow("Concurrency:", self.concurrency_spin)

        return options

    def _build_action_buttons(self) -> QtWidgets.QHBoxLayout:
        buttons = QtWidgets.QHBoxLayout()

        self.write_button = QtWidgets.QPushButton("Write sweep...")
        self.write_button.clicked.connect(self.write_sweep)
        buttons.addWidget(self.write_button)

        self.run_button = QtWidgets.QPushButton("Run locally")
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self._on_run_clicked)
        buttons.addWidget(self.run_button)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._controller.cancel)
        buttons.addWidget(self.cancel_button)

        self.slurm_button = QtWidgets.QPushButton("Export SLURM script...")
        self.slurm_button.clicked.connect(self._on_export_slurm_clicked)
        buttons.addWidget(self.slurm_button)

        return buttons

    # --------------------------------------------------------------- add row

    def _on_add_parameter_clicked(self) -> None:
        dialog = ParameterPickerDialog(self._session, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        picked = dialog.selected()
        if picked is not None:
            self.add_field(*picked)

    def add_field(self, path: str, spec: Optional[ParamSpec] = None) -> int:
        """Add one sweep row for ``path``, starting on an empty list entry.

        Args:
            path: A dotted config path, normally one of
                ``sweep_paths(session.config)``.
            spec: That path's ``ParamSpec``, when known (used to decide
                whether values are rounded to ``int``).

        Returns:
            The new row's index.
        """
        row_index = self.table.rowCount()
        self.table.insertRow(row_index)

        path_item = QtWidgets.QTableWidgetItem(path)
        path_item.setFlags(path_item.flags() & ~QtCore.Qt.ItemIsEditable)
        self.table.setItem(row_index, _COL_PATH, path_item)

        mode_combo = QtWidgets.QComboBox()
        for key in FIELD_MODES:
            mode_combo.addItem(_MODE_LABELS[key], key)
        self.table.setCellWidget(row_index, _COL_MODE, mode_combo)

        values_edit = QtWidgets.QLineEdit()
        values_edit.setPlaceholderText("10, 20, 40  (or start, stop, count)")
        self.table.setCellWidget(row_index, _COL_VALUES, values_edit)

        status_label = QtWidgets.QLabel("enter values")
        self.table.setCellWidget(row_index, _COL_STATUS, status_label)

        remove_button = QtWidgets.QPushButton("Remove")
        self.table.setCellWidget(row_index, _COL_REMOVE, remove_button)

        row = {
            "path": path,
            "spec": spec,
            "mode_combo": mode_combo,
            "values_edit": values_edit,
            "status_label": status_label,
        }
        self._rows.insert(row_index, row)

        mode_combo.currentIndexChanged.connect(self._update_summary)
        values_edit.textChanged.connect(self._update_summary)
        remove_button.clicked.connect(lambda: self._remove_row(row))

        self._update_summary()
        return row_index

    def _remove_row(self, row: Dict) -> None:
        if row not in self._rows:
            return
        index = self._rows.index(row)
        self._rows.pop(index)
        self.table.removeRow(index)
        self._update_summary()

    # ------------------------------------------------------------- computing

    def _row_values(self, row: Dict) -> Tuple[Optional[List], Optional[str]]:
        mode = row["mode_combo"].currentData()
        is_int = infer_is_int(self._session, row["path"], row["spec"])
        return parse_field_values(row["values_edit"].text(), mode, is_int)

    def _valid_fields(self) -> Tuple[List[Tuple[str, List]], Optional[str]]:
        """Every row's ``(path, values)``, or the first row error found.

        Also updates each row's status label as a side effect, so the table
        always reflects the latest parse.
        """
        fields: List[Tuple[str, List]] = []
        first_error: Optional[str] = None
        for row in self._rows:
            values, error = self._row_values(row)
            if error is not None:
                row["status_label"].setText(error)
                row["status_label"].setStyleSheet("color: #D6394A;")
                if first_error is None:
                    first_error = f"{row['path']}: {error}"
            else:
                row["status_label"].setText(f"{len(values)} value(s)")
                row["status_label"].setStyleSheet("")
                fields.append((row["path"], values))
        return fields, first_error

    def base_combinations(self) -> Tuple[List[Dict], Optional[str]]:
        """The base (pre-repetition) combinations, or an error message.

        Returns:
            ``(combos, error)`` -- ``error`` is set (and ``combos`` is
            ``[]``) when a row fails to parse, or ``"zipped"`` mode is chosen
            with mismatched value-list lengths.
        """
        fields, error = self._valid_fields()
        if error is not None:
            return [], error
        if not fields:
            return [], "add at least one parameter"
        try:
            return build_combinations(fields, self.mode_combo.currentData()), None
        except ValueError as exc:
            return [], str(exc)

    def jobs(self) -> Tuple[List[Dict], Optional[str]]:
        """``base_combinations()`` expanded by repetitions, with distinct seeds."""
        combos, error = self.base_combinations()
        if error is not None:
            return [], error
        expanded = expand_repetitions(
            combos, self.reps_spin.value(), self.base_seed_spin.value()
        )
        return expanded, None

    def job_count(self) -> int:
        """How many jobs the current sweep settings describe (0 if invalid)."""
        jobs, error = self.jobs()
        return 0 if error is not None else len(jobs)

    # -------------------------------------------------------------- summary

    def _update_summary(self, *_args) -> None:
        combos, error = self.base_combinations()
        reps = self.reps_spin.value()
        can_run = error is None and bool(combos)

        if error is not None:
            self.summary_label.setText(f"<span style='color:#D6394A;'>{error}</span>")
        else:
            jobs = len(combos) * reps
            lines = [
                f"{len(combos)} combination(s) x {reps} repetition(s) "
                f"= {jobs} job(s)"
            ]
            for combo in combos[:_SUMMARY_PREVIEW_COUNT]:
                lines.append(f"  - {format_combo(combo)}")
            self.summary_label.setText("<br/>".join(lines))

        self.write_button.setEnabled(can_run)
        self.run_button.setEnabled(can_run and not self._controller.running)
        self.slurm_button.setEnabled(can_run)

        self._update_sweep_preview()

    def _update_sweep_preview(self) -> None:
        jobs, error = self.jobs()
        if error is not None or not jobs:
            self.preview.set_sweep_preview("", "")
            return
        first = jobs[0]
        try:
            cfg = preview_config(self._session.config, first)
        except ValueError as exc:
            self.preview.set_sweep_preview(f"# {exc}", "")
            return
        manifest = build_preview_manifest(self._planned_root(), jobs)
        argv = sweep_command(manifest, 0, self.duration_spin.value())
        self.preview.set_sweep_preview(cfg.to_yaml(), " ".join(argv))

    def _planned_root(self) -> Path:
        """Where a sweep *would* land if written now (may not exist yet)."""
        name = self.sweep_name_edit.text().strip() or "sweep"
        if self._session.project is not None:
            return _timestamped_root(self._session.project.root / "sweeps", name)
        return _timestamped_root(Path("<choose a directory>"), name)

    def _refresh_export(self) -> None:
        config = self._session.config
        if self._session.project is not None:
            config_path = str(self._session.project.config_path)
            bundle_dir = str(self._session.project.new_run_dir("run"))
        else:
            config_path = "config.yml"
            bundle_dir = "<bundle-dir>"
        duration = config.simulation.duration_ms or self.duration_spin.value()
        self.preview.set_export(
            config.to_yaml(), _single_run_command(config_path, duration, bundle_dir)
        )

    # -------------------------------------------------------------- writing

    def _resolve_sweep_root(self) -> Optional[Path]:
        name = self.sweep_name_edit.text().strip() or "sweep"
        if self._session.project is not None:
            return _timestamped_root(self._session.project.root / "sweeps", name)
        directory = self.choose_directory(self, "Choose a directory for the sweep")
        if not directory:
            return None
        return _timestamped_root(Path(directory), name)

    def write_sweep(self) -> Optional[SweepManifest]:
        """Write the current sweep to disk, returning its manifest.

        Returns:
            The written :class:`SweepManifest`, or ``None`` when the sweep is
            invalid or the user cancelled the directory picker (no project
            open). Never mutates :attr:`Session.config`.
        """
        jobs, error = self.jobs()
        if error is not None:
            if self.show_dialogs:
                QtWidgets.QMessageBox.warning(self, "Cannot write sweep", error)
            return None
        root = self._resolve_sweep_root()
        if root is None:
            return None
        manifest = write_combo_sweep(
            self._session.config,
            jobs,
            root=root,
            duration_ms=self.duration_spin.value(),
        )
        self._last_manifest = manifest
        self.written_path_label.setText(f"Sweep written to {manifest.root}")
        self.run_panel.reset(manifest)
        return manifest

    # ------------------------------------------------------------- run/cancel

    def _on_run_clicked(self) -> None:
        manifest = self.write_sweep()
        if manifest is None:
            return
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._controller.start(
            manifest,
            duration_ms=self.duration_spin.value(),
            parallel=self.concurrency_spin.value(),
        )

    def _on_log(self, line: str) -> None:
        self.run_panel.on_log(line)

    def _on_progress(self, done: int, total: int) -> None:
        self.run_panel.on_progress(done, total)

    def _on_finished(self, n_failed: int) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.run_panel.on_finished(n_failed)

    def _on_failed(self, message: str) -> None:
        self.run_panel.on_failed(message)

    # ---------------------------------------------------------------- slurm

    def export_slurm(self, settings: Dict) -> Optional[Path]:
        """Write a SLURM array script for the sweep, writing it first if needed.

        Args:
            settings: The dialog's settings (job_name, partition, time,
                mem_gb, cpus_per_task, gpus, conda_env).

        Returns:
            The written script path, or ``None`` if the sweep could not be
            written (see :meth:`write_sweep`).
        """
        manifest = self._last_manifest or self.write_sweep()
        if manifest is None:
            return None
        settings = dict(settings)
        settings["duration_ms"] = self.duration_spin.value()
        script_path = write_slurm_script(manifest, settings=settings)
        self.run_panel.append(f"SLURM script written to {script_path}")
        self.run_panel.append(f"sbatch {script_path}")
        return script_path

    def _on_export_slurm_clicked(self) -> None:
        dialog = SlurmSettingsDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        script_path = self.export_slurm(dialog.settings())
        if script_path is not None and self.show_dialogs:
            QtWidgets.QMessageBox.information(
                self,
                "SLURM script written",
                f"{script_path}\n\nsbatch {script_path}",
            )
