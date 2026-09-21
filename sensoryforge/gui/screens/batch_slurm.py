"""The "Export SLURM script..." settings dialog for the Batch screen.

Fields mirror the old ``gui/tabs/batch_tab.py::_SlurmSettingsDialog`` (job
name, partition, walltime, memory, CPUs, GPUs, conda env) so a returning user
finds the same knobs; values persist across sessions through
:func:`sensoryforge.gui.settings.gui_settings` (never a Qt settings object built directly, F-072).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt5 import QtWidgets

from sensoryforge.gui.settings import gui_settings

#: The ``gui_settings()`` key prefix every field is stored under.
SETTINGS_PREFIX = "gui/batch_screen/slurm"

_DEFAULTS: Dict[str, Any] = {
    "job_name": "sensoryforge_sweep",
    "partition": "gpu",
    "time": "04:00:00",
    "mem_gb": 32,
    "cpus_per_task": 4,
    "gpus": 1,
    "conda_env": "sensoryforge",
}


def _load_defaults() -> Dict[str, Any]:
    """The dialog's starting values: persisted settings, falling back to defaults."""
    settings = gui_settings()
    values = dict(_DEFAULTS)
    for key, default in _DEFAULTS.items():
        stored = settings.value(f"{SETTINGS_PREFIX}/{key}", default)
        if isinstance(default, int):
            try:
                values[key] = int(stored)
            except (TypeError, ValueError):
                values[key] = default
        else:
            values[key] = str(stored)
    return values


class SlurmSettingsDialog(QtWidgets.QDialog):
    """Collects the SLURM array-job settings :func:`write_slurm_script` needs.

    Args:
        parent: Qt parent.

    Example:
        >>> dialog = SlurmSettingsDialog()             # doctest: +SKIP
        >>> if dialog.exec_() == QtWidgets.QDialog.Accepted:
        ...     settings = dialog.settings()            # doctest: +SKIP
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export SLURM script")
        defaults = _load_defaults()

        form = QtWidgets.QFormLayout(self)

        self.job_name = QtWidgets.QLineEdit(defaults["job_name"])
        form.addRow("Job name:", self.job_name)

        self.partition = QtWidgets.QLineEdit(defaults["partition"])
        form.addRow("Partition:", self.partition)

        self.time = QtWidgets.QLineEdit(defaults["time"])
        form.addRow("Time (HH:MM:SS):", self.time)

        self.mem_gb = QtWidgets.QSpinBox()
        self.mem_gb.setRange(1, 4096)
        self.mem_gb.setValue(defaults["mem_gb"])
        form.addRow("Memory (GB):", self.mem_gb)

        self.cpus_per_task = QtWidgets.QSpinBox()
        self.cpus_per_task.setRange(1, 256)
        self.cpus_per_task.setValue(defaults["cpus_per_task"])
        form.addRow("CPUs per task:", self.cpus_per_task)

        self.gpus = QtWidgets.QSpinBox()
        self.gpus.setRange(0, 8)
        self.gpus.setValue(defaults["gpus"])
        form.addRow("GPUs:", self.gpus)

        self.conda_env = QtWidgets.QLineEdit(defaults["conda_env"])
        form.addRow("Conda env:", self.conda_env)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def settings(self) -> Dict[str, Any]:
        """The dialog's current values, in :func:`~...sweep_controller.write_slurm_script`'s shape.

        Also persists them via :func:`~sensoryforge.gui.settings.gui_settings`
        so the next dialog opens with what was last used.

        Returns:
            ``{"job_name", "partition", "time", "mem_gb", "cpus_per_task",
            "gpus", "conda_env"}``.
        """
        values: Dict[str, Any] = {
            "job_name": self.job_name.text().strip() or _DEFAULTS["job_name"],
            "partition": self.partition.text().strip() or _DEFAULTS["partition"],
            "time": self.time.text().strip() or _DEFAULTS["time"],
            "mem_gb": self.mem_gb.value(),
            "cpus_per_task": self.cpus_per_task.value(),
            "gpus": self.gpus.value(),
            "conda_env": self.conda_env.text().strip() or _DEFAULTS["conda_env"],
        }
        settings = gui_settings()
        for key, value in values.items():
            settings.setValue(f"{SETTINGS_PREFIX}/{key}", value)
        return values
