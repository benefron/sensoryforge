"""The bottom run bar: device, duration, dt, seed, Run/Cancel, progress.

``RunBar`` is written against the ``RunController`` names from section 7 of
the phase 1 interface contract (``started``, ``progress(int, int, str)``,
``finished(object)``, ``failed(str)``, ``cancelled``,
``run(duration_ms=..., bundle=..., quick_population=...)``, ``cancel()``,
``running``) rather than importing ``execution.run_controller`` directly, so
it can be built and tested before that lane lands -- ``set_controller`` binds
to a controller (or, in tests, a small stub with the same signals).
"""

from __future__ import annotations

from functools import partial
from typing import Optional

from PyQt5 import QtCore, QtWidgets

from sensoryforge.gui import theme
from sensoryforge.config.defaults import resolve_duration_ms
from sensoryforge.gui.session import Session

#: Placement of a "no seed" spinbox value, since ``QSpinBox`` has no notion
#: of ``None``.
NO_SEED_VALUE = -1


class RunBar(QtWidgets.QWidget):
    """Device, run parameters, Run/Cancel, progress -- bound to a ``Session``.

    Args:
        session: The experiment this bar runs.
        parent: Qt parent.

    Example:
        >>> bar = RunBar(session)                  # doctest: +SKIP
        >>> bar.set_controller(run_controller)      # doctest: +SKIP
    """

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._controller = None
        #: Guard so tests can disable ``QMessageBox.critical`` on failure.
        self.show_dialogs = True

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        layout.addWidget(QtWidgets.QLabel("Device"))
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(session.available_devices)
        self.device_combo.setCurrentText(session.device)
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        layout.addWidget(self.device_combo)

        layout.addWidget(QtWidgets.QLabel("Duration (ms)"))
        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 600000.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setValue(
            resolve_duration_ms(session.config.simulation.duration_ms)
        )
        # The duration is part of the experiment: written into the config, so
        # a saved project, an exported YAML and `sensoryforge run` on it all
        # run the length shown here.
        self.duration_spin.editingFinished.connect(self._on_duration_edited)
        layout.addWidget(self.duration_spin)

        layout.addWidget(QtWidgets.QLabel("dt (ms)"))
        self.dt_spin = QtWidgets.QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 100.0)
        self.dt_spin.setDecimals(4)
        self.dt_spin.setValue(session.config.simulation.dt_ms)
        self.dt_spin.editingFinished.connect(self._on_dt_edited)
        layout.addWidget(self.dt_spin)

        layout.addWidget(QtWidgets.QLabel("Seed"))
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(NO_SEED_VALUE, 2**31 - 1)
        self.seed_spin.setSpecialValueText("none")
        self.seed_spin.setValue(self._seed_value())
        self.seed_spin.editingFinished.connect(self._on_seed_edited)
        layout.addWidget(self.seed_spin)

        self.run_button = QtWidgets.QPushButton("Run")
        self.run_button.setObjectName("Primary")
        self.run_button.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_button)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar, 1)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        layout.addWidget(self.cancel_button)

        self.bundle_label = QtWidgets.QLabel("")
        layout.addWidget(self.bundle_label)

        # Why Run is unavailable: the first problem that would stop the
        # engine building this config (Session.errors), all of them on hover.
        self.problem_label = QtWidgets.QLabel("")
        self.problem_label.setObjectName("ProblemLabel")
        self.problem_label.setStyleSheet(f"color: {theme.PALETTE['error']};")
        self.problem_label.setVisible(False)
        layout.addWidget(self.problem_label, 1)
        self._running = False

        session.deviceChanged.connect(self._on_session_device_changed)
        session.configChanged.connect(self._on_config_changed)
        session.configReplaced.connect(self._refresh_from_config)
        session.validationChanged.connect(self._update_run_available)
        self._update_run_available()

    # -------------------------------------------------------------- controller

    def set_controller(self, controller: QtCore.QObject) -> None:
        """Bind this bar to a run controller (or a test stub with the names).

        Args:
            controller: An object exposing ``started``, ``progress(int, int,
                str)``, ``finished(object)``, ``failed(str)``, ``cancelled``
                signals, ``run(duration_ms=..., bundle=..., quick_population=...)``
                and ``cancel()`` methods, and a ``running`` property.
        """
        self._controller = controller
        controller.started.connect(self._on_started)
        controller.progress.connect(self._on_progress)
        controller.finished.connect(self._on_finished)
        controller.failed.connect(self._on_failed)
        controller.cancelled.connect(self._on_cancelled)

    # --------------------------------------------------------------- bindings

    def _refresh_from_config(self) -> None:
        """Show the run settings of the config now in the session."""
        simulation = self._session.config.simulation
        for spin, value in (
            (self.duration_spin, resolve_duration_ms(simulation.duration_ms)),
            (self.dt_spin, simulation.dt_ms),
            (self.seed_spin, self._seed_value()),
        ):
            if spin.value() != value:
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)

    def _on_config_changed(self, path: str) -> None:
        if path in ("", "simulation") or path.startswith("simulation."):
            self._refresh_from_config()

    def _seed_value(self) -> int:
        seed = self._session.config.simulation.seed
        return NO_SEED_VALUE if seed is None else int(seed)

    def _on_device_changed(self, device: str) -> None:
        if device and device != self._session.device:
            self._session.set_device(device)

    def _on_session_device_changed(self, device: str) -> None:
        if self.device_combo.currentText() != device:
            self.device_combo.blockSignals(True)
            self.device_combo.setCurrentText(device)
            self.device_combo.blockSignals(False)

    def _on_duration_edited(self) -> None:
        value = self.duration_spin.value()
        if self._session.config.simulation.duration_ms != value:
            self._session.set_by_path("simulation.duration_ms", value)

    def _on_dt_edited(self) -> None:
        self._session.set_by_path("simulation.dt_ms", self.dt_spin.value())

    def _on_seed_edited(self) -> None:
        value = self.seed_spin.value()
        seed = None if value == NO_SEED_VALUE else value
        self._session.set_by_path("simulation.seed", seed)

    # -------------------------------------------------------------- run/cancel

    def _update_run_available(self, *_args: object) -> None:
        """Enable Run only when idle and the config would build."""
        errors = self._session.errors
        messages = [errors[key] for key in sorted(errors)]
        if messages:
            first = messages[0].splitlines()[0]
            more = f" (+{len(messages) - 1} more)" if len(messages) > 1 else ""
            self.problem_label.setText(f"Cannot run: {first}{more}")
            self.problem_label.setToolTip("\n\n".join(messages))
            self.run_button.setToolTip("\n\n".join(messages))
        else:
            self.problem_label.setText("")
            self.problem_label.setToolTip("")
            self.run_button.setToolTip("")
        self.problem_label.setVisible(bool(messages))
        self.run_button.setEnabled(not self._running and not messages)

    def _on_run_clicked(self) -> None:
        if self._controller is None or self._session.errors:
            return
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._controller.run(
            duration_ms=self.duration_spin.value(),
            bundle=self._session.project is not None,
        )

    def _on_cancel_clicked(self) -> None:
        if self._controller is None:
            return
        self._controller.cancel()

    # ----------------------------------------------------------- controller cb

    def _on_started(self) -> None:
        self._running = True
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("running...")
        self.bundle_label.setText("")

    def _on_progress(self, index: int, total: int, name: str) -> None:
        if total > 0:
            self.progress_bar.setValue(int(100 * (index + 1) / total))
        self.progress_bar.setFormat(f"{name} ({index + 1}/{total})")

    def _on_finished(self, result: object) -> None:
        self._running = False
        self._update_run_available()
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("done")
        bundle_dir = getattr(result, "bundle_dir", None)
        self.bundle_label.setText(
            str(bundle_dir) if bundle_dir else "not saved (no project)"
        )

    def _on_failed(self, message: str) -> None:
        self._running = False
        self._update_run_available()
        self.cancel_button.setEnabled(False)
        self.progress_bar.setFormat("failed")
        if self.show_dialogs:
            QtWidgets.QMessageBox.critical(self, "Run failed", message)

    def _on_cancelled(self) -> None:
        self._running = False
        self._update_run_available()
        self.cancel_button.setEnabled(False)
        self.progress_bar.setFormat("cancelled")
