"""The DSL neuron-model editor (Task 2.4).

A dialog over one population's ``dsl_config``
(:class:`~sensoryforge.neurons.model_dsl.NeuronModel`'s equations/threshold/
reset/parameters/state_vars). **Compile** builds a real
:class:`~sensoryforge.neurons.model_dsl.NeuronModel` and compiles it -- the
same construction
:class:`~sensoryforge.core.simulation_engine.SimulationEngine` performs for a
DSL population -- showing the error text inline on failure and, on success, a
"spiking" (has a threshold) or "analog" (no threshold) badge plus a 200 ms
step-current trace. **OK** writes ``dsl_config`` through the session only if
the equations compiled since the last edit; it also resets ``readout`` back
to ``"auto"`` when the compiled spiking/analog-ness would make the
population's current explicit ``readout`` choice invalid
(:class:`~sensoryforge.core.simulation_engine.SimulationEngine._build_populations`
raises on that combination otherwise).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PyQt5 import QtWidgets

from sensoryforge.gui import theme
from sensoryforge.gui.bench import find_population, population_index
from sensoryforge.gui.session import Session
from sensoryforge.gui.widgets import plot_factory
from sensoryforge.neurons.model_dsl import NeuronModel

#: Trace preview duration and step-current amplitude/onset.
_TRACE_DURATION_MS = 200.0
_TRACE_AMPLITUDE = 1.0
_TRACE_ONSET_FRACTION = 0.2

#: A minimal leaky-integrator starting point for a brand-new population.
_DEFAULT_EQUATIONS = "dv/dt = -v + I"


class _KeyValueTable(QtWidgets.QWidget):
    """A tiny editable key -> float table, with add/remove row buttons."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["name", "value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("+ row")
        add_btn.clicked.connect(self._add_row)
        remove_btn = QtWidgets.QPushButton("- row")
        remove_btn.clicked.connect(self._remove_selected_row)
        buttons.addWidget(add_btn)
        buttons.addWidget(remove_btn)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    def _add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem("0.0"))

    def _remove_selected_row(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def load(self, values: Dict[str, float]) -> None:
        self.table.setRowCount(0)
        for name, value in values.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(name)))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))

    def values(self) -> Dict[str, float]:
        """Read the table as ``{name: float(value)}``.

        Raises:
            ValueError: If a name is blank or repeated, or a value does not
                parse as a float -- named with the offending row.
        """
        result: Dict[str, float] = {}
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            value_item = self.table.item(row, 1)
            name = name_item.text().strip() if name_item else ""
            if not name:
                continue
            if name in result:
                raise ValueError(f"row {row + 1}: duplicate name {name!r}")
            text = value_item.text().strip() if value_item else "0"
            try:
                result[name] = float(text)
            except ValueError:
                raise ValueError(
                    f"row {row + 1} ({name!r}): {text!r} is not a number"
                ) from None
        return result


class DslEditorDialog(QtWidgets.QDialog):
    """Edit and compile one population's ``dsl_config``.

    Args:
        session: The experiment ``population_name`` belongs to.
        population_name: The population being edited.
        parent: Qt parent.

    Raises:
        ValueError: If no population named ``population_name`` exists in
            ``session.config``.
    """

    def __init__(
        self,
        session: Session,
        population_name: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        pop_cfg = find_population(session.config, population_name)
        if pop_cfg is None:
            raise ValueError(f"no population named {population_name!r}")
        self._session = session
        self._population_name = population_name
        self._compiled_ok = False
        self._compiled_config: Optional[Dict[str, Any]] = None

        self.setWindowTitle(f"Edit equations — {population_name}")
        self.resize(560, 640)
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("Equations (one 'dX/dt = ...' per line):"))
        self.eq_edit = QtWidgets.QPlainTextEdit()
        self.eq_edit.setPlaceholderText(_DEFAULT_EQUATIONS)
        layout.addWidget(self.eq_edit)

        form = QtWidgets.QFormLayout()
        self.threshold_edit = QtWidgets.QLineEdit()
        self.threshold_edit.setPlaceholderText(
            "e.g. v >= 30  (blank = analog, no spikes)"
        )
        form.addRow("Threshold:", self.threshold_edit)
        self.reset_edit = QtWidgets.QLineEdit()
        self.reset_edit.setPlaceholderText("e.g. v = -65")
        form.addRow("Reset:", self.reset_edit)
        layout.addLayout(form)

        tables = QtWidgets.QHBoxLayout()
        param_box = QtWidgets.QGroupBox("Parameters")
        param_layout = QtWidgets.QVBoxLayout(param_box)
        self.param_table = _KeyValueTable()
        param_layout.addWidget(self.param_table)
        tables.addWidget(param_box)

        state_box = QtWidgets.QGroupBox("State variables")
        state_layout = QtWidgets.QVBoxLayout(state_box)
        self.state_table = _KeyValueTable()
        state_layout.addWidget(self.state_table)
        tables.addWidget(state_box)
        layout.addLayout(tables)

        compile_row = QtWidgets.QHBoxLayout()
        self.compile_btn = QtWidgets.QPushButton("Compile")
        self.compile_btn.clicked.connect(self._on_compile)
        compile_row.addWidget(self.compile_btn)
        self.badge = QtWidgets.QLabel("")
        self.badge.setObjectName("SectionTitle")
        compile_row.addWidget(self.badge)
        compile_row.addStretch(1)
        layout.addLayout(compile_row)

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setStyleSheet(f"color: {theme.PALETTE['error']};")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        self.trace_plot = plot_factory.make_plot(
            "Step response (preview)", "Time", "State", x_unit="ms"
        )
        self.trace_curve = self.trace_plot.plot(pen=theme.pen(theme.PALETTE["accent"]))
        layout.addWidget(self.trace_plot, 1)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self._on_ok)
        # Any edit after a compile invalidates it: OK must save what was
        # compiled, and what was compiled must be what is on screen.
        self.eq_edit.textChanged.connect(self._on_edited)
        self.threshold_edit.textChanged.connect(self._on_edited)
        self.reset_edit.textChanged.connect(self._on_edited)
        for kv in (self.param_table, self.state_table):
            kv.table.itemChanged.connect(self._on_edited)
            kv.table.model().rowsInserted.connect(self._on_edited)
            kv.table.model().rowsRemoved.connect(self._on_edited)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._load(pop_cfg.dsl_config or {})
        self._set_ok_enabled(False)

    # -------------------------------------------------------------- loading

    def _load(self, dsl_config: Dict[str, Any]) -> None:
        self.eq_edit.setPlainText(dsl_config.get("equations") or _DEFAULT_EQUATIONS)
        self.threshold_edit.setText(dsl_config.get("threshold") or "")
        self.reset_edit.setText(dsl_config.get("reset") or "")
        self.param_table.load(dict(dsl_config.get("parameters") or {}))
        self.state_table.load(dict(dsl_config.get("state_vars") or {}))

    def _collect_config(self) -> Dict[str, Any]:
        """Read the form into a ``dsl_config`` dict.

        Raises:
            ValueError: If a parameter/state-var table row is malformed.
        """
        equations = self.eq_edit.toPlainText()
        threshold = self.threshold_edit.text().strip() or None
        reset = self.reset_edit.text().strip() or None
        parameters = self.param_table.values()
        state_vars = self.state_table.values()
        return {
            "equations": equations,
            "threshold": threshold,
            "reset": reset,
            "parameters": parameters,
            "state_vars": state_vars,
        }

    def _on_edited(self, *_args: object) -> None:
        if self._compiled_ok:
            self._compiled_ok = False
            self._compiled_config = None
            self._set_ok_enabled(False)
            self.badge.setText("edited: compile again")

    def _set_ok_enabled(self, enabled: bool) -> None:
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(enabled)

    # ------------------------------------------------------------- compile

    def _on_compile(self) -> None:
        self._compiled_ok = False
        self._compiled_config = None
        self._set_ok_enabled(False)
        try:
            config_dict = self._collect_config()
            model = NeuronModel.from_config(config_dict)
            integrate_dt_ms = self._session.config.simulation.integrate_dt_ms
            compiled = model.compile(dt=integrate_dt_ms, device="cpu", noise_std=0.0)
        except (ValueError, KeyError, TypeError) as exc:
            self.error_label.setText(str(exc))
            self.error_label.setVisible(True)
            self.badge.setText("")
            self.trace_curve.setData([], [])
            return

        self.error_label.setVisible(False)
        self._compiled_ok = True
        self._compiled_config = config_dict
        has_threshold = model.threshold_str is not None
        self.badge.setText("spiking" if has_threshold else "analog")
        self._set_ok_enabled(True)
        self._plot_trace(compiled, integrate_dt_ms)

    def _plot_trace(self, compiled_module: Any, integrate_dt_ms: float) -> None:
        n_steps = max(1, int(round(_TRACE_DURATION_MS / integrate_dt_ms)))
        onset = max(1, int(n_steps * _TRACE_ONSET_FRACTION))
        current = torch.zeros(1, n_steps, 1)
        current[:, onset:, :] = _TRACE_AMPLITUDE
        output = compiled_module(current)
        trace = output[0] if isinstance(output, tuple) else output
        trace_np = trace.detach().cpu().numpy().reshape(trace.shape[1])
        time_ms = np.arange(trace_np.shape[0], dtype=np.float64) * integrate_dt_ms
        self.trace_curve.setData(time_ms, trace_np)

    # ------------------------------------------------------------------- ok

    def _on_ok(self) -> None:
        if not self._compiled_ok or self._compiled_config is None:
            # Defence in depth: the Ok button is disabled in this state, but
            # a caller invoking accept()/this slot directly must not write a
            # config that never compiled.
            return
        pop_cfg = find_population(self._session.config, self._population_name)
        if pop_cfg is None:
            self.reject()
            return
        index = population_index(self._session.config, self._population_name)
        has_threshold = self._compiled_config.get("threshold") is not None
        self._session.set_by_path(
            f"populations.{index}.dsl_config", self._compiled_config
        )
        # SimulationEngine._build_populations raises if readout='spiking'
        # names a threshold-less dsl_config, or 'analog' one with a
        # threshold; a forced choice that the new equations invalidate
        # falls back to 'auto', which always follows the model itself.
        current_readout = (pop_cfg.readout or "auto").lower()
        if current_readout == "spiking" and not has_threshold:
            self._session.set_by_path(f"populations.{index}.readout", "auto")
        elif current_readout == "analog" and has_threshold:
            self._session.set_by_path(f"populations.{index}.readout", "auto")
        self.accept()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        plot_factory.teardown(self.trace_plot)
        super().closeEvent(event)
