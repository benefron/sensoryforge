"""Membrane voltage panel.

Shows the full voltage trace for a single selected neuron and a raster dot
marker at the current time step.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5 import QtGui, QtWidgets
import pyqtgraph as pg  # type: ignore

from .base_panel import VisualizationPanel, VisData


class VoltagePanel(VisualizationPanel):
    """Membrane voltage trace for one neuron with playback cursor."""

    PANEL_DISPLAY_NAME = "Voltage Trace"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._pop_name: Optional[str] = None
        self._neuron_idx: int = 0
        self._show_drive: bool = True
        super().__init__(title="Voltage Trace", parent=parent)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_content(self, layout: QtWidgets.QVBoxLayout) -> None:
        self._pw = pg.PlotWidget(background="w")
        self._pw.setMenuEnabled(False)
        self._pw.getAxis("bottom").setLabel("Time (ms)")
        self._pw.getAxis("left").setLabel("V (mV)")
        self._pw.showGrid(x=True, y=True, alpha=0.2)

        self._v_curve = self._pw.plot(
            pen=pg.mkPen("#4287f5", width=1.5),
            name="Voltage",
        )
        self._drive_curve = self._pw.plot(
            pen=pg.mkPen("#f5a442", width=1.2, style=pg.QtCore.Qt.DashLine),
            name="Drive",
        )
        self._cursor = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen(color=(200, 50, 50), width=1.5),
        )
        self._pw.addItem(self._cursor)

        # Spike tick marks
        self._spike_scatter = pg.ScatterPlotItem(
            size=6,
            brush=pg.mkBrush(220, 80, 80, 200),
            pen=None,
        )
        self._pw.addItem(self._spike_scatter)

        self._pw.addLegend(offset=(10, 10))
        layout.addWidget(self._pw)

    # ------------------------------------------------------------------
    # VisualizationPanel interface
    # ------------------------------------------------------------------

    def _on_data_set(self) -> None:
        if self._data is None:
            self._v_curve.setData([], [])
            self._drive_curve.setData([], [])
            return
        if self._pop_name not in self._data.population_names:
            self._pop_name = (
                self._data.population_names[0]
                if self._data.population_names else None
            )
        self._redraw_full_trace()
        self._render_frame(0)

    def _redraw_full_trace(self) -> None:
        self._v_curve.setData([], [])
        self._drive_curve.setData([], [])
        self._spike_scatter.setData([], [])
        if self._data is None or self._pop_name is None:
            return
        res = self._data.population_results.get(self._pop_name, {})
        v_trace = res.get("v_trace")    # [T, N]
        drive   = res.get("drive")      # [T, N]
        spikes  = res.get("spikes")     # [T, N]
        time_ms = self._data.time_ms

        n_neurons = v_trace.shape[1] if v_trace is not None else 0
        idx = min(self._neuron_idx, max(0, n_neurons - 1))

        if v_trace is not None:
            T = min(len(time_ms), v_trace.shape[0])
            self._v_curve.setData(time_ms[:T], v_trace[:T, idx])

        if self._show_drive and drive is not None:
            T = min(len(time_ms), drive.shape[0])
            # Scale drive to voltage range for visual overlay
            d = drive[:T, idx]
            v_min = float(v_trace[:T, idx].min()) if v_trace is not None else 0.0
            v_max = float(v_trace[:T, idx].max()) if v_trace is not None else 1.0
            d_min, d_max = float(d.min()), float(d.max())
            span_d = d_max - d_min or 1.0
            d_scaled = (d - d_min) / span_d * (v_max - v_min) * 0.3 + v_min
            self._drive_curve.setData(time_ms[:T], d_scaled)

        if spikes is not None:
            T = min(len(time_ms), spikes.shape[0])
            spike_times = time_ms[:T][spikes[:T, idx] > 0.5]
            if v_trace is not None:
                v_at_spikes = v_trace[:T, idx][spikes[:T, idx] > 0.5]
            else:
                v_at_spikes = np.ones(len(spike_times)) * 0.0
            self._spike_scatter.setData(
                x=spike_times.tolist(),
                y=v_at_spikes.tolist(),
            )

        if time_ms.size:
            self._cursor.setPos(float(time_ms[0]))

    def _render_frame(self, t_idx: int) -> None:
        if self._data is None or not self._data.time_ms.size:
            return
        t_ms = float(self._data.time_ms[min(t_idx, len(self._data.time_ms) - 1)])
        self._cursor.setPos(t_ms)

    # ------------------------------------------------------------------
    # Settings widget
    # ------------------------------------------------------------------

    def build_settings_widget(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        form.setContentsMargins(8, 8, 8, 8)

        if self._data:
            pop_cmb = QtWidgets.QComboBox()
            pop_cmb.addItems(self._data.population_names)
            if self._pop_name:
                pop_cmb.setCurrentText(self._pop_name)
            def _pop_changed(name):
                self._pop_name = name
                self._neuron_idx = 0
                self._redraw_full_trace()
            pop_cmb.currentTextChanged.connect(_pop_changed)
            form.addRow("Population:", pop_cmb)

            n_neurons = 0
            if self._pop_name:
                res = self._data.population_results.get(self._pop_name, {})
                v = res.get("v_trace")
                n_neurons = v.shape[1] if v is not None else 0

            neuron_spin = QtWidgets.QSpinBox()
            neuron_spin.setRange(0, max(0, n_neurons - 1))
            neuron_spin.setValue(self._neuron_idx)
            def _neuron_changed(idx):
                self._neuron_idx = idx
                self._redraw_full_trace()
            neuron_spin.valueChanged.connect(_neuron_changed)
            form.addRow("Neuron #:", neuron_spin)

        drive_chk = QtWidgets.QCheckBox("Overlay drive (scaled)")
        drive_chk.setChecked(self._show_drive)
        def _drive_changed(s):
            self._show_drive = bool(s)
            self._redraw_full_trace()
        drive_chk.stateChanged.connect(_drive_changed)
        form.addRow(drive_chk)
        return w
