"""Population firing rate panel.

Plots mean firing rate over time for one or more populations as a line chart,
with a vertical cursor tracking the current frame.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtGui, QtWidgets
import pyqtgraph as pg  # type: ignore

from .base_panel import VisualizationPanel, VisData
from .neuron_panel import _smooth_firing_rate


class FiringRatePanel(VisualizationPanel):
    """Mean population firing rate over time with playback cursor."""

    PANEL_DISPLAY_NAME = "Firing Rate"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._selected_populations: List[str] = []
        self._bin_ms: float = 20.0
        # Pre-computed mean rates per population [T]
        self._rate_curves: Dict[str, np.ndarray] = {}
        self._curves: Dict[str, pg.PlotDataItem] = {}
        super().__init__(title="Firing Rate", parent=parent)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_content(self, layout: QtWidgets.QVBoxLayout) -> None:
        self._pw = pg.PlotWidget(background="w")
        self._pw.setMenuEnabled(False)
        self._pw.getAxis("bottom").setLabel("Time (ms)")
        self._pw.getAxis("left").setLabel("Rate (Hz)")
        self._pw.showGrid(x=True, y=True, alpha=0.2)
        self._pw.addLegend(offset=(10, 10))

        self._cursor = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen(color=(180, 180, 180), width=1.5, style=pg.QtCore.Qt.DashLine),
        )
        self._pw.addItem(self._cursor)
        layout.addWidget(self._pw)

    # ------------------------------------------------------------------
    # VisualizationPanel interface
    # ------------------------------------------------------------------

    def _on_data_set(self) -> None:
        for c in self._curves.values():
            self._pw.removeItem(c)
        self._curves.clear()
        self._rate_curves.clear()

        if self._data is None:
            return

        self._selected_populations = list(self._data.population_names)
        self._precompute_rates()
        self._draw_curves()
        self._render_frame(0)

    def _precompute_rates(self) -> None:
        """Compute smooth mean firing rate for each selected population."""
        self._rate_curves.clear()
        if self._data is None:
            return
        for name in self._selected_populations:
            res = self._data.population_results.get(name, {})
            spk = res.get("spikes")     # [T, N]
            if spk is None:
                continue
            rate = _smooth_firing_rate(spk, self._data.dt_ms, self._bin_ms)
            self._rate_curves[name] = rate.mean(axis=1)  # mean over neurons [T]

    def _draw_curves(self) -> None:
        for c in self._curves.values():
            self._pw.removeItem(c)
        self._curves.clear()
        if self._data is None:
            return
        time_ms = self._data.time_ms
        for name, rate in self._rate_curves.items():
            T = min(len(time_ms), len(rate))
            color = self._data.population_colors.get(name, QtGui.QColor(66, 135, 245))
            if isinstance(color, QtGui.QColor):
                pen_color = color.name()
            else:
                pen_color = pg.mkColor(*color)
            curve = self._pw.plot(
                time_ms[:T],
                rate[:T],
                pen=pg.mkPen(pen_color, width=2),
                name=name,
            )
            self._curves[name] = curve

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
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addWidget(QtWidgets.QLabel("Populations:"))
        if self._data:
            for name in self._data.population_names:
                chk = QtWidgets.QCheckBox(name)
                chk.setChecked(name in self._selected_populations)
                def _toggle(checked, n=name):
                    if checked and n not in self._selected_populations:
                        self._selected_populations.append(n)
                    elif not checked and n in self._selected_populations:
                        self._selected_populations.remove(n)
                    self._precompute_rates()
                    self._draw_curves()
                chk.stateChanged.connect(_toggle)
                layout.addWidget(chk)

        bin_spin = QtWidgets.QDoubleSpinBox()
        bin_spin.setRange(1.0, 200.0)
        bin_spin.setSingleStep(5.0)
        bin_spin.setValue(self._bin_ms)
        bin_spin.setSuffix(" ms")
        def _bin_changed(v):
            self._bin_ms = v
            self._precompute_rates()
            self._draw_curves()
        bin_spin.valueChanged.connect(_bin_changed)
        lbl = QtWidgets.QLabel("Bin size:")
        layout.addWidget(lbl)
        layout.addWidget(bin_spin)
        layout.addStretch()
        return w
