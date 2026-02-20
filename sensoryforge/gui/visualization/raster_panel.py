"""Spike raster panel.

Displays spikes progressively in time (animated build-up). PSTH at bottom
with adjustable time bin, also appearing through time.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg  # type: ignore

from .base_panel import VisualizationPanel, VisData

_PSTH_BIN_MS_DEFAULT = 20.0


class RasterPanel(VisualizationPanel):
    """Spike raster: rows = neurons. Spikes appear in time. PSTH below."""

    PANEL_DISPLAY_NAME = "Spike Raster"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._selected_populations: List[str] = []
        self._sort_by_rate = True
        self._raster_image: Optional[np.ndarray] = None
        self._pop_row_starts: Dict[str, int] = {}
        self._psth_bin_ms = _PSTH_BIN_MS_DEFAULT
        self._sep_lines: List[pg.InfiniteLine] = []
        self._pop_labels: List[pg.TextItem] = []
        super().__init__(title="Spike Raster", parent=parent)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_content(self, layout: QtWidgets.QVBoxLayout) -> None:
        # Raster plot
        self._pw = pg.PlotWidget(background="k")
        self._pw.setMenuEnabled(False)
        self._pw.getAxis("bottom").setLabel("Time (ms)")
        self._pw.getAxis("left").setLabel("Neuron #")
        self._pw.showGrid(x=False, y=False)

        self._img = pg.ImageItem()
        self._img.setZValue(-10)
        self._pw.addItem(self._img)

        lut = np.zeros((2, 4), dtype=np.uint8)
        lut[0] = [20, 20, 20, 255]
        lut[1] = [255, 220, 100, 255]
        self._img.setLookupTable(lut)

        self._cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("y", width=1.5))
        self._pw.addItem(self._cursor)

        layout.addWidget(self._pw)

        # PSTH plot
        self._psth_pw = pg.PlotWidget(background="k")
        self._psth_pw.setMenuEnabled(False)
        self._psth_pw.getAxis("bottom").setLabel("Time (ms)")
        self._psth_pw.getAxis("left").setLabel("Rate (Hz)")
        self._psth_pw.showGrid(x=True, y=True, alpha=0.2)
        self._psth_curves: Dict[str, pg.PlotDataItem] = {}
        layout.addWidget(self._psth_pw)

    # ------------------------------------------------------------------
    # VisualizationPanel interface
    # ------------------------------------------------------------------

    def _on_data_set(self) -> None:
        if self._data is None:
            return
        self._selected_populations = list(self._data.population_names)
        self._build_raster_image()
        self._build_psth_curves()

    def _build_raster_image(self) -> None:
        for ln in self._sep_lines:
            self._pw.removeItem(ln)
        self._sep_lines.clear()
        for lbl in self._pop_labels:
            self._pw.removeItem(lbl)
        self._pop_labels.clear()

        if self._data is None or not self._selected_populations:
            self._img.clear()
            return

        T = self._data.n_steps
        time_ms = self._data.time_ms

        blocks: List[np.ndarray] = []
        self._pop_row_starts = {}
        row = 0
        for name in self._selected_populations:
            res = self._data.population_results.get(name, {})
            spk = res.get("spikes")
            if spk is None:
                continue
            if spk.shape[0] != T:
                spk = spk[:T]
            n_neurons = spk.shape[1]
            if self._sort_by_rate:
                rates = spk.sum(axis=0)
                order = np.argsort(rates)[::-1]
                spk = spk[:, order]
            blocks.append(spk.T.astype(np.float32))
            self._pop_row_starts[name] = row
            row += n_neurons

        if not blocks:
            self._img.clear()
            return

        raster = np.concatenate(blocks, axis=0)
        self._raster_image = raster
        n_total = raster.shape[0]

        tr = pg.QtGui.QTransform()
        dt = time_ms[1] - time_ms[0] if len(time_ms) > 1 else self._data.dt_ms
        tr.translate(float(time_ms[0] if len(time_ms) else 0), 0)
        tr.scale(float(dt), 1.0)
        self._img.setTransform(tr)

        self._pw.setRange(
            xRange=[float(time_ms[0] if len(time_ms) else 0),
                    float(time_ms[-1] if len(time_ms) else T)],
            yRange=[-0.5, n_total - 0.5],
            padding=0.02,
        )

        row = 0
        for name in self._selected_populations:
            if name not in self._pop_row_starts:
                continue
            res = self._data.population_results.get(name, {})
            spk = res.get("spikes")
            if spk is None:
                continue
            n_neurons = spk.shape[1]
            color = self._data.population_colors.get(name, QtGui.QColor(200, 200, 200))
            qt_color = color if isinstance(color, QtGui.QColor) else QtGui.QColor(*color)
            if row > 0:
                sep = pg.InfiniteLine(
                    pos=row - 0.5,
                    angle=0,
                    pen=pg.mkPen(qt_color.lighter(150), width=1, style=QtCore.Qt.DashLine),
                )
                self._pw.addItem(sep)
                self._sep_lines.append(sep)
            lbl = pg.TextItem(
                text=name,
                color=qt_color.name(),
                anchor=(0, 1),
            )
            lbl.setPos(
                float(time_ms[0] if len(time_ms) else 0),
                row + n_neurons,
            )
            lbl.setFont(QtGui.QFont("sans-serif", 8))
            self._pw.addItem(lbl)
            self._pop_labels.append(lbl)
            row += n_neurons

        self._cursor.setPos(float(time_ms[0] if len(time_ms) else 0))

    def _build_psth_curves(self) -> None:
        """Create PSTH plot items per population."""
        self._psth_pw.clear()
        self._psth_curves.clear()
        if self._data is None or not self._selected_populations:
            return
        for name in self._selected_populations:
            color = self._data.population_colors.get(name, QtGui.QColor(200, 200, 200))
            qt_color = color if isinstance(color, QtGui.QColor) else QtGui.QColor(*color)
            curve = self._psth_pw.plot(
                [], [],
                pen=pg.mkPen(qt_color, width=2),
                name=name,
            )
            self._psth_curves[name] = curve
        self._psth_pw.addLegend(offset=(10, 10))

    def _compute_psth(self, name: str, t_max: int) -> tuple:
        """Return (time_centers, rates) for PSTH up to t_max."""
        if self._data is None:
            return np.array([]), np.array([])
        res = self._data.population_results.get(name, {})
        spk = res.get("spikes")
        if spk is None:
            return np.array([]), np.array([])
        spk = spk[: t_max + 1]
        dt_ms = self._data.dt_ms
        bin_steps = max(1, int(round(self._psth_bin_ms / dt_ms)))
        T = spk.shape[0]
        n_bins = (T + bin_steps - 1) // bin_steps
        rates = np.zeros(n_bins, dtype=np.float32)
        for b in range(n_bins):
            start = b * bin_steps
            end = min(start + bin_steps, T)
            if end > start:
                n_neurons = spk.shape[1]
                count = spk[start:end].sum()
                bin_dur_s = (end - start) * dt_ms * 1e-3
                rates[b] = count / (n_neurons * bin_dur_s) if bin_dur_s > 0 else 0
        t_centers = (np.arange(n_bins) + 0.5) * bin_steps * dt_ms
        return t_centers, rates

    def _render_frame(self, t_idx: int) -> None:
        if self._data is None or self._raster_image is None:
            return
        time_ms = self._data.time_ms
        if len(time_ms) <= t_idx:
            return

        # Show only spikes up to current time (crop)
        crop = self._raster_image[:, : t_idx + 1]
        dt = time_ms[1] - time_ms[0] if len(time_ms) > 1 else self._data.dt_ms
        self._img.setImage(
            crop.T,
            autoLevels=False,
            levels=(0.0, 1.0),
        )
        tr = pg.QtGui.QTransform()
        tr.translate(float(time_ms[0]), 0)
        tr.scale(float(dt), 1.0)
        self._img.setTransform(tr)

        self._cursor.setPos(float(time_ms[t_idx]))

        # PSTH up to current time
        for name, curve in self._psth_curves.items():
            t_centers, rates = self._compute_psth(name, t_idx)
            if t_centers.size > 0:
                curve.setData(t_centers, rates)

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
                    self._build_raster_image()
                    self._build_psth_curves()
                    self._render_frame(self._t_idx)
                chk.stateChanged.connect(_toggle)
                layout.addWidget(chk)

        sort_chk = QtWidgets.QCheckBox("Sort by firing rate")
        sort_chk.setChecked(self._sort_by_rate)
        def _sort_changed(s):
            self._sort_by_rate = bool(s)
            self._build_raster_image()
            self._render_frame(self._t_idx)
        sort_chk.stateChanged.connect(_sort_changed)
        layout.addWidget(sort_chk)

        bin_spin = QtWidgets.QDoubleSpinBox()
        bin_spin.setRange(1.0, 200.0)
        bin_spin.setSingleStep(5.0)
        bin_spin.setValue(self._psth_bin_ms)
        bin_spin.setSuffix(" ms")
        bin_spin.setToolTip("PSTH time bin")

        def _bin_changed(v):
            self._psth_bin_ms = v
            self._render_frame(self._t_idx)

        bin_spin.valueChanged.connect(_bin_changed)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("PSTH bin:"))
        row.addWidget(bin_spin)
        layout.addLayout(row)

        layout.addStretch()
        return w
