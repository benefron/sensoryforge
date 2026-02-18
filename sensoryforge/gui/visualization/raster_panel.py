"""Spike raster panel.

Displays spikes as a 2-D raster image [neuron × time] with a moving cursor
showing the current time step.  Each population gets its own row block.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg  # type: ignore

from .base_panel import VisualizationPanel, VisData


class RasterPanel(VisualizationPanel):
    """Spike raster: rows = neurons, columns = time.  Vertical cursor tracks t."""

    PANEL_DISPLAY_NAME = "Spike Raster"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._selected_populations: List[str] = []
        self._sort_by_rate = False
        self._raster_image: Optional[np.ndarray] = None  # [N_total, T] float32
        self._row_labels: List[str] = []                  # population per row block
        self._pop_row_starts: dict = {}
        super().__init__(title="Spike Raster", parent=parent)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_content(self, layout: QtWidgets.QVBoxLayout) -> None:
        self._pw = pg.PlotWidget(background="k")
        self._pw.setMenuEnabled(False)
        self._pw.getAxis("bottom").setLabel("Time (ms)")
        self._pw.getAxis("left").setLabel("Neuron #")
        self._pw.showGrid(x=False, y=False)

        self._img = pg.ImageItem()
        self._img.setZValue(-10)
        self._pw.addItem(self._img)

        # Black-and-color LUT: background black, spikes use population color
        # Default single-population LUT (white spikes on black)
        lut = np.zeros((2, 4), dtype=np.uint8)
        lut[0] = [20, 20, 20, 255]    # no spike — near-black
        lut[1] = [255, 255, 255, 255] # spike — white
        self._img.setLookupTable(lut)

        # Vertical time cursor
        self._cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("y", width=1.5))
        self._pw.addItem(self._cursor)

        # Population separator lines (added dynamically)
        self._sep_lines: List[pg.InfiniteLine] = []

        # Population label text items
        self._pop_labels: List[pg.TextItem] = []

        layout.addWidget(self._pw)

    # ------------------------------------------------------------------
    # VisualizationPanel interface
    # ------------------------------------------------------------------

    def _on_data_set(self) -> None:
        if self._data is None:
            return
        self._selected_populations = list(self._data.population_names)
        self._build_raster_image()

    def _build_raster_image(self) -> None:
        """Assemble combined raster image from selected populations."""
        # Remove old separator lines and labels
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
            blocks.append(spk.T.astype(np.float32))  # [N, T]
            self._pop_row_starts[name] = row
            row += n_neurons

        if not blocks:
            self._img.clear()
            return

        raster = np.concatenate(blocks, axis=0)   # [N_total, T]
        self._raster_image = raster
        n_total = raster.shape[0]

        # Set image: pg.ImageItem with [W, H] convention → transpose to [T, N]
        self._img.setImage(
            raster.T,
            autoLevels=False,
            levels=(0.0, 1.0),
        )

        # Set spatial transform: x = time_ms, y = neuron index
        tr = pg.QtGui.QTransform()
        if len(time_ms) > 1:
            dt = time_ms[1] - time_ms[0]
        else:
            dt = self._data.dt_ms
        tr.translate(float(time_ms[0] if len(time_ms) else 0), 0)
        tr.scale(float(dt), 1.0)
        self._img.setTransform(tr)

        self._pw.setRange(
            xRange=[float(time_ms[0] if len(time_ms) else 0),
                    float(time_ms[-1] if len(time_ms) else T)],
            yRange=[-0.5, n_total - 0.5],
            padding=0.02,
        )

        # Draw population separators and labels
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
            # Separator line at top of block (except first)
            if row > 0:
                sep = pg.InfiniteLine(
                    pos=row - 0.5,
                    angle=0,
                    pen=pg.mkPen(qt_color.lighter(150), width=1, style=QtCore.Qt.DashLine),
                )
                self._pw.addItem(sep)
                self._sep_lines.append(sep)
            # Population label
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

        # Build multi-population LUT (color-coded by population)
        self._build_lut()

        self._cursor.setPos(float(time_ms[0] if len(time_ms) else 0))

    def _build_lut(self) -> None:
        """Single-population: white spikes.  Multi-population: keep white for now."""
        lut = np.zeros((2, 4), dtype=np.uint8)
        lut[0] = [20, 20, 20, 255]
        lut[1] = [255, 220, 100, 255]  # warm yellow spikes
        self._img.setLookupTable(lut)

    def _render_frame(self, t_idx: int) -> None:
        if self._data is None:
            return
        if len(self._data.time_ms) > t_idx:
            self._cursor.setPos(float(self._data.time_ms[t_idx]))

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
                chk.stateChanged.connect(_toggle)
                layout.addWidget(chk)

        sort_chk = QtWidgets.QCheckBox("Sort by firing rate")
        sort_chk.setChecked(self._sort_by_rate)
        def _sort_changed(s):
            self._sort_by_rate = bool(s)
            self._build_raster_image()
        sort_chk.stateChanged.connect(_sort_changed)
        layout.addWidget(sort_chk)
        layout.addStretch()
        return w
