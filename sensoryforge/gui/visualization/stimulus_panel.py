"""Stimulus heatmap panel.

Displays the spatial pressure field at the current time step as a false-color
heatmap.  The colormap and contrast limits are user-adjustable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg  # type: ignore

from .base_panel import VisualizationPanel, VisData


class StimulusPanel(VisualizationPanel):
    """Animated heatmap of the stimulus pressure field [T, H, W]."""

    PANEL_DISPLAY_NAME = "Stimulus"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._cmap_name = "viridis"
        self._auto_levels = True
        self._global_min: float = 0.0
        self._global_max: float = 1.0
        super().__init__(title="Stimulus", parent=parent)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_content(self, layout: QtWidgets.QVBoxLayout) -> None:
        self._pw = pg.PlotWidget(background="w")
        self._pw.setAspectLocked(True)
        self._pw.showGrid(x=False, y=False)
        self._pw.getAxis("bottom").setLabel("X (mm)")
        self._pw.getAxis("left").setLabel("Y (mm)")
        self._pw.setMenuEnabled(False)

        self._img = pg.ImageItem()
        self._pw.addItem(self._img)

        # Colormap
        self._set_colormap(self._cmap_name)

        # Colorbar (optional — API varies by pyqtgraph version)
        try:
            self._colorbar = pg.ColorBarItem(
                interactive=False,
                colorMap=pg.colormap.get(self._cmap_name, source="matplotlib"),
            )
            self._colorbar.setImageItem(self._img, insert_in=self._pw.getPlotItem())
        except Exception:
            self._colorbar = None

        layout.addWidget(self._pw)
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self._img.clear()

    # ------------------------------------------------------------------
    # VisualizationPanel interface
    # ------------------------------------------------------------------

    def _on_data_set(self) -> None:
        if self._data is None or self._data.stimulus_frames is None:
            self._show_placeholder()
            return
        frames = self._data.stimulus_frames       # [T, H, W]
        self._global_min = float(frames.min())
        self._global_max = float(frames.max()) or 1.0
        # Set spatial transform so image axes match mm
        # stimulus_frames [T, H, W]: H = n_x (cols), W = n_y (rows); dim0=x, dim1=y
        xl, xr = self._data.stimulus_xlim
        yb, yt = self._data.stimulus_ylim
        n_x, n_y = frames.shape[1], frames.shape[2]
        dx = (xr - xl) / max(n_x, 1)
        dy = (yt - yb) / max(n_y, 1)
        tr = pg.QtGui.QTransform()
        tr.translate(xl, yb)
        tr.scale(dx, dy)
        self._img.setTransform(tr)
        self._pw.setRange(
            xRange=[xl, xr],
            yRange=[yb, yt],
            padding=0.05,
        )
        if self._colorbar is not None:
            self._colorbar.setLevels((self._global_min, self._global_max))
        self._render_frame(0)

    def _render_frame(self, t_idx: int) -> None:
        if self._data is None or self._data.stimulus_frames is None:
            return
        frame = self._data.stimulus_frames[t_idx]    # [n_x, n_y] = (x, y)
        self._img.setImage(
            frame,
            autoLevels=False,
            levels=(self._global_min, self._global_max),
        )

    # ------------------------------------------------------------------
    # Settings widget
    # ------------------------------------------------------------------

    def build_settings_widget(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(w)
        form.setContentsMargins(8, 8, 8, 8)

        cmap_cmb = QtWidgets.QComboBox()
        cmap_cmb.addItems(["viridis", "plasma", "inferno", "magma", "hot", "jet", "grey"])
        cmap_cmb.setCurrentText(self._cmap_name)
        cmap_cmb.currentTextChanged.connect(self._set_colormap)
        form.addRow("Colormap:", cmap_cmb)

        self._auto_chk = QtWidgets.QCheckBox("Auto levels")
        self._auto_chk.setChecked(self._auto_levels)
        self._auto_chk.stateChanged.connect(
            lambda s: setattr(self, "_auto_levels", bool(s))
        )
        form.addRow(self._auto_chk)
        return w

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_colormap(self, name: str) -> None:
        self._cmap_name = name
        try:
            lut = pg.colormap.get(name, source="matplotlib").getLookupTable(
                nPts=256, alpha=False
            )
            self._img.setLookupTable(lut)
        except Exception:
            pass
