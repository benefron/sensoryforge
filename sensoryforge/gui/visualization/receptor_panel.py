"""Receptor drive scatter panel.

Shows each receptor position as a dot whose color encodes the instantaneous
weighted drive delivered by the selected population at time t.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg  # type: ignore

from .base_panel import VisualizationPanel, VisData


class ReceptorPanel(VisualizationPanel):
    """Spatial scatter of receptor positions colored by drive at time t."""

    PANEL_DISPLAY_NAME = "Receptor Drive"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._pop_name: Optional[str] = None
        self._cmap_name = "plasma"
        self._log_scale = False
        self._lut: Optional[np.ndarray] = None  # [256, 4] uint8
        # Pre-computed per-receptor aggregate drive [T, M]
        self._receptor_drive: Optional[np.ndarray] = None
        self._global_min: float = 0.0
        self._global_max: float = 1.0
        super().__init__(title="Receptor Drive", parent=parent)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_content(self, layout: QtWidgets.QVBoxLayout) -> None:
        self._pw = pg.PlotWidget(background="w")
        self._pw.setAspectLocked(True)
        self._pw.setMenuEnabled(False)
        self._pw.getAxis("bottom").setLabel("X (mm)")
        self._pw.getAxis("left").setLabel("Y (mm)")

        self._scatter = pg.ScatterPlotItem(pxMode=True)
        self._scatter.setZValue(5)
        self._pw.addItem(self._scatter)

        # Color scale bar (manual, lightweight)
        self._cbar_label = QtWidgets.QLabel()
        self._cbar_label.setAlignment(QtCore.Qt.AlignCenter)
        self._cbar_label.setStyleSheet("font-size: 10px; color: #555;")

        layout.addWidget(self._pw)
        layout.addWidget(self._cbar_label)

        self._rebuild_lut()

    # ------------------------------------------------------------------
    # VisualizationPanel interface
    # ------------------------------------------------------------------

    def _on_data_set(self) -> None:
        if self._data is None:
            self._scatter.clear()
            return
        # Default to first population
        if self._pop_name not in self._data.population_names:
            self._pop_name = (
                self._data.population_names[0]
                if self._data.population_names else None
            )
        self._build_receptor_drive()
        self._set_view_range()
        self._render_frame(0)

    def _build_receptor_drive(self) -> None:
        """Aggregate drive onto each receptor: sum over neurons weighted by innervation.

        For now we use population ``drive`` [T, N_neurons] directly — the drive
        already reflects the innervation-weighted sum at each time step.
        For spatial display we need per-receptor values.  We store the mean
        drive across all neurons per time step (a reasonable proxy when
        receptor-level drive isn't separately exported).
        """
        self._receptor_drive = None
        if self._data is None or self._pop_name is None:
            return
        res = self._data.population_results.get(self._pop_name, {})
        drive = res.get("drive")    # [T, N_neurons]
        if drive is None:
            return

        coords = self._data.receptor_positions  # [M, 2]
        if coords is None or coords.shape[0] == 0:
            return

        # Map: for each receptor m, its drive = mean of all neurons' drive
        # (approximation; exact would need innervation weights per receptor)
        T = drive.shape[0]
        # drive [T, N] → mean across neurons → [T, 1] broadcast
        mean_drive = drive.mean(axis=1, keepdims=True)  # [T, 1]
        # Per-receptor variation: weight each receptor by its distance from
        # the stimulus centroid — we don't have innervation weights here, so
        # we use the mean drive as a uniform heat at each receptor position.
        self._receptor_drive = np.broadcast_to(
            mean_drive, (T, coords.shape[0])
        ).copy()  # [T, M]

        nonzero = self._receptor_drive[self._receptor_drive > 0]
        if nonzero.size:
            self._global_min = float(np.percentile(nonzero, 2))
            self._global_max = float(np.percentile(nonzero, 98)) or 1.0
        else:
            self._global_min, self._global_max = 0.0, 1.0

    def _set_view_range(self) -> None:
        if self._data is None:
            return
        coords = self._data.receptor_positions
        if coords is not None and coords.shape[0] > 0:
            margin = 0.5
            self._pw.setRange(
                xRange=[float(coords[:, 0].min()) - margin,
                        float(coords[:, 0].max()) + margin],
                yRange=[float(coords[:, 1].min()) - margin,
                        float(coords[:, 1].max()) + margin],
                padding=0,
            )

    def _render_frame(self, t_idx: int) -> None:
        if self._data is None:
            return
        coords = self._data.receptor_positions
        if coords is None or coords.shape[0] == 0:
            self._scatter.clear()
            return

        if self._receptor_drive is not None:
            drive_t = self._receptor_drive[t_idx]  # [M]
        else:
            drive_t = np.zeros(coords.shape[0])

        colors = self._drive_to_colors(drive_t)

        spots = [
            {
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "size": 5,
                "brush": pg.mkBrush(*colors[i]),
                "pen": None,
            }
            for i in range(len(coords))
        ]
        self._scatter.setData(spots)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_lut(self) -> None:
        self._lut = self._colormap_lut(self._cmap_name, 256)

    def _drive_to_colors(self, values: np.ndarray) -> np.ndarray:
        """Map drive values → RGBA via LUT.  Returns [M, 4] uint8."""
        span = self._global_max - self._global_min
        if span < 1e-9:
            norm = np.zeros_like(values)
        else:
            if self._log_scale:
                safe = np.clip(values - self._global_min + 1e-9, 1e-9, None)
                log_span = np.log(max(span + 1e-9, 1e-9))
                norm = np.clip(np.log(safe) / log_span, 0.0, 1.0)
            else:
                norm = np.clip((values - self._global_min) / span, 0.0, 1.0)
        indices = (norm * 255).astype(np.int32)
        return self._lut[indices]

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
                self._build_receptor_drive()
                self._render_frame(self._t_idx)
            pop_cmb.currentTextChanged.connect(_pop_changed)
            form.addRow("Population:", pop_cmb)

        cmap_cmb = QtWidgets.QComboBox()
        cmap_cmb.addItems(["plasma", "viridis", "inferno", "hot", "magma"])
        cmap_cmb.setCurrentText(self._cmap_name)
        def _cmap_changed(name):
            self._cmap_name = name
            self._rebuild_lut()
            self._render_frame(self._t_idx)
        cmap_cmb.currentTextChanged.connect(_cmap_changed)
        form.addRow("Colormap:", cmap_cmb)

        log_chk = QtWidgets.QCheckBox("Log scale")
        log_chk.setChecked(self._log_scale)
        def _log_changed(s):
            self._log_scale = bool(s)
            self._render_frame(self._t_idx)
        log_chk.stateChanged.connect(_log_changed)
        form.addRow(log_chk)
        return w
