"""Sensory neuron spatial panel.

Displays the neuron positions (from mechanoreceptor tab) with color encoding
one of: instantaneous drive, membrane voltage, or local firing rate.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5 import QtGui, QtWidgets
import pyqtgraph as pg  # type: ignore

from .base_panel import VisualizationPanel, VisData

_OVERLAY_OPTIONS = ["Drive", "Voltage", "Firing Rate"]
_BIN_SIZE_MS_DEFAULT = 20.0


class NeuronPanel(VisualizationPanel):
    """Spatial scatter of sensory neuron positions, colored by signal at t."""

    PANEL_DISPLAY_NAME = "Neuron Activity"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._pop_name: Optional[str] = None
        self._overlay = "Drive"       # "Drive" | "Voltage" | "Firing Rate"
        self._cmap_name = "viridis"
        self._bin_ms = _BIN_SIZE_MS_DEFAULT
        self._lut: Optional[np.ndarray] = None
        self._global_min: float = 0.0
        self._global_max: float = 1.0
        # Pre-computed firing rate [T, N] for speed
        self._firing_rate: Optional[np.ndarray] = None
        super().__init__(title="Neuron Activity", parent=parent)

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

        layout.addWidget(self._pw)
        self._rebuild_lut()

    # ------------------------------------------------------------------
    # VisualizationPanel interface
    # ------------------------------------------------------------------

    def _on_data_set(self) -> None:
        if self._data is None:
            self._scatter.clear()
            return
        if self._pop_name not in self._data.population_names:
            self._pop_name = (
                self._data.population_names[0]
                if self._data.population_names else None
            )
        self._precompute()
        self._set_view_range()
        self._render_frame(0)

    def _precompute(self) -> None:
        """Pre-compute firing rates and determine global color range."""
        self._firing_rate = None
        if self._data is None or self._pop_name is None:
            return
        res = self._data.population_results.get(self._pop_name, {})
        spk = res.get("spikes")    # [T, N]
        if spk is not None and self._overlay == "Firing Rate":
            self._firing_rate = _smooth_firing_rate(spk, self._data.dt_ms, self._bin_ms)

        signal = self._get_signal()
        if signal is not None and signal.size:
            nonzero = signal[signal != 0]
            if nonzero.size:
                self._global_min = float(np.percentile(nonzero, 2))
                self._global_max = float(np.percentile(nonzero, 98)) or 1.0
            else:
                self._global_min, self._global_max = 0.0, 1.0

    def _get_signal(self) -> Optional[np.ndarray]:
        """Return [T, N] signal array for the current overlay mode."""
        if self._data is None or self._pop_name is None:
            return None
        res = self._data.population_results.get(self._pop_name, {})
        if self._overlay == "Drive":
            return res.get("drive")
        if self._overlay == "Voltage":
            return res.get("v_trace")
        if self._overlay == "Firing Rate":
            return self._firing_rate
        return None

    def _set_view_range(self) -> None:
        if self._data is None or self._pop_name is None:
            return
        pos = self._data.neuron_positions.get(self._pop_name)
        if pos is not None and pos.shape[0] > 0:
            margin = 0.5
            self._pw.setRange(
                xRange=[float(pos[:, 0].min()) - margin,
                        float(pos[:, 0].max()) + margin],
                yRange=[float(pos[:, 1].min()) - margin,
                        float(pos[:, 1].max()) + margin],
                padding=0,
            )

    def _render_frame(self, t_idx: int) -> None:
        if self._data is None or self._pop_name is None:
            self._scatter.clear()
            return
        pos = self._data.neuron_positions.get(self._pop_name)
        if pos is None or pos.shape[0] == 0:
            self._scatter.clear()
            return

        signal = self._get_signal()
        if signal is not None and t_idx < signal.shape[0]:
            values = signal[t_idx]  # [N]
        else:
            values = np.zeros(pos.shape[0])

        colors = self._values_to_colors(values)

        spots = [
            {
                "x": float(pos[i, 0]),
                "y": float(pos[i, 1]),
                "size": 9,
                "brush": pg.mkBrush(*colors[i]),
                "pen": pg.mkPen("#444", width=0.8),
            }
            for i in range(len(pos))
        ]
        self._scatter.setData(spots)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_lut(self) -> None:
        self._lut = self._colormap_lut(self._cmap_name, 256)

    def _values_to_colors(self, values: np.ndarray) -> np.ndarray:
        span = self._global_max - self._global_min
        if span < 1e-9:
            norm = np.zeros_like(values, dtype=np.float32)
        else:
            norm = np.clip((values - self._global_min) / span, 0.0, 1.0).astype(
                np.float32
            )
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
                self._precompute()
                self._set_view_range()
                self._render_frame(self._t_idx)
            pop_cmb.currentTextChanged.connect(_pop_changed)
            form.addRow("Population:", pop_cmb)

        overlay_cmb = QtWidgets.QComboBox()
        overlay_cmb.addItems(_OVERLAY_OPTIONS)
        overlay_cmb.setCurrentText(self._overlay)
        def _overlay_changed(name):
            self._overlay = name
            self._precompute()
            self._render_frame(self._t_idx)
        overlay_cmb.currentTextChanged.connect(_overlay_changed)
        form.addRow("Overlay:", overlay_cmb)

        cmap_cmb = QtWidgets.QComboBox()
        cmap_cmb.addItems(["viridis", "plasma", "inferno", "hot", "coolwarm"])
        cmap_cmb.setCurrentText(self._cmap_name)
        def _cmap_changed(name):
            self._cmap_name = name
            self._rebuild_lut()
            self._render_frame(self._t_idx)
        cmap_cmb.currentTextChanged.connect(_cmap_changed)
        form.addRow("Colormap:", cmap_cmb)

        bin_spin = QtWidgets.QDoubleSpinBox()
        bin_spin.setRange(1.0, 200.0)
        bin_spin.setSingleStep(5.0)
        bin_spin.setValue(self._bin_ms)
        bin_spin.setSuffix(" ms")
        def _bin_changed(v):
            self._bin_ms = v
            if self._overlay == "Firing Rate":
                self._precompute()
                self._render_frame(self._t_idx)
        bin_spin.valueChanged.connect(_bin_changed)
        form.addRow("Rate bin:", bin_spin)
        return w


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth_firing_rate(
    spikes: np.ndarray,      # [T, N]
    dt_ms: float,
    bin_ms: float,
) -> np.ndarray:
    """Return instantaneous firing rate [T, N] in Hz using a boxcar window."""
    if dt_ms <= 0:
        return np.zeros_like(spikes, dtype=np.float32)
    win = max(1, int(round(bin_ms / dt_ms)))
    kernel = np.ones(win, dtype=np.float32) / (win * dt_ms * 1e-3)  # spikes/s
    T, N = spikes.shape
    out = np.empty((T, N), dtype=np.float32)
    for n in range(N):
        out[:, n] = np.convolve(spikes[:, n].astype(np.float32), kernel, mode="same")
    return out
