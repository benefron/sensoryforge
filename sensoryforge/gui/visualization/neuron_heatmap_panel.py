"""Neuron heatmap panel.

Shows the receptor grid with stimulus weighted by summed innervation (across
all neurons), normalized. Indicates how the stimulus is represented by the
population. Neurons overlaid and colored by selected signal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg  # type: ignore

from .base_panel import VisualizationPanel, VisData

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None
try:
    from scipy.interpolate import RegularGridInterpolator, griddata
except ImportError:
    RegularGridInterpolator = None
    griddata = None

_SIGNAL_OPTIONS = [
    "Drive (filtered)",
    "Drive (raw)",
    "Voltage",
    "Firing rate",
]
_BIN_MS_DEFAULT = 20.0


def _smooth_firing_rate(
    spikes: np.ndarray,
    dt_ms: float,
    bin_ms: float,
) -> np.ndarray:
    """Return [T, N] firing rate in Hz using boxcar window."""
    if dt_ms <= 0:
        return np.zeros_like(spikes, dtype=np.float32)
    win = max(1, int(round(bin_ms / dt_ms)))
    kernel = np.ones(win, dtype=np.float32) / (win * dt_ms * 1e-3)
    T, N = spikes.shape
    out = np.empty((T, N), dtype=np.float32)
    for n in range(N):
        out[:, n] = np.convolve(spikes[:, n].astype(np.float32), kernel, mode="same")
    return out


class NeuronHeatmapPanel(VisualizationPanel):
    """Receptor grid heatmap: stimulus × summed innervation, normalized."""

    PANEL_DISPLAY_NAME = "Neuron Heatmap"

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._pop_name: Optional[str] = None
        self._signal = "Drive (filtered)"  # colors neuron points
        self._bin_ms = _BIN_MS_DEFAULT
        self._cmap_name = "viridis"
        self._smooth_sigma = 0.0  # spatial smoothing in pixels (0 = none)
        self._global_min: float = 0.0
        self._global_max: float = 1.0
        self._firing_rate: Optional[np.ndarray] = None
        super().__init__(title="Neuron Heatmap", parent=parent)

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

        self._neuron_scatter = pg.ScatterPlotItem(pxMode=True)
        self._neuron_scatter.setZValue(11)
        self._pw.addItem(self._neuron_scatter)

        self._set_colormap(self._cmap_name)

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
        self._neuron_scatter.clear()

    # ------------------------------------------------------------------
    # VisualizationPanel interface
    # ------------------------------------------------------------------

    def _on_data_set(self) -> None:
        if self._data is None:
            self._show_placeholder()
            return
        if self._pop_name not in self._data.population_names:
            self._pop_name = (
                self._data.population_names[0]
                if self._data.population_names else None
            )
        self._precompute()
        self._set_view_range()
        self._render_frame(0)

    def _get_neuron_signal(self) -> Optional[np.ndarray]:
        """Return [T, N] for coloring neuron points."""
        if self._data is None or self._pop_name is None:
            return None
        res = self._data.population_results.get(self._pop_name, {})
        if self._signal == "Drive (filtered)":
            return res.get("drive")
        if self._signal == "Drive (raw)":
            raw = res.get("raw_drive")
            return raw if raw is not None else res.get("drive")
        if self._signal == "Voltage":
            return res.get("v_trace")
        if self._signal == "Firing rate":
            return self._firing_rate
        return None

    def _precompute(self) -> None:
        self._firing_rate = None
        if self._data is None or self._pop_name is None:
            return
        res = self._data.population_results.get(self._pop_name, {})
        spk = res.get("spikes")
        if spk is not None and self._signal == "Firing rate":
            self._firing_rate = _smooth_firing_rate(spk, self._data.dt_ms, self._bin_ms)

    def _set_view_range(self) -> None:
        if self._data is None:
            return
        xl, xr = self._data.stimulus_xlim
        yb, yt = self._data.stimulus_ylim
        margin = 0.5
        self._pw.setRange(
            xRange=[xl - margin, xr + margin],
            yRange=[yb - margin, yt + margin],
            padding=0.05,
        )

    def _render_frame(self, t_idx: int) -> None:
        if self._data is None or self._pop_name is None:
            self._show_placeholder()
            return
        stimulus = self._data.stimulus_frames
        weights = self._data.innervation_weights.get(self._pop_name)
        pos = self._data.neuron_positions.get(self._pop_name)
        if stimulus is None or weights is None or pos is None:
            self._show_placeholder()
            return
        if t_idx >= stimulus.shape[0]:
            return

        n_neurons = pos.shape[0]
        # Sum innervation over all neurons: total weight per receptor
        w_sum = np.sum(weights, axis=0)

        # Receptor grid: stimulus × summed weights (how population represents stimulus)
        if w_sum.ndim == 2:
            # Grid module: stimulus [H,W], w_sum [H,W]
            stim_t = stimulus[t_idx]
            heat = stim_t * w_sum
        else:
            # Flat: w_sum [M], interpolate stimulus at receptor positions
            rpos = self._data.receptor_positions
            if rpos is None or w_sum.shape[0] != rpos.shape[0]:
                self._show_placeholder()
                return
            stim_t = stimulus[t_idx]
            H, W = stim_t.shape
            xl, xr = self._data.stimulus_xlim
            yb, yt = self._data.stimulus_ylim
            if RegularGridInterpolator is None:
                self._show_placeholder()
                return
            x_vals = np.linspace(xl, xr, W)
            y_vals = np.linspace(yb, yt, H)
            interp = RegularGridInterpolator((y_vals, x_vals), stim_t)
            receptor_signal = interp(rpos[:, [1, 0]])
            weighted = receptor_signal * w_sum
            # Interpolate back onto stimulus grid for display
            if griddata is None:
                self._show_placeholder()
                return
            xi = np.linspace(xl, xr, W)
            yi = np.linspace(yb, yt, H)
            Xi, Yi = np.meshgrid(xi, yi)
            heat = griddata(
                (rpos[:, 0], rpos[:, 1]),
                weighted,
                (Xi, Yi),
                method="linear",
                fill_value=0.0,
            )

        heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
        # Normalize: 0–1 for display
        if heat.size and np.any(heat > 0):
            heat = heat / (np.max(heat) + 1e-9)

        # Optional spatial smoothing
        if self._smooth_sigma > 0 and gaussian_filter is not None:
            heat = gaussian_filter(heat, sigma=self._smooth_sigma)

        # Dim background for subtlety
        heat = heat * 0.7
        self._global_min, self._global_max = 0.0, 1.0

        # heat [n_x, n_y] = (x, y), same convention as stimulus
        self._img.setImage(
            heat,
            autoLevels=False,
            levels=(self._global_min, self._global_max),
        )
        xl, xr = self._data.stimulus_xlim
        yb, yt = self._data.stimulus_ylim
        n_x, n_y = heat.shape
        dx = (xr - xl) / max(n_x, 1)
        dy = (yt - yb) / max(n_y, 1)
        tr = pg.QtGui.QTransform()
        tr.translate(xl, yb)
        tr.scale(dx, dy)
        self._img.setTransform(tr)

        # Neuron overlay: all neurons colored by signal
        self._neuron_scatter.clear()
        neuron_signal = self._get_neuron_signal()
        if (
            neuron_signal is not None
            and t_idx < neuron_signal.shape[0]
            and neuron_signal.shape[1] == n_neurons
        ):
            vals = neuron_signal[t_idx]
            span = float(np.ptp(vals)) or 1.0
            vmin = float(np.percentile(vals, 2))
            lut = self._colormap_lut(self._cmap_name, 256)
            for i in range(n_neurons):
                v = vals[i]
                norm = np.clip((v - vmin) / span, 0, 1)
                cidx = int(norm * 255)
                c = lut[cidx]
                brush = pg.mkBrush(int(c[0]), int(c[1]), int(c[2]), 200)
                self._neuron_scatter.addPoints(
                    x=[float(pos[i, 0])],
                    y=[float(pos[i, 1])],
                    size=8,
                    brush=brush,
                    pen=pg.mkPen("#333", width=0.5),
                )
        else:
            for i in range(n_neurons):
                self._neuron_scatter.addPoints(
                    x=[float(pos[i, 0])],
                    y=[float(pos[i, 1])],
                    size=8,
                    brush=pg.mkBrush(100, 100, 100, 180),
                    pen=pg.mkPen("#333", width=0.5),
                )

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

        sig_cmb = QtWidgets.QComboBox()
        sig_cmb.addItems(_SIGNAL_OPTIONS)
        sig_cmb.setCurrentText(self._signal)
        sig_cmb.setToolTip("Signal used to color neuron points")
        def _sig_changed(name):
            self._signal = name
            self._precompute()
            self._render_frame(self._t_idx)
        sig_cmb.currentTextChanged.connect(_sig_changed)
        form.addRow("Neuron color:", sig_cmb)

        bin_spin = QtWidgets.QDoubleSpinBox()
        bin_spin.setRange(1.0, 200.0)
        bin_spin.setSingleStep(5.0)
        bin_spin.setValue(self._bin_ms)
        bin_spin.setSuffix(" ms")
        bin_spin.setToolTip("Temporal bin for firing rate")
        def _bin_changed(v):
            self._bin_ms = v
            if self._signal == "Firing rate":
                self._precompute()
                self._render_frame(self._t_idx)
        bin_spin.valueChanged.connect(_bin_changed)
        form.addRow("Rate bin:", bin_spin)

        smooth_spin = QtWidgets.QDoubleSpinBox()
        smooth_spin.setRange(0.0, 5.0)
        smooth_spin.setSingleStep(0.25)
        smooth_spin.setValue(self._smooth_sigma)
        smooth_spin.setToolTip("Spatial smoothing (0 = none)")
        def _smooth_changed(v):
            self._smooth_sigma = v
            self._render_frame(self._t_idx)
        smooth_spin.valueChanged.connect(_smooth_changed)
        form.addRow("Smooth (px):", smooth_spin)

        cmap_cmb = QtWidgets.QComboBox()
        cmap_cmb.addItems(["viridis", "plasma", "inferno", "magma", "hot", "coolwarm"])
        cmap_cmb.setCurrentText(self._cmap_name)
        cmap_cmb.currentTextChanged.connect(self._set_colormap)
        form.addRow("Colormap:", cmap_cmb)

        return w

    def _set_colormap(self, name: str) -> None:
        self._cmap_name = name
        try:
            lut = pg.colormap.get(name, source="matplotlib").getLookupTable(
                nPts=256, alpha=False
            )
            self._img.setLookupTable(lut)
        except Exception:
            pass
