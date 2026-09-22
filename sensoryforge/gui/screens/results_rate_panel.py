"""Population rate panel: spikes/s per neuron, smoothed, one curve per population.

An analog population has no rate; its mean state is drawn instead, on a
second (right-hand) axis so its very different units and scale do not
squash the spiking curves.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from sensoryforge.gui import theme
from sensoryforge.gui.screens.results_data import ResultsView
from sensoryforge.gui.widgets import plot_factory

DEFAULT_WINDOW_MS = 20.0


def _sync_state_view(
    plot_item: "pg.PlotItem", state_viewbox: "pg.ViewBox", *_args
) -> None:
    """Keep the overlaid state ``ViewBox`` matching the rate plot's geometry.

    Plain module-level function so it can be wired through
    :func:`plot_factory.connect` (ledger F-035: never a bound method).

    Args:
        plot_item: The rate plot's ``PlotItem``, whose ``ViewBox`` resized.
        state_viewbox: The overlaid ``ViewBox`` to resize to match.
        *_args: Absorbs whatever ``sigResized`` emits; unused.
    """
    state_viewbox.setGeometry(plot_item.vb.sceneBoundingRect())


def smoothed_rate(spikes: np.ndarray, dt_ms: float, window_ms: float) -> np.ndarray:
    """Mean per-neuron firing rate (Hz) in a sliding window, one value per bin.

    Args:
        spikes: ``[T, N]`` sub-step spike counts.
        dt_ms: Time between rows of ``spikes``, in ms.
        window_ms: Smoothing window, in ms; at least one bin wide.

    Returns:
        ``[T]`` rate in Hz, averaged over neurons, using a centred moving sum
        (``numpy.convolve(..., mode="same")``) over ``max(1, round(window_ms
        / dt_ms))`` bins, clamped to at most ``T`` bins so the result always
        has exactly ``T`` samples -- a run shorter than the window still
        returns one value per bin instead of ``numpy.convolve`` padding the
        output out to the kernel's length.
    """
    spikes = np.asarray(spikes, dtype=np.float64)
    t, n = spikes.shape
    if t == 0:
        return np.zeros(0)
    per_step = spikes.mean(axis=1) if n else np.zeros(t)
    window_steps = max(1, int(round(window_ms / dt_ms))) if dt_ms > 0 else 1
    window_steps = min(window_steps, t)
    kernel = np.ones(window_steps) / window_steps
    smoothed = np.convolve(per_step, kernel, mode="same")
    return smoothed * (1000.0 / dt_ms) if dt_ms > 0 else smoothed


class RatePanel(QtWidgets.QWidget):
    """Rate curves for spiking populations; mean state for analog ones."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._view: Optional[ResultsView] = None
        self._curves: Dict[str, pg.PlotDataItem] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header_widget = QtWidgets.QWidget()
        header = QtWidgets.QHBoxLayout(header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(QtWidgets.QLabel("Smoothing window (ms)"))
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(1.0, 5000.0)
        self.window_spin.setValue(DEFAULT_WINDOW_MS)
        self.window_spin.valueChanged.connect(self._redraw)
        header.addWidget(self.window_spin)
        header.addStretch(1)
        layout.addWidget(header_widget)

        self.plot = plot_factory.make_plot(
            "Rate", "Time", "Rate", x_unit="ms", y_unit="Hz"
        )
        layout.addWidget(self.plot)
        self.cursor = pg.InfiniteLine(
            pos=0, angle=90, pen=theme.pen(theme.PALETTE["text_secondary"])
        )
        self.plot.addItem(self.cursor)

        # Second axis (state) shares the same ViewBox geometry, laid over it.
        # Built but left hidden: shown only when the view has an analog
        # population (set_view/_redraw), never unconditionally.
        self._state_viewbox = pg.ViewBox()
        self.plot.getPlotItem().scene().addItem(self._state_viewbox)
        self.plot.getPlotItem().getAxis("right").linkToView(self._state_viewbox)
        self._state_viewbox.setXLink(self.plot.getPlotItem())
        self.plot.getPlotItem().getAxis("right").setLabel(
            plot_factory.axis_label("State")
        )
        self.plot.getPlotItem().hideAxis("right")
        plot_factory.connect(
            self.plot.getPlotItem().vb.sigResized,
            _sync_state_view,
            self.plot.getPlotItem(),
            self._state_viewbox,
            owner=self.plot,
        )

    def set_view(self, view: Optional[ResultsView]) -> None:
        """Rebuild the rate/state curves for a new :class:`ResultsView`."""
        self.plot.getPlotItem().clear()
        self.plot.addItem(self.cursor)
        self._state_viewbox.clear()
        self._curves.clear()
        self._view = view
        if view is None:
            self.plot.getPlotItem().hideAxis("right")
            return
        self._redraw()

    def _redraw(self) -> None:
        view = self._view
        if view is None:
            return
        self.plot.getPlotItem().clear()
        self.plot.addItem(self.cursor)
        self._state_viewbox.clear()
        self._curves.clear()

        time_ms = view.time_ms.detach().cpu().numpy()
        dt_ms = float(time_ms[1] - time_ms[0]) if len(time_ms) > 1 else 1.0
        window_ms = self.window_spin.value()

        has_analog = any(pop.is_analog for pop in view.populations)
        if has_analog:
            self.plot.getPlotItem().showAxis("right")
        else:
            self.plot.getPlotItem().hideAxis("right")

        for pop in view.populations:
            color = theme.population_color(pop.index, pop.neuron_type)
            if pop.is_analog:
                state = pop.state.detach().cpu().numpy()
                mean_state = state.mean(axis=1) if state.shape[1] else state[:, 0]
                curve = pg.PlotCurveItem(
                    time_ms, mean_state, pen=theme.pen(color, width=1.2)
                )
                self._state_viewbox.addItem(curve)
            else:
                spikes = pop.spikes.detach().cpu().numpy()
                rate = smoothed_rate(spikes, dt_ms, window_ms)
                curve = self.plot.getPlotItem().plot(
                    time_ms, rate, pen=theme.pen(color), name=pop.name
                )
            self._curves[pop.name] = curve
        _sync_state_view(self.plot.getPlotItem(), self._state_viewbox)

    def set_cursor(self, time_value: float) -> None:
        """Move the shared cursor line to ``time_value`` ms."""
        self.cursor.setPos(time_value)

    def teardown(self) -> None:
        """Release the plot's pyqtgraph resources."""
        plot_factory.teardown(self.plot)
