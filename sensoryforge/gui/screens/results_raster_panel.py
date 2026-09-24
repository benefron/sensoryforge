"""Raster panel: every population, stacked in colour-coded bands.

A spiking population draws one ``|`` marker per (time, neuron) spike, neuron
index within its own band on the y axis. An analog population (no
``spikes``, only ``state``) has nothing to tick, so it draws its state trace
as a heat-map band instead, one row per neuron, occupying the same band
height a spiking population's neurons would.

A run with more spike events than :data:`MAX_RASTER_POINTS` is decimated by
an even stride over the flattened (time, neuron) spike list, never by
changing which neurons or times are shown preferentially -- see
:func:`raster_points`. The panel's caption always states the plotted count
and, when decimated, the true count.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from sensoryforge.gui import theme
from sensoryforge.gui.screens.results_data import ResultsView
from sensoryforge.gui.widgets import plot_factory

#: Spike events (time, neuron) pairs drawn before the raster decimates.
MAX_RASTER_POINTS = 20000


def raster_points(spikes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """The (time-index, neuron-index) pairs to draw for one population's spikes.

    Args:
        spikes: ``[T, N]`` sub-step spike counts.

    Returns:
        ``(t_idx, n_idx, total, decimated)``: the indices to plot, the true
        number of spike events (``int((spikes > 0).sum())``), and whether the
        plotted indices are a strided subset of them. Decimation takes an
        even stride over the flattened event list, so ``len(t_idx)`` is
        exactly ``math.ceil(total / stride)`` for the smallest stride that
        keeps it at or under :data:`MAX_RASTER_POINTS` -- reproducible and
        testable, never a random subset.
    """
    t_idx, n_idx = np.nonzero(np.asarray(spikes) > 0)
    total = int(len(t_idx))
    if total <= MAX_RASTER_POINTS:
        return t_idx, n_idx, total, False
    stride = math.ceil(total / MAX_RASTER_POINTS)
    keep = slice(0, total, stride)
    return t_idx[keep], n_idx[keep], total, True


class RasterPanel(QtWidgets.QWidget):
    """Stacked per-population raster/heat-map bands with a shared cursor."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._view: Optional[ResultsView] = None
        self._band_offsets: dict = {}
        self._image_items: List[pg.ImageItem] = []
        self._scatter_items: List[pg.ScatterPlotItem] = []
        #: Population name -> (plotted count, true count, decimated?), for
        #: tests and captions.
        self.spike_counts: dict = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = plot_factory.make_plot("Raster", "Time", "Neuron", x_unit="ms")
        layout.addWidget(self.plot)
        self.cursor = pg.InfiniteLine(
            pos=0, angle=90, pen=theme.pen(theme.PALETTE["text_secondary"])
        )
        self.plot.addItem(self.cursor)
        self.caption = QtWidgets.QLabel("")
        self.caption.setStyleSheet(f"color: {theme.PALETTE['text_secondary']};")
        layout.addWidget(self.caption)

    def set_view(self, view: Optional[ResultsView]) -> None:
        """Rebuild every band for a new :class:`ResultsView`."""
        for item in self._scatter_items + self._image_items:
            self.plot.getPlotItem().removeItem(item)
        self._scatter_items.clear()
        self._image_items.clear()
        self._band_offsets.clear()
        self.spike_counts.clear()
        self._view = view
        if view is None:
            self.caption.setText("")
            return

        offset = 0
        captions: List[str] = []
        time0 = float(view.time_ms[0]) if len(view.time_ms) else 0.0
        dt = float(view.time_ms[1] - view.time_ms[0]) if len(view.time_ms) > 1 else 1.0
        for pop in view.populations:
            self._band_offsets[pop.name] = offset
            n = max(pop.n_neurons, 1)
            color = theme.population_color(pop.index, pop.neuron_type)
            if pop.is_analog:
                image = pg.ImageItem()
                lut = theme.colormap().getLookupTable(nPts=256)
                image.setLookupTable(lut)
                state = pop.state.detach().cpu().numpy()
                image.setImage(state, autoLevels=True)
                tr = pg.QtGui.QTransform()
                tr.translate(time0, offset)
                tr.scale(dt, 1.0)
                image.setTransform(tr)
                self.plot.getPlotItem().addItem(image)
                self._image_items.append(image)
            else:
                spikes = pop.spikes.detach().cpu().numpy()
                t_idx, n_idx, total, decimated = raster_points(spikes)
                self.spike_counts[pop.name] = (len(t_idx), total, decimated)
                scatter = plot_factory.make_raster_item(color)
                xs = (
                    view.time_ms[t_idx].detach().cpu().numpy()
                    if total
                    else np.array([])
                )
                ys = (n_idx + offset).astype(float)
                scatter.setData(xs, ys)
                self.plot.getPlotItem().addItem(scatter)
                self._scatter_items.append(scatter)
                if decimated:
                    captions.append(f"{pop.name}: {len(t_idx)}/{total} spikes shown")
            offset += n

        self.plot.setYRange(-0.5, max(offset - 0.5, 0.5), padding=0)
        self.caption.setText("; ".join(captions) if captions else "All spikes shown.")

    def set_cursor(self, time_value: float) -> None:
        """Move the shared cursor line to ``time_value`` ms."""
        self.cursor.setPos(time_value)

    def teardown(self) -> None:
        """Release the plot's pyqtgraph resources."""
        plot_factory.teardown(self.plot)
