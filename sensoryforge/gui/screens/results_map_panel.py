"""Neuron-map panel: neuron centres over the receptor extent, click to select.

Colour encodes total spike count for a spiking population, mean state for an
analog one. Clicking a neuron emits :attr:`NeuronMapPanel.neuronClicked`
``(population_name, neuron_index)`` -- the Populations screen's map and this
one share that convention.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from sensoryforge.gui import theme
from sensoryforge.gui.screens.results_data import ResultsView
from sensoryforge.gui.widgets import plot_factory


class NeuronMapPanel(QtWidgets.QWidget):
    """Scatter of every population's neuron centres, coloured by activity."""

    neuronClicked = QtCore.pyqtSignal(str, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._view: Optional[ResultsView] = None
        self._scatters: list = []
        #: Flat arrays parallel to every scatter point, for click lookup.
        self._point_population: list = []
        self._point_index: list = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = plot_factory.make_plot(
            "Neuron map", "x", "y", x_unit="mm", y_unit="mm"
        )
        self.plot.setAspectLocked(True)
        layout.addWidget(self.plot)

    def set_view(self, view: Optional[ResultsView]) -> None:
        """Rebuild the scatter for a new :class:`ResultsView`."""
        for scatter in self._scatters:
            self.plot.getPlotItem().removeItem(scatter)
        self._scatters.clear()
        self._point_population.clear()
        self._point_index.clear()
        self._view = view
        if view is None:
            return

        cmap = theme.colormap()
        for pop in view.populations:
            if pop.neuron_centers is None or pop.n_neurons == 0:
                continue
            centers = pop.neuron_centers.detach().cpu().numpy()
            if pop.is_analog:
                activity = pop.state.detach().cpu().numpy().mean(axis=0)
            else:
                activity = pop.spikes.detach().cpu().numpy().sum(axis=0)
            lo, hi = float(activity.min()), float(activity.max())
            norm = (activity - lo) / (hi - lo) if hi > lo else np.zeros_like(activity)
            colors = [cmap.map(v, mode="qcolor") for v in norm]
            scatter = pg.ScatterPlotItem(
                x=centers[:, 0],
                y=centers[:, 1],
                size=8,
                pen=None,
                brush=[pg.mkBrush(c) for c in colors],
            )
            plot_factory.connect(
                scatter.sigClicked, self._on_clicked, pop.name, owner=self.plot
            )
            self.plot.getPlotItem().addItem(scatter)
            self._scatters.append(scatter)
            self._point_population.append(pop.name)
            self._point_index.append(np.arange(len(activity)))

    def _on_clicked(self, population_name: str, scatter, points) -> None:
        """``plot_factory.connect`` callback: emit the clicked neuron index."""
        if not points:
            return
        index = int(points[0].index())
        self.neuronClicked.emit(population_name, index)

    def teardown(self) -> None:
        """Release the plot's pyqtgraph resources."""
        plot_factory.teardown(self.plot)
