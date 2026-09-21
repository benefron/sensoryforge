"""One-neuron trace panel: drive, filtered response, voltage/state, spikes.

Two stacked, x-linked plots so current (mA) and the readout (mV, or a DSL
state variable in its own unit) never share an axis: the top plot holds
drive + filtered current, the bottom plot holds the voltage/state readout
and the spike ticks.

The population and neuron index are picked from a combo box and a spin box,
or set programmatically by :meth:`TraceNeuronPanel.select_neuron` (wired to
:attr:`~sensoryforge.gui.screens.results_map_panel.NeuronMapPanel.neuronClicked`
by the screen).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets

from sensoryforge.gui import theme
from sensoryforge.gui.screens.results_data import ResultsView
from sensoryforge.gui.widgets import plot_factory


class TraceNeuronPanel(QtWidgets.QWidget):
    """Per-neuron drive/filtered/voltage-or-state trace, with spike ticks."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._view: Optional[ResultsView] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(QtWidgets.QLabel("Population"))
        self.population_combo = QtWidgets.QComboBox()
        self.population_combo.currentIndexChanged.connect(self._on_selection_changed)
        header_layout.addWidget(self.population_combo)
        header_layout.addWidget(QtWidgets.QLabel("Neuron"))
        self.neuron_spin = QtWidgets.QSpinBox()
        self.neuron_spin.setMinimum(0)
        self.neuron_spin.valueChanged.connect(self._redraw)
        header_layout.addWidget(self.neuron_spin)
        header_layout.addStretch(1)
        layout.addWidget(header)

        #: Top plot: drive + filtered current, both in mA.
        self.current_plot = plot_factory.make_plot(
            "Neuron trace", "", "Input", y_unit="mA"
        )
        self.current_plot.getPlotItem().addLegend(offset=(10, 10))
        layout.addWidget(self.current_plot)

        #: Bottom plot: membrane voltage (mV) or a DSL state variable.
        self.readout_plot = plot_factory.make_plot(
            "", "Time", "Potential", x_unit="ms", y_unit="mV"
        )
        layout.addWidget(self.readout_plot)
        self.readout_plot.setXLink(self.current_plot)

        #: The plot the shared cursor used to live on, kept as an alias of
        #: :attr:`cursor_readout` for callers that only ever moved one line.
        self.cursor_current = pg.InfiniteLine(
            pos=0, angle=90, pen=theme.pen(theme.PALETTE["text_secondary"])
        )
        self.current_plot.addItem(self.cursor_current)
        self.cursor_readout = pg.InfiniteLine(
            pos=0, angle=90, pen=theme.pen(theme.PALETTE["text_secondary"])
        )
        self.readout_plot.addItem(self.cursor_readout)

        #: The curve last drawn for ``(population, neuron)``'s readout --
        #: tests read this to check the trace's y-data.
        self.active_curve: Optional[pg.PlotDataItem] = None

    def set_view(self, view: Optional[ResultsView]) -> None:
        """Rebuild the population list for a new :class:`ResultsView`."""
        self._view = view
        self.population_combo.blockSignals(True)
        self.population_combo.clear()
        if view is not None:
            self.population_combo.addItems([pop.name for pop in view.populations])
        self.population_combo.blockSignals(False)
        self._on_selection_changed()
        self._select_most_active_neuron()

    def _select_most_active_neuron(self) -> None:
        """Start on the neuron that spiked most: neuron 0 is usually a silent corner."""
        pop = self._selected_population()
        if pop is None or pop.spikes is None or pop.n_neurons == 0:
            return
        counts = pop.spikes.sum(dim=0)
        if float(counts.max()) > 0:
            self.neuron_spin.setValue(int(counts.argmax()))

    def _on_selection_changed(self) -> None:
        pop = self._selected_population()
        self.neuron_spin.blockSignals(True)
        self.neuron_spin.setMaximum(max(pop.n_neurons - 1, 0) if pop is not None else 0)
        self.neuron_spin.blockSignals(False)
        self._redraw()

    def _selected_population(self):
        if self._view is None:
            return None
        name = self.population_combo.currentText()
        return self._view.population(name)

    def select_neuron(self, population_name: str, neuron_index: int) -> None:
        """Show ``population_name``'s neuron ``neuron_index``."""
        idx = self.population_combo.findText(population_name)
        if idx < 0:
            raise ValueError(
                f"no population named {population_name!r} in the current results"
            )
        self.population_combo.setCurrentIndex(idx)
        self.neuron_spin.setValue(neuron_index)

    def _redraw(self) -> None:
        current_item = self.current_plot.getPlotItem()
        readout_item = self.readout_plot.getPlotItem()
        current_item.clear()
        readout_item.clear()
        legend = current_item.legend
        if legend is not None:
            legend.clear()
        current_item.addItem(self.cursor_current)
        readout_item.addItem(self.cursor_readout)
        self.active_curve = None
        pop = self._selected_population()
        if pop is None or pop.n_neurons == 0:
            readout_item.setLabel("left", plot_factory.axis_label("Potential", "mV"))
            return
        n = min(self.neuron_spin.value(), pop.n_neurons - 1)
        time_ms = self._view.time_ms.detach().cpu().numpy()
        color = theme.population_color(pop.index, pop.neuron_type)

        if pop.drive is not None:
            current_item.plot(
                time_ms,
                pop.drive[:, n].detach().cpu().numpy(),
                pen=theme.pen(theme.PALETTE["text_disabled"]),
                name="drive",
            )
        if pop.filtered is not None:
            current_item.plot(
                time_ms,
                pop.filtered[:, n].detach().cpu().numpy(),
                pen=theme.pen(color),
                name="filtered",
            )

        if pop.is_analog:
            readout_label = pop.state_var_name or "State"
            readout_item.setLabel("left", plot_factory.axis_label(readout_label))
        else:
            readout_label = "Potential"
            readout_item.setLabel("left", plot_factory.axis_label(readout_label, "mV"))

        readout = pop.voltages if pop.voltages is not None else pop.state
        if readout is not None:
            y = readout[:, n].detach().cpu().numpy()
            self.active_curve = readout_item.plot(
                time_ms, y, pen=theme.pen(color, width=2.0), name=readout_label
            )
        elif pop.filtered is not None:
            # No voltage/state kept (e.g. a bundle without intermediates):
            # fall back to the filtered current as the readout curve, drawn
            # on the current plot where it already lives.
            self.active_curve = current_item.listDataItems()[-1]

        if pop.spikes is not None:
            spikes = pop.spikes[:, n].detach().cpu().numpy()
            spike_times = time_ms[np.nonzero(spikes > 0)[0]]
            if len(spike_times):
                y_top = (
                    float(np.max(readout[:, n].detach().cpu().numpy()))
                    if readout is not None
                    else 1.0
                )
                ticks = plot_factory.make_raster_item(color)
                ticks.setData(spike_times, np.full_like(spike_times, y_top))
                readout_item.addItem(ticks)

    def set_cursor(self, time_value: float) -> None:
        """Move both plots' shared cursor line to ``time_value`` ms."""
        self.cursor_current.setPos(time_value)
        self.cursor_readout.setPos(time_value)

    def teardown(self) -> None:
        """Release both plots' pyqtgraph resources."""
        plot_factory.teardown(self.current_plot)
        plot_factory.teardown(self.readout_plot)
