"""The receptor / receptive-field preview, lifted from the old Mechanoreceptor tab.

``GridPreview`` draws the receptor scatter for one or more
:class:`~sensoryforge.config.schema.GridConfig` grids (mm axes, equal
aspect), overlays neuron centres for populations added through
:meth:`GridPreview.set_population`, and can highlight one neuron's
receptive-field "footprint" (the receptors its
:class:`~sensoryforge.core.rf_bank.ReceptiveFieldBank` row weights nonzero).

Grids are built with :func:`sensoryforge.core.simulation_engine.build_grid`
-- the exact function ``SimulationEngine._build_grids()`` uses per grid
entry -- so this widget never re-implements arrangement logic (grid, hex,
poisson, jittered, blue_noise, composite, coords_file all go through it).

The widget knows only ``GridConfig``, the ``ReceptorGrid``/
``CompositeReceptorGrid`` objects ``build_grid`` returns, and
``ReceptiveFieldBank`` -- no ``QSettings``, no expert-mode lists, no
``GridEntry``/``NeuronPopulation`` (those are the old tab's bookkeeping
classes, retired here).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from sensoryforge.config.schema import GridConfig
from sensoryforge.core.rf_bank import ReceptiveFieldBank
from sensoryforge.core.simulation_engine import build_grid
from sensoryforge.gui import theme
from sensoryforge.gui.widgets import plot_factory

#: Marker diameter (px) for the plain receptor scatter -- copied from the
#: old Mechanoreceptor tab's ``_update_grid_visualization``.
_RECEPTOR_SIZE = 5.0
#: Marker diameter (px) for neuron centres (phase1-contract.md section 6).
_NEURON_SIZE = 7.0
#: Marker diameter (px) for the receptive-field footprint overlay.
_FOOTPRINT_SIZE = 12.0
#: Ring diameter (px) around the selected neuron in a footprint.
_RING_SIZE = 18.0


def _emit_receptor_click(
    widget: "GridPreview", base_offset: int, scatter, points, ev
) -> None:
    """F-035-safe ``sigClicked`` handler for a grid's receptor scatter.

    A plain module-level function (not a bound method, no closure over a
    widget) -- ``widget``/``base_offset`` arrive as ``connect()``'s bound
    ``*args``, ``scatter``/``points``/``ev`` as pyqtgraph's own signal
    payload.

    Args:
        widget: The ``GridPreview`` to emit ``receptorClicked`` on.
        base_offset: This grid's receptor index offset into the widget's
            concatenated receptor numbering.
        scatter: The clicked ``pg.ScatterPlotItem`` (pyqtgraph's payload).
        points: The clicked ``SpotItem`` list.
        ev: The originating mouse event.
    """
    if not points:
        return
    widget.receptorClicked.emit(int(points[0].index()) + base_offset)


def _emit_neuron_click(widget: "GridPreview", name: str, scatter, points, ev) -> None:
    """F-035-safe ``sigClicked`` handler for a population's neuron scatter.

    Args:
        widget: The ``GridPreview`` to emit ``neuronClicked`` on.
        name: The population name this scatter belongs to.
        scatter: The clicked ``pg.ScatterPlotItem``.
        points: The clicked ``SpotItem`` list.
        ev: The originating mouse event.
    """
    if not points:
        return
    widget.neuronClicked.emit(name, int(points[0].index()))


class GridPreview(QtWidgets.QWidget):
    """Receptor scatter + neuron centres + receptive-field footprint, on one plot.

    Args:
        parent: Qt parent.

    Example:
        >>> preview = GridPreview()                          # doctest: +SKIP
        >>> preview.set_grids([GridConfig(name="skin", rows=20, cols=20)])
        >>> preview.receptor_count()
        400
    """

    #: Receptor index (into the widget's concatenated receptor numbering
    #: across every grid passed to ``set_grids``).
    receptorClicked = QtCore.pyqtSignal(int)
    #: Population name, neuron index.
    neuronClicked = QtCore.pyqtSignal(str, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self.plot = plot_factory.make_plot(
            xlabel="X (mm)", ylabel="Y (mm)", x_unit="mm", y_unit="mm"
        )
        self.plot.setAspectLocked(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

        # One scatter item per grid, plus the receptor count each covers
        # (for receptor_count() and the click-offset each scatter's index
        # needs added to keep receptor indices unique across grids).
        self._grid_scatters: List[Any] = []
        self._grid_counts: List[int] = []

        # name -> {"bank": ReceptiveFieldBank, "color": QColor,
        #          "scatter": pg.ScatterPlotItem, "visible": bool}
        self._populations: Dict[str, Dict[str, Any]] = {}

        # The footprint overlay (receptors coloured by one neuron's weight
        # row) and the ring marking that neuron; both None when clear.
        self._footprint_scatter: Optional[Any] = None
        self._footprint_ring: Optional[Any] = None
        self._footprint_name: Optional[str] = None
        self._footprint_index: Optional[int] = None

    # -- grids -----------------------------------------------------------

    def set_grids(
        self, grids: List[GridConfig], *, seed_override: Optional[int] = None
    ) -> None:
        """Rebuild the receptor scatter(s) from ``grids``.

        Builds each grid with :func:`build_grid` -- the same construction
        :class:`~sensoryforge.core.simulation_engine.SimulationEngine` uses
        -- and draws one grey scatter per grid.

        Args:
            grids: The grid configurations to preview.
            seed_override: When given, every grid is built as if its
                ``seed`` field were this value (for a reproducible preview
                of a random arrangement independent of the config's own
                seed), regardless of what ``grid_cfg.seed`` holds.
        """
        self._clear_grid_items()

        color = theme.PALETTE["border_strong"]
        offset = 0
        for grid_cfg in grids:
            cfg = grid_cfg
            if seed_override is not None:
                cfg = dataclasses.replace(grid_cfg, seed=seed_override)
            grid_obj = build_grid(cfg, device="cpu")
            coords = grid_obj.get_all_coordinates().detach().cpu().numpy()

            scatter = plot_factory.make_scatter(size=_RECEPTOR_SIZE, color=color)
            scatter.setData(x=coords[:, 0], y=coords[:, 1])
            scatter.setZValue(-2)
            self.plot.addItem(scatter)
            plot_factory.connect(
                scatter.sigClicked,
                _emit_receptor_click,
                self,
                offset,
                owner=self.plot,
            )

            self._grid_scatters.append(scatter)
            self._grid_counts.append(int(coords.shape[0]))
            offset += int(coords.shape[0])

        self.plot.getPlotItem().vb.autoRange()

    def receptor_count(self) -> int:
        """Total receptors across every grid passed to the last :meth:`set_grids`."""
        return sum(self._grid_counts)

    def _clear_grid_items(self) -> None:
        for scatter in self._grid_scatters:
            self.plot.removeItem(scatter)
        self._grid_scatters = []
        self._grid_counts = []

    # -- populations -------------------------------------------------------

    def set_population(
        self,
        name: str,
        bank: ReceptiveFieldBank,
        color: QtGui.QColor,
        *,
        visible: bool = True,
    ) -> None:
        """Add or replace a population's neuron-centre scatter.

        Args:
            name: Population name; a second call with the same name
                replaces the previous scatter.
            bank: The population's receptive-field bank; ``bank.weights``
                and ``bank.receptor_coords`` are read again by
                :meth:`show_rf_footprint`.
            color: Neuron marker colour.
            visible: Initial visibility of the neuron scatter.
        """
        self._remove_population_item(name)

        centers = bank.neuron_centers.detach().cpu().numpy()
        scatter = plot_factory.make_scatter(size=_NEURON_SIZE, color=color)
        scatter.setData(x=centers[:, 0], y=centers[:, 1])
        scatter.setZValue(5)
        scatter.setVisible(visible)
        self.plot.addItem(scatter)
        plot_factory.connect(
            scatter.sigClicked, _emit_neuron_click, self, name, owner=self.plot
        )

        self._populations[name] = {
            "bank": bank,
            "color": QtGui.QColor(color),
            "scatter": scatter,
            "visible": visible,
        }

    def clear_populations(self) -> None:
        """Remove every population's neuron scatter and any active footprint."""
        self.clear_footprint()
        for name in list(self._populations):
            self._remove_population_item(name)
        self._populations = {}

    def _remove_population_item(self, name: str) -> None:
        entry = self._populations.pop(name, None)
        if entry is None:
            return
        self.plot.removeItem(entry["scatter"])
        if self._footprint_name == name:
            self.clear_footprint()

    # -- receptive-field footprint ------------------------------------------

    def show_rf_footprint(self, name: Optional[str], neuron_index: int) -> None:
        """Highlight one neuron's receptive field.

        Colours the receptors with a nonzero weight in ``bank.weights[neuron_index]``
        (``bank`` being population ``name``'s bank) by that weight, normalised
        0-1 and mapped through :func:`sensoryforge.gui.theme.colormap`
        (viridis), and rings the neuron itself. Reads straight from the
        bank passed to :meth:`set_population` -- not from any of this
        widget's own scatter bookkeeping -- so the highlight is exactly
        what the bank says, every time.

        Args:
            name: Population name, or ``None`` to clear (equivalent to
                :meth:`clear_footprint`).
            neuron_index: Row of ``bank.weights`` to show.

        Raises:
            KeyError: If ``name`` was never passed to :meth:`set_population`.
        """
        self.clear_footprint()
        if name is None:
            return

        entry = self._populations[name]
        bank = entry["bank"]
        weights_row = bank.weights[neuron_index].detach().cpu().numpy()
        nonzero = weights_row != 0.0
        if np.any(nonzero):
            coords = bank.receptor_coords.detach().cpu().numpy()[nonzero]
            values = weights_row[nonzero].astype(np.float32)
            peak = float(np.abs(values).max())
            normalized = values / peak if peak > 0.0 else np.zeros_like(values)
            colors = theme.colormap().map(normalized, mode="byte")

            footprint = plot_factory.make_scatter(size=_FOOTPRINT_SIZE)
            footprint.setData(x=coords[:, 0], y=coords[:, 1], brush=colors, pen=None)
            footprint.setZValue(8)
            self.plot.addItem(footprint)
            self._footprint_scatter = footprint

        center = bank.neuron_centers.detach().cpu().numpy()[neuron_index]
        ring_item = plot_factory.make_scatter(size=_RING_SIZE)
        ring_item.setData(
            x=[float(center[0])],
            y=[float(center[1])],
            brush=pg.mkBrush(None),
            pen=theme.pen(entry["color"], width=2.0),
        )
        ring_item.setZValue(9)
        self.plot.addItem(ring_item)
        self._footprint_ring = ring_item

        self._footprint_name = name
        self._footprint_index = neuron_index

    def clear_footprint(self) -> None:
        """Remove the receptive-field footprint overlay, restoring plain scatters."""
        if self._footprint_scatter is not None:
            self.plot.removeItem(self._footprint_scatter)
            self._footprint_scatter = None
        if self._footprint_ring is not None:
            self.plot.removeItem(self._footprint_ring)
            self._footprint_ring = None
        self._footprint_name = None
        self._footprint_index = None

    # -- teardown ------------------------------------------------------------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt override)
        """Release pyqtgraph connections/items before the widget is destroyed (F-035)."""
        plot_factory.teardown(self.plot)
        super().closeEvent(event)
