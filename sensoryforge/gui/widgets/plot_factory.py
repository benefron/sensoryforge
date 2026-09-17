"""The one themed pyqtgraph construction path for GUI v2.

Every ``pg.PlotWidget`` / ``pg.ImageItem`` / ``pg.ScatterPlotItem`` built by
the new GUI goes through this module, so theming (``sensoryforge.gui.theme``)
stays in one place and pyqtgraph signal wiring stays safe.

Rule (ledger F-035): **never pass a bound method, or a lambda that closes
over a ``QWidget``/plot item, to a pyqtgraph signal.** pyqtgraph's C-side
event dispatch keeps such a slot alive as part of a reference cycle between
the Qt object graph and Python; sweeping that cycle with CPython's cyclic
garbage collector has reproduced a segfault inside
``ScatterPlotItem.renderSymbol`` (see ``docs_root/LEDGER.md`` F-035). Always
connect through :func:`connect`, which wraps the callback in a plain
``functools.partial`` (no bound method, no closure over a widget) and
records the connection so it can be released with :func:`teardown` before
the plot is destroyed.
"""

from __future__ import annotations

import functools
import weakref
from typing import Callable, List, Optional, Tuple

import pyqtgraph as pg
from PyQt5 import QtWidgets

from sensoryforge.gui import theme

# owner (a pg.PlotWidget, or any other hashable/weak-referenceable object
# passed as `owner=`) -> list of (signal, slot) pairs made through connect().
_CONNECTIONS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

# Connections made without an `owner`: never torn down automatically by
# teardown(plot) since there is no owning plot to key them under. Kept only
# so connect()'s bookkeeping is total; callers that care about cleanup should
# always pass `owner=`.
_UNOWNED_CONNECTIONS: List[Tuple[object, Callable]] = []


def make_plot(
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    *,
    x_unit: str = "",
    y_unit: str = "",
) -> pg.PlotWidget:
    """Build a themed, empty plot.

    White background, themed axis pens, a light grid, axis labels with
    units, no right-click context menu, and no corner auto-range button.

    Args:
        title: Plot title, shown above the axes. Omitted if empty.
        xlabel: Bottom axis label text. Omitted if empty.
        ylabel: Left axis label text. Omitted if empty.
        x_unit: Bottom axis unit string (e.g. ``"ms"``), passed to
            ``setLabel``'s ``units=``.
        y_unit: Left axis unit string (e.g. ``"mA"``).

    Returns:
        A themed ``pg.PlotWidget``, empty of data items.

    Example:
        >>> plot = make_plot("Drive", "Time", "Current", x_unit="ms", y_unit="mA")
    """
    plot = pg.PlotWidget(background=theme.PALETTE["bg_panel"])
    plot_item = plot.getPlotItem()

    for axis_name in ("bottom", "left"):
        axis = plot_item.getAxis(axis_name)
        axis.setPen(theme.AXIS_PEN)
        axis.setTextPen(theme.AXIS_PEN)

    plot.showGrid(x=True, y=True, alpha=theme.GRID_ALPHA)

    if title:
        plot_item.setTitle(title)
    if xlabel:
        plot_item.setLabel("bottom", xlabel, units=x_unit or None)
    if ylabel:
        plot_item.setLabel("left", ylabel, units=y_unit or None)

    plot.setMenuEnabled(False)
    plot.hideButtons()
    return plot


def make_image_plot(
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    *,
    x_unit: str = "",
    y_unit: str = "",
    colorbar_label: str = "",
) -> Tuple[pg.PlotWidget, pg.ImageItem, pg.ColorBarItem]:
    """Build a themed plot holding one ``pg.ImageItem`` with a colorbar.

    Args:
        title: Plot title. Omitted if empty.
        xlabel: Bottom axis label text. Omitted if empty.
        ylabel: Left axis label text. Omitted if empty.
        x_unit: Bottom axis unit string.
        y_unit: Left axis unit string.
        colorbar_label: Label shown on the attached colorbar.

    Returns:
        ``(plot, image_item, colorbar)``. The contract (phase1-contract.md
        section 5) describes this as returning ``(plot, image_item)``; a
        third element, the ``pg.ColorBarItem`` the image is attached to, is
        returned here since callers need it to update value limits as data
        changes (documented deviation, see task-1.4-report.md).

    Example:
        >>> plot, image, colorbar = make_image_plot("Footprint", "x", "y", x_unit="mm", y_unit="mm")
        >>> image.setImage(data)
    """
    plot = make_plot(title, xlabel, ylabel, x_unit=x_unit, y_unit=y_unit)
    plot.setAspectLocked(True)

    image_item = pg.ImageItem()
    plot.getPlotItem().addItem(image_item)

    cmap = theme.colormap()
    lut = cmap.getLookupTable(nPts=256, alpha=True)
    image_item.setLookupTable(lut)

    colorbar = pg.ColorBarItem(colorMap=cmap, label=colorbar_label or None)
    colorbar.setImageItem(image_item, insert_in=plot.getPlotItem())

    return plot, image_item, colorbar


def make_scatter(size: float = 6, color=None) -> pg.ScatterPlotItem:
    """Build a themed circular scatter item.

    Args:
        size: Marker diameter in px.
        color: Marker fill color (str, tuple, or ``QtGui.QColor``); defaults
            to the theme's accent color.

    Returns:
        A ``pg.ScatterPlotItem`` with symbol ``'o'`` and no outline pen.
    """
    brush = pg.mkBrush(color if color is not None else theme.PALETTE["accent"])
    return pg.ScatterPlotItem(symbol="o", size=size, pen=None, brush=brush)


def make_raster_item(color) -> pg.ScatterPlotItem:
    """Build a themed spike-raster tick item.

    Args:
        color: Tick color (str, tuple, or ``QtGui.QColor``).

    Returns:
        A ``pg.ScatterPlotItem`` with symbol ``'|'`` sized
        ``theme.RASTER_SIZE`` and no outline pen.
    """
    brush = pg.mkBrush(color)
    return pg.ScatterPlotItem(symbol="|", size=theme.RASTER_SIZE, pen=None, brush=brush)


def connect(signal, callback: Callable, *args, owner: Optional[object] = None) -> Callable:
    """Connect a pyqtgraph signal without risking a stale widget-closing slot.

    Wraps ``callback`` in ``functools.partial(callback, *args)`` -- never a
    bound method, never a lambda closing over a widget -- and connects that
    to ``signal``. The connection is recorded so :func:`teardown` can
    disconnect it later; pass ``owner`` (typically the ``pg.PlotWidget`` the
    signal's item lives on) so ``teardown(owner)`` finds it.

    Args:
        signal: A pyqtgraph/Qt bound signal (e.g. ``scatter.sigClicked``).
        callback: A plain function (not a bound method of a widget) to
            invoke on emission. Called as ``callback(*args, *signal_payload)``.
        *args: Extra positional arguments bound ahead of the signal's own
            emitted arguments.
        owner: The object connections should be torn down with. If omitted,
            the connection is recorded but not associated with any plot, so
            it will not be released by :func:`teardown`.

    Returns:
        The connected slot (the ``functools.partial``), for anyone who wants
        to disconnect it manually.

    Example:
        >>> scatter = make_scatter()
        >>> connect(scatter.sigClicked, on_click, "population-a", owner=plot)
    """
    slot = functools.partial(callback, *args)
    signal.connect(slot)
    if owner is not None:
        _CONNECTIONS.setdefault(owner, []).append((signal, slot))
    else:
        _UNOWNED_CONNECTIONS.append((signal, slot))
    return slot


def teardown(plot: pg.PlotWidget) -> None:
    """Release everything :func:`connect` and :func:`make_plot` attached to ``plot``.

    Disconnects every ``(signal, slot)`` recorded for ``plot`` as an
    ``owner`` (a ``TypeError`` from a slot already disconnected, e.g. by Qt
    itself when its item was destroyed first, is ignored), removes every
    item from the plot's ``PlotItem``, clears the plot, and drops its
    connection record. Call this before ``plot.close()``/``deleteLater()``.

    Args:
        plot: The plot (or other owner passed to ``connect``) to tear down.
    """
    for signal, slot in _CONNECTIONS.pop(plot, []):
        try:
            signal.disconnect(slot)
        except TypeError:
            pass

    plot_item = plot.getPlotItem() if hasattr(plot, "getPlotItem") else None
    if plot_item is not None:
        for item in list(plot_item.items):
            plot_item.removeItem(item)

    plot.clear()


class PlotHost(QtWidgets.QWidget):
    """Thin widget wrapper that tears down its plot on close.

    Optional convenience for callers that want ``teardown()`` called
    automatically. Only ``closeEvent`` is wired (not ``destroyed``/
    ``deleteLater``, which fire during Qt/sip object teardown, too late to
    safely touch the plot) -- callers that never call ``close()`` on the
    host must call :func:`teardown` themselves.
    """

    def __init__(self, plot: pg.PlotWidget, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.plot = plot
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(plot)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        teardown(self.plot)
        super().closeEvent(event)
