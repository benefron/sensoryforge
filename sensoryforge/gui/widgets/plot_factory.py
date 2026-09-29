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
records the connection on its owner so it can be released with
:func:`teardown` before the plot is destroyed.

Where the record lives (ledger F-0eaa5d9): on the owner itself, as an
attribute, never in a module-level table. A module-level
``WeakKeyDictionary`` keyed by the owner cannot release it when the value's
slot holds the owner, as it often does (``grid_preview.py`` binds its own
widget, ``results_map_panel.py`` a method of its panel, and both hold the
plot): the key is never unreferenced, so every torn-down window's plots,
and everything they reference, stayed alive for the life of the process.
Held by the owner, the record is part of the owner's own reference graph
and is freed with it. Connections made without an owner are not recorded
anywhere.
"""

from __future__ import annotations

import functools
import inspect
import weakref
from typing import Any, Callable, List, Optional, Tuple

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from sensoryforge.gui import theme

#: The attribute of an ``owner`` under which :func:`connect` keeps that
#: owner's ``(signal, slot)`` pairs, until :func:`teardown` releases them.
_RECORD_ATTR = "_plot_factory_connections"


def axis_label(text: str, unit: str = "") -> str:
    """The text of an axis label: ``"Time (ms)"``.

    The unit is part of the text, never pyqtgraph's ``units=``: with
    ``units="ms"`` pyqtgraph rescales the ticks and prefixes the unit, so a
    1000 ms run reads ``0 .. 1.0 (kms)`` and a 12 mm array ``(kmm)``.
    SensoryForge's units are fixed (ms, mm, mA, mV, Hz); they are shown as
    they are.

    Args:
        text: What the axis measures.
        unit: Its unit, or ``""`` for none.

    Returns:
        ``"text (unit)"``, or ``text`` when there is no unit.
    """
    return f"{text} ({unit})" if unit else text


def fix_axis_units(plot_item: "pg.PlotItem") -> None:
    """Turn off SI-prefix rescaling on every axis of ``plot_item``.

    Args:
        plot_item: The plot whose axes should show values as they are.
    """
    for axis_name in ("bottom", "left", "right", "top"):
        axis = plot_item.getAxis(axis_name)
        axis.enableAutoSIPrefix(False)
        # An axis that both grows and shrinks its label space to fit its tick
        # labels can oscillate: a narrower axis widens the view, which changes
        # the ticks, which widens the axis again. Growing only converges.
        # (Suspected cause of CI hangs in ViewBox.updateViewRange/resizeEvent
        # and AxisItem.generateDrawSpecs; Linux font metrics differ.)
        axis.setStyle(autoReduceTextSpace=False)


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
        plot_item.setLabel("bottom", axis_label(xlabel, x_unit))
    if ylabel:
        plot_item.setLabel("left", axis_label(ylabel, y_unit))

    fix_axis_units(plot_item)
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
        >>> plot, image, colorbar = make_image_plot(
        ...     "Footprint", "x", "y", x_unit="mm", y_unit="mm"
        ... )
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


def _hold_weakly(value: Any) -> Tuple[bool, Any]:
    """``(True, weakref)`` for a Qt object, ``(False, value)`` for anything else."""
    if isinstance(value, (QtCore.QObject, QtWidgets.QGraphicsItem)):
        return True, weakref.ref(value)
    return False, value


def _call_weakly(callback_ref, packed, *signal_args) -> None:
    """Resolve the weakly held callback and arguments; skip the call if any died."""
    callback = callback_ref()
    if callback is None:
        return
    values = []
    for weak, value in packed:
        if weak:
            value = value()
            if value is None:
                return
        values.append(value)
    callback(*values, *signal_args)


def connect(
    signal, callback: Callable, *args, owner: Optional[object] = None
) -> Callable:
    """Connect a pyqtgraph signal without a reference cycle through the slot.

    The slot holds a bound-method ``callback`` through a
    :class:`weakref.WeakMethod` and every Qt-object argument (a widget, a
    plot, a view box, a graphics item) through a :func:`weakref.ref`, so
    connecting a panel's own method, or passing the panel itself, cannot
    make a cycle ``owner -> record -> slot -> panel -> owner`` that only the
    cyclic collector would free (ledger F-1d91063; F-085 is what such a
    cycle once did). Plain values (names, indices, numbers) are held as
    given. Once any weakly held object has been freed the slot does nothing.
    With an ``owner`` (typically the ``pg.PlotWidget`` the signal's item
    lives on) the connection is recorded on the owner itself, so
    :func:`teardown` can disconnect it later and the record is freed with
    the owner (ledger F-0eaa5d9); :func:`connections` lists it.

    Args:
        signal: A pyqtgraph/Qt bound signal (e.g. ``scatter.sigClicked``).
        callback: The function to invoke on emission, a plain function or a
            bound method (held weakly). Called as
            ``callback(*args, *signal_payload)``.
        *args: Extra positional arguments bound ahead of the signal's own
            emitted arguments; Qt objects among them are held weakly.
        owner: The object connections should be torn down with. It must
            take attributes (every Qt and pyqtgraph object does). If
            omitted, the connection is not recorded anywhere and
            :func:`teardown` cannot release it: keep the returned slot and
            disconnect it yourself, or let it end with the signal's sender.

    Returns:
        The connected slot (the ``functools.partial``), for anyone who wants
        to disconnect it manually.

    Raises:
        TypeError: If ``owner`` cannot take attributes (it has no
            ``__dict__``). Nothing is connected in that case.

    Example:
        >>> scatter = make_scatter()
        >>> connect(scatter.sigClicked, on_click, "population-a", owner=plot)
    """
    record: Optional[List[Tuple[object, Callable]]] = None
    if owner is not None:
        try:
            # vars(), not getattr/setattr: pg.PlotWidget forwards unknown
            # attribute reads to its PlotItem.
            record = vars(owner).setdefault(_RECORD_ATTR, [])
        except TypeError:
            raise TypeError(
                f"plot_factory.connect owner must take attributes, got a "
                f"{type(owner).__name__} with no __dict__"
            ) from None
    if inspect.ismethod(callback):
        callback_ref = weakref.WeakMethod(callback)
    else:
        callback_ref = functools.partial(_identity, callback)
    packed = tuple(_hold_weakly(arg) for arg in args)
    slot = functools.partial(_call_weakly, callback_ref, packed)
    signal.connect(slot)
    if record is not None:
        record.append((signal, slot))
    return slot


def _identity(value: Any) -> Any:
    return value


def connections(owner: object) -> List[Tuple[object, Callable]]:
    """The ``(signal, slot)`` pairs :func:`connect` recorded for ``owner``.

    Args:
        owner: An object passed as ``owner=`` to :func:`connect`.

    Returns:
        A copy of its record, in connection order; empty if it has none
        (never connected, or already torn down).
    """
    return list(getattr(owner, "__dict__", {}).get(_RECORD_ATTR, ()))


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
    for signal, slot in getattr(plot, "__dict__", {}).pop(_RECORD_ATTR, []):
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

    def __init__(
        self, plot: pg.PlotWidget, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.plot = plot
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(plot)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        teardown(self.plot)
        super().closeEvent(event)
