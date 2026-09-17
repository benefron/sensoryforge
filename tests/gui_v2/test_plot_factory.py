"""Tests for sensoryforge.gui.widgets.plot_factory.

Covers the themed construction helpers, the F-035-safe connect/teardown
signal wiring, and a GC-enabled regression test for the segfault ledger
entry F-035 describes.
"""

import gc

import pytest

pytest.importorskip("PyQt5")

import pyqtgraph as pg  # noqa: E402
from PyQt5 import QtGui  # noqa: E402

from sensoryforge.gui import theme  # noqa: E402
from sensoryforge.gui.widgets import plot_factory  # noqa: E402

pytestmark = pytest.mark.gui


class TestMakePlot:
    def test_returns_plot_widget(self, qtbot):
        plot = plot_factory.make_plot()
        assert isinstance(plot, pg.PlotWidget)

    def test_white_background(self, qtbot):
        plot = plot_factory.make_plot()
        assert plot.backgroundBrush().color().name().upper() == theme.PALETTE["bg_panel"].upper()

    def test_axis_pens_from_theme(self, qtbot):
        plot = plot_factory.make_plot()
        bottom = plot.getPlotItem().getAxis("bottom")
        left = plot.getPlotItem().getAxis("left")
        assert bottom.pen().color() == theme.AXIS_PEN.color()
        assert left.pen().color() == theme.AXIS_PEN.color()

    def test_grid_enabled(self, qtbot):
        plot = plot_factory.make_plot()
        plot_item = plot.getPlotItem()
        assert plot_item.ctrl.xGridCheck.isChecked()
        assert plot_item.ctrl.yGridCheck.isChecked()

    def test_labels_and_units(self, qtbot):
        plot = plot_factory.make_plot(
            title="Drive", xlabel="Time", ylabel="Current", x_unit="ms", y_unit="mA"
        )
        plot_item = plot.getPlotItem()
        assert "Time" in plot_item.getAxis("bottom").labelText
        assert plot_item.getAxis("bottom").labelUnits == "ms"
        assert "Current" in plot_item.getAxis("left").labelText
        assert plot_item.getAxis("left").labelUnits == "mA"

    def test_no_labels_when_omitted(self, qtbot):
        plot = plot_factory.make_plot()
        plot_item = plot.getPlotItem()
        assert plot_item.getAxis("bottom").labelText == ""
        assert plot_item.getAxis("left").labelText == ""

    def test_menu_disabled(self, qtbot):
        plot = plot_factory.make_plot()
        assert plot.getPlotItem().ctrlMenu is None or plot.plotItem.vb.menuEnabled() is False

    def test_buttons_hidden(self, qtbot):
        plot = plot_factory.make_plot()
        # hideButtons() hides the autorange corner button; verify it's not visible.
        assert plot.getPlotItem().buttonsHidden is True


class TestMakeImagePlot:
    def test_returns_plot_image_colorbar(self, qtbot):
        plot, image_item, colorbar = plot_factory.make_image_plot()
        assert isinstance(plot, pg.PlotWidget)
        assert isinstance(image_item, pg.ImageItem)
        assert isinstance(colorbar, pg.ColorBarItem)

    def test_aspect_locked(self, qtbot):
        plot, image_item, colorbar = plot_factory.make_image_plot()
        assert plot.getPlotItem().vb.state["aspectLocked"] is not False

    def test_image_item_has_lut(self, qtbot):
        plot, image_item, colorbar = plot_factory.make_image_plot()
        assert image_item.lut is not None
        assert len(image_item.lut) == 256

    def test_image_item_added_to_plot(self, qtbot):
        plot, image_item, colorbar = plot_factory.make_image_plot()
        assert image_item in plot.getPlotItem().items


class TestMakeScatterAndRaster:
    def test_make_scatter_symbol(self, qtbot):
        scatter = plot_factory.make_scatter()
        assert isinstance(scatter, pg.ScatterPlotItem)

    def test_make_raster_item_size(self, qtbot):
        raster = plot_factory.make_raster_item("#2563EB")
        assert isinstance(raster, pg.ScatterPlotItem)


class TestConnectTeardown:
    def test_connect_records_and_invokes_callback(self, qtbot):
        plot = plot_factory.make_plot()
        scatter = plot_factory.make_scatter()
        plot.addItem(scatter)

        calls = []

        def on_click(tag, *emitted):
            calls.append(tag)

        plot_factory.connect(scatter.sigClicked, on_click, "tag-a", owner=plot)
        scatter.sigClicked.emit(scatter, [], None)
        assert calls == ["tag-a"]

    def test_teardown_disconnects_recorded_slots(self, qtbot):
        plot = plot_factory.make_plot()
        scatter = plot_factory.make_scatter()
        plot.addItem(scatter)

        calls = []

        def on_click(*args):
            calls.append(args)

        plot_factory.connect(scatter.sigClicked, on_click, owner=plot)
        plot_factory.teardown(plot)

        # Emitting after teardown must not invoke the callback.
        scatter.sigClicked.emit(scatter, [], None)
        assert calls == []

    def test_teardown_removes_items_and_clears(self, qtbot):
        plot = plot_factory.make_plot()
        scatter = plot_factory.make_scatter()
        plot.addItem(scatter)
        assert scatter in plot.getPlotItem().items

        plot_factory.teardown(plot)
        assert scatter not in plot.getPlotItem().items

    def test_teardown_ignores_typeerror_from_prior_disconnect(self, qtbot):
        plot = plot_factory.make_plot()
        scatter = plot_factory.make_scatter()
        plot.addItem(scatter)

        def cb(*args):
            pass

        slot = plot_factory.connect(scatter.sigClicked, cb, owner=plot)
        scatter.sigClicked.disconnect(slot)  # disconnect out from under teardown

        # Must not raise even though the slot is already disconnected.
        plot_factory.teardown(plot)


class TestF035Regression:
    """Regression test for ledger F-035: no segfault with GC enabled."""

    def test_repeated_plot_scatter_build_teardown_with_gc_enabled(self, qtbot):
        was_enabled = gc.isenabled()
        gc.enable()
        try:
            for i in range(50):
                plot = plot_factory.make_plot()
                scatter = plot_factory.make_scatter()
                scatter.setData(
                    x=list(range(500)),
                    y=[float(j % 7) for j in range(500)],
                )
                plot.addItem(scatter)

                def cb(*args):
                    pass

                def cb2(*args):
                    pass

                plot_factory.connect(scatter.sigClicked, cb, "tag", owner=plot)
                plot_factory.connect(scatter.sigHovered, cb2, owner=plot)

                plot.show()
                qtbot.wait(1)

                plot_factory.teardown(plot)
                plot.close()
                plot.deleteLater()
                gc.collect()
        finally:
            if not was_enabled:
                gc.disable()
