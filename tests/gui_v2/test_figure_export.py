"""Plots can be saved as figures (PNG and SVG), as the old Grid tab could."""

import pytest

pytestmark = pytest.mark.gui

import numpy as np  # noqa: E402
from PyQt5 import QtGui  # noqa: E402

from sensoryforge.config.schema import GridConfig, SensoryForgeConfig  # noqa: E402
from sensoryforge.gui.widgets import plot_factory  # noqa: E402
from sensoryforge.gui.widgets.figure_export import export_plot  # noqa: E402
from sensoryforge.gui.widgets.grid_preview import GridPreview  # noqa: E402


def _drawn_fraction(png_path) -> float:
    """Fraction of pixels that are not the (white) background."""
    image = QtGui.QImage(str(png_path)).convertToFormat(QtGui.QImage.Format_RGB32)
    ptr = image.constBits()
    ptr.setsize(image.byteCount())
    pixels = np.frombuffer(ptr, np.uint8).reshape(image.height(), image.width(), 4)
    return float((pixels[..., :3].min(axis=2) < 200).mean())


def test_a_plot_is_saved_as_png_at_the_requested_width_and_as_svg(qtbot, tmp_path):
    plot = plot_factory.make_plot("Test", "Time", "Rate", x_unit="ms", y_unit="Hz")
    qtbot.addWidget(plot)
    plot.resize(600, 400)
    plot.plot([0, 1, 2, 3], [0, 3, 1, 4])
    png = export_plot(plot, tmp_path / "f.png", width_px=1200)
    image = QtGui.QImage(str(png))
    assert image.width() == 1200 and image.height() > 0
    assert _drawn_fraction(png) > 0.001
    svg = export_plot(plot, tmp_path / "f.svg")
    text = svg.read_text()
    assert "<svg" in text and "Time (ms)" in text
    with pytest.raises(ValueError):
        export_plot(plot, tmp_path / "f.jpg")


def test_the_receptor_preview_saves_what_it_shows(qtbot, tmp_path):
    preview = GridPreview()
    qtbot.addWidget(preview)
    preview.resize(500, 500)
    preview.set_grids([GridConfig(name="g", rows=10, cols=10)])
    target = tmp_path / "receptors.png"
    preview.export_button.choose_path = lambda *a: str(target)
    preview.export_button.click()
    assert preview.export_button.last_path == target
    assert _drawn_fraction(target) > 0.005


def test_results_export_writes_every_visible_panel(qtbot, tmp_path):
    from sensoryforge.gui.app import SensoryForgeApp
    from sensoryforge.gui.session import Session

    config = SensoryForgeConfig.from_yaml_file("sensoryforge/presets/tactile_sa1_ra1.yml")
    config.grids[0].rows = 12
    config.grids[0].cols = 12
    session = Session(config)
    window = SensoryForgeApp(session)
    qtbot.addWidget(window)
    window.run_bar.duration_spin.setValue(30.0)
    with qtbot.waitSignal(window.run_controller.finished, timeout=60000):
        window.run_bar.run_button.click()
    results = window._screens["results"]
    written = results.export_figures(tmp_path / "figs")
    names = sorted(p.name for p in written)
    for stem in ("stimulus", "raster", "rate", "map", "trace_1", "trace_2"):
        assert f"{stem}.png" in names and f"{stem}.svg" in names, names
    assert _drawn_fraction(tmp_path / "figs" / "raster.png") > 0.001
