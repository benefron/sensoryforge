"""Save plots as figures: PNG (raster) or SVG (vector), exactly as shown.

:func:`export_plot` writes one ``pg.PlotWidget``'s scene -- axes, labels,
colour bar and all -- with pyqtgraph's own exporters, so the file matches
the screen. :class:`ExportFigureButton` asks for a file and calls it;
:func:`export_plots` writes several plots into one folder (the Results
screen's "Export figures").
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import pyqtgraph as pg
from PyQt5 import QtWidgets

#: Default raster width in pixels; height follows the plot's aspect ratio.
DEFAULT_WIDTH_PX = 1600

_SUFFIXES = (".png", ".svg")


def export_plot(
    plot: pg.PlotWidget, path: Path, *, width_px: int = DEFAULT_WIDTH_PX
) -> Path:
    """Write ``plot`` to ``path`` as PNG or SVG, chosen by the suffix.

    Args:
        plot: The plot to save.
        path: Destination ending in ``.png`` or ``.svg``.
        width_px: Width of a PNG in pixels (ignored for SVG).

    Returns:
        The path written.

    Raises:
        ValueError: If the suffix is neither ``.png`` nor ``.svg``.
    """
    from pyqtgraph.exporters import ImageExporter, SVGExporter

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _SUFFIXES:
        raise ValueError(f"figure must be .png or .svg, got {path.name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    item = plot.getPlotItem()
    if suffix == ".svg":
        exporter = SVGExporter(item)
    else:
        exporter = ImageExporter(item)
        exporter.parameters()["width"] = int(width_px)
        exporter.parameters()["antialias"] = True
    exporter.export(str(path))
    return path


def plots_in(widget: QtWidgets.QWidget) -> List[pg.PlotWidget]:
    """Every visible ``pg.PlotWidget`` inside ``widget``, in layout order."""
    if isinstance(widget, pg.PlotWidget):
        return [widget]
    return [p for p in widget.findChildren(pg.PlotWidget) if p.isVisibleTo(widget)]


def export_plots(
    panels: Dict[str, QtWidgets.QWidget], folder: Path, *, suffix: str = ".png"
) -> List[Path]:
    """Write every plot of every panel into ``folder``, one file per plot.

    Args:
        panels: Name -> panel widget; a panel holding several plots writes
            ``name_1``, ``name_2``, ...
        folder: Destination directory (created if needed).
        suffix: ``".png"`` or ``".svg"``.

    Returns:
        The files written.
    """
    written = []
    for name, panel in panels.items():
        plots = plots_in(panel)
        for i, plot in enumerate(plots, start=1):
            stem = name if len(plots) == 1 else f"{name}_{i}"
            written.append(export_plot(plot, Path(folder) / f"{stem}{suffix}"))
    return written


class ExportFigureButton(QtWidgets.QToolButton):
    """ "Save figure..." for one plot: asks for a PNG or SVG file and writes it.

    Args:
        plot: The plot, or a function returning it (for a plot rebuilt later).
        default_name: File name the dialog proposes, without suffix.
        parent: Qt parent.

    Attributes:
        choose_path: ``(parent, title, default) -> str`` returning the chosen
            file ("" to cancel); a file dialog by default, replaceable in tests.
        last_path: The last file written, or ``None``.
    """

    def __init__(
        self,
        plot,
        default_name: str = "figure",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setText("Save figure…")
        self.setToolTip("Save this plot as PNG (image) or SVG (vector), as shown.")
        self._plot = plot
        self._default_name = default_name
        self.choose_path: Callable[..., str] = _ask_for_file
        self.last_path: Optional[Path] = None
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self, *_args: object) -> None:
        path = self.choose_path(self, "Save figure", f"{self._default_name}.png")
        if not path:
            return
        plot = self._plot() if callable(self._plot) else self._plot
        try:
            self.last_path = export_plot(plot, Path(path))
        except (ValueError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "Could not save figure", str(exc))


def _ask_for_file(parent: QtWidgets.QWidget, title: str, default: str) -> str:
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent, title, default, "PNG image (*.png);;SVG vector (*.svg)"
    )
    return path
