"""Figure export: PyQtGraph plot export + standalone colorbar for compositing.

Export the plot directly from PyQtGraph (exact on-screen appearance) and a
separate colorbar image you can add in Inkscape, Illustrator, or PowerPoint.
"""

from pathlib import Path
from typing import Any, Optional, Tuple

from PyQt5 import QtWidgets

# Matplotlib for colorbar only (small, simple)
try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.figure import Figure
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Discrete levels for innervation strength (matches receptor dot coloring)
NUM_WEIGHT_LEVELS = 12


def _populations_with_weights(tab: Any):
    """Yield (population, weight_min, weight_max, base_color) for each with weights."""
    for pop in getattr(tab, "populations", []):
        if not pop.visible:
            continue
        if pop.innervation_weights is not None:
            c = pop.color
            rgb = (c.red() / 255, c.green() / 255, c.blue() / 255)
            yield pop, pop.weight_min, pop.weight_max, rgb


def _export_colorbar(
    path: str,
    weight_min: float,
    weight_max: float,
    base_color: Optional[Tuple[float, float, float]] = None,
    fmt: str = "png",
    dpi: int = 150,
    num_levels: int = NUM_WEIGHT_LEVELS,
) -> None:
    """Export a standalone vertical colorbar with discrete levels for compositing."""
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for colorbar export")
    if base_color is None:
        base_color = (0.2, 0.4, 0.8)  # default blue
    # Discrete color bands: light (min) to dark (max), matching receptor dots
    fractions = [i / (num_levels - 1) for i in range(num_levels)]
    colors = []
    for frac in fractions:
        factor = 1.0 - frac * 0.6  # 1.0 at min (light), 0.4 at max (dark)
        colors.append((
            base_color[0] * factor,
            base_color[1] * factor,
            base_color[2] * factor,
        ))
    cmap = LinearSegmentedColormap.from_list("weight", colors, N=num_levels)
    boundaries = [
        weight_min + (weight_max - weight_min) * i / num_levels
        for i in range(num_levels + 1)
    ]
    norm = BoundaryNorm(boundaries, cmap.N)
    fig = Figure(figsize=(1.2, 3.0), dpi=100)
    ax = fig.add_subplot(111)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label("Innervation weight", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", format=fmt)


class ExportDialog(QtWidgets.QDialog):
    """Export plot (PyQtGraph) and/or standalone colorbar for easy compositing."""

    def __init__(self, tab: Any, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._tab = tab
        self.setWindowTitle("Export Figure")
        self.setMinimumWidth(400)
        layout = QtWidgets.QVBoxLayout(self)

        # Plot export
        plot_group = QtWidgets.QGroupBox("Export Plot")
        plot_layout = QtWidgets.QFormLayout(plot_group)
        self.spin_width = QtWidgets.QSpinBox()
        self.spin_width.setRange(400, 4000)
        self.spin_width.setValue(1200)
        self.spin_width.setSuffix(" px")
        plot_layout.addRow("Width:", self.spin_width)
        self.spin_height = QtWidgets.QSpinBox()
        self.spin_height.setRange(300, 3000)
        self.spin_height.setValue(900)
        self.spin_height.setSuffix(" px")
        plot_layout.addRow("Height:", self.spin_height)
        self.btn_export_plot = QtWidgets.QPushButton("Export Plot (PNG/SVG)...")
        self.btn_export_plot.setToolTip(
            "Export the current plot exactly as shown (PyQtGraph). "
            "Use PNG or SVG in the file dialog."
        )
        self.btn_export_plot.clicked.connect(self._export_plot)
        plot_layout.addRow("", self.btn_export_plot)
        layout.addWidget(plot_group)

        # Colorbar export
        colorbar_group = QtWidgets.QGroupBox("Export Colorbar")
        colorbar_layout = QtWidgets.QVBoxLayout(colorbar_group)
        colorbar_layout.addWidget(
            QtWidgets.QLabel(
                "Export one colorbar per population (discrete levels for clarity). "
                "Add next to the plot in Inkscape, Illustrator, or PowerPoint."
            )
        )
        self.btn_export_colorbar = QtWidgets.QPushButton("Export All Colorbars...")
        self.btn_export_colorbar.setToolTip(
            "Export one colorbar per population to a folder. "
            "Each file: {population_name}_colorbar.png (or .svg)."
        )
        self.btn_export_colorbar.clicked.connect(self._export_colorbar)
        colorbar_layout.addWidget(self.btn_export_colorbar)
        layout.addWidget(colorbar_group)

        layout.addWidget(
            QtWidgets.QLabel(
                "Tip: Export both, then composite in your figure editor."
            )
        )

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def _export_plot(self) -> None:
        """Export the PyQtGraph plot directly (exact on-screen appearance)."""
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "",
            "PNG image (*.png);;SVG vector (*.svg);;All files (*)",
        )
        if not path:
            return
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".png" if "PNG" in selected_filter else ".svg")
        try:
            from pyqtgraph.exporters import ImageExporter, SVGExporter

            scene = self._tab.plot_widget.scene()
            if path.suffix.lower() == ".svg":
                exporter = SVGExporter(scene)
            else:
                exporter = ImageExporter(scene)
                exporter.parameters()["width"] = self.spin_width.value()
                exporter.parameters()["height"] = self.spin_height.value()
                exporter.parameters()["antialias"] = True
            exporter.export(str(path))
            QtWidgets.QMessageBox.information(
                self, "Exported", f"Plot saved to {path}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Export failed", f"Could not export plot:\n{exc}",
            )

    def _export_colorbar(self) -> None:
        """Export one colorbar per population to a chosen directory."""
        pops = list(_populations_with_weights(self._tab))
        if not pops:
            QtWidgets.QMessageBox.warning(
                self,
                "No data",
                "Generate a population with innervation first.",
            )
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose folder for colorbars",
            "",
        )
        if not out_dir:
            return
        out_dir = Path(out_dir)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Format",
            "Export as SVG (vector) or PNG (raster)?\nYes=SVG, No=PNG",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes,
        )
        if reply == QtWidgets.QMessageBox.Cancel:
            return
        use_svg = reply == QtWidgets.QMessageBox.Yes
        fmt = "svg" if use_svg else "png"
        ext = ".svg" if use_svg else ".png"
        try:
            for pop, w_min, w_max, rgb in pops:
                safe_name = "".join(
                    c if c.isalnum() or c in "_-" else "_"
                    for c in pop.name
                ).strip() or "population"
                path = out_dir / f"{safe_name}_colorbar{ext}"
                _export_colorbar(
                    str(path),
                    w_min,
                    w_max,
                    base_color=rgb,
                    fmt=fmt,
                    dpi=150,
                )
            QtWidgets.QMessageBox.information(
                self,
                "Exported",
                f"Saved {len(pops)} colorbar(s) to {out_dir}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Export failed", f"Could not export colorbars:\n{exc}",
            )
