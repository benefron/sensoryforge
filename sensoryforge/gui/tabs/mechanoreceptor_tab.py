import json
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg  # type: ignore
from pyqtgraph.exporters import ImageExporter, SVGExporter  # type: ignore

# Ensure repository root on sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sensoryforge.core.grid import GridManager  # noqa: E402
from sensoryforge.core.innervation import (  # noqa: E402
    InnervationModule,
    FlatInnervationModule,
    create_innervation,
    create_neuron_centers,
)
from sensoryforge.core.composite_grid import CompositeReceptorGrid  # noqa: E402


CONFIG_SCHEMA_VERSION = "1.0.0"
CONFIG_JSON_NAME = "config.json"

# Default color palette for new grid layers
_GRID_COLORS = [
    QtGui.QColor(66, 135, 245, 200),   # Blue
    QtGui.QColor(245, 135, 66, 200),   # Orange
    QtGui.QColor(66, 245, 135, 200),   # Green
    QtGui.QColor(245, 66, 135, 200),   # Pink
    QtGui.QColor(135, 66, 245, 200),   # Purple
    QtGui.QColor(245, 245, 66, 200),   # Yellow
]


def _default_population_name(neuron_type: str, index: int) -> str:
    return f"{neuron_type} #{index}" if neuron_type else f"Population #{index}"


class CollapsibleGroupBox(QtWidgets.QWidget):
    """Group box that collapses/expands its content when the header is toggled."""

    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._base_title = title
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._header = QtWidgets.QPushButton("▶ " + title)
        self._header.setCheckable(True)
        self._header.setChecked(False)
        self._header.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; border: none; padding: 4px; }"
            "QPushButton:hover { background-color: palette(midlight); }"
        )
        self._header.toggled.connect(self._on_toggled)
        layout.addWidget(self._header)
        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QFormLayout(self._content)
        layout.addWidget(self._content)
        self._content.setVisible(False)

    def _on_toggled(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._header.setText(("▼ " if checked else "▶ ") + self._base_title)

    def layout(self) -> QtWidgets.QFormLayout:
        return self._content_layout

    def addRow(self, *args) -> None:
        self._content_layout.addRow(*args)


@dataclass
class GridEntry:
    """Configuration for one receptor grid layer in the unified grid workspace."""

    name: str
    arrangement: str = "grid"
    rows: int = 40
    cols: int = 40
    density: float = 100.0
    spacing: float = 0.15
    center_x: float = 0.0
    center_y: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor(66, 135, 245, 200))
    visible: bool = True

    def to_dict(self) -> dict:
        """Serialize to plain dict for config round-trip."""
        return {
            "name": self.name,
            "arrangement": self.arrangement,
            "rows": self.rows,
            "cols": self.cols,
            "density": self.density,
            "spacing": self.spacing,
            "center": [self.center_x, self.center_y],
            "offset": [self.offset_x, self.offset_y],
            "color": [self.color.red(), self.color.green(), self.color.blue(), self.color.alpha()],
            "visible": self.visible,
        }

    @staticmethod
    def from_dict(d: dict) -> "GridEntry":
        """Deserialize from a config dict."""
        center = d.get("center", [0.0, 0.0])
        offset = d.get("offset", [0.0, 0.0])
        c = d.get("color", [66, 135, 245, 200])
        return GridEntry(
            name=d.get("name", "grid"),
            arrangement=d.get("arrangement", "grid"),
            rows=int(d.get("rows", 40)),
            cols=int(d.get("cols", 40)),
            density=float(d.get("density", 100.0)),
            spacing=float(d.get("spacing", 0.15)),
            center_x=float(center[0]),
            center_y=float(center[1]),
            offset_x=float(offset[0]),
            offset_y=float(offset[1]),
            color=QtGui.QColor(c[0], c[1], c[2], c[3] if len(c) > 3 else 200),
            visible=bool(d.get("visible", True)),
        )


@dataclass
class NeuronPopulation:
    """Configuration and visualization handles for a population layer."""

    name: str
    neuron_type: str
    color: QtGui.QColor
    neurons_per_row: int
    connections_per_neuron: float
    sigma_d_mm: float
    weight_min: float
    weight_max: float
    innervation_method: str = "gaussian"
    neuron_rows: Optional[int] = None
    neuron_cols: Optional[int] = None
    neuron_arrangement: str = "grid"
    use_distance_weights: bool = False
    far_connection_fraction: float = 0.0
    far_sigma_factor: float = 5.0
    max_distance_mm: float = 1.0
    decay_function: str = "exponential"
    decay_rate: float = 2.0
    seed: Optional[int] = None
    edge_offset: Optional[float] = None
    target_grid: Optional[str] = None  # Name of target grid layer
    module: Optional[InnervationModule] = None
    flat_module: Optional[FlatInnervationModule] = None
    scatter_item: Optional[pg.ScatterPlotItem] = None
    connection_items: List[Tuple[pg.PlotDataItem, QtGui.QColor, float]] = field(
        default_factory=list
    )
    heatmap_item: Optional[pg.ImageItem] = None
    receptor_item: Optional[pg.ScatterPlotItem] = None
    highlight_neuron_item: Optional[pg.ScatterPlotItem] = None
    highlight_shadow_item: Optional[pg.ScatterPlotItem] = None
    highlight_connection_items: List = field(default_factory=list)
    highlight_receptor_item: Optional[pg.ScatterPlotItem] = None
    highlight_receptor_shadow_item: Optional[pg.ScatterPlotItem] = None
    visible: bool = True

    def instantiate(self, grid_manager: GridManager) -> None:
        """Build the PyTorch innervation module for this configuration (grid-based)."""
        weight_range: Tuple[float, float] = (self.weight_min, self.weight_max)
        kwargs = {
            "neuron_type": self.neuron_type,
            "grid_manager": grid_manager,
            "neurons_per_row": self.neurons_per_row,
            "neuron_rows": self.neuron_rows,
            "neuron_cols": self.neuron_cols,
            "neuron_arrangement": self.neuron_arrangement,
            "connections_per_neuron": self.connections_per_neuron,
            "sigma_d_mm": self.sigma_d_mm,
            "weight_range": weight_range,
            "use_distance_weights": self.use_distance_weights,
            "far_connection_fraction": self.far_connection_fraction,
            "far_sigma_factor": self.far_sigma_factor,
            "seed": self.seed,
            "edge_offset": self.edge_offset,
        }
        self.module = InnervationModule(**kwargs)
        self.flat_module = None
        if self.innervation_method != "gaussian":
            xx, yy = grid_manager.get_coordinates()
            receptor_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
            neuron_centers = self.module.neuron_centers

            if self.innervation_method == "one_to_one":
                method_params = {
                    "connections_per_neuron": self.connections_per_neuron,
                    "sigma_d_mm": self.sigma_d_mm,
                    "weight_range": (self.weight_min, self.weight_max),
                    "seed": self.seed,
                }
            elif self.innervation_method == "distance_weighted":
                method_params = {
                    "connections_per_neuron": self.connections_per_neuron,
                    "sigma_d_mm": self.sigma_d_mm,
                    "max_distance_mm": self.max_distance_mm,
                    "decay_function": self.decay_function,
                    "decay_rate": self.decay_rate,
                    "weight_range": (self.weight_min, self.weight_max),
                    "seed": self.seed,
                }
            else:
                method_params = {}
            weights = create_innervation(
                receptor_coords=receptor_coords,
                neuron_centers=neuron_centers,
                method=self.innervation_method,
                device=grid_manager.get_grid_properties()["device"],
                **method_params,
            )
            grid_h, grid_w = grid_manager.grid_size
            reshaped = weights.view(neuron_centers.shape[0], grid_h, grid_w)
            self.module.innervation_map = reshaped
            self.module.innervation_weights.data.copy_(reshaped)

    def instantiate_flat(self, receptor_coords: torch.Tensor,
                         xlim: Tuple[float, float],
                         ylim: Tuple[float, float]) -> None:
        """Build innervation from flat receptor coordinates (composite grid)."""
        self.flat_module = FlatInnervationModule(
            neuron_type=self.neuron_type,
            receptor_coords=receptor_coords,
            neurons_per_row=self.neurons_per_row,
            neuron_rows=self.neuron_rows,
            neuron_cols=self.neuron_cols,
            neuron_arrangement=self.neuron_arrangement,
            xlim=xlim,
            ylim=ylim,
            innervation_method=self.innervation_method,
            connections_per_neuron=self.connections_per_neuron,
            sigma_d_mm=self.sigma_d_mm,
            max_sigma_distance=3.0,
            weight_range=(self.weight_min, self.weight_max),
            use_distance_weights=self.use_distance_weights,
            far_connection_fraction=self.far_connection_fraction,
            far_sigma_factor=self.far_sigma_factor,
            max_distance_mm=self.max_distance_mm,
            decay_function=self.decay_function,
            decay_rate=self.decay_rate,
            seed=self.seed,
            edge_offset=self.edge_offset,
        )
        self.module = None

    @property
    def neuron_centers(self) -> Optional[torch.Tensor]:
        """Get neuron centers from whichever module is active."""
        if self.module is not None:
            return self.module.neuron_centers
        if self.flat_module is not None:
            return self.flat_module.neuron_centers
        return None

    @property
    def innervation_weights(self) -> Optional[torch.Tensor]:
        """Get innervation weights from whichever module is active."""
        if self.module is not None:
            return self.module.innervation_weights
        if self.flat_module is not None:
            return self.flat_module.innervation_weights
        return None

    @property
    def num_neurons(self) -> int:
        if self.module is not None:
            return self.module.num_neurons
        if self.flat_module is not None:
            return self.flat_module.num_neurons
        return 0

    def delete_graphics(self, plot: pg.PlotItem) -> None:
        if self.scatter_item is not None:
            plot.removeItem(self.scatter_item)
        for item, _, _ in self.connection_items:
            plot.removeItem(item)
        if self.heatmap_item is not None:
            plot.removeItem(self.heatmap_item)
        if self.receptor_item is not None:
            plot.removeItem(self.receptor_item)
        if self.highlight_neuron_item is not None:
            plot.removeItem(self.highlight_neuron_item)
        if self.highlight_shadow_item is not None:
            plot.removeItem(self.highlight_shadow_item)
        for item in self.highlight_connection_items:
            plot.removeItem(item)
        if self.highlight_receptor_item is not None:
            plot.removeItem(self.highlight_receptor_item)
        if self.highlight_receptor_shadow_item is not None:
            plot.removeItem(self.highlight_receptor_shadow_item)
        self.scatter_item = None
        self.connection_items.clear()
        self.heatmap_item = None
        self.receptor_item = None
        self.highlight_neuron_item = None
        self.highlight_shadow_item = None
        self.highlight_connection_items.clear()
        self.highlight_receptor_item = None
        self.highlight_receptor_shadow_item = None


class MechanoreceptorTab(QtWidgets.QWidget):
    """Configure mechanoreceptor grid and sensory neuron overlays."""

    grid_changed = QtCore.pyqtSignal(object)
    configuration_directory_changed = QtCore.pyqtSignal(object)
    populations_changed = QtCore.pyqtSignal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        self.grid_manager: Optional[GridManager] = None
        self._composite_grid = None
        self._grid_type = "standard"  # kept for backward compat in save/load
        self.grid_scatter: Optional[pg.ScatterPlotItem] = None
        self._grid_scatter_items: List[pg.ScatterPlotItem] = []
        self._grid_scatter_map: List[Tuple[GridEntry, pg.ScatterPlotItem]] = []
        self._block_grid_item_changed = False
        self.populations: List[NeuronPopulation] = []
        self._population_counter = 1
        self._selected_population: Optional[NeuronPopulation] = None
        self._selected_neuron_idx: Optional[int] = None
        self._block_population_item_changed = False
        self._current_config_dir: Optional[Path] = None
        self._composite_populations: List[dict] = []
        # Unified grid workspace
        self._grid_entries: List[GridEntry] = []
        self._grid_counter = 1
        self._block_grid_editor = False
        self._setup_ui()
        self._configure_plot()
        if self._grid_entries:
            self._generate_grids()
        self.populations_changed.emit(list(self.populations))

    @staticmethod
    def _sanitize_name(name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
        return sanitized or "bundle"

    def _grid_descriptor(self) -> str:
        if self.grid_manager is not None:
            rows, cols = self.grid_manager.grid_size
        else:
            rows = self.spin_grid_rows.value()
            cols = self.spin_grid_cols.value()
        rows = int(rows)
        cols = int(cols)
        if rows == cols:
            return f"grid{rows}"
        return f"grid{rows}x{cols}"

    def _population_descriptor(self) -> str:
        if not self.populations:
            return "nopop"
        bucket: Counter[Tuple[str, int]] = Counter()
        for population in self.populations:
            neuron_type = population.neuron_type or "SA"
            neurons_per_row = int(population.neurons_per_row)
            bucket[(neuron_type, neurons_per_row)] += 1
        tokens: List[str] = []
        for (neuron_type, neurons_per_row), count in sorted(bucket.items()):
            base = f"{neuron_type}{neurons_per_row}"
            if count > 1:
                base = f"{base}x{count}"
            tokens.append(base)
        return "_".join(tokens)

    def _default_configuration_name(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        parts = [
            self._grid_descriptor(),
            self._population_descriptor(),
            timestamp,
        ]
        return self._sanitize_name("_".join(parts))

    @staticmethod
    def _color_to_rgba(color: QtGui.QColor) -> List[int]:
        r, g, b, a = color.getRgb()
        return [int(r), int(g), int(b), int(a)]

    @staticmethod
    def _rgba_to_color(rgba: List[int]) -> QtGui.QColor:
        if len(rgba) == 4:
            return QtGui.QColor(*rgba)
        if len(rgba) == 3:
            return QtGui.QColor(rgba[0], rgba[1], rgba[2], 255)
        return QtGui.QColor(128, 128, 128, 255)

    def _clear_populations(self) -> None:
        for population in self.populations:
            population.delete_graphics(self.plot)
        self.populations.clear()
        self.population_list.clear()
        self._population_counter = 1
        self._selected_population = None
        self._selected_neuron_idx = None
        self.populations_changed.emit([])

    def _setup_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        # Plot area
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget, stretch=3)
        self.plot = self.plot_widget.addPlot()

        # Control panel with scroll area to fit smaller displays
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        layout.addWidget(scroll_area, stretch=2)

        control_panel = QtWidgets.QWidget()
        control_panel.setMinimumWidth(320)
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setAlignment(QtCore.Qt.AlignTop)
        scroll_area.setWidget(control_panel)

        # Global seed (top-level, used for both grid and neuron populations)
        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(QtWidgets.QLabel("Seed (-1 = random):"))
        self.spin_global_seed = QtWidgets.QSpinBox()
        self.spin_global_seed.setRange(-1, 1000000)
        self.spin_global_seed.setValue(42)
        self.spin_global_seed.setToolTip("Global random seed for grid and neuron arrangements.")
        seed_row.addWidget(self.spin_global_seed)
        control_layout.addLayout(seed_row)

        # Grid Workspace — collapsible (Phase 3 B.1-B.2)
        grid_group = CollapsibleGroupBox("Grid Workspace")
        grid_layout = QtWidgets.QVBoxLayout()
        grid_layout.setContentsMargins(8, 8, 8, 12)

        # Grid list (with checkboxes for visibility, linked to visualization)
        self.grid_list = QtWidgets.QListWidget()
        self.grid_list.setMaximumHeight(140)
        self.grid_list.currentRowChanged.connect(self._on_grid_selected)
        self.grid_list.itemChanged.connect(self._on_grid_item_changed)
        grid_layout.addWidget(self.grid_list)

        grid_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_add_grid = QtWidgets.QPushButton("Add Grid")
        self.btn_remove_grid = QtWidgets.QPushButton("Remove Grid")
        self.btn_add_grid.clicked.connect(self._on_add_grid)
        self.btn_remove_grid.clicked.connect(self._on_remove_grid)
        grid_btn_layout.addWidget(self.btn_add_grid)
        grid_btn_layout.addWidget(self.btn_remove_grid)
        grid_layout.addLayout(grid_btn_layout)

        # Per-grid editor panel
        self._grid_editor = QtWidgets.QWidget()
        ge_layout = QtWidgets.QFormLayout(self._grid_editor)
        ge_layout.setContentsMargins(0, 4, 0, 0)

        self.txt_grid_name = QtWidgets.QLineEdit()
        self.txt_grid_name.editingFinished.connect(self._on_grid_editor_changed)
        ge_layout.addRow("Name:", self.txt_grid_name)

        self.cmb_grid_arrangement = QtWidgets.QComboBox()
        self.cmb_grid_arrangement.addItems(["grid", "poisson", "hex", "jittered_grid", "blue_noise"])
        self.cmb_grid_arrangement.currentTextChanged.connect(self._on_grid_arrangement_changed)
        ge_layout.addRow("Arrangement:", self.cmb_grid_arrangement)

        self.spin_grid_rows = QtWidgets.QSpinBox()
        self.spin_grid_rows.setRange(4, 256)
        self.spin_grid_rows.setValue(40)
        self.spin_grid_rows.valueChanged.connect(self._on_grid_editor_changed)
        ge_layout.addRow("Rows:", self.spin_grid_rows)

        self.spin_grid_cols = QtWidgets.QSpinBox()
        self.spin_grid_cols.setRange(4, 256)
        self.spin_grid_cols.setValue(40)
        self.spin_grid_cols.valueChanged.connect(self._on_grid_editor_changed)
        ge_layout.addRow("Cols:", self.spin_grid_cols)

        self.dbl_spacing = QtWidgets.QDoubleSpinBox()
        self.dbl_spacing.setDecimals(4)
        self.dbl_spacing.setRange(0.01, 1.0)
        self.dbl_spacing.setSingleStep(0.01)
        self.dbl_spacing.setValue(0.15)
        self.dbl_spacing.valueChanged.connect(self._on_grid_editor_changed)
        ge_layout.addRow("Spacing (mm):", self.dbl_spacing)

        pos_group = CollapsibleGroupBox("Position (center, offset)")
        pos_layout = pos_group.layout()
        self.dbl_center_x = QtWidgets.QDoubleSpinBox()
        self.dbl_center_x.setRange(-50.0, 50.0)
        self.dbl_center_x.setDecimals(3)
        self.dbl_center_x.setValue(0.0)
        self.dbl_center_x.valueChanged.connect(self._on_grid_editor_changed)
        pos_layout.addRow("Center X (mm):", self.dbl_center_x)
        self.dbl_center_y = QtWidgets.QDoubleSpinBox()
        self.dbl_center_y.setRange(-50.0, 50.0)
        self.dbl_center_y.setDecimals(3)
        self.dbl_center_y.setValue(0.0)
        self.dbl_center_y.valueChanged.connect(self._on_grid_editor_changed)
        pos_layout.addRow("Center Y (mm):", self.dbl_center_y)
        self.dbl_offset_x = QtWidgets.QDoubleSpinBox()
        self.dbl_offset_x.setRange(-10.0, 10.0)
        self.dbl_offset_x.setDecimals(3)
        self.dbl_offset_x.setValue(0.0)
        self.dbl_offset_x.valueChanged.connect(self._on_grid_editor_changed)
        pos_layout.addRow("Offset X (mm):", self.dbl_offset_x)
        self.dbl_offset_y = QtWidgets.QDoubleSpinBox()
        self.dbl_offset_y.setRange(-10.0, 10.0)
        self.dbl_offset_y.setDecimals(3)
        self.dbl_offset_y.setValue(0.0)
        self.dbl_offset_y.valueChanged.connect(self._on_grid_editor_changed)
        pos_layout.addRow("Offset Y (mm):", self.dbl_offset_y)
        ge_layout.addRow(pos_group)

        self.btn_grid_color = QtWidgets.QPushButton("Pick Color")
        self.btn_grid_color.clicked.connect(self._on_pick_grid_color)
        ge_layout.addRow("Color:", self.btn_grid_color)
        ge_layout.setContentsMargins(0, 4, 0, 16)

        self._grid_editor.setVisible(False)
        grid_layout.addWidget(self._grid_editor)

        self.btn_generate_grid = QtWidgets.QPushButton("Generate Grid(s)")
        self.btn_generate_grid.clicked.connect(self._on_generate_grid)
        grid_layout.addWidget(self.btn_generate_grid)
        grid_group.addRow(grid_layout)
        control_layout.addWidget(grid_group)

        # Population controls (collapsible)
        pop_group = CollapsibleGroupBox("Neuron Populations")
        pop_layout = pop_group.layout()
        self._pop_layout = pop_layout
        self.txt_population_name = QtWidgets.QLineEdit()
        self.spin_neurons_per_row = QtWidgets.QSpinBox()
        self.spin_neurons_per_row.setRange(1, 128)
        self.spin_neurons_per_row.setValue(10)
        self.dbl_connections = QtWidgets.QDoubleSpinBox()
        self.dbl_connections.setDecimals(1)
        self.dbl_connections.setRange(1.0, 500.0)
        self.dbl_connections.setValue(28.0)
        self.dbl_sigma = QtWidgets.QDoubleSpinBox()
        self.dbl_sigma.setDecimals(4)
        self.dbl_sigma.setRange(0.01, 3.0)
        self.dbl_sigma.setValue(0.3)
        self.cmb_innervation_method = QtWidgets.QComboBox()
        self.cmb_innervation_method.addItems(
            ["gaussian", "one_to_one", "uniform", "distance_weighted"]
        )
        self.cmb_innervation_method.setCurrentText("gaussian")
        self.dbl_max_distance = QtWidgets.QDoubleSpinBox()
        self.dbl_max_distance.setDecimals(3)
        self.dbl_max_distance.setRange(0.01, 10.0)
        self.dbl_max_distance.setValue(1.0)
        self.cmb_decay_function = QtWidgets.QComboBox()
        self.cmb_decay_function.addItems(
            ["exponential", "linear", "inverse_square"]
        )
        self.cmb_decay_function.setCurrentText("exponential")
        self.dbl_decay_rate = QtWidgets.QDoubleSpinBox()
        self.dbl_decay_rate.setDecimals(3)
        self.dbl_decay_rate.setRange(0.01, 10.0)
        self.dbl_decay_rate.setValue(2.0)
        self.dbl_weight_min = QtWidgets.QDoubleSpinBox()
        self.dbl_weight_min.setDecimals(2)
        self.dbl_weight_min.setRange(0.0, 10.0)
        self.dbl_weight_min.setValue(0.1)
        self.dbl_weight_min.setSingleStep(0.05)
        self.dbl_weight_max = QtWidgets.QDoubleSpinBox()
        self.dbl_weight_max.setDecimals(2)
        self.dbl_weight_max.setRange(0.0, 10.0)
        self.dbl_weight_max.setValue(1.0)
        self.dbl_weight_max.setSingleStep(0.05)
        self.dbl_edge_offset = QtWidgets.QDoubleSpinBox()
        self.dbl_edge_offset.setDecimals(4)
        self.dbl_edge_offset.setRange(0.0, 20.0)
        self.dbl_edge_offset.setValue(0.15)
        self.dbl_edge_offset.setSingleStep(0.01)
        self.btn_pick_color = QtWidgets.QPushButton()
        self.btn_pick_color.clicked.connect(self._on_pick_color)
        self.btn_pick_color.setText("Pick Color")
        self._population_color = QtGui.QColor(66, 135, 245)
        self._update_color_button()
        self.cmb_innervation_method.currentTextChanged.connect(
            self._on_innervation_method_changed
        )
        self.cmb_target_grid = QtWidgets.QComboBox()
        self.cmb_target_grid.addItem("(all receptors)")
        self.cmb_target_grid.setToolTip(
            "Select which grid layer this population targets."
        )
        self.cmb_target_grid.setEnabled(True)
        pop_layout.addRow("Name:", self.txt_population_name)
        pop_layout.addRow("Target Grid:", self.cmb_target_grid)
        pop_layout.addRow("Neurons/row:", self.spin_neurons_per_row)
        pop_layout.addRow("Connections:", self.dbl_connections)
        pop_layout.addRow("Sigma d (mm):", self.dbl_sigma)
        pop_layout.addRow("Innervation Method:", self.cmb_innervation_method)

        weights_group = CollapsibleGroupBox("Weights")
        weights_group.layout().addRow("Weight min:", self.dbl_weight_min)
        weights_group.layout().addRow("Weight max:", self.dbl_weight_max)
        pop_layout.addRow(weights_group)

        pop_layout.addRow("Edge offset (mm):", self.dbl_edge_offset)
        pop_layout.addRow("Color:", self.btn_pick_color)
        self._dist_params_group = CollapsibleGroupBox("Distance-weighted params")
        self._dist_params_group.addRow("Max Distance (mm):", self.dbl_max_distance)
        self._dist_params_group.addRow("Decay Function:", self.cmb_decay_function)
        self._dist_params_group.addRow("Decay Rate:", self.dbl_decay_rate)
        pop_layout.addRow(self._dist_params_group)
        self._dist_params_group.setVisible(False)
        adv_group = CollapsibleGroupBox("Advanced")
        self.dbl_far_connection_fraction = QtWidgets.QDoubleSpinBox()
        self.dbl_far_connection_fraction.setDecimals(3)
        self.dbl_far_connection_fraction.setRange(0.0, 1.0)
        self.dbl_far_connection_fraction.setSingleStep(0.05)
        self.dbl_far_connection_fraction.setValue(0.0)
        self.dbl_far_connection_fraction.setToolTip(
            "Fraction of connections from far receptors (beyond 5*sigma) to break coherence."
        )
        self.dbl_far_sigma_factor = QtWidgets.QDoubleSpinBox()
        self.dbl_far_sigma_factor.setDecimals(1)
        self.dbl_far_sigma_factor.setRange(2.0, 20.0)
        self.dbl_far_sigma_factor.setSingleStep(0.5)
        self.dbl_far_sigma_factor.setValue(5.0)
        self.dbl_far_sigma_factor.setToolTip("Receptors beyond this × sigma are 'far'.")
        adv_group.addRow("Far connection %:", self.dbl_far_connection_fraction)
        adv_group.addRow("Far sigma factor:", self.dbl_far_sigma_factor)
        self.spin_neuron_rows = QtWidgets.QSpinBox()
        self.spin_neuron_rows.setRange(1, 128)
        self.spin_neuron_rows.setValue(10)
        self.spin_neuron_rows.setToolTip("Number of neuron rows (vertical).")
        self.spin_neuron_cols = QtWidgets.QSpinBox()
        self.spin_neuron_cols.setRange(1, 128)
        self.spin_neuron_cols.setValue(10)
        self.spin_neuron_cols.setToolTip("Number of neuron columns (horizontal).")
        self.cmb_neuron_arrangement = QtWidgets.QComboBox()
        self.cmb_neuron_arrangement.addItems(["grid", "poisson", "hex", "blue_noise", "jittered_grid"])
        self.cmb_neuron_arrangement.setToolTip("Spatial distribution of neuron centers.")
        adv_group.addRow("Neuron rows:", self.spin_neuron_rows)
        adv_group.addRow("Neuron cols:", self.spin_neuron_cols)
        adv_group.addRow("Neuron arrangement:", self.cmb_neuron_arrangement)
        pop_layout.addRow(adv_group)
        self.spin_neurons_per_row.valueChanged.connect(self._sync_neuron_rows_cols_from_simple)
        self.spin_neuron_rows.valueChanged.connect(self._sync_simple_from_rows_cols)
        self.spin_neuron_cols.valueChanged.connect(self._sync_simple_from_rows_cols)
        self._block_neuron_sync = False
        self._on_innervation_method_changed(
            self.cmb_innervation_method.currentText()
        )
        self.btn_add_population = QtWidgets.QPushButton("Add Population")
        self.btn_add_population.clicked.connect(self._on_add_population)
        pop_layout.addRow(self.btn_add_population)
        control_layout.addWidget(pop_group)

        # Population list and visibility toggles
        list_group = QtWidgets.QGroupBox("Active Populations")
        list_layout = QtWidgets.QVBoxLayout(list_group)
        self.population_list = QtWidgets.QListWidget()
        self.population_list.currentRowChanged.connect(self._on_population_selected)
        self.population_list.itemChanged.connect(self._on_population_item_changed)
        list_layout.addWidget(self.population_list)

        list_controls = QtWidgets.QHBoxLayout()
        self.btn_show_selected = QtWidgets.QPushButton("Show")
        self.btn_hide_selected = QtWidgets.QPushButton("Hide")
        self.btn_show_selected.clicked.connect(
            lambda: self._set_selected_visibility(True)
        )
        self.btn_hide_selected.clicked.connect(
            lambda: self._set_selected_visibility(False)
        )
        list_controls.addWidget(self.btn_show_selected)
        list_controls.addWidget(self.btn_hide_selected)
        btn_remove = QtWidgets.QPushButton("Remove")
        btn_remove.clicked.connect(self._on_remove_population)
        list_controls.addWidget(btn_remove)
        list_layout.addLayout(list_controls)
        visibility_group = QtWidgets.QGroupBox("Layers")
        visibility_layout = QtWidgets.QVBoxLayout(visibility_group)
        self.chk_show_mechanoreceptors = QtWidgets.QCheckBox("Show mechanoreceptors")
        self.chk_show_mechanoreceptors.setChecked(True)
        self.chk_show_mechanoreceptors.stateChanged.connect(
            self._update_layer_visibility
        )
        self.chk_show_neuron_centers = QtWidgets.QCheckBox("Show neuron centers")
        self.chk_show_neuron_centers.setChecked(True)
        self.chk_show_neuron_centers.stateChanged.connect(self._update_layer_visibility)
        self.chk_show_innervation = QtWidgets.QCheckBox("Show innervation")
        self.chk_show_innervation.setChecked(True)
        self.chk_show_innervation.stateChanged.connect(self._update_layer_visibility)
        visibility_layout.addWidget(self.chk_show_mechanoreceptors)
        visibility_layout.addWidget(self.chk_show_neuron_centers)
        visibility_layout.addWidget(self.chk_show_innervation)
        list_layout.addWidget(visibility_group)

        legend_group = QtWidgets.QGroupBox("Innervation Weight Legend")
        legend_layout = QtWidgets.QVBoxLayout(legend_group)
        self.population_legend_label = QtWidgets.QLabel(
            "Select a population to view weight shading."
        )
        self.population_legend_label.setAlignment(QtCore.Qt.AlignCenter)
        self.population_legend_label.setMinimumHeight(36)
        legend_layout.addWidget(self.population_legend_label)
        list_layout.addWidget(legend_group)
        control_layout.addWidget(list_group)

        # Snapshot controls
        export_group = QtWidgets.QGroupBox("Snapshot")
        export_layout = QtWidgets.QVBoxLayout(export_group)
        self.btn_export_snapshot = QtWidgets.QPushButton("Export Snapshot...")
        self.btn_export_snapshot.clicked.connect(self._on_export_snapshot)
        export_layout.addWidget(self.btn_export_snapshot)
        control_layout.addWidget(export_group)

        persistence_group = QtWidgets.QGroupBox("Configuration")
        persistence_layout = QtWidgets.QVBoxLayout(persistence_group)
        self.btn_save_configuration = QtWidgets.QPushButton("Save (Update)")
        self.btn_save_as_configuration = QtWidgets.QPushButton("Save As...")
        self.btn_load_configuration = QtWidgets.QPushButton("Load Configuration...")
        self.btn_save_configuration.clicked.connect(self._on_save_configuration)
        self.btn_save_as_configuration.clicked.connect(self._on_save_as_configuration)
        self.btn_load_configuration.clicked.connect(self._on_load_configuration)
        persistence_layout.addWidget(self.btn_save_configuration)
        persistence_layout.addWidget(self.btn_save_as_configuration)
        persistence_layout.addWidget(self.btn_load_configuration)
        control_layout.addWidget(persistence_group)
        control_layout.addStretch(1)

    def _configure_plot(self) -> None:
        self.plot.setAspectLocked(True, 1)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "X (mm)")
        self.plot.setLabel("left", "Y (mm)")
        self.plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

    # ------------------------------------------------------------------ #
    #  Unified Grid Workspace Methods (Phase 3 B.1-B.4)                   #
    # ------------------------------------------------------------------ #

    def _add_grid_entry(self, entry: GridEntry, *, regenerate: bool = True) -> None:
        """Add a GridEntry to the grid list and refresh the UI."""
        self._grid_entries.append(entry)
        item = QtWidgets.QListWidgetItem()
        self._refresh_grid_list_item(item, entry)
        self.grid_list.addItem(item)
        self.grid_list.setCurrentRow(self.grid_list.count() - 1)
        self._grid_counter += 1
        self._refresh_target_grid_dropdown()
        if regenerate:
            self._generate_grids()

    def _refresh_grid_list_item(self, item: QtWidgets.QListWidgetItem,
                                 entry: GridEntry) -> None:
        """Update a list item's display text, icon, and visibility from its GridEntry."""
        arr = entry.arrangement
        detail = f"{entry.rows}×{entry.cols}"
        item.setText(f"{entry.name}  [{arr}, {detail}]")
        pixmap = QtGui.QPixmap(12, 12)
        pixmap.fill(entry.color)
        item.setIcon(QtGui.QIcon(pixmap))
        item.setData(QtCore.Qt.UserRole, entry)
        item.setFlags(
            item.flags()
            | QtCore.Qt.ItemIsUserCheckable
            | QtCore.Qt.ItemIsSelectable
            | QtCore.Qt.ItemIsEnabled
        )
        self._block_grid_item_changed = True
        item.setCheckState(QtCore.Qt.Checked if entry.visible else QtCore.Qt.Unchecked)
        self._block_grid_item_changed = False

    def _on_add_grid(self) -> None:
        """Add a new grid layer."""
        idx = self._grid_counter
        color = _GRID_COLORS[(len(self._grid_entries)) % len(_GRID_COLORS)]
        entry = GridEntry(name=f"Grid {idx}", color=QtGui.QColor(color))
        self._add_grid_entry(entry)

    def _on_remove_grid(self) -> None:
        """Remove the selected grid from the list and update visualization."""
        row = self.grid_list.currentRow()
        if row < 0 or len(self._grid_entries) == 0:
            return
        self._grid_entries.pop(row)
        self.grid_list.takeItem(row)
        if self.grid_list.count() > 0:
            self.grid_list.setCurrentRow(min(row, self.grid_list.count() - 1))
        self._refresh_target_grid_dropdown()
        self._generate_grids()

    def _on_grid_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        """Handle grid list checkbox toggle for per-grid visibility."""
        if self._block_grid_item_changed:
            return
        entry = item.data(QtCore.Qt.UserRole)
        if not isinstance(entry, GridEntry):
            return
        entry.visible = item.checkState() == QtCore.Qt.Checked
        self._update_layer_visibility()

    def _highlight_grid_in_plot(self, entry: Optional[GridEntry]) -> None:
        """Highlight the selected grid in the plot (larger points, bolder) for list linkage."""
        for e, scatter in self._grid_scatter_map:
            is_selected = e is entry
            if is_selected:
                scatter.setSize(7)
                scatter.setPen(pg.mkPen(e.color.darker(130), width=1.2))
                scatter.setZValue(-1)
            else:
                scatter.setSize(5)
                scatter.setPen(None)
                scatter.setZValue(-2)

    def _on_grid_selected(self, row: int) -> None:
        """Load the selected grid entry into the editor panel and highlight in plot."""
        if row < 0 or row >= len(self._grid_entries):
            self._grid_editor.setVisible(False)
            self._highlight_grid_in_plot(None)
            return
        self._grid_editor.setVisible(True)
        entry = self._grid_entries[row]
        self._block_grid_editor = True
        self.txt_grid_name.setText(entry.name)
        self.cmb_grid_arrangement.setCurrentText(entry.arrangement)
        self.spin_grid_rows.setValue(entry.rows)
        self.spin_grid_cols.setValue(entry.cols)
        self.dbl_spacing.setValue(entry.spacing)
        self.dbl_center_x.setValue(entry.center_x)
        self.dbl_center_y.setValue(entry.center_y)
        self.dbl_offset_x.setValue(entry.offset_x)
        self.dbl_offset_y.setValue(entry.offset_y)
        self._update_grid_color_button(entry.color)
        self._block_grid_editor = False
        self._update_grid_editor_visibility(entry.arrangement)
        self._highlight_grid_in_plot(entry)

    def _on_grid_editor_changed(self, *_args: object) -> None:
        """Sync editor widgets back to the selected GridEntry."""
        if self._block_grid_editor:
            return
        row = self.grid_list.currentRow()
        if row < 0 or row >= len(self._grid_entries):
            return
        entry = self._grid_entries[row]
        entry.name = self.txt_grid_name.text().strip() or entry.name
        entry.arrangement = self.cmb_grid_arrangement.currentText()
        entry.rows = self.spin_grid_rows.value()
        entry.cols = self.spin_grid_cols.value()
        entry.spacing = self.dbl_spacing.value()
        entry.density = self._density_from_grid_params(entry.rows, entry.cols, entry.spacing)
        entry.center_x = self.dbl_center_x.value()
        entry.center_y = self.dbl_center_y.value()
        entry.offset_x = self.dbl_offset_x.value()
        entry.offset_y = self.dbl_offset_y.value()
        item = self.grid_list.item(row)
        if item is not None:
            self._refresh_grid_list_item(item, entry)
        self._refresh_target_grid_dropdown()

    def _on_grid_arrangement_changed(self, arrangement: str) -> None:
        """Show/hide rows/cols vs density based on arrangement type."""
        self._update_grid_editor_visibility(arrangement)
        self._on_grid_editor_changed()

    @staticmethod
    def _density_from_grid_params(rows: int, cols: int, spacing: float) -> float:
        """Derive receptor density (receptors/mm²) from rows, cols, spacing."""
        extent_w = max((cols - 1) * spacing, 1e-6)
        extent_h = max((rows - 1) * spacing, 1e-6)
        area = extent_w * extent_h
        return (rows * cols) / area

    def _update_grid_editor_visibility(self, arrangement: str) -> None:
        """All arrangements use rows, cols, spacing; no visibility toggling."""
        pass

    def _on_pick_grid_color(self) -> None:
        """Open color picker for the selected grid layer."""
        row = self.grid_list.currentRow()
        if row < 0 or row >= len(self._grid_entries):
            return
        entry = self._grid_entries[row]
        color = QtWidgets.QColorDialog.getColor(entry.color, self, "Select Grid Color")
        if color.isValid():
            entry.color = color
            self._update_grid_color_button(color)
            item = self.grid_list.item(row)
            if item is not None:
                self._refresh_grid_list_item(item, entry)

    def _update_grid_color_button(self, color: QtGui.QColor) -> None:
        """Update the grid color button swatch."""
        palette = self.btn_grid_color.palette()
        palette.setColor(QtGui.QPalette.Button, color)
        self.btn_grid_color.setPalette(palette)
        self.btn_grid_color.setAutoFillBackground(True)

    def _refresh_target_grid_dropdown(self) -> None:
        """Repopulate the target grid dropdown from the current grid entries."""
        # Guard against being called during initialization before widget exists
        if not hasattr(self, 'cmb_target_grid'):
            return
        current_text = self.cmb_target_grid.currentText()
        self.cmb_target_grid.blockSignals(True)
        self.cmb_target_grid.clear()
        self.cmb_target_grid.addItem("(all receptors)")
        for entry in self._grid_entries:
            self.cmb_target_grid.addItem(entry.name)
        # Restore previous selection if still valid
        idx = self.cmb_target_grid.findText(current_text)
        self.cmb_target_grid.setCurrentIndex(max(0, idx))
        self.cmb_target_grid.blockSignals(False)

    def _on_generate_grid(self) -> None:
        """Generate grid(s) from all grid entries."""
        if not self._grid_entries:
            return
        self._generate_grids()

    def _generate_grids(self) -> None:
        """Build GridManager (single grid) or CompositeReceptorGrid (multiple grids)."""
        entries = self._grid_entries
        if not entries:
            return

        if len(entries) == 1:
            self._generate_single_grid(entries[0])
        else:
            self._generate_multi_grid(entries)

    def _generate_single_grid(self, entry: GridEntry) -> None:
        """Generate a single GridManager from one grid entry."""
        seed_val = self.spin_global_seed.value()
        if seed_val >= 0:
            torch.manual_seed(seed_val)
        arrangement = entry.arrangement
        center = (entry.center_x + entry.offset_x,
                  entry.center_y + entry.offset_y)
        # grid_size: (n_x, n_y) = (cols, rows) for horizontal x, vertical y
        grid_size = (entry.cols, entry.rows)
        spacing = entry.spacing

        self.grid_manager = GridManager(
            grid_size=grid_size,
            spacing=spacing,
            center=center,
            arrangement=arrangement,
            density=None,
            device="cpu",
        )
        self._composite_grid = None
        self._grid_type = "standard"
        self._composite_populations = []

        self._refresh_target_grid_dropdown()
        self._sync_edge_offset_from_grid(entry.spacing)
        self._update_grid_visualization()
        self._rebuild_populations()
        self.grid_changed.emit(self.grid_manager)

    def _sync_edge_offset_from_grid(self, spacing: float) -> None:
        """Set edge offset default and step to grid spacing scale."""
        self.dbl_edge_offset.setValue(spacing)
        step = max(spacing / 5.0, 0.01)
        self.dbl_edge_offset.setSingleStep(step)

    def _generate_multi_grid(self, entries: List[GridEntry]) -> None:
        """Generate a CompositeReceptorGrid from multiple entries."""
        seed_val = self.spin_global_seed.value()
        if seed_val >= 0:
            torch.manual_seed(seed_val)
        # Compute bounds from all grids
        all_xmin, all_xmax = float("inf"), float("-inf")
        all_ymin, all_ymax = float("inf"), float("-inf")
        for e in entries:
            cx = e.center_x + e.offset_x
            cy = e.center_y + e.offset_y
            half_w = (e.cols - 1) * e.spacing / 2.0
            half_h = (e.rows - 1) * e.spacing / 2.0
            all_xmin = min(all_xmin, cx - half_w - 0.5)
            all_xmax = max(all_xmax, cx + half_w + 0.5)
            all_ymin = min(all_ymin, cy - half_h - 0.5)
            all_ymax = max(all_ymax, cy + half_h + 0.5)

        xlim = (all_xmin, all_xmax)
        ylim = (all_ymin, all_ymax)

        cg = CompositeReceptorGrid(xlim=xlim, ylim=ylim, device="cpu")

        self._composite_populations = []
        area = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])
        area = max(area, 1e-6)
        for e in entries:
            offset = (e.offset_x, e.offset_y)
            color_rgba = (e.color.red(), e.color.green(), e.color.blue(), e.color.alpha())
            density = self._density_from_grid_params(e.rows, e.cols, e.spacing)
            expected_count = e.rows * e.cols
            layer_density = expected_count / area
            cg.add_layer(
                name=e.name,
                density=layer_density,
                arrangement=e.arrangement,
                offset=offset,
                color=color_rgba,
            )
            self._composite_populations.append({
                "name": e.name,
                "rows": e.rows,
                "cols": e.cols,
                "spacing": e.spacing,
                "density": layer_density,
                "arrangement": e.arrangement,
                "offset": list(offset),
                "color": list(color_rgba),
            })

        self._composite_grid = cg
        self.grid_manager = None
        self._grid_type = "composite"

        self._refresh_target_grid_dropdown()
        first_spacing = entries[0].spacing if entries else 0.15
        self._sync_edge_offset_from_grid(first_spacing)
        self._update_grid_visualization()
        self._rebuild_populations()
        self.grid_changed.emit({
            "type": "composite",
            "grid": cg,
            "xlim": list(xlim),
            "ylim": list(ylim),
            "layers": list(self._composite_populations),
        })

    def _update_grid_visualization(self) -> None:
        """Redraw grid scatter points using per-grid colors and visibility."""
        # Remove old scatter items
        if self.grid_scatter is not None:
            self.plot.removeItem(self.grid_scatter)
            self.grid_scatter = None
        for item in self._grid_scatter_items:
            self.plot.removeItem(item)
        self._grid_scatter_items.clear()
        self._grid_scatter_map.clear()

        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []

        if self.grid_manager is not None and len(self._grid_entries) == 1:
            # Single grid mode — use grid entry color
            entry = self._grid_entries[0]
            if hasattr(self.grid_manager, "xx") and self.grid_manager.xx is not None:
                xx, yy = self.grid_manager.get_coordinates()
                x = xx.detach().cpu().numpy().ravel()
                y = yy.detach().cpu().numpy().ravel()
            else:
                coords = self.grid_manager.get_receptor_coordinates()
                x = coords[:, 0].detach().cpu().numpy()
                y = coords[:, 1].detach().cpu().numpy()
            color = entry.color if self._grid_entries else QtGui.QColor(80, 80, 80)
            brush = pg.mkBrush(color.red(), color.green(), color.blue(), 150)
            scatter = pg.ScatterPlotItem(x, y, size=5, pen=None, brush=brush)
            scatter.setZValue(-2)
            self.plot.addItem(scatter)
            self._grid_scatter_items.append(scatter)
            self._grid_scatter_map.append((entry, scatter))
            all_x.append(x)
            all_y.append(y)
        elif self._composite_grid is not None:
            # Multi-grid mode — color per layer, one scatter per entry
            for entry in self._grid_entries:
                coords = self._composite_grid.get_population_coordinates(entry.name)
                if coords is None or coords.shape[0] == 0:
                    continue
                x = coords[:, 0].detach().cpu().numpy()
                y = coords[:, 1].detach().cpu().numpy()
                brush = pg.mkBrush(
                    entry.color.red(), entry.color.green(),
                    entry.color.blue(), 150
                )
                scatter = pg.ScatterPlotItem(
                    x, y, size=5, pen=None, brush=brush, name=entry.name
                )
                scatter.setZValue(-2)
                self.plot.addItem(scatter)
                self._grid_scatter_items.append(scatter)
                self._grid_scatter_map.append((entry, scatter))
                all_x.append(x)
                all_y.append(y)

        if all_x:
            combined_x = np.concatenate(all_x)
            combined_y = np.concatenate(all_y)
            self._auto_range_plot(combined_x, combined_y)

        self._update_layer_visibility()

    def _update_mechanoreceptor_points(self) -> None:
        """Legacy helper — delegates to unified visualization."""
        self._update_grid_visualization()
        self._update_layer_visibility()

    def _auto_range_plot(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.size == 0 or y.size == 0:
            return
        padding = max(x.ptp(), y.ptp()) * 0.05 + 1e-6
        self.plot.setXRange(x.min() - padding, x.max() + padding, padding=0)
        self.plot.setYRange(y.min() - padding, y.max() + padding, padding=0)

    def _update_composite_visualization(self) -> None:
        """Legacy composite visualization — delegates to unified method."""
        self._update_grid_visualization()

    def _on_population_type_changed(self, text: str) -> None:
        if text == "SA":
            self.spin_neurons_per_row.setValue(10)
            if hasattr(self, "spin_neuron_rows"):
                self.spin_neuron_rows.setValue(10)
            if hasattr(self, "spin_neuron_cols"):
                self.spin_neuron_cols.setValue(10)
            self.dbl_connections.setValue(28.0)
            self.dbl_sigma.setValue(0.3)
            self._set_weight_defaults((0.1, 1.0))
        elif text == "RA":
            self.spin_neurons_per_row.setValue(14)
            if hasattr(self, "spin_neuron_rows"):
                self.spin_neuron_rows.setValue(14)
            if hasattr(self, "spin_neuron_cols"):
                self.spin_neuron_cols.setValue(14)
            self.dbl_connections.setValue(28.0)
            self.dbl_sigma.setValue(0.39)
            self._set_weight_defaults((0.05, 0.8))
        else:
            pass

    def _on_plot_clicked(self, event) -> None:
        """Handle plot click: select nearest neuron and highlight it with its connections.
        Click same neuron again to unselect."""
        if self._selected_population is None or self._selected_population.neuron_centers is None:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        pos = event.scenePos()
        vb = self.plot.getViewBox()
        data_pos = vb.mapSceneToView(pos)
        x_click, y_click = data_pos.x(), data_pos.y()
        centers = self._selected_population.neuron_centers.detach().cpu().numpy()
        dists = (centers[:, 0] - x_click) ** 2 + (centers[:, 1] - y_click) ** 2
        nearest = int(np.argmin(dists))
        if self._selected_neuron_idx == nearest:
            self._selected_neuron_idx = None
        else:
            self._selected_neuron_idx = nearest
        self._update_neuron_highlight()

    def _update_neuron_highlight(self) -> None:
        """Update the highlight overlay for the selected neuron and its connections."""
        pop = self._selected_population
        idx = self._selected_neuron_idx
        for p in self.populations:
            if p.highlight_neuron_item is not None:
                self.plot.removeItem(p.highlight_neuron_item)
                p.highlight_neuron_item = None
            if p.highlight_shadow_item is not None:
                self.plot.removeItem(p.highlight_shadow_item)
                p.highlight_shadow_item = None
            for item in p.highlight_connection_items:
                self.plot.removeItem(item)
            p.highlight_connection_items.clear()
            if p.highlight_receptor_item is not None:
                self.plot.removeItem(p.highlight_receptor_item)
                p.highlight_receptor_item = None
            if p.highlight_receptor_shadow_item is not None:
                self.plot.removeItem(p.highlight_receptor_shadow_item)
                p.highlight_receptor_shadow_item = None
        if pop is None or idx is None or pop.neuron_centers is None:
            self._update_heatmap_for_selection()
            self._dim_non_selected_elements(None)
            for p in self.populations:
                self._apply_population_visibility(p)
            return
        centers = pop.neuron_centers.detach().cpu().numpy()
        weights = pop.innervation_weights
        if weights is None:
            return
        if self.grid_manager is not None:
            if hasattr(self.grid_manager, "xx") and self.grid_manager.xx is not None:
                xx, yy = self.grid_manager.get_coordinates()
                x_flat = xx.detach().cpu().numpy().ravel()
                y_flat = yy.detach().cpu().numpy().ravel()
            else:
                coords = self.grid_manager.get_receptor_coordinates()
                x_flat = coords[:, 0].detach().cpu().numpy()
                y_flat = coords[:, 1].detach().cpu().numpy()
        elif pop.flat_module is not None:
            rc = pop.flat_module.receptor_coords.detach().cpu().numpy()
            x_flat, y_flat = rc[:, 0], rc[:, 1]
        else:
            return
        w = weights[idx].detach().cpu().numpy().ravel()
        nz = np.nonzero(w > 0)[0]
        if nz.size == 0:
            return
        cx, cy = centers[idx, 0], centers[idx, 1]
        rx, ry = x_flat[nz], y_flat[nz]
        w_vals = w[nz]
        w_min, w_max = float(w_vals.min()), float(w_vals.max())
        if np.isclose(w_max, w_min):
            norm_w = np.full_like(w_vals, 0.5)
        else:
            norm_w = (w_vals - w_min) / (w_max - w_min)

        self._update_heatmap_for_selection()

        # Connection lines with weight-proportional thickness (thicker = stronger)
        num_bins = 6
        width_min, width_max = 2.0, 5.0
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        for bin_idx in range(num_bins):
            lower, upper = bins[bin_idx], bins[bin_idx + 1]
            mask = (norm_w >= lower) & (norm_w < upper) if bin_idx < num_bins - 1 else (norm_w >= lower) & (norm_w <= upper)
            if not np.any(mask):
                continue
            t = (bin_idx + 0.5) / num_bins
            width = width_min + t * (width_max - width_min)
            line_color = QtGui.QColor(pop.color)
            line_color.setAlpha(220)
            line_x = np.empty(mask.sum() * 3)
            line_y = np.empty(mask.sum() * 3)
            line_x[0::3], line_x[1::3], line_x[2::3] = cx, rx[mask], np.nan
            line_y[0::3], line_y[1::3], line_y[2::3] = cy, ry[mask], np.nan
            conn_item = pg.PlotDataItem(
                line_x, line_y,
                pen=pg.mkPen(line_color, width=width, cap=QtCore.Qt.RoundCap),
            )
            conn_item.setZValue(10)
            self.plot.addItem(conn_item)
            pop.highlight_connection_items.append(conn_item)

        # Shadow layer behind neuron (larger, darker - makes it pop)
        shadow_item = pg.ScatterPlotItem(
            [cx], [cy],
            size=26,
            brush=pg.mkBrush(30, 30, 30, 140),
            pen=pg.mkPen(None),
        )
        shadow_item.setPxMode(True)
        shadow_item.setZValue(9)
        self.plot.addItem(shadow_item)
        pop.highlight_shadow_item = shadow_item

        # Highlighted neuron (larger, bold outline)
        neuron_item = pg.ScatterPlotItem(
            [cx], [cy],
            size=18,
            brush=pg.mkBrush(pop.color),
            pen=pg.mkPen(QtGui.QColor(0, 0, 0), width=4),
        )
        neuron_item.setPxMode(True)
        neuron_item.setZValue(11)
        self.plot.addItem(neuron_item)
        pop.highlight_neuron_item = neuron_item

        # Receptor shadows (behind receptors - makes them pop)
        receptor_shadow = pg.ScatterPlotItem(
            x=rx, y=ry,
            size=14,
            brush=pg.mkBrush(40, 40, 40, 130),
            pen=pg.mkPen(None),
        )
        receptor_shadow.setPxMode(True)
        receptor_shadow.setZValue(11)
        self.plot.addItem(receptor_shadow)
        pop.highlight_receptor_shadow_item = receptor_shadow

        # Highlighted receptors with weight-based color (stronger = darker/saturated)
        receptor_colors = [
            pg.mkBrush(self._weight_to_color(pop.color, float(norm_w[i])))
            for i in range(len(rx))
        ]
        receptor_item = pg.ScatterPlotItem(
            x=rx, y=ry,
            size=11,
            brush=receptor_colors,
            pen=pg.mkPen(QtGui.QColor(80, 80, 80), width=2),
        )
        receptor_item.setPxMode(True)
        receptor_item.setZValue(12)
        self.plot.addItem(receptor_item)
        pop.highlight_receptor_item = receptor_item

        self._apply_population_visibility(pop)
        self._dim_non_selected_elements(pop)

    def _dim_non_selected_elements(self, selected_pop: NeuronPopulation) -> None:
        """Reduce opacity of non-selected populations and grid when neuron highlighted."""
        if self._selected_neuron_idx is None:
            # Restore full opacity when no selection
            for pop in self.populations:
                if pop.scatter_item is not None:
                    pop.scatter_item.setOpacity(1.0)
                for conn_item, _, _ in pop.connection_items:
                    conn_item.setOpacity(1.0)
                if pop.heatmap_item is not None:
                    pop.heatmap_item.setOpacity(0.55)
                if pop.receptor_item is not None:
                    pop.receptor_item.setOpacity(1.0)
            for scatter_item in self._grid_scatter_items:
                scatter_item.setOpacity(1.0)
        else:
            # Dim non-selected populations and grid
            for pop in self.populations:
                is_selected = (pop is selected_pop)
                opacity = 1.0 if is_selected else 0.25
                if pop.scatter_item is not None:
                    pop.scatter_item.setOpacity(opacity)
                for conn_item, _, _ in pop.connection_items:
                    conn_item.setOpacity(opacity)
                if pop.heatmap_item is not None:
                    # Heatmap already handled by _update_heatmap_for_selection
                    pass
                if pop.receptor_item is not None:
                    pop.receptor_item.setOpacity(opacity)
            for scatter_item in self._grid_scatter_items:
                scatter_item.setOpacity(0.3)

    def _on_innervation_method_changed(self, method: str) -> None:
        is_distance = method == "distance_weighted"
        if hasattr(self, "_dist_params_group"):
            self._dist_params_group.setVisible(is_distance)

    def _sync_neuron_rows_cols_from_simple(self, value: int) -> None:
        if getattr(self, "_block_neuron_sync", False):
            return
        self._block_neuron_sync = True
        if hasattr(self, "spin_neuron_rows"):
            self.spin_neuron_rows.setValue(value)
        if hasattr(self, "spin_neuron_cols"):
            self.spin_neuron_cols.setValue(value)
        self._block_neuron_sync = False

    def _sync_simple_from_rows_cols(self) -> None:
        if getattr(self, "_block_neuron_sync", False):
            return
        if not hasattr(self, "spin_neuron_rows") or not hasattr(self, "spin_neuron_cols"):
            return
        r, c = self.spin_neuron_rows.value(), self.spin_neuron_cols.value()
        self._block_neuron_sync = True
        self.spin_neurons_per_row.setValue(r if r == c else int((r * c) ** 0.5))
        self._block_neuron_sync = False

    def _set_weight_defaults(self, weights: Tuple[float, float]) -> None:
        min_w, max_w = weights
        self.dbl_weight_min.setValue(min_w)
        self.dbl_weight_max.setValue(max_w)

    def _on_pick_color(self) -> None:
        color = QtWidgets.QColorDialog.getColor(
            self._population_color,
            self,
            "Select Population Color",
        )
        if color.isValid():
            self._population_color = color
            self._update_color_button()

    def _update_color_button(self) -> None:
        palette = self.btn_pick_color.palette()
        role = QtGui.QPalette.Button
        palette.setColor(role, self._population_color)
        self.btn_pick_color.setPalette(palette)
        self.btn_pick_color.setAutoFillBackground(True)

    def _on_add_population(self) -> None:
        if self.grid_manager is None and self._composite_grid is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Grid required",
                "Create the mechanoreceptor grid before adding populations.",
            )
            return
        name = self.txt_population_name.text().strip()
        neuron_type = "SA"  # Filter determines type; SA/RA used for defaults
        if not name:
            name = _default_population_name(
                neuron_type,
                self._population_counter,
            )
        seed_value = self.spin_global_seed.value()
        seed = None if seed_value < 0 else seed_value

        # Determine target grid for composite mode
        target_grid = None
        if self._composite_grid is not None and self.cmb_target_grid.isEnabled():
            target_text = self.cmb_target_grid.currentText()
            if target_text != "(all receptors)":
                target_grid = target_text

        use_dw = self.cmb_innervation_method.currentText() == "distance_weighted"
        far_frac = self.dbl_far_connection_fraction.value() if hasattr(self, "dbl_far_connection_fraction") else 0.0
        far_sigma = self.dbl_far_sigma_factor.value() if hasattr(self, "dbl_far_sigma_factor") else 5.0
        nr = self.spin_neuron_rows.value() if hasattr(self, "spin_neuron_rows") else self.spin_neurons_per_row.value()
        nc = self.spin_neuron_cols.value() if hasattr(self, "spin_neuron_cols") else self.spin_neurons_per_row.value()
        arr = self.cmb_neuron_arrangement.currentText() if hasattr(self, "cmb_neuron_arrangement") else "grid"
        population = NeuronPopulation(
            name=name,
            neuron_type=neuron_type,
            color=self._population_color,
            neurons_per_row=self.spin_neurons_per_row.value(),
            connections_per_neuron=self.dbl_connections.value(),
            neuron_rows=nr,
            neuron_cols=nc,
            neuron_arrangement=arr,
            sigma_d_mm=self.dbl_sigma.value(),
            weight_min=self.dbl_weight_min.value(),
            weight_max=self.dbl_weight_max.value(),
            innervation_method=self.cmb_innervation_method.currentText(),
            use_distance_weights=use_dw,
            far_connection_fraction=far_frac,
            far_sigma_factor=far_sigma,
            max_distance_mm=self.dbl_max_distance.value(),
            decay_function=self.cmb_decay_function.currentText(),
            decay_rate=self.dbl_decay_rate.value(),
            seed=seed,
            edge_offset=self.dbl_edge_offset.value() or None,
            target_grid=target_grid,
        )

        if self.grid_manager is not None:
            if hasattr(self.grid_manager, "xx") and self.grid_manager.xx is not None:
                population.instantiate(self.grid_manager)
            else:
                coords = self.grid_manager.get_receptor_coordinates()
                props = self.grid_manager.get_grid_properties()
                population.instantiate_flat(coords, props["xlim"], props["ylim"])
        elif self._composite_grid is not None:
            # Use FlatInnervationModule with composite grid coordinates
            if target_grid is not None:
                coords = self._composite_grid.get_population_coordinates(target_grid)
            else:
                coords = self._composite_grid.get_all_coordinates()
            if coords is None or coords.shape[0] == 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No receptors",
                    f"Grid layer '{target_grid or 'all'}' has no receptor coordinates.",
                )
                return
            bounds = self._composite_grid.computed_bounds
            xlim = bounds[0]
            ylim = bounds[1]
            population.instantiate_flat(coords, xlim, ylim)

        self.populations.append(population)
        self._population_counter += 1
        self._add_population_to_list(population)
        self._create_population_graphics(population)
        self._update_layer_visibility()
        self.populations_changed.emit(list(self.populations))
        self.txt_population_name.clear()

    def _add_population_to_list(self, population: NeuronPopulation) -> None:
        list_item = QtWidgets.QListWidgetItem(
            f"{population.name} ({population.neuron_type})"
        )
        list_item.setData(QtCore.Qt.UserRole, population)
        color = population.color
        list_item.setForeground(QtGui.QBrush(color))
        self._block_population_item_changed = True
        list_item.setFlags(
            list_item.flags()
            | QtCore.Qt.ItemIsUserCheckable
            | QtCore.Qt.ItemIsSelectable
            | QtCore.Qt.ItemIsEnabled
        )
        list_item.setCheckState(QtCore.Qt.Checked)
        self._block_population_item_changed = False
        self.population_list.addItem(list_item)
        self.population_list.setCurrentItem(list_item)

    def _on_population_selected(self, row: int) -> None:
        population = self._population_from_row(row)
        self._selected_population = population
        self._selected_neuron_idx = None
        self._update_neuron_highlight()
        self._highlight_population(population)

    def _on_population_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._block_population_item_changed:
            return
        population = item.data(QtCore.Qt.UserRole)
        if not isinstance(population, NeuronPopulation):
            return
        population.visible = item.checkState() == QtCore.Qt.Checked
        self._apply_population_visibility(population)

    def _highlight_population(self, population: Optional[NeuronPopulation]) -> None:
        for idx in range(self.population_list.count()):
            item = self.population_list.item(idx)
            pop = item.data(QtCore.Qt.UserRole)
            if not isinstance(pop, NeuronPopulation):
                continue
            is_selected = pop is population
            scatter = pop.scatter_item
            if scatter is not None:
                size = 8 if is_selected else 5
                width = 1.2 if is_selected else 0.7
                pen_color = (
                    pop.color.darker(130) if is_selected else pop.color.darker(150)
                )
                scatter.setBrush(pg.mkBrush(pop.color))
                scatter.setPen(pg.mkPen(pen_color, width=width))
                scatter.setSize(size)
                scatter.setPxMode(True)
                scatter.setZValue(7 if is_selected else 5)
            for conn_item, base_color, base_width in pop.connection_items:
                color = QtGui.QColor(base_color)
                if is_selected:
                    color = color.darker(115)
                width = base_width * (1.35 if is_selected else 1.0)
                conn_item.setPen(pg.mkPen(color, width=width))
                conn_item.setZValue(4 if is_selected else 2)
            if pop.receptor_item is not None:
                pop.receptor_item.setOpacity(1.0)
                pop.receptor_item.setZValue(2)
        self._update_population_legend(population)

    def _update_population_legend(self, population: Optional[NeuronPopulation]) -> None:
        if not hasattr(self, "population_legend_label"):
            return
        if population is None or not population.connection_items:
            self.population_legend_label.setText(
                "Select a population to view weight shading."
            )
            self.population_legend_label.setPixmap(QtGui.QPixmap())
            return
        width = 220
        height = 28
        image = QtGui.QImage(
            width,
            height,
            QtGui.QImage.Format_ARGB32,
        )
        image.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(image)
        for x in range(width):
            fraction = x / max(1, width - 1)
            color = self._weight_to_color(population.color, fraction)
            painter.setPen(QtGui.QPen(color))
            painter.drawLine(x, 0, x, height - 1)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QtCore.Qt.black)
        painter.drawText(4, height - 6, f"{population.weight_min:.2f}")
        painter.drawText(
            width - 40,
            height - 6,
            f"{population.weight_max:.2f}",
        )
        painter.end()
        pixmap = QtGui.QPixmap.fromImage(image)
        self.population_legend_label.setText("")
        self.population_legend_label.setPixmap(pixmap)
        self.population_legend_label.setToolTip(
            "Weight shading: lighter = weaker, darker = stronger. "
            "Connection line thickness also encodes strength (thicker = stronger). "
            "Click a neuron to highlight it and its connections; click again to unselect."
        )

    def _apply_population_visibility(self, population: NeuronPopulation) -> None:
        show_centers = population.visible and self.chk_show_neuron_centers.isChecked()
        show_innervation = population.visible and self.chk_show_innervation.isChecked()
        is_highlighted = (
            population is self._selected_population
            and self._selected_neuron_idx is not None
        )
        if population.scatter_item is not None:
            population.scatter_item.setVisible(show_centers)
        for conn_item, _, _ in population.connection_items:
            conn_item.setVisible(show_innervation and not is_highlighted)
        if population.heatmap_item is not None:
            population.heatmap_item.setVisible(show_innervation)
        if population.receptor_item is not None:
            population.receptor_item.setVisible(show_innervation and not is_highlighted)

    def _set_population_visibility(
        self,
        population: NeuronPopulation,
        visible: bool,
    ) -> None:
        population.visible = visible
        self._apply_population_visibility(population)
        row = self._row_for_population(population)
        if row is None:
            return
        item = self.population_list.item(row)
        self._block_population_item_changed = True
        item.setCheckState(QtCore.Qt.Checked if visible else QtCore.Qt.Unchecked)
        self._block_population_item_changed = False

    def _set_selected_visibility(self, visible: bool) -> None:
        population = self._population_from_row(self.population_list.currentRow())
        if population is None:
            return
        self._set_population_visibility(population, visible)

    def _row_for_population(self, population: NeuronPopulation) -> Optional[int]:
        for idx in range(self.population_list.count()):
            item = self.population_list.item(idx)
            if item.data(QtCore.Qt.UserRole) is population:
                return idx
        return None

    def _population_from_row(self, row: int) -> Optional[NeuronPopulation]:
        if row < 0 or row >= self.population_list.count():
            return None
        item = self.population_list.item(row)
        pop = item.data(QtCore.Qt.UserRole)
        return pop if isinstance(pop, NeuronPopulation) else None

    def _create_population_graphics(self, population: NeuronPopulation) -> None:
        centers = population.neuron_centers
        if centers is None:
            return
        centers_np = centers.detach().cpu().numpy()
        scatter = pg.ScatterPlotItem(
            centers_np[:, 0],
            centers_np[:, 1],
            size=5,
            brush=pg.mkBrush(population.color),
            pen=pg.mkPen(population.color.darker(150)),
        )
        scatter.setPxMode(True)
        scatter.setZValue(6)
        self.plot.addItem(scatter)
        population.scatter_item = scatter

        if population.module is not None and self.grid_manager is not None:
            self._update_innervation_graphics(population)
        elif population.flat_module is not None:
            self._update_innervation_graphics_flat(population)

        self._highlight_population(self._selected_population or population)

    def _weight_to_color(
        self, base_color: QtGui.QColor, fraction: float
    ) -> QtGui.QColor:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        graded = QtGui.QColor(base_color)
        lighten_factor = 130.0 + (1.0 - fraction) * 45.0
        lighten_factor = max(110.0, min(200.0, lighten_factor))
        graded = graded.lighter(int(lighten_factor))
        alpha = int(160.0 + fraction * 90.0)
        graded.setAlpha(max(150, min(255, alpha)))
        return graded

    def _update_heatmap_for_selection(self) -> None:
        """Update heatmap to show single-neuron innervation when selected, else cumulative."""
        pop = self._selected_population
        idx = self._selected_neuron_idx
        if pop is None or pop.heatmap_item is None or pop.module is None:
            return
        weights = pop.module.innervation_weights.detach().cpu().numpy()
        if idx is not None:
            weight_map = weights[idx].astype(np.float32)
        else:
            weight_map = weights.sum(axis=0).astype(np.float32)
        nonzero_mask = weight_map > 0.0
        if not np.any(nonzero_mask):
            pop.heatmap_item.setImage(np.zeros_like(weight_map))
            return
        max_weight = float(weight_map[nonzero_mask].max())
        normalized = np.zeros_like(weight_map, dtype=np.float32)
        normalized[nonzero_mask] = weight_map[nonzero_mask] / max_weight
        pop.heatmap_item.setImage(normalized)
        nonzero_values = normalized[nonzero_mask]
        lower = float(np.quantile(nonzero_values, 0.05))
        upper = float(np.quantile(nonzero_values, 0.95))
        if np.isclose(upper, lower):
            lower = max(0.0, lower - 0.05)
            upper = min(1.0, upper + 0.15)
        pop.heatmap_item.setLevels(
            (max(0.0, lower * 0.8), min(1.0, upper * 1.05))
        )
        pop.heatmap_item.setOpacity(0.75 if idx is not None else 0.55)

    def _population_heatmap_lookup(
        self, base_color: QtGui.QColor, steps: int = 256
    ) -> np.ndarray:
        fractions = np.linspace(0.0, 1.0, steps)
        lookup = np.zeros((steps, 4), dtype=np.ubyte)
        for idx, frac in enumerate(fractions):
            color = self._weight_to_color(base_color, float(frac))
            lookup[idx, 0] = color.red()
            lookup[idx, 1] = color.green()
            lookup[idx, 2] = color.blue()
            lookup[idx, 3] = color.alpha()
        return lookup

    def _update_innervation_graphics(
        self,
        population: NeuronPopulation,
    ) -> None:
        if population.module is None or self.grid_manager is None:
            return

        for item, _, _ in population.connection_items:
            self.plot.removeItem(item)
        if population.heatmap_item is not None:
            self.plot.removeItem(population.heatmap_item)
            population.heatmap_item = None
        if population.receptor_item is not None:
            self.plot.removeItem(population.receptor_item)
            population.receptor_item = None
        population.connection_items.clear()

        weights = population.module.innervation_weights.detach().cpu().numpy()
        centers = population.module.neuron_centers.detach().cpu().numpy()
        xx, yy = self.grid_manager.get_coordinates()
        x_flat = xx.detach().cpu().numpy().reshape(-1)
        y_flat = yy.detach().cpu().numpy().reshape(-1)
        x_grid = xx.detach().cpu().numpy()
        y_grid = yy.detach().cpu().numpy()
        grid_props = self.grid_manager.get_grid_properties()
        dx = float(grid_props.get("dx", 0.0) or grid_props.get("spacing", 0.0))
        dy = float(grid_props.get("dy", 0.0) or grid_props.get("spacing", 0.0))

        receptor_x_list: List[np.ndarray] = []
        receptor_y_list: List[np.ndarray] = []
        neuron_x_list: List[np.ndarray] = []
        neuron_y_list: List[np.ndarray] = []
        weight_list: List[np.ndarray] = []

        num_neurons = centers.shape[0]
        for idx in range(num_neurons):
            weights_flat = weights[idx].reshape(-1)
            nz = np.nonzero(weights_flat > 0.0)[0]
            if nz.size == 0:
                continue
            weight_list.append(weights_flat[nz])
            receptor_x_list.append(x_flat[nz])
            receptor_y_list.append(y_flat[nz])
            neuron_x_list.append(np.full(nz.size, centers[idx, 0]))
            neuron_y_list.append(np.full(nz.size, centers[idx, 1]))

        if not weight_list:
            return

        weights_all = np.concatenate(weight_list)
        receptor_x_all = np.concatenate(receptor_x_list)
        receptor_y_all = np.concatenate(receptor_y_list)
        neuron_x_all = np.concatenate(neuron_x_list)
        neuron_y_all = np.concatenate(neuron_y_list)

        # Generate heatmap overlay from total innervation density
        weight_map = weights.sum(axis=0).astype(np.float32)
        nonzero_mask = weight_map > 0.0
        if np.any(nonzero_mask):
            max_weight = float(weight_map[nonzero_mask].max())
            normalized = np.zeros_like(weight_map, dtype=np.float32)
            normalized[nonzero_mask] = weight_map[nonzero_mask] / max_weight
            image_item = pg.ImageItem(normalized)
            lookup = self._population_heatmap_lookup(population.color)
            image_item.setLookupTable(lookup)
            nonzero_values = normalized[nonzero_mask]
            lower = float(np.quantile(nonzero_values, 0.05))
            upper = float(np.quantile(nonzero_values, 0.95))
            if np.isclose(upper, lower):
                lower = max(0.0, lower - 0.05)
                upper = min(1.0, upper + 0.15)
            image_item.setLevels(
                (
                    max(0.0, lower * 0.8),
                    min(1.0, upper * 1.05),
                )
            )
            image_item.setOpacity(1.0)
            x_min = float(x_grid.min())
            x_max = float(x_grid.max())
            y_min = float(y_grid.min())
            y_max = float(y_grid.max())
            rect_width = (x_max - x_min) + (dx if dx != 0.0 else 0.0)
            rect_height = (y_max - y_min) + (dy if dy != 0.0 else 0.0)
            rect_x = x_min - (dx / 2.0 if dx != 0.0 else 0.0)
            rect_y = y_min - (dy / 2.0 if dy != 0.0 else 0.0)
            image_item.setRect(QtCore.QRectF(rect_x, rect_y, rect_width, rect_height))
            image_item.setZValue(1)
            image_item.setOpacity(0.55)
            self.plot.addItem(image_item)
            population.heatmap_item = image_item

        w_min = float(weights_all.min())
        w_max = float(weights_all.max())
        if np.isclose(w_max, w_min):
            norm_weights = np.full_like(weights_all, 0.5)
        else:
            norm_weights = (weights_all - w_min) / (w_max - w_min)

        # Scatter of innervated mechanoreceptors (visible overlay)
        neutral_brush = pg.mkBrush(80, 80, 80, 180)
        receptor_size = 5.0
        receptor_item = pg.ScatterPlotItem(
            x=receptor_x_all,
            y=receptor_y_all,
            size=receptor_size,
            brush=neutral_brush,
            pen=pg.mkPen(60, 60, 60, 100),
        )
        receptor_item.setOpacity(1.0)
        receptor_item.setPxMode(True)
        receptor_item.setZValue(2)
        self.plot.addItem(receptor_item)
        population.receptor_item = receptor_item

        # Connection lines: thickness encodes innervation strength (thicker = stronger)
        # Contrast: darker lines with higher alpha vs. semi-transparent heatmap
        num_bins = 8
        width_min, width_max = 0.4, 2.8
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        for bin_idx in range(num_bins):
            lower = bins[bin_idx]
            upper = bins[bin_idx + 1]
            if bin_idx == num_bins - 1:
                mask = (norm_weights >= lower) & (norm_weights <= upper)
            else:
                mask = (norm_weights >= lower) & (norm_weights < upper)
            if not np.any(mask):
                continue

            x0 = neuron_x_all[mask]
            x1 = receptor_x_all[mask]
            y0 = neuron_y_all[mask]
            y1 = receptor_y_all[mask]
            line_x = np.empty(mask.sum() * 3)
            line_y = np.empty(mask.sum() * 3)
            line_x[0::3] = x0
            line_x[1::3] = x1
            line_x[2::3] = np.nan
            line_y[0::3] = y0
            line_y[1::3] = y1
            line_y[2::3] = np.nan

            # Line width scales with weight bin (stronger = thicker)
            t = (bin_idx + 0.5) / num_bins
            width = width_min + t * (width_max - width_min)
            line_color = QtGui.QColor(population.color)
            line_color.setAlpha(int(200))
            line_color = line_color.darker(115)
            connection_item = pg.PlotDataItem(
                line_x,
                line_y,
                pen=pg.mkPen(
                    line_color,
                    width=width,
                    cap=QtCore.Qt.RoundCap,
                    join=QtCore.Qt.RoundJoin,
                    cosmetic=True,
                ),
                connect="finite",
            )
            connection_item.setZValue(3)
            self.plot.addItem(connection_item)
            population.connection_items.append((connection_item, line_color, width))

        self._apply_population_visibility(population)

    def _update_innervation_graphics_flat(
        self,
        population: NeuronPopulation,
    ) -> None:
        """Update innervation graphics for flat-coordinate (composite grid) populations."""
        if population.flat_module is None:
            return

        for item, _, _ in population.connection_items:
            self.plot.removeItem(item)
        if population.heatmap_item is not None:
            self.plot.removeItem(population.heatmap_item)
            population.heatmap_item = None
        if population.receptor_item is not None:
            self.plot.removeItem(population.receptor_item)
            population.receptor_item = None
        population.connection_items.clear()

        weights = population.flat_module.innervation_weights.detach().cpu().numpy()
        centers = population.flat_module.neuron_centers.detach().cpu().numpy()
        receptor_coords = population.flat_module.receptor_coords.detach().cpu().numpy()
        x_flat = receptor_coords[:, 0]
        y_flat = receptor_coords[:, 1]

        receptor_x_list: List[np.ndarray] = []
        receptor_y_list: List[np.ndarray] = []
        neuron_x_list: List[np.ndarray] = []
        neuron_y_list: List[np.ndarray] = []
        weight_list: List[np.ndarray] = []

        num_neurons = centers.shape[0]
        for idx in range(num_neurons):
            weights_flat = weights[idx].reshape(-1)
            nz = np.nonzero(weights_flat > 0.0)[0]
            if nz.size == 0:
                continue
            weight_list.append(weights_flat[nz])
            receptor_x_list.append(x_flat[nz])
            receptor_y_list.append(y_flat[nz])
            neuron_x_list.append(np.full(nz.size, centers[idx, 0]))
            neuron_y_list.append(np.full(nz.size, centers[idx, 1]))

        if not weight_list:
            return

        weights_all = np.concatenate(weight_list)
        receptor_x_all = np.concatenate(receptor_x_list)
        receptor_y_all = np.concatenate(receptor_y_list)
        neuron_x_all = np.concatenate(neuron_x_list)
        neuron_y_all = np.concatenate(neuron_y_list)

        # Scatter of innervated mechanoreceptors (visible overlay)
        neutral_brush = pg.mkBrush(80, 80, 80, 180)
        receptor_size = 5.0
        receptor_item = pg.ScatterPlotItem(
            x=receptor_x_all,
            y=receptor_y_all,
            size=receptor_size,
            brush=neutral_brush,
            pen=pg.mkPen(60, 60, 60, 100),
        )
        receptor_item.setOpacity(1.0)
        receptor_item.setPxMode(True)
        receptor_item.setZValue(2)
        self.plot.addItem(receptor_item)
        population.receptor_item = receptor_item

        w_min = float(weights_all.min())
        w_max = float(weights_all.max())
        if np.isclose(w_max, w_min):
            norm_weights = np.full_like(weights_all, 0.5)
        else:
            norm_weights = (weights_all - w_min) / (w_max - w_min)

        # Connection lines: thickness encodes innervation strength (thicker = stronger)
        # Contrast: darker lines with higher alpha vs. background
        num_bins = 8
        width_min, width_max = 0.4, 2.8
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        for bin_idx in range(num_bins):
            lower = bins[bin_idx]
            upper = bins[bin_idx + 1]
            if bin_idx == num_bins - 1:
                mask = (norm_weights >= lower) & (norm_weights <= upper)
            else:
                mask = (norm_weights >= lower) & (norm_weights < upper)
            if not np.any(mask):
                continue

            x0 = neuron_x_all[mask]
            x1 = receptor_x_all[mask]
            y0 = neuron_y_all[mask]
            y1 = receptor_y_all[mask]
            line_x = np.empty(mask.sum() * 3)
            line_y = np.empty(mask.sum() * 3)
            line_x[0::3] = x0
            line_x[1::3] = x1
            line_x[2::3] = np.nan
            line_y[0::3] = y0
            line_y[1::3] = y1
            line_y[2::3] = np.nan

            # Line width scales with weight bin (stronger = thicker)
            t = (bin_idx + 0.5) / num_bins
            width = width_min + t * (width_max - width_min)
            line_color = QtGui.QColor(population.color)
            line_color.setAlpha(int(200))
            line_color = line_color.darker(115)
            connection_item = pg.PlotDataItem(
                line_x,
                line_y,
                pen=pg.mkPen(
                    line_color,
                    width=width,
                    cap=QtCore.Qt.RoundCap,
                    join=QtCore.Qt.RoundJoin,
                    cosmetic=True,
                ),
                connect="finite",
            )
            connection_item.setZValue(3)
            self.plot.addItem(connection_item)
            population.connection_items.append((connection_item, line_color, width))

        self._apply_population_visibility(population)

    def _rebuild_populations(self) -> None:
        if self.grid_manager is None and self._composite_grid is None:
            return
        for population in self.populations:
            population.delete_graphics(self.plot)
        if self.grid_scatter is not None:
            self.plot.removeItem(self.grid_scatter)
            self.grid_scatter = None
        for item in self._grid_scatter_items:
            self.plot.removeItem(item)
        self._grid_scatter_items.clear()
        self._grid_scatter_map.clear()
        self.plot.clear()
        self._configure_plot()

        # Redraw grid points
        self._update_grid_visualization()

        # Rebuild all population innervation
        if self.grid_manager is not None:
            for population in self.populations:
                if hasattr(self.grid_manager, "xx") and self.grid_manager.xx is not None:
                    population.instantiate(self.grid_manager)
                else:
                    coords = self.grid_manager.get_receptor_coordinates()
                    props = self.grid_manager.get_grid_properties()
                    population.instantiate_flat(coords, props["xlim"], props["ylim"])
                self._create_population_graphics(population)
        elif self._composite_grid is not None:
            bounds = self._composite_grid.computed_bounds
            xlim = bounds[0]
            ylim = bounds[1]
            for population in self.populations:
                target = population.target_grid
                if target is not None:
                    coords = self._composite_grid.get_population_coordinates(target)
                else:
                    coords = self._composite_grid.get_all_coordinates()
                if coords is not None and coords.shape[0] > 0:
                    population.instantiate_flat(coords, xlim, ylim)
                    self._create_population_graphics(population)

        self._highlight_population(self._selected_population)
        self._update_layer_visibility()
        self.populations_changed.emit(list(self.populations))

    def _on_remove_population(self) -> None:
        row = self.population_list.currentRow()
        population = self._population_from_row(row)
        if population is None:
            return
        population.delete_graphics(self.plot)
        self.populations.remove(population)
        self._block_population_item_changed = True
        self.population_list.takeItem(row)
        self._block_population_item_changed = False
        self._selected_population = None
        self._highlight_population(None)
        self._update_layer_visibility()
        self.populations_changed.emit(list(self.populations))

    def _ensure_grid_ready(self) -> bool:
        if self.grid_manager is not None or self._composite_grid is not None:
            return True
        QtWidgets.QMessageBox.warning(
            self,
            "Grid required",
            "Create a mechanoreceptor grid before saving a configuration.",
        )
        return False

    def _prompt_save_directory(self) -> Optional[Path]:
        base_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select destination folder",
        )
        if not base_dir:
            return None
        default_name = self._default_configuration_name()
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Configuration Name",
            "Folder name:",
            text=default_name,
        )
        if not ok:
            return None
        folder_name = self._sanitize_name(name)
        return Path(base_dir) / folder_name

    def _perform_save(
        self,
        config_dir: Path,
        *,
        confirm_overwrite: bool,
    ) -> bool:
        try:
            if config_dir.exists():
                if confirm_overwrite and any(config_dir.iterdir()):
                    message = (
                        "The folder '"
                        f"{config_dir.name}"
                        "' already exists and contains files. Overwrite it?"
                    )
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "Overwrite configuration?",
                        message,
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No,
                    )
                    if reply != QtWidgets.QMessageBox.Yes:
                        return False
                for child in config_dir.iterdir():
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    else:
                        shutil.rmtree(child)
            else:
                config_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Save failed",
                f"Could not prepare configuration folder:\n{exc}",
            )
            return False

        if self.grid_manager is None and self._composite_grid is None:
            return False
        
        grid_entry = {}
        if self._grid_type == "composite" and self._composite_grid is not None:
            bounds = self._composite_grid.computed_bounds
            grid_entry = {
                "type": "composite",
                "xlim": [float(bounds[0][0]), float(bounds[0][1])],
                "ylim": [float(bounds[1][0]), float(bounds[1][1])],
                "populations": self._composite_populations,
            }
        elif self.grid_manager is not None:
            grid_props = self.grid_manager.get_grid_properties()
            rows, cols = self.grid_manager.grid_size
            grid_entry = {
                "type": "standard",
                "rows": int(rows),
                "cols": int(cols),
                "spacing_mm": float(grid_props.get("spacing", 0.0)),
                "center_mm": [
                    float(grid_props.get("center", (0.0, 0.0))[0]),
                    float(grid_props.get("center", (0.0, 0.0))[1]),
                ],
                "device": str(grid_props.get("device", "cpu")),
            }
        else:
            return False

        manifest = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "kind": "mechanoreceptor_bundle",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "seed": self.spin_global_seed.value(),
            "grid": grid_entry,
            "populations": [],
        }

        for idx, population in enumerate(self.populations, start=1):
            if population.module is None and population.flat_module is None:
                if self.grid_manager is not None:
                    if hasattr(self.grid_manager, "xx") and self.grid_manager.xx is not None:
                        population.instantiate(self.grid_manager)
                    else:
                        coords = self.grid_manager.get_receptor_coordinates()
                        props = self.grid_manager.get_grid_properties()
                        population.instantiate_flat(coords, props["xlim"], props["ylim"])
                elif self._composite_grid is not None:
                    target = population.target_grid
                    if target is not None:
                        coords = self._composite_grid.get_population_coordinates(target)
                    else:
                        coords = self._composite_grid.get_all_coordinates()
                    if coords is not None and coords.shape[0] > 0:
                        bounds = self._composite_grid.computed_bounds
                        population.instantiate_flat(coords, bounds[0], bounds[1])

            # Collect weights/centers from whichever module is active
            weights = population.innervation_weights
            centers = population.neuron_centers
            if weights is None or centers is None:
                continue

            base_name = population.name or f"population_{idx:02d}"
            pop_slug = self._sanitize_name(base_name)
            tensor_filename = f"population_{idx:02d}_{pop_slug}.pt"
            tensor_path = config_dir / tensor_filename
            try:
                payload = {
                    "innervation_weights": weights.detach().cpu(),
                    "neuron_centers": centers.detach().cpu(),
                }
                torch.save(payload, tensor_path)
            except OSError as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Save failed",
                    f"Unable to write tensor file '{tensor_filename}':\n{exc}",
                )
                return False

            pop_entry = {
                "name": population.name,
                "neuron_type": population.neuron_type,
                "color": self._color_to_rgba(population.color),
                "parameters": {
                    "neurons_per_row": int(population.neurons_per_row),
                    "neuron_rows": population.neuron_rows,
                    "neuron_cols": population.neuron_cols,
                    "neuron_arrangement": population.neuron_arrangement,
                    "connections_per_neuron": float(
                        population.connections_per_neuron
                    ),
                    "sigma_d_mm": float(population.sigma_d_mm),
                    "innervation_method": population.innervation_method,
                    "use_distance_weights": bool(population.use_distance_weights),
                    "far_connection_fraction": float(population.far_connection_fraction),
                    "far_sigma_factor": float(population.far_sigma_factor),
                    "max_distance_mm": float(population.max_distance_mm),
                    "decay_function": population.decay_function,
                    "decay_rate": float(population.decay_rate),
                    "weight_min": float(population.weight_min),
                    "weight_max": float(population.weight_max),
                    "seed": population.seed,
                    "edge_offset": population.edge_offset,
                },
                "tensors": tensor_filename,
                "visible": bool(population.visible),
            }
            if population.target_grid is not None:
                pop_entry["target_grid"] = population.target_grid
            manifest["populations"].append(pop_entry)

        manifest_path = config_dir / CONFIG_JSON_NAME
        try:
            with manifest_path.open("w", encoding="utf-8") as fp:
                json.dump(manifest, fp, indent=2)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Save failed",
                f"Unable to write configuration manifest:\n{exc}",
            )
            return False

        self._current_config_dir = config_dir
        self.configuration_directory_changed.emit(self._current_config_dir)
        return True

    def _on_save_configuration(self) -> None:
        if not self._ensure_grid_ready():
            return
        if self._current_config_dir is None:
            QtWidgets.QMessageBox.information(
                self,
                "Select folder",
                "Choose a destination with 'Save As...' before updating.",
            )
            self._on_save_as_configuration()
            return
        if self._perform_save(
            self._current_config_dir,
            confirm_overwrite=False,
        ):
            manifest_path = self._current_config_dir / CONFIG_JSON_NAME
            QtWidgets.QMessageBox.information(
                self,
                "Configuration saved",
                f"Updated configuration in '{manifest_path}'.",
            )

    def _on_save_as_configuration(self) -> None:
        if not self._ensure_grid_ready():
            return
        config_dir = self._prompt_save_directory()
        if config_dir is None:
            return
        if self._perform_save(config_dir, confirm_overwrite=True):
            manifest_path = config_dir / CONFIG_JSON_NAME
            QtWidgets.QMessageBox.information(
                self,
                "Configuration saved",
                f"Saved configuration to '{manifest_path}'.",
            )

    def _on_load_configuration(self) -> None:
        json_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "Configuration JSON (*.json)",
        )
        if not json_path:
            return

        manifest_path = Path(json_path)
        try:
            with manifest_path.open("r", encoding="utf-8") as fp:
                manifest = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Load failed",
                f"Unable to read configuration:\n{exc}",
            )
            return

        if manifest.get("schema_version") != CONFIG_SCHEMA_VERSION:
            warning_message = (
                "The configuration schema version differs from the current "
                "GUI expectations. Attempting to load anyway."
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Schema mismatch",
                warning_message,
            )

        grid_entry = manifest.get("grid")
        if not grid_entry:
            QtWidgets.QMessageBox.critical(
                self,
                "Load failed",
                "Configuration file is missing grid information.",
            )
            return

        grid_type = grid_entry.get("type", "standard")

        if "seed" in manifest:
            self.spin_global_seed.setValue(int(manifest["seed"]))

        self._clear_populations()

        # Convert legacy config.json grid to grid entries
        self._grid_entries.clear()
        self.grid_list.clear()
        self._grid_counter = 1

        try:
            if grid_type == "composite":
                comp_pops = grid_entry.get("populations", [])
                for i, pop in enumerate(comp_pops):
                    color = _GRID_COLORS[i % len(_GRID_COLORS)]
                    rows = int(pop.get("rows", 40))
                    cols = int(pop.get("cols", 40))
                    spacing = float(pop.get("spacing", 0.15))
                    entry = GridEntry(
                        name=pop.get("name", f"layer{i+1}"),
                        arrangement=pop.get("arrangement", "grid"),
                        rows=rows,
                        cols=cols,
                        spacing=spacing,
                        density=MechanoreceptorTab._density_from_grid_params(
                            rows, cols, spacing
                        ),
                        color=QtGui.QColor(color),
                    )
                    self._add_grid_entry(entry, regenerate=False)
            else:
                rows = int(grid_entry.get("rows"))
                cols = int(grid_entry.get("cols"))
                spacing = float(grid_entry.get("spacing_mm", 0.0))
                center_vals = grid_entry.get("center_mm", [0.0, 0.0])
                center = (float(center_vals[0]), float(center_vals[1]))
                entry = GridEntry(
                    name="Grid 1",
                    rows=rows,
                    cols=cols,
                    spacing=spacing,
                    center_x=center[0],
                    center_y=center[1],
                )
                self._add_grid_entry(entry, regenerate=False)

            self._generate_grids()
        except (TypeError, ValueError, IndexError) as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Load failed",
                f"Invalid grid parameters in configuration:\n{exc}",
            )
            return

        if self.grid_list.count() > 0:
            self.grid_list.setCurrentRow(0)

        populations_data = manifest.get("populations", [])
        self.population_list.blockSignals(True)
        for idx, entry in enumerate(populations_data, start=1):
            params = entry.get("parameters", {})
            color = self._rgba_to_color(entry.get("color", [66, 135, 245, 255]))
            target_grid = entry.get("target_grid", None)
            population = NeuronPopulation(
                name=entry.get("name", _default_population_name("SA", idx)),
                neuron_type=entry.get("neuron_type", "SA"),
                color=color,
                neurons_per_row=int(params.get("neurons_per_row", 10)),
                connections_per_neuron=float(
                    params.get("connections_per_neuron", 28.0)
                ),
                sigma_d_mm=float(params.get("sigma_d_mm", 0.3)),
                neuron_rows=params.get("neuron_rows"),
                neuron_cols=params.get("neuron_cols"),
                neuron_arrangement=params.get("neuron_arrangement", "grid"),
                innervation_method=params.get("innervation_method", "gaussian"),
                use_distance_weights=bool(params.get("use_distance_weights", False)),
                far_connection_fraction=float(params.get("far_connection_fraction", 0.0)),
                far_sigma_factor=float(params.get("far_sigma_factor", 5.0)),
                max_distance_mm=float(params.get("max_distance_mm", 1.0)),
                decay_function=params.get("decay_function", "exponential"),
                decay_rate=float(params.get("decay_rate", 2.0)),
                weight_min=float(params.get("weight_min", 0.1)),
                weight_max=float(params.get("weight_max", 1.0)),
                seed=params.get("seed"),
                edge_offset=params.get("edge_offset"),
                target_grid=target_grid,
            )

            # Instantiate with appropriate grid type
            if self.grid_manager is not None:
                if hasattr(self.grid_manager, "xx") and self.grid_manager.xx is not None:
                    population.instantiate(self.grid_manager)
                else:
                    coords = self.grid_manager.get_receptor_coordinates()
                    props = self.grid_manager.get_grid_properties()
                    population.instantiate_flat(coords, props["xlim"], props["ylim"])
            elif self._composite_grid is not None:
                if target_grid is not None:
                    coords = self._composite_grid.get_population_coordinates(target_grid)
                else:
                    coords = self._composite_grid.get_all_coordinates()
                if coords is not None and coords.shape[0] > 0:
                    bounds = self._composite_grid.computed_bounds
                    population.instantiate_flat(coords, bounds[0], bounds[1])

            tensor_file = entry.get("tensors")
            if tensor_file:
                tensor_path = (manifest_path.parent / tensor_file).resolve()
                try:
                    payload = torch.load(tensor_path, map_location="cpu")
                    weights_tensor = payload.get("innervation_weights")
                    centers_tensor = payload.get("neuron_centers")
                    if population.module is not None:
                        if weights_tensor is not None:
                            device = population.module.innervation_weights.device
                            population.module.innervation_map = weights_tensor.to(device)
                            population.module.innervation_weights.data.copy_(
                                population.module.innervation_map
                            )
                        if centers_tensor is not None:
                            population.module.neuron_centers = centers_tensor.to(
                                population.module.neuron_centers.device
                            )
                    elif population.flat_module is not None:
                        if weights_tensor is not None:
                            population.flat_module.innervation_weights.data.copy_(
                                weights_tensor.to(population.flat_module.innervation_weights.device)
                            )
                        if centers_tensor is not None:
                            population.flat_module.neuron_centers = centers_tensor.to(
                                population.flat_module.neuron_centers.device
                            )
                except (OSError, RuntimeError) as exc:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Load failed",
                        f"Unable to load tensor file '{tensor_file}':\n{exc}",
                    )
                    self.population_list.blockSignals(False)
                    return

            self.populations.append(population)
            self._add_population_to_list(population)
            self._create_population_graphics(population)
            visible = bool(entry.get("visible", True))
            population.visible = visible
            if not visible:
                self._set_population_visibility(population, False)
            else:
                self._apply_population_visibility(population)

        self.population_list.blockSignals(False)
        self._population_counter = len(self.populations) + 1
        if self.population_list.count() > 0:
            self.population_list.setCurrentRow(0)
        else:
            self._highlight_population(None)

        self.grid_changed.emit(self.grid_manager)
        self._current_config_dir = manifest_path.parent
        self.configuration_directory_changed.emit(self._current_config_dir)
        self.populations_changed.emit(list(self.populations))

        QtWidgets.QMessageBox.information(
            self,
            "Configuration loaded",
            f"Loaded configuration from '{manifest_path}'.",
        )
        self._update_layer_visibility()

    def current_grid_manager(self) -> Optional[GridManager]:
        return self.grid_manager

    def current_configuration_directory(self) -> Optional[Path]:
        return self._current_config_dir

    def active_populations(self) -> List[NeuronPopulation]:
        """Return a shallow copy of the configured neuron populations."""

        return list(self.populations)

    def _on_export_snapshot(self) -> None:
        if self.plot is None:
            return
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Snapshot",
            "tactile_snapshot",
            "Images (*.png *.svg)",
            options=options,
        )
        if not file_path:
            return
        exporter = None
        plot_item = self.plot
        if file_path.lower().endswith(".svg"):
            exporter = SVGExporter(plot_item)
        else:
            if not file_path.lower().endswith(".png"):
                file_path = f"{file_path}.png"
            exporter = ImageExporter(plot_item)
        exporter.export(file_path)

    def _update_layer_visibility(self) -> None:
        if not hasattr(self, "chk_show_mechanoreceptors"):
            return
        show_mech = self.chk_show_mechanoreceptors.isChecked()
        if self.grid_scatter is not None:
            self.grid_scatter.setVisible(show_mech)
        for entry, scatter in self._grid_scatter_map:
            scatter.setVisible(show_mech and entry.visible)
        for population in self.populations:
            self._apply_population_visibility(population)

    # ------------------------------------------------------------------ #
    #  Phase B — YAML ↔ GUI bidirectional config API                      #
    # ------------------------------------------------------------------ #

    def get_config(self) -> dict:
        """Export current tab state as a plain dict suitable for YAML.

        Returns:
            Dictionary with ``grids`` list and ``populations`` list.
            Supports round-trip fidelity: ``save → load → save``
            produces identical output.
        """
        # --- Grids section (Phase 3: unified grid list) ---
        grids = [entry.to_dict() for entry in self._grid_entries]

        # --- Populations section ---
        populations = []
        for pop in self.populations:
            c = pop.color
            pop_dict = {
                "name": pop.name,
                "neuron_type": pop.neuron_type,
                "neurons_per_row": pop.neurons_per_row,
                "neuron_rows": pop.neuron_rows,
                "neuron_cols": pop.neuron_cols,
                "neuron_arrangement": pop.neuron_arrangement,
                "connections_per_neuron": pop.connections_per_neuron,
                "sigma_d_mm": pop.sigma_d_mm,
                "innervation_method": pop.innervation_method,
                "use_distance_weights": pop.use_distance_weights,
                "far_connection_fraction": pop.far_connection_fraction,
                "far_sigma_factor": pop.far_sigma_factor,
                "max_distance_mm": pop.max_distance_mm,
                "decay_function": pop.decay_function,
                "decay_rate": pop.decay_rate,
                "weight_range": [pop.weight_min, pop.weight_max],
                "edge_offset": pop.edge_offset if pop.edge_offset else 0.0,
                "seed": pop.seed if pop.seed is not None else 42,
                "color": [c.red(), c.green(), c.blue(), c.alpha()],
                "visible": pop.visible,
            }
            if pop.target_grid is not None:
                pop_dict["target_grid"] = pop.target_grid
            populations.append(pop_dict)

        return {"grids": grids, "populations": populations}

    def set_config(self, config: dict) -> None:
        """Restore tab state from a config dict (inverse of ``get_config``).

        Args:
            config: Dictionary with ``grids`` list and ``populations`` list.
                Also accepts legacy format with ``grid`` dict for backward
                compatibility.
        """
        # --- Grid entries ---
        grids_cfg = config.get("grids", None)
        if grids_cfg is not None:
            # New Phase 3 format: list of grid entries
            self._grid_entries.clear()
            self.grid_list.clear()
            self._grid_counter = 1
            for g in grids_cfg:
                entry = GridEntry.from_dict(g)
                self._add_grid_entry(entry, regenerate=False)
        else:
            # Legacy format: single "grid" dict
            grid_cfg = config.get("grid", {})
            grid_type = grid_cfg.get("type", "standard")
            self._grid_entries.clear()
            self.grid_list.clear()
            self._grid_counter = 1

            if grid_type == "composite":
                for i, pop in enumerate(grid_cfg.get("composite_populations", [])):
                    color = _GRID_COLORS[i % len(_GRID_COLORS)]
                    rows = int(pop.get("rows", 40))
                    cols = int(pop.get("cols", 40))
                    spacing = float(pop.get("spacing", 0.15))
                    entry = GridEntry(
                        name=pop.get("name", f"layer{i+1}"),
                        arrangement=pop.get("arrangement", "grid"),
                        rows=rows,
                        cols=cols,
                        spacing=spacing,
                        density=MechanoreceptorTab._density_from_grid_params(
                            rows, cols, spacing
                        ),
                        color=QtGui.QColor(color),
                    )
                    self._add_grid_entry(entry, regenerate=False)
            else:
                # Convert legacy standard to single grid entry
                center = grid_cfg.get("center", [0.0, 0.0])
                entry = GridEntry(
                    name="Grid 1",
                    arrangement=grid_cfg.get("arrangement", "grid"),
                    rows=int(grid_cfg.get("rows", 40)),
                    cols=int(grid_cfg.get("cols", 40)),
                    spacing=float(grid_cfg.get("spacing_mm", 0.15)),
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                )
                self._add_grid_entry(entry, regenerate=False)

        # Generate grids from entries
        self._generate_grids()

        # Select first grid in list
        if self.grid_list.count() > 0:
            self.grid_list.setCurrentRow(0)

        # --- Populations ---
        self._clear_populations()
        for pop_cfg in config.get("populations", []):
            c = pop_cfg.get("color", [66, 135, 245, 255])
            wrange = pop_cfg.get("weight_range", [0.1, 1.0])
            target_grid = pop_cfg.get("target_grid", None)
            pop = NeuronPopulation(
                name=pop_cfg.get("name", "Population"),
                neuron_type=pop_cfg.get("neuron_type", "SA"),
                color=QtGui.QColor(c[0], c[1], c[2], c[3] if len(c) > 3 else 255),
                neurons_per_row=int(pop_cfg.get("neurons_per_row", 10)),
                neuron_rows=pop_cfg.get("neuron_rows"),
                neuron_cols=pop_cfg.get("neuron_cols"),
                neuron_arrangement=pop_cfg.get("neuron_arrangement", "grid"),
                connections_per_neuron=float(pop_cfg.get("connections_per_neuron", 28.0)),
                sigma_d_mm=float(pop_cfg.get("sigma_d_mm", 0.3)),
                innervation_method=pop_cfg.get("innervation_method", "gaussian"),
                use_distance_weights=bool(pop_cfg.get("use_distance_weights", False)),
                far_connection_fraction=float(pop_cfg.get("far_connection_fraction", 0.0)),
                far_sigma_factor=float(pop_cfg.get("far_sigma_factor", 5.0)),
                max_distance_mm=float(pop_cfg.get("max_distance_mm", 1.0)),
                decay_function=pop_cfg.get("decay_function", "exponential"),
                decay_rate=float(pop_cfg.get("decay_rate", 2.0)),
                weight_min=float(wrange[0]),
                weight_max=float(wrange[1]),
                seed=pop_cfg.get("seed", 42),
                edge_offset=pop_cfg.get("edge_offset", 0.0),
                target_grid=target_grid,
                visible=pop_cfg.get("visible", True),
            )
            if self.grid_manager is not None:
                try:
                    if hasattr(self.grid_manager, "xx") and self.grid_manager.xx is not None:
                        pop.instantiate(self.grid_manager)
                    else:
                        coords = self.grid_manager.get_receptor_coordinates()
                        props = self.grid_manager.get_grid_properties()
                        pop.instantiate_flat(coords, props["xlim"], props["ylim"])
                except Exception:
                    pass
            elif self._composite_grid is not None:
                try:
                    if target_grid is not None:
                        coords = self._composite_grid.get_population_coordinates(target_grid)
                    else:
                        coords = self._composite_grid.get_all_coordinates()
                    if coords is not None and coords.shape[0] > 0:
                        bounds = self._composite_grid.computed_bounds
                        pop.instantiate_flat(coords, bounds[0], bounds[1])
                except Exception:
                    pass
            self.populations.append(pop)
            self._add_population_to_list(pop)
            if pop.module is not None or pop.flat_module is not None:
                self._create_population_graphics(pop)
            if not pop.visible:
                self._set_population_visibility(pop, False)

        self.populations_changed.emit(list(self.populations))


__all__ = ["MechanoreceptorTab", "NeuronPopulation", "GridEntry"]
