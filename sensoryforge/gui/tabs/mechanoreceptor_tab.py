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
from sensoryforge.core.innervation import InnervationModule  # noqa: E402


CONFIG_SCHEMA_VERSION = "1.0.0"
CONFIG_JSON_NAME = "config.json"


def _default_population_name(neuron_type: str, index: int) -> str:
    return f"{neuron_type} #{index}" if neuron_type else f"Population #{index}"


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
    seed: Optional[int] = None
    edge_offset: Optional[float] = None
    module: Optional[InnervationModule] = None
    scatter_item: Optional[pg.ScatterPlotItem] = None
    connection_items: List[Tuple[pg.PlotDataItem, QtGui.QColor]] = field(
        default_factory=list
    )
    heatmap_item: Optional[pg.ImageItem] = None
    receptor_item: Optional[pg.ScatterPlotItem] = None
    visible: bool = True

    def instantiate(self, grid_manager: GridManager) -> None:
        """Build the PyTorch innervation module for this configuration."""
        weight_range: Tuple[float, float] = (self.weight_min, self.weight_max)
        kwargs = {
            "neuron_type": self.neuron_type,
            "grid_manager": grid_manager,
            "neurons_per_row": self.neurons_per_row,
            "connections_per_neuron": self.connections_per_neuron,
            "sigma_d_mm": self.sigma_d_mm,
            "weight_range": weight_range,
            "seed": self.seed,
            "edge_offset": self.edge_offset,
        }
        self.module = InnervationModule(**kwargs)

    def delete_graphics(self, plot: pg.PlotItem) -> None:
        if self.scatter_item is not None:
            plot.removeItem(self.scatter_item)
        for item, _ in self.connection_items:
            plot.removeItem(item)
        if self.heatmap_item is not None:
            plot.removeItem(self.heatmap_item)
        if self.receptor_item is not None:
            plot.removeItem(self.receptor_item)
        self.scatter_item = None
        self.connection_items.clear()
        self.heatmap_item = None
        self.receptor_item = None


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
        self._grid_type = "standard"
        self.grid_scatter: Optional[pg.ScatterPlotItem] = None
        self.populations: List[NeuronPopulation] = []
        self._population_counter = 1
        self._selected_population: Optional[NeuronPopulation] = None
        self._block_population_item_changed = False
        self._current_config_dir: Optional[Path] = None
        self._composite_populations: List[dict] = []
        self._load_default_params()
        self._setup_ui()
        self._configure_plot()
        self.populations_changed.emit(list(self.populations))

    def _load_default_params(self) -> None:
        """Load default parameters from default_params.json."""
        try:
            params_path = Path(__file__).parent.parent / "default_params.json"
            with open(params_path, "r") as f:
                self._default_params = json.load(f)
        except Exception:
            self._default_params = {
                "gui": {"default_grid_type": "standard"},
                "phase2": {
                    "composite_grid": {
                        "populations": {
                            "sa1": {"density": 100.0, "arrangement": "grid", "filter": "SA"},
                            "ra1": {"density": 70.0, "arrangement": "hex", "filter": "RA"},
                            "sa2": {"density": 30.0, "arrangement": "poisson", "filter": "SA"},
                        }
                    }
                }
            }

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

        # Grid controls
        grid_group = QtWidgets.QGroupBox("Mechanoreceptor Grid")
        grid_layout = QtWidgets.QFormLayout(grid_group)
        
        # Grid type selector
        self.cmb_grid_type = QtWidgets.QComboBox()
        self.cmb_grid_type.addItems(["Standard Grid", "Composite Grid"])
        default_type = self._default_params.get("gui", {}).get("default_grid_type", "standard")
        self.cmb_grid_type.setCurrentIndex(0 if default_type == "standard" else 1)
        self.cmb_grid_type.currentIndexChanged.connect(self._on_grid_type_changed)
        grid_layout.addRow("Grid Type:", self.cmb_grid_type)
        
        # Standard grid controls
        self.standard_grid_widget = QtWidgets.QWidget()
        standard_grid_layout = QtWidgets.QFormLayout(self.standard_grid_widget)
        standard_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.spin_grid_rows = QtWidgets.QSpinBox()
        self.spin_grid_rows.setRange(4, 256)
        self.spin_grid_rows.setValue(40)
        self.spin_grid_cols = QtWidgets.QSpinBox()
        self.spin_grid_cols.setRange(4, 256)
        self.spin_grid_cols.setValue(40)
        self.dbl_spacing = QtWidgets.QDoubleSpinBox()
        self.dbl_spacing.setDecimals(4)
        self.dbl_spacing.setRange(0.01, 1.0)
        self.dbl_spacing.setSingleStep(0.01)
        self.dbl_spacing.setValue(0.15)
        self.dbl_center_x = QtWidgets.QDoubleSpinBox()
        self.dbl_center_x.setRange(-50.0, 50.0)
        self.dbl_center_x.setDecimals(3)
        self.dbl_center_x.setValue(0.0)
        self.dbl_center_y = QtWidgets.QDoubleSpinBox()
        self.dbl_center_y.setRange(-50.0, 50.0)
        self.dbl_center_y.setDecimals(3)
        self.dbl_center_y.setValue(0.0)
        standard_grid_layout.addRow("Rows:", self.spin_grid_rows)
        standard_grid_layout.addRow("Cols:", self.spin_grid_cols)
        standard_grid_layout.addRow("Spacing (mm):", self.dbl_spacing)
        standard_grid_layout.addRow("Center X (mm):", self.dbl_center_x)
        standard_grid_layout.addRow("Center Y (mm):", self.dbl_center_y)
        grid_layout.addRow(self.standard_grid_widget)
        
        # Composite grid controls
        self.composite_grid_widget = QtWidgets.QWidget()
        composite_grid_layout = QtWidgets.QVBoxLayout(self.composite_grid_widget)
        composite_grid_layout.setContentsMargins(0, 0, 0, 0)
        
        bounds_layout = QtWidgets.QFormLayout()
        self.dbl_xlim_min = QtWidgets.QDoubleSpinBox()
        self.dbl_xlim_min.setRange(-100.0, 100.0)
        self.dbl_xlim_min.setDecimals(2)
        self.dbl_xlim_min.setValue(-5.0)
        self.dbl_xlim_max = QtWidgets.QDoubleSpinBox()
        self.dbl_xlim_max.setRange(-100.0, 100.0)
        self.dbl_xlim_max.setDecimals(2)
        self.dbl_xlim_max.setValue(5.0)
        self.dbl_ylim_min = QtWidgets.QDoubleSpinBox()
        self.dbl_ylim_min.setRange(-100.0, 100.0)
        self.dbl_ylim_min.setDecimals(2)
        self.dbl_ylim_min.setValue(-5.0)
        self.dbl_ylim_max = QtWidgets.QDoubleSpinBox()
        self.dbl_ylim_max.setRange(-100.0, 100.0)
        self.dbl_ylim_max.setDecimals(2)
        self.dbl_ylim_max.setValue(5.0)
        bounds_layout.addRow("X min (mm):", self.dbl_xlim_min)
        bounds_layout.addRow("X max (mm):", self.dbl_xlim_max)
        bounds_layout.addRow("Y min (mm):", self.dbl_ylim_min)
        bounds_layout.addRow("Y max (mm):", self.dbl_ylim_max)
        composite_grid_layout.addLayout(bounds_layout)
        
        self.composite_pop_table = QtWidgets.QTableWidget()
        self.composite_pop_table.setColumnCount(4)
        self.composite_pop_table.setHorizontalHeaderLabels(["Name", "Density", "Arrangement", "Filter"])
        self.composite_pop_table.horizontalHeader().setStretchLastSection(True)
        self.composite_pop_table.setMaximumHeight(150)
        composite_grid_layout.addWidget(QtWidgets.QLabel("Populations:"))
        composite_grid_layout.addWidget(self.composite_pop_table)
        
        comp_pop_buttons = QtWidgets.QHBoxLayout()
        self.btn_add_comp_pop = QtWidgets.QPushButton("Add Population")
        self.btn_remove_comp_pop = QtWidgets.QPushButton("Remove Selected")
        self.btn_add_comp_pop.clicked.connect(self._on_add_composite_population)
        self.btn_remove_comp_pop.clicked.connect(self._on_remove_composite_population)
        comp_pop_buttons.addWidget(self.btn_add_comp_pop)
        comp_pop_buttons.addWidget(self.btn_remove_comp_pop)
        composite_grid_layout.addLayout(comp_pop_buttons)
        
        grid_layout.addRow(self.composite_grid_widget)
        self.composite_grid_widget.setVisible(False)
        
        self.btn_generate_grid = QtWidgets.QPushButton("Generate Grid")
        self.btn_generate_grid.clicked.connect(self._on_generate_grid)
        grid_layout.addRow(self.btn_generate_grid)
        control_layout.addWidget(grid_group)

        # Population controls
        pop_group = QtWidgets.QGroupBox("Neuron Populations")
        pop_layout = QtWidgets.QFormLayout(pop_group)
        self.txt_population_name = QtWidgets.QLineEdit()
        self.cmb_population_type = QtWidgets.QComboBox()
        self.cmb_population_type.addItems(["SA", "RA", "Custom"])
        self.spin_neurons_per_row = QtWidgets.QSpinBox()
        self.spin_neurons_per_row.setRange(1, 128)
        self.spin_neurons_per_row.setValue(10)
        self.dbl_connections = QtWidgets.QDoubleSpinBox()
        self.dbl_connections.setDecimals(1)
        self.dbl_connections.setRange(1.0, 500.0)
        self.dbl_connections.setValue(28.0)
        self.dbl_sigma = QtWidgets.QDoubleSpinBox()
        self.dbl_sigma.setDecimals(4)
        self.dbl_sigma.setRange(0.01, 5.0)
        self.dbl_sigma.setValue(0.3)
        self.dbl_weight_min = QtWidgets.QDoubleSpinBox()
        self.dbl_weight_min.setDecimals(4)
        self.dbl_weight_min.setRange(0.0, 10.0)
        self.dbl_weight_min.setValue(0.1)
        self.dbl_weight_max = QtWidgets.QDoubleSpinBox()
        self.dbl_weight_max.setDecimals(4)
        self.dbl_weight_max.setRange(0.0, 10.0)
        self.dbl_weight_max.setValue(1.0)
        self.dbl_edge_offset = QtWidgets.QDoubleSpinBox()
        self.dbl_edge_offset.setDecimals(3)
        self.dbl_edge_offset.setRange(0.0, 20.0)
        self.dbl_edge_offset.setValue(0.0)
        self.spin_seed = QtWidgets.QSpinBox()
        self.spin_seed.setRange(-1, 1000000)
        self.spin_seed.setValue(42)
        self.btn_pick_color = QtWidgets.QPushButton()
        self.btn_pick_color.clicked.connect(self._on_pick_color)
        self.btn_pick_color.setText("Pick Color")
        self._population_color = QtGui.QColor(66, 135, 245)
        self._update_color_button()
        self.cmb_population_type.currentTextChanged.connect(
            self._on_population_type_changed
        )
        pop_layout.addRow("Name:", self.txt_population_name)
        pop_layout.addRow("Type:", self.cmb_population_type)
        pop_layout.addRow("Neurons/row:", self.spin_neurons_per_row)
        pop_layout.addRow("Connections:", self.dbl_connections)
        pop_layout.addRow("Sigma d (mm):", self.dbl_sigma)
        pop_layout.addRow("Weight min:", self.dbl_weight_min)
        pop_layout.addRow("Weight max:", self.dbl_weight_max)
        pop_layout.addRow("Edge offset (mm):", self.dbl_edge_offset)
        pop_layout.addRow("Seed (-1 for random):", self.spin_seed)
        pop_layout.addRow("Color:", self.btn_pick_color)
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

    def _on_grid_type_changed(self, index: int) -> None:
        """Handle grid type selection change."""
        if index == 0:  # Standard Grid
            self._grid_type = "standard"
            self.standard_grid_widget.setVisible(True)
            self.composite_grid_widget.setVisible(False)
        else:  # Composite Grid
            self._grid_type = "composite"
            self.standard_grid_widget.setVisible(False)
            self.composite_grid_widget.setVisible(True)
            self._load_default_composite_populations()

    def _load_default_composite_populations(self) -> None:
        """Load default composite grid populations from config."""
        if self.composite_pop_table.rowCount() > 0:
            return
        
        default_pops = self._default_params.get("phase2", {}).get("composite_grid", {}).get("populations", {})
        for name, config in default_pops.items():
            self._add_composite_population_row(
                name=name,
                density=config.get("density", 100.0),
                arrangement=config.get("arrangement", "grid"),
                filter_type=config.get("filter", "SA")
            )

    def _add_composite_population_row(self, name: str = "", density: float = 100.0, 
                                      arrangement: str = "grid", filter_type: str = "SA") -> None:
        """Add a row to the composite population table."""
        row = self.composite_pop_table.rowCount()
        self.composite_pop_table.insertRow(row)
        
        name_item = QtWidgets.QTableWidgetItem(name)
        self.composite_pop_table.setItem(row, 0, name_item)
        
        density_item = QtWidgets.QTableWidgetItem(str(density))
        self.composite_pop_table.setItem(row, 1, density_item)
        
        arrangement_combo = QtWidgets.QComboBox()
        arrangement_combo.addItems(["grid", "poisson", "hex", "jittered_grid"])
        arrangement_combo.setCurrentText(arrangement)
        self.composite_pop_table.setCellWidget(row, 2, arrangement_combo)
        
        filter_combo = QtWidgets.QComboBox()
        filter_combo.addItems(["SA", "RA", "None"])
        filter_combo.setCurrentText(filter_type)
        self.composite_pop_table.setCellWidget(row, 3, filter_combo)

    def _on_add_composite_population(self) -> None:
        """Add a new population row to the composite grid table."""
        row_num = self.composite_pop_table.rowCount() + 1
        self._add_composite_population_row(
            name=f"pop{row_num}",
            density=100.0,
            arrangement="grid",
            filter_type="SA"
        )

    def _on_remove_composite_population(self) -> None:
        """Remove selected population row from the composite grid table."""
        current_row = self.composite_pop_table.currentRow()
        if current_row >= 0:
            self.composite_pop_table.removeRow(current_row)

    def _on_generate_grid(self) -> None:
        if self._grid_type == "standard":
            self._generate_standard_grid()
        else:
            self._generate_composite_grid()

    def _generate_standard_grid(self) -> None:
        """Generate standard GridManager grid."""
        rows = self.spin_grid_rows.value()
        cols = self.spin_grid_cols.value()
        spacing = self.dbl_spacing.value()
        center = (self.dbl_center_x.value(), self.dbl_center_y.value())
        grid_size = (rows, cols)

        self.grid_manager = GridManager(
            grid_size=grid_size,
            spacing=spacing,
            center=center,
            device="cpu",
        )
        self._composite_grid = None

        self._update_mechanoreceptor_points()
        self._rebuild_populations()
        self.grid_changed.emit(self.grid_manager)

    def _generate_composite_grid(self) -> None:
        """Generate CompositeGrid from table configuration."""
        from sensoryforge.core.composite_grid import CompositeGrid
        
        xlim = (self.dbl_xlim_min.value(), self.dbl_xlim_max.value())
        ylim = (self.dbl_ylim_min.value(), self.dbl_ylim_max.value())
        
        cg = CompositeGrid(xlim=xlim, ylim=ylim, device="cpu")
        
        self._composite_populations = []
        for row in range(self.composite_pop_table.rowCount()):
            name_item = self.composite_pop_table.item(row, 0)
            density_item = self.composite_pop_table.item(row, 1)
            arrangement_combo = self.composite_pop_table.cellWidget(row, 2)
            filter_combo = self.composite_pop_table.cellWidget(row, 3)
            
            if name_item and density_item:
                name = name_item.text()
                density = float(density_item.text())
                arrangement = arrangement_combo.currentText() if arrangement_combo else "grid"
                filter_type = filter_combo.currentText() if filter_combo else "SA"
                
                cg.add_population(name=name, density=density, arrangement=arrangement)
                self._composite_populations.append({
                    "name": name,
                    "density": density,
                    "arrangement": arrangement,
                    "filter": filter_type
                })
        
        self._composite_grid = cg
        self.grid_manager = None
        
        self._update_composite_visualization()
        self.grid_changed.emit({"type": "composite", "grid": cg, "populations": self._composite_populations})

    def _update_mechanoreceptor_points(self) -> None:
        if self.grid_manager is None:
            return
        xx, yy = self.grid_manager.get_coordinates()
        x = xx.detach().cpu().numpy().ravel()
        y = yy.detach().cpu().numpy().ravel()
        brush = pg.mkBrush(80, 80, 80, 110)
        size = 4
        if self.grid_scatter is None:
            self.grid_scatter = pg.ScatterPlotItem(
                x,
                y,
                size=size,
                pen=None,
                brush=brush,
            )
            self.grid_scatter.setZValue(-2)
            self.plot.addItem(self.grid_scatter)
        else:
            self.grid_scatter.setData(
                x=x,
                y=y,
                size=size,
                pen=None,
                brush=brush,
            )
            self.grid_scatter.setZValue(-2)
        self._auto_range_plot(x, y)
        self._update_layer_visibility()

    def _auto_range_plot(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.size == 0 or y.size == 0:
            return
        padding = max(x.ptp(), y.ptp()) * 0.05 + 1e-6
        self.plot.setXRange(x.min() - padding, x.max() + padding, padding=0)
        self.plot.setYRange(y.min() - padding, y.max() + padding, padding=0)

    def _update_composite_visualization(self) -> None:
        """Update visualization for composite grid with color-coded populations."""
        if self._composite_grid is None:
            return
        
        if self.grid_scatter is not None:
            self.plot.removeItem(self.grid_scatter)
            self.grid_scatter = None
        
        pop_names = self._composite_grid.list_populations()
        colors = [
            (66, 135, 245),   # Blue
            (245, 135, 66),   # Orange
            (66, 245, 135),   # Green
            (245, 66, 135),   # Pink
            (135, 66, 245),   # Purple
            (245, 245, 66),   # Yellow
        ]
        
        all_x = []
        all_y = []
        
        for idx, name in enumerate(pop_names):
            coords = self._composite_grid.get_population_coordinates(name)
            if coords is None or coords.shape[0] == 0:
                continue
            
            x = coords[:, 0].detach().cpu().numpy()
            y = coords[:, 1].detach().cpu().numpy()
            all_x.extend(x)
            all_y.extend(y)
            
            color = colors[idx % len(colors)]
            brush = pg.mkBrush(*color, 150)
            scatter = pg.ScatterPlotItem(
                x, y, size=6, pen=None, brush=brush, name=name
            )
            scatter.setZValue(-2)
            self.plot.addItem(scatter)
        
        if all_x and all_y:
            self._auto_range_plot(np.array(all_x), np.array(all_y))
        
        self._update_layer_visibility()

    def _on_population_type_changed(self, text: str) -> None:
        if text == "SA":
            self.spin_neurons_per_row.setValue(10)
            self.dbl_connections.setValue(28.0)
            self.dbl_sigma.setValue(0.3)
            self._set_weight_defaults((0.1, 1.0))
        elif text == "RA":
            self.spin_neurons_per_row.setValue(14)
            self.dbl_connections.setValue(28.0)
            self.dbl_sigma.setValue(0.39)
            self._set_weight_defaults((0.05, 0.8))
        else:
            # Custom retains current values
            pass

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
        if self.grid_manager is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Grid required",
                "Create the mechanoreceptor grid before adding populations.",
            )
            return
        name = self.txt_population_name.text().strip()
        neuron_type = self.cmb_population_type.currentText()
        if neuron_type == "Custom":
            neuron_type = "SA"
        if not name:
            name = _default_population_name(
                neuron_type,
                self._population_counter,
            )
        seed_value = self.spin_seed.value()
        seed = None if seed_value < 0 else seed_value
        population = NeuronPopulation(
            name=name,
            neuron_type=neuron_type,
            color=self._population_color,
            neurons_per_row=self.spin_neurons_per_row.value(),
            connections_per_neuron=self.dbl_connections.value(),
            sigma_d_mm=self.dbl_sigma.value(),
            weight_min=self.dbl_weight_min.value(),
            weight_max=self.dbl_weight_max.value(),
            seed=seed,
            edge_offset=self.dbl_edge_offset.value() or None,
        )
        population.instantiate(self.grid_manager)
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
            for conn_item, base_color in pop.connection_items:
                color = QtGui.QColor(base_color)
                if is_selected:
                    color = color.darker(115)
                width = 2.2 if is_selected else 1.4
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
            "Lighter tint = weaker innervation; darker tint = stronger"
        )

    def _apply_population_visibility(self, population: NeuronPopulation) -> None:
        show_centers = population.visible and self.chk_show_neuron_centers.isChecked()
        show_innervation = population.visible and self.chk_show_innervation.isChecked()
        if population.scatter_item is not None:
            population.scatter_item.setVisible(show_centers)
        for conn_item, _ in population.connection_items:
            conn_item.setVisible(show_innervation)
        if population.heatmap_item is not None:
            population.heatmap_item.setVisible(show_innervation)
        if population.receptor_item is not None:
            population.receptor_item.setVisible(show_innervation)

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
        if population.module is None or self.grid_manager is None:
            return
        centers = population.module.neuron_centers.detach().cpu().numpy()
        scatter = pg.ScatterPlotItem(
            centers[:, 0],
            centers[:, 1],
            size=5,
            brush=pg.mkBrush(population.color),
            pen=pg.mkPen(population.color.darker(150)),
        )
        scatter.setPxMode(True)
        scatter.setZValue(6)
        self.plot.addItem(scatter)
        population.scatter_item = scatter
        self._update_innervation_graphics(population)
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

        for item, _ in population.connection_items:
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
            self.plot.addItem(image_item)
            population.heatmap_item = image_item

        w_min = float(weights_all.min())
        w_max = float(weights_all.max())
        if np.isclose(w_max, w_min):
            norm_weights = np.full_like(weights_all, 0.5)
        else:
            norm_weights = (weights_all - w_min) / (w_max - w_min)

        # Scatter of innervated mechanoreceptors with neutral appearance
        neutral_brush = pg.mkBrush(80, 80, 80, 110)
        receptor_size = 4.0
        receptor_item = pg.ScatterPlotItem(
            x=receptor_x_all,
            y=receptor_y_all,
            size=receptor_size,
            brush=neutral_brush,
            pen=None,
        )
        receptor_item.setOpacity(1.0)
        receptor_item.setPxMode(True)
        receptor_item.setZValue(2)
        self.plot.addItem(receptor_item)
        population.receptor_item = receptor_item

        num_bins = 8
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

            width = 0.05
            line_color = QtGui.QColor(population.color)
            line_color.setAlpha(int(50))
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
            population.connection_items.append((connection_item, line_color))

        self._apply_population_visibility(population)

    def _rebuild_populations(self) -> None:
        if self.grid_manager is None:
            return
        for population in self.populations:
            population.delete_graphics(self.plot)
        if self.grid_scatter is not None:
            self.plot.removeItem(self.grid_scatter)
            self.grid_scatter = None
        self.plot.clear()
        self._configure_plot()
        self._update_mechanoreceptor_points()
        for population in self.populations:
            population.instantiate(self.grid_manager)
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
        if self.grid_manager is not None:
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
            grid_entry = {
                "type": "composite",
                "xlim": [float(self.dbl_xlim_min.value()), float(self.dbl_xlim_max.value())],
                "ylim": [float(self.dbl_ylim_min.value()), float(self.dbl_ylim_max.value())],
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
            "grid": grid_entry,
            "populations": [],
        }

        for idx, population in enumerate(self.populations, start=1):
            if population.module is None and self.grid_manager is not None:
                population.instantiate(self.grid_manager)
            module = population.module
            if module is None:
                continue
            base_name = population.name or f"population_{idx:02d}"
            pop_slug = self._sanitize_name(base_name)
            tensor_filename = f"population_{idx:02d}_{pop_slug}.pt"
            tensor_path = config_dir / tensor_filename
            try:
                payload = {
                    "innervation_weights": module.innervation_weights.detach().cpu(),
                    "neuron_centers": module.neuron_centers.detach().cpu(),
                }
                torch.save(payload, tensor_path)
            except OSError as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Save failed",
                    f"Unable to write tensor file '{tensor_filename}':\n{exc}",
                )
                return False

            manifest["populations"].append(
                {
                    "name": population.name,
                    "neuron_type": population.neuron_type,
                    "color": self._color_to_rgba(population.color),
                    "parameters": {
                        "neurons_per_row": int(population.neurons_per_row),
                        "connections_per_neuron": float(
                            population.connections_per_neuron
                        ),
                        "sigma_d_mm": float(population.sigma_d_mm),
                        "weight_min": float(population.weight_min),
                        "weight_max": float(population.weight_max),
                        "seed": population.seed,
                        "edge_offset": population.edge_offset,
                    },
                    "tensors": tensor_filename,
                    "visible": bool(population.visible),
                }
            )

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
        
        self._clear_populations()
        
        if grid_type == "composite":
            try:
                xlim = grid_entry.get("xlim", [-5.0, 5.0])
                ylim = grid_entry.get("ylim", [-5.0, 5.0])
                comp_pops = grid_entry.get("populations", [])
                
                self.cmb_grid_type.setCurrentIndex(1)
                self.dbl_xlim_min.setValue(float(xlim[0]))
                self.dbl_xlim_max.setValue(float(xlim[1]))
                self.dbl_ylim_min.setValue(float(ylim[0]))
                self.dbl_ylim_max.setValue(float(ylim[1]))
                
                self.composite_pop_table.setRowCount(0)
                for pop in comp_pops:
                    self._add_composite_population_row(
                        name=pop.get("name", "pop"),
                        density=pop.get("density", 100.0),
                        arrangement=pop.get("arrangement", "grid"),
                        filter_type=pop.get("filter", "SA")
                    )
                
                self._generate_composite_grid()
            except (TypeError, ValueError, IndexError) as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Load failed",
                    f"Invalid composite grid parameters in configuration:\n{exc}",
                )
                return
        else:
            try:
                rows = int(grid_entry.get("rows"))
                cols = int(grid_entry.get("cols"))
                spacing = float(grid_entry.get("spacing_mm", 0.0))
                center_vals = grid_entry.get("center_mm", [0.0, 0.0])
                center = (float(center_vals[0]), float(center_vals[1]))
            except (TypeError, ValueError, IndexError) as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Load failed",
                    f"Invalid grid parameters in configuration:\n{exc}",
                )
                return

            self.cmb_grid_type.setCurrentIndex(0)
            for spin, value in (
                (self.spin_grid_rows, rows),
                (self.spin_grid_cols, cols),
            ):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
            for dbl, value in (
                (self.dbl_spacing, spacing),
                (self.dbl_center_x, center[0]),
                (self.dbl_center_y, center[1]),
            ):
                dbl.blockSignals(True)
                dbl.setValue(value)
                dbl.blockSignals(False)

            self.grid_manager = GridManager(
                grid_size=(rows, cols),
                spacing=spacing,
                center=center,
                device="cpu",
            )
            self._update_mechanoreceptor_points()

        populations_data = manifest.get("populations", [])
        self.population_list.blockSignals(True)
        for idx, entry in enumerate(populations_data, start=1):
            params = entry.get("parameters", {})
            color = self._rgba_to_color(entry.get("color", [66, 135, 245, 255]))
            population = NeuronPopulation(
                name=entry.get("name", _default_population_name("SA", idx)),
                neuron_type=entry.get("neuron_type", "SA"),
                color=color,
                neurons_per_row=int(params.get("neurons_per_row", 10)),
                connections_per_neuron=float(
                    params.get("connections_per_neuron", 28.0)
                ),
                sigma_d_mm=float(params.get("sigma_d_mm", 0.3)),
                weight_min=float(params.get("weight_min", 0.1)),
                weight_max=float(params.get("weight_max", 1.0)),
                seed=params.get("seed"),
                edge_offset=params.get("edge_offset"),
            )
            population.instantiate(self.grid_manager)

            tensor_file = entry.get("tensors")
            if tensor_file:
                tensor_path = (manifest_path.parent / tensor_file).resolve()
                try:
                    payload = torch.load(tensor_path, map_location="cpu")
                    weights = payload.get("innervation_weights")
                    if weights is not None:
                        device = population.module.innervation_weights.device
                        population.module.innervation_map = weights.to(device)
                        population.module.innervation_weights.data.copy_(
                            population.module.innervation_map
                        )
                    centers_tensor = payload.get("neuron_centers")
                    if centers_tensor is not None:
                        population.module.neuron_centers = centers_tensor.to(
                            population.module.neuron_centers.device
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
        if self.grid_scatter is not None:
            self.grid_scatter.setVisible(self.chk_show_mechanoreceptors.isChecked())
        for population in self.populations:
            self._apply_population_visibility(population)

    # ------------------------------------------------------------------ #
    #  Phase B — YAML ↔ GUI bidirectional config API                      #
    # ------------------------------------------------------------------ #

    def get_config(self) -> dict:
        """Export current tab state as a plain dict suitable for YAML.

        Returns:
            Dictionary with ``grid`` and ``populations`` sections. The
            structure supports round-trip fidelity:
            ``save → load → save`` produces identical output.
        """
        # --- Grid section ---
        grid: dict = {"type": self._grid_type}
        if self._grid_type == "standard":
            grid["rows"] = self.spin_grid_rows.value()
            grid["cols"] = self.spin_grid_cols.value()
            grid["spacing_mm"] = self.dbl_spacing.value()
            grid["center"] = [self.dbl_center_x.value(), self.dbl_center_y.value()]
        else:
            grid["xlim"] = [self.dbl_xlim_min.value(), self.dbl_xlim_max.value()]
            grid["ylim"] = [self.dbl_ylim_min.value(), self.dbl_ylim_max.value()]
            comp_pops = []
            for row in range(self.composite_pop_table.rowCount()):
                name_item = self.composite_pop_table.item(row, 0)
                density_item = self.composite_pop_table.item(row, 1)
                arr_combo = self.composite_pop_table.cellWidget(row, 2)
                filt_combo = self.composite_pop_table.cellWidget(row, 3)
                if name_item and density_item:
                    comp_pops.append({
                        "name": name_item.text(),
                        "density": float(density_item.text()),
                        "arrangement": arr_combo.currentText() if arr_combo else "grid",
                        "filter": filt_combo.currentText() if filt_combo else "SA",
                    })
            grid["composite_populations"] = comp_pops

        # --- Populations section ---
        populations = []
        for pop in self.populations:
            c = pop.color
            populations.append({
                "name": pop.name,
                "neuron_type": pop.neuron_type,
                "neurons_per_row": pop.neurons_per_row,
                "connections_per_neuron": pop.connections_per_neuron,
                "sigma_d_mm": pop.sigma_d_mm,
                "weight_range": [pop.weight_min, pop.weight_max],
                "edge_offset": pop.edge_offset if pop.edge_offset else 0.0,
                "seed": pop.seed if pop.seed is not None else 42,
                "color": [c.red(), c.green(), c.blue(), c.alpha()],
                "visible": pop.visible,
            })

        return {"grid": grid, "populations": populations}

    def set_config(self, config: dict) -> None:
        """Restore tab state from a config dict (inverse of ``get_config``).

        Args:
            config: Dictionary with ``grid`` and ``populations`` keys.
                Missing keys fall back to current widget defaults.
        """
        grid_cfg = config.get("grid", {})
        grid_type = grid_cfg.get("type", "standard")

        # --- Grid ---
        if grid_type == "composite":
            self.cmb_grid_type.blockSignals(True)
            self.cmb_grid_type.setCurrentIndex(1)
            self.cmb_grid_type.blockSignals(False)
            self._grid_type = "composite"
            self.standard_grid_widget.setVisible(False)
            self.composite_grid_widget.setVisible(True)

            xlim = grid_cfg.get("xlim", [-5.0, 5.0])
            ylim = grid_cfg.get("ylim", [-5.0, 5.0])
            for widget, val in [
                (self.dbl_xlim_min, xlim[0]),
                (self.dbl_xlim_max, xlim[1]),
                (self.dbl_ylim_min, ylim[0]),
                (self.dbl_ylim_max, ylim[1]),
            ]:
                widget.blockSignals(True)
                widget.setValue(float(val))
                widget.blockSignals(False)

            self.composite_pop_table.setRowCount(0)
            for pop in grid_cfg.get("composite_populations", []):
                self._add_composite_population_row(
                    name=pop.get("name", "pop"),
                    density=pop.get("density", 100.0),
                    arrangement=pop.get("arrangement", "grid"),
                    filter_type=pop.get("filter", "SA"),
                )
            self._generate_composite_grid()
        else:
            self.cmb_grid_type.blockSignals(True)
            self.cmb_grid_type.setCurrentIndex(0)
            self.cmb_grid_type.blockSignals(False)
            self._grid_type = "standard"
            self.standard_grid_widget.setVisible(True)
            self.composite_grid_widget.setVisible(False)

            rows = int(grid_cfg.get("rows", self.spin_grid_rows.value()))
            cols = int(grid_cfg.get("cols", self.spin_grid_cols.value()))
            spacing = float(grid_cfg.get("spacing_mm", self.dbl_spacing.value()))
            center = grid_cfg.get("center", [self.dbl_center_x.value(), self.dbl_center_y.value()])

            for spin, val in [(self.spin_grid_rows, rows), (self.spin_grid_cols, cols)]:
                spin.blockSignals(True)
                spin.setValue(val)
                spin.blockSignals(False)
            for dbl, val in [
                (self.dbl_spacing, spacing),
                (self.dbl_center_x, float(center[0])),
                (self.dbl_center_y, float(center[1])),
            ]:
                dbl.blockSignals(True)
                dbl.setValue(val)
                dbl.blockSignals(False)
            self._generate_standard_grid()

        # --- Populations ---
        self._clear_populations()
        for pop_cfg in config.get("populations", []):
            c = pop_cfg.get("color", [66, 135, 245, 255])
            wrange = pop_cfg.get("weight_range", [0.1, 1.0])
            pop = NeuronPopulation(
                name=pop_cfg.get("name", "Population"),
                neuron_type=pop_cfg.get("neuron_type", "SA"),
                color=QtGui.QColor(c[0], c[1], c[2], c[3] if len(c) > 3 else 255),
                neurons_per_row=int(pop_cfg.get("neurons_per_row", 10)),
                connections_per_neuron=float(pop_cfg.get("connections_per_neuron", 28.0)),
                sigma_d_mm=float(pop_cfg.get("sigma_d_mm", 0.3)),
                weight_min=float(wrange[0]),
                weight_max=float(wrange[1]),
                seed=pop_cfg.get("seed", 42),
                edge_offset=pop_cfg.get("edge_offset", 0.0),
                visible=pop_cfg.get("visible", True),
            )
            if self.grid_manager is not None:
                try:
                    pop.instantiate(self.grid_manager)
                except Exception:
                    pass
            self.populations.append(pop)
            item = QtWidgets.QListWidgetItem(pop.name)
            item.setForeground(QtGui.QBrush(pop.color))
            self.population_list.addItem(item)

        self.populations_changed.emit(list(self.populations))


__all__ = ["MechanoreceptorTab", "NeuronPopulation"]
