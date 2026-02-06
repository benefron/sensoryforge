import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg  # type: ignore

from sensoryforge.stimuli.stimulus import (
    StimulusGenerator,
    edge_stimulus_torch,
    gaussian_pressure_torch,
    point_pressure_torch,
)
from sensoryforge.core.grid import GridManager


STIMULUS_SCHEMA_VERSION = "1.0.0"
MIN_TIME_STEP_MS = 0.1


@dataclass
class StimulusConfig:
    """Container for the current stimulus parameters."""

    name: str
    stimulus_type: str
    motion: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    spread: float
    orientation_deg: float
    amplitude: float
    ramp_up_ms: float
    plateau_ms: float
    ramp_down_ms: float
    total_ms: float
    dt_ms: float
    speed_mm_s: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "type": self.stimulus_type,
            "motion": self.motion,
            "start": list(self.start),
            "end": list(self.end),
            "spread": float(self.spread),
            "orientation_deg": float(self.orientation_deg),
            "amplitude": float(self.amplitude),
            "ramp_up_ms": float(self.ramp_up_ms),
            "plateau_ms": float(self.plateau_ms),
            "ramp_down_ms": float(self.ramp_down_ms),
            "total_ms": float(self.total_ms),
            "dt_ms": float(self.dt_ms),
            "speed_mm_s": float(self.speed_mm_s),
        }


class StimulusDesignerTab(QtWidgets.QWidget):
    """Interactive designer for spatial-temporal tactile stimuli."""

    PREVIEW_THROTTLE_MS = 120

    def __init__(
        self,
        mechanoreceptor_tab,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.mechanoreceptor_tab = mechanoreceptor_tab
        self.grid_manager: Optional[GridManager] = (
            mechanoreceptor_tab.current_grid_manager()
            if hasattr(mechanoreceptor_tab, "current_grid_manager")
            else None
        )
        self.generator: Optional[StimulusGenerator] = None
        if self.grid_manager is not None:
            self.generator = StimulusGenerator(self.grid_manager)

        self._current_project_dir: Optional[Path] = (
            mechanoreceptor_tab.current_configuration_directory()
            if hasattr(mechanoreceptor_tab, "current_configuration_directory")
            else None
        )
        self._library_dir: Optional[Path] = None
        self._current_stimulus_path: Optional[Path] = None
        self._time_syncing = False
        self._speed_syncing = False
        self._loading = False
        self._preview_frame_count = 0
        self._preview_times = np.zeros(0, dtype=float)
        self._preview_dt_ms = 1.0
        self._playback_active = False

        self._setup_ui()
        self._connect_signals()

        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview)
        self._animation_timer = QtCore.QTimer(self)
        self._animation_timer.setInterval(50)
        self._animation_timer.setSingleShot(True)
        self._animation_timer.timeout.connect(self._advance_frame)
        self._reset_animation_controls()

        self._on_grid_changed(self.grid_manager)
        self._on_config_dir_changed(self._current_project_dir)

        if hasattr(mechanoreceptor_tab, "grid_changed"):
            mechanoreceptor_tab.grid_changed.connect(self._on_grid_changed)
        if hasattr(mechanoreceptor_tab, "configuration_directory_changed"):
            mechanoreceptor_tab.configuration_directory_changed.connect(
                self._on_config_dir_changed
            )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        control_widget = QtWidgets.QWidget()
        self.control_layout = QtWidgets.QVBoxLayout(control_widget)
        self.control_layout.setAlignment(QtCore.Qt.AlignTop)
        scroll_area.setWidget(control_widget)
        layout.addWidget(scroll_area, stretch=2)

        self._build_metadata_section()
        self._build_type_section()
        self._build_motion_section()
        self._build_spatial_section()
        self._build_temporal_section()
        self._build_library_section()
        self.control_layout.addStretch(1)

        preview_container = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)

        status_row = QtWidgets.QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(6)
        self.lbl_preview_status = QtWidgets.QLabel(
            "No grid available. Generate or load a grid in the " "Mechanoreceptor tab."
        )
        self.lbl_preview_status.setWordWrap(True)
        self.lbl_preview_status.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )
        status_row.addWidget(self.lbl_preview_status, stretch=1)
        self.btn_confirm_preview = QtWidgets.QPushButton("Confirm Settings")
        self.btn_confirm_preview.setEnabled(False)
        status_row.addWidget(self.btn_confirm_preview)
        preview_layout.addLayout(status_row)

        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.histogram.hide()
        self.image_view.view.setAspectLocked(True)
        self.image_view.view.invertY(False)
        preview_layout.addWidget(self.image_view, stretch=5)

        frame_controls = QtWidgets.QHBoxLayout()
        frame_controls.setContentsMargins(0, 0, 0, 0)
        frame_controls.setSpacing(6)
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setEnabled(False)
        frame_controls.addWidget(self.btn_play)
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(1)
        frame_controls.addWidget(self.frame_slider, stretch=1)
        self.lbl_frame_info = QtWidgets.QLabel("Frame 0 / 0 (0.0 ms)")
        frame_controls.addWidget(self.lbl_frame_info)
        preview_layout.addLayout(frame_controls)

        self.time_plot = pg.PlotWidget()
        self.time_plot.setLabel("bottom", "Time (ms)")
        self.time_plot.setLabel("left", "Amplitude")
        preview_layout.addWidget(self.time_plot, stretch=2)

        layout.addWidget(preview_container, stretch=3)

    def _build_metadata_section(self) -> None:
        group = QtWidgets.QGroupBox("Stimulus Metadata")
        form = QtWidgets.QFormLayout(group)
        self.txt_stimulus_name = QtWidgets.QLineEdit()
        self.txt_stimulus_name.setPlaceholderText("Stimulus name")
        form.addRow("Name:", self.txt_stimulus_name)
        self.control_layout.addWidget(group)

    def _reset_animation_controls(self) -> None:
        self._animation_timer.stop()
        self._playback_active = False
        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(False)
        self.btn_play.blockSignals(False)
        self.btn_play.setText("Play")
        self.btn_play.setEnabled(False)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.lbl_frame_info.setText("Frame 0 / 0 (0.0 ms)")
        self.lbl_time_steps.setText("-- frames")
        self._preview_frame_count = 0
        self._preview_times = np.zeros(0, dtype=float)
        self._preview_dt_ms = max(float(self.spin_dt.value()), MIN_TIME_STEP_MS)

    def _build_type_section(self) -> None:
        group = QtWidgets.QGroupBox("Stimulus Type")
        layout = QtWidgets.QHBoxLayout(group)
        layout.setSpacing(6)
        self.type_button_group = QtWidgets.QButtonGroup(self)
        self.type_buttons: Dict[str, QtWidgets.QToolButton] = {}

        for key, label in (
            ("gaussian", "🎯 Gaussian"),
            ("point", "🔵 Point"),
            ("edge", "📐 Edge"),
        ):
            button = QtWidgets.QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            button.setProperty("stimulusType", key)
            layout.addWidget(button)
            self.type_button_group.addButton(button)
            self.type_buttons[key] = button
        layout.addStretch(1)

        # Default selection
        self.type_buttons["gaussian"].setChecked(True)
        self._selected_type = "gaussian"

        self.control_layout.addWidget(group)

    def _build_motion_section(self) -> None:
        group = QtWidgets.QGroupBox("Motion Mode")
        layout = QtWidgets.QHBoxLayout(group)
        layout.setSpacing(12)

        self.radio_static = QtWidgets.QRadioButton("Static")
        self.radio_moving = QtWidgets.QRadioButton("Moving")
        self.radio_static.setChecked(True)

        layout.addWidget(self.radio_static)
        layout.addWidget(self.radio_moving)
        layout.addStretch(1)
        self.control_layout.addWidget(group)

    def _build_spatial_section(self) -> None:
        group = QtWidgets.QGroupBox("Spatial Parameters")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        def make_spin(
            min_val: float,
            max_val: float,
            value: float,
        ) -> QtWidgets.QDoubleSpinBox:
            box = QtWidgets.QDoubleSpinBox()
            box.setDecimals(3)
            box.setRange(min_val, max_val)
            box.setSingleStep(0.1)
            box.setValue(value)
            return box

        self.spin_start_x = make_spin(-50.0, 50.0, 0.0)
        self.spin_start_y = make_spin(-50.0, 50.0, 0.0)
        self.spin_end_x = make_spin(-50.0, 50.0, 0.0)
        self.spin_end_y = make_spin(-50.0, 50.0, 0.0)
        self.spin_end_x.setEnabled(False)
        self.spin_end_y.setEnabled(False)

        self.dbl_spread = QtWidgets.QDoubleSpinBox()
        self.dbl_spread.setDecimals(3)
        self.dbl_spread.setRange(0.01, 30.0)
        self.dbl_spread.setSingleStep(0.05)
        self.dbl_spread.setValue(0.3)
        self.lbl_spread = QtWidgets.QLabel("Sigma (mm):")

        self.dbl_orientation = QtWidgets.QDoubleSpinBox()
        self.dbl_orientation.setDecimals(1)
        self.dbl_orientation.setRange(0.0, 360.0)
        self.dbl_orientation.setSingleStep(5.0)
        self.dbl_orientation.setValue(0.0)

        self.dbl_amplitude = QtWidgets.QDoubleSpinBox()
        self.dbl_amplitude.setDecimals(2)
        self.dbl_amplitude.setRange(0.0, 10.0)
        self.dbl_amplitude.setSingleStep(0.1)
        self.dbl_amplitude.setValue(1.0)

        self.spin_speed = QtWidgets.QDoubleSpinBox()
        self.spin_speed.setDecimals(2)
        self.spin_speed.setRange(0.0, 500.0)
        self.spin_speed.setSingleStep(1.0)
        self.spin_speed.setValue(0.0)
        self.spin_speed.setSuffix(" mm/s")
        self.spin_speed.setEnabled(False)

        form.addRow("Start X (mm):", self.spin_start_x)
        form.addRow("Start Y (mm):", self.spin_start_y)
        form.addRow("End X (mm):", self.spin_end_x)
        form.addRow("End Y (mm):", self.spin_end_y)
        form.addRow(self.lbl_spread, self.dbl_spread)
        form.addRow("Edge angle (°):", self.dbl_orientation)
        form.addRow("Amplitude:", self.dbl_amplitude)
        form.addRow("Speed:", self.spin_speed)

        self.control_layout.addWidget(group)
        self._update_spread_label()
        self.dbl_orientation.setVisible(False)
        self.lbl_orientation_label = form.labelForField(self.dbl_orientation)
        if self.lbl_orientation_label is not None:
            self.lbl_orientation_label.setVisible(False)

    def _build_temporal_section(self) -> None:
        group = QtWidgets.QGroupBox("Temporal Parameters")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        def make_time_spin(value: float) -> QtWidgets.QDoubleSpinBox:
            box = QtWidgets.QDoubleSpinBox()
            box.setDecimals(1)
            box.setRange(0.0, 20000.0)
            box.setSingleStep(10.0)
            box.setValue(value)
            return box

        self.spin_ramp_up = make_time_spin(50.0)
        self.spin_plateau = make_time_spin(200.0)
        self.spin_ramp_down = make_time_spin(50.0)
        self.spin_total_time = make_time_spin(300.0)
        self.spin_dt = QtWidgets.QDoubleSpinBox()
        self.spin_dt.setDecimals(2)
        self.spin_dt.setRange(MIN_TIME_STEP_MS, 100.0)
        self.spin_dt.setSingleStep(0.1)
        self.spin_dt.setValue(1.0)
        self.spin_dt.setSuffix(" ms")

        self.lbl_time_steps = QtWidgets.QLabel("-- frames")

        form.addRow("Ramp up (ms):", self.spin_ramp_up)
        form.addRow("Plateau (ms):", self.spin_plateau)
        form.addRow("Ramp down (ms):", self.spin_ramp_down)
        form.addRow("Total (ms):", self.spin_total_time)
        form.addRow("Δt (ms):", self.spin_dt)
        form.addRow("Samples:", self.lbl_time_steps)

        self.control_layout.addWidget(group)

    def _build_library_section(self) -> None:
        group = QtWidgets.QGroupBox("Stimulus Library")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        self.library_status = QtWidgets.QLabel("No project loaded.")
        self.library_status.setWordWrap(True)
        layout.addWidget(self.library_status)

        self.stimulus_list = QtWidgets.QListWidget()
        self.stimulus_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        layout.addWidget(self.stimulus_list, stretch=1)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(6)
        self.btn_save_stimulus = QtWidgets.QPushButton("Save (Update)")
        self.btn_save_stimulus_as = QtWidgets.QPushButton("Save As...")
        self.btn_load_stimulus = QtWidgets.QPushButton("Load Selected")
        self.btn_delete_stimulus = QtWidgets.QPushButton("Delete")
        self.btn_refresh_library = QtWidgets.QPushButton("Refresh")

        button_row.addWidget(self.btn_save_stimulus)
        button_row.addWidget(self.btn_save_stimulus_as)
        button_row.addWidget(self.btn_load_stimulus)
        button_row.addWidget(self.btn_delete_stimulus)
        button_row.addWidget(self.btn_refresh_library)

        layout.addLayout(button_row)

        self.control_layout.addWidget(group)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.type_button_group.buttonClicked.connect(self._on_type_selected)
        self.radio_static.toggled.connect(self._on_motion_toggled)
        self.radio_moving.toggled.connect(self._on_motion_toggled)

        for spin in (
            self.spin_start_x,
            self.spin_start_y,
            self.spin_end_x,
            self.spin_end_y,
        ):
            spin.valueChanged.connect(self._on_position_changed)

        self.dbl_spread.valueChanged.connect(self._handle_preview_request)
        self.dbl_orientation.valueChanged.connect(self._handle_preview_request)
        self.dbl_amplitude.valueChanged.connect(self._handle_preview_request)
        self.spin_dt.valueChanged.connect(self._handle_preview_request)

        self.spin_speed.valueChanged.connect(
            lambda _: self._update_motion_dependencies(from_speed=True)
        )

        self.spin_ramp_up.valueChanged.connect(self._on_ramp_component_changed)
        self.spin_plateau.valueChanged.connect(self._on_plateau_changed)
        self.spin_ramp_down.valueChanged.connect(self._on_ramp_component_changed)
        self.spin_total_time.valueChanged.connect(self._on_total_time_changed)

        self.txt_stimulus_name.textChanged.connect(self._on_name_changed)

        self.btn_save_stimulus.clicked.connect(self._save_stimulus_update)
        self.btn_save_stimulus_as.clicked.connect(self._save_stimulus_as)
        self.btn_load_stimulus.clicked.connect(self._load_selected_stimulus)
        self.btn_delete_stimulus.clicked.connect(self._delete_selected_stimulus)
        self.btn_refresh_library.clicked.connect(self._refresh_stimulus_library)

        self.stimulus_list.itemSelectionChanged.connect(
            self._on_library_selection_changed
        )
        self.stimulus_list.itemDoubleClicked.connect(
            lambda _: self._load_selected_stimulus()
        )

        self.btn_play.toggled.connect(self._toggle_animation)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.btn_confirm_preview.clicked.connect(self._on_confirm_preview)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_type_selected(self, button: QtWidgets.QAbstractButton) -> None:
        stimulus_type = button.property("stimulusType")
        if not stimulus_type or stimulus_type == self._selected_type:
            return
        self._selected_type = str(stimulus_type)
        self._update_spread_label()
        is_edge = self._selected_type == "edge"
        self.dbl_orientation.setVisible(is_edge)
        if self.lbl_orientation_label is not None:
            self.lbl_orientation_label.setVisible(is_edge)
        self._request_preview()

    def _on_motion_toggled(self, _: bool) -> None:
        moving = self.radio_moving.isChecked()
        for widget in (self.spin_end_x, self.spin_end_y, self.spin_speed):
            widget.setEnabled(moving)
        if not moving:
            self.spin_speed.setValue(0.0)
            self.spin_end_x.setValue(self.spin_start_x.value())
            self.spin_end_y.setValue(self.spin_start_y.value())
        self._update_motion_dependencies(from_speed=False)
        self._request_preview()

    def _on_confirm_preview(self) -> None:
        if self._loading:
            return
        self._preview_timer.stop()
        self._update_preview()

    def _on_position_changed(self, _: float) -> None:
        self._update_motion_dependencies(from_speed=False)
        self._request_preview()

    def _handle_preview_request(self, *_: float) -> None:
        self._request_preview()

    def _on_ramp_component_changed(self, _: float) -> None:
        if self._loading:
            return
        if self._time_syncing:
            return
        self._time_syncing = True
        total = (
            self.spin_ramp_up.value()
            + self.spin_plateau.value()
            + self.spin_ramp_down.value()
        )
        self.spin_total_time.setValue(total)
        self._time_syncing = False
        self._update_motion_dependencies(from_speed=False)
        self._request_preview()

    def _on_plateau_changed(self, _: float) -> None:
        if self._loading:
            return
        if self._speed_syncing:
            return
        self._on_ramp_component_changed(0.0)

    def _on_total_time_changed(self, value: float) -> None:
        if self._loading:
            return
        if self._time_syncing:
            return
        available = max(
            value - self.spin_ramp_up.value() - self.spin_ramp_down.value(),
            0.0,
        )
        self._time_syncing = True
        self.spin_plateau.setValue(available)
        self._time_syncing = False
        self._update_motion_dependencies(from_speed=False)
        self._request_preview()

    def _on_name_changed(self, _: str) -> None:
        if self._loading:
            return
        # Editing the name resets the current selection
        self._current_stimulus_path = None
        self._update_library_buttons()

    def _on_library_selection_changed(self) -> None:
        items = self.stimulus_list.selectedItems()
        if not items:
            self._current_stimulus_path = None
        else:
            path_str = items[0].data(QtCore.Qt.UserRole)
            self._current_stimulus_path = Path(path_str) if path_str else None
        self._update_library_buttons()

    # ------------------------------------------------------------------
    # Grid and project context updates
    # ------------------------------------------------------------------
    def _on_grid_changed(self, grid_manager: Optional[GridManager]) -> None:
        self.grid_manager = grid_manager
        self._reset_animation_controls()
        if grid_manager is None:
            self.generator = None
            self.lbl_preview_status.setText(
                "No grid available. Generate or load a grid in the "
                "Mechanoreceptor tab."
            )
            self.image_view.clear()
            self.time_plot.clear()
            self.btn_confirm_preview.setEnabled(False)
            return

        self.generator = StimulusGenerator(grid_manager)
        self.btn_confirm_preview.setEnabled(True)
        xx, yy = grid_manager.get_coordinates()
        x_vals = xx.detach().cpu().numpy()
        y_vals = yy.detach().cpu().numpy()
        min_x, max_x = float(np.min(x_vals)), float(np.max(x_vals))
        min_y, max_y = float(np.min(y_vals)), float(np.max(y_vals))

        grid_props = grid_manager.get_grid_properties()
        spacing = float(grid_props.get("spacing", 1.0))
        x_extent = max_x - min_x
        y_extent = max_y - min_y
        x_margin = max(spacing, 0.1 * x_extent)
        y_margin = max(spacing, 0.1 * y_extent)

        for spin, lower, upper in (
            (self.spin_start_x, min_x - x_margin, max_x + x_margin),
            (self.spin_end_x, min_x - x_margin, max_x + x_margin),
            (self.spin_start_y, min_y - y_margin, max_y + y_margin),
            (self.spin_end_y, min_y - y_margin, max_y + y_margin),
        ):
            spin.blockSignals(True)
            spin.setRange(lower, upper)
            spin.blockSignals(False)

        rows, cols = grid_manager.grid_size
        self.lbl_preview_status.setText(f"Grid ready: {rows}×{cols}")
        self._request_preview()

    def _on_config_dir_changed(self, directory: Optional[Path]) -> None:
        self._current_project_dir = directory
        if directory is None:
            self.library_status.setText(
                "No project loaded. Save a configuration first."
            )
            self._library_dir = None
            self.stimulus_list.clear()
            self._current_stimulus_path = None
            self._update_library_buttons()
            return

        library_dir = directory / "stimuli"
        library_dir.mkdir(parents=True, exist_ok=True)
        self._library_dir = library_dir
        self.library_status.setText(f"Library folder: {library_dir}")
        self._refresh_stimulus_library()

    # ------------------------------------------------------------------
    # Library operations
    # ------------------------------------------------------------------
    def _refresh_stimulus_library(self) -> None:
        self.stimulus_list.blockSignals(True)
        self.stimulus_list.clear()
        if self._library_dir is None:
            self.stimulus_list.blockSignals(False)
            self._update_library_buttons()
            return

        entries: List[Tuple[str, Path]] = []
        for json_path in sorted(self._library_dir.glob("*.json")):
            try:
                with json_path.open("r", encoding="utf-8") as fp:
                    payload = json.load(fp)
                name = payload.get("name") or json_path.stem
                stim_type = payload.get("type", "unknown")
                motion = payload.get("motion", "static")
                display = f"{name} — {stim_type}/{motion}"
                entries.append((display, json_path))
            except (OSError, json.JSONDecodeError):
                continue

        for display, path in entries:
            item = QtWidgets.QListWidgetItem(display)
            item.setData(QtCore.Qt.UserRole, str(path))
            self.stimulus_list.addItem(item)

        self.stimulus_list.blockSignals(False)
        self._on_library_selection_changed()

    def _save_stimulus_update(self) -> None:
        if self._library_dir is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Project required",
                "Save a mechanoreceptor configuration before storing stimuli.",
            )
            return
        if self._current_stimulus_path is None:
            self._save_stimulus_as()
            return
        self._write_stimulus(self._current_stimulus_path)

    def _save_stimulus_as(self) -> None:
        if self._library_dir is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Project required",
                "Save a mechanoreceptor configuration before storing stimuli.",
            )
            return

        default_name = self.txt_stimulus_name.text().strip() or "stimulus"
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Stimulus name",
            "Enter a stimulus name:",
            text=default_name,
        )
        if not ok or not name.strip():
            return
        sanitized = self._sanitize_name(name)
        target_path = self._library_dir / f"{sanitized}.json"
        if target_path.exists():
            reply = QtWidgets.QMessageBox.question(
                self,
                "Overwrite?",
                f"Stimulus '{sanitized}' already exists. Overwrite it?",
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        self.txt_stimulus_name.setText(name.strip())
        self._current_stimulus_path = target_path
        self._write_stimulus(target_path)

    def _write_stimulus(self, path: Path) -> None:
        config = self._collect_config()
        bundle = {
            "schema_version": STIMULUS_SCHEMA_VERSION,
            "kind": "stimulus",
            "name": config.name,
            **config.as_dict(),
            "grid": self._grid_signature(),
        }
        try:
            with path.open("w", encoding="utf-8") as fp:
                json.dump(bundle, fp, indent=2)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Save failed",
                f"Unable to write stimulus file:\n{exc}",
            )
            return
        self.library_status.setText(f"Saved stimulus to {path}")
        self._refresh_stimulus_library()
        self._select_library_item(path)

    def _load_selected_stimulus(self) -> None:
        if self._current_stimulus_path is None:
            return
        self._load_stimulus_from_file(self._current_stimulus_path)

    def _load_stimulus_from_file(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Load failed",
                f"Could not read stimulus:\n{exc}",
            )
            return

        self._loading = True
        try:
            name = payload.get("name", path.stem)
            self.txt_stimulus_name.setText(name)

            stim_type = payload.get("type", "gaussian")
            if stim_type in self.type_buttons:
                self.type_buttons[stim_type].setChecked(True)
                self._selected_type = stim_type
            else:
                self.type_buttons["gaussian"].setChecked(True)
                self._selected_type = "gaussian"
            self._update_spread_label()

            motion = payload.get("motion", "static")
            self.radio_moving.setChecked(motion == "moving")
            self.radio_static.setChecked(motion != "moving")

            start = payload.get("start", [0.0, 0.0])
            end = payload.get("end", start)
            self._set_spin_values(
                (
                    (self.spin_start_x, float(start[0])),
                    (self.spin_start_y, float(start[1])),
                    (self.spin_end_x, float(end[0])),
                    (self.spin_end_y, float(end[1])),
                )
            )

            self.dbl_spread.setValue(float(payload.get("spread", 0.3)))
            self.dbl_orientation.setValue(float(payload.get("orientation_deg", 0.0)))
            self.dbl_amplitude.setValue(float(payload.get("amplitude", 1.0)))
            self.spin_speed.setValue(float(payload.get("speed_mm_s", 0.0)))

            for widget, key in (
                (self.spin_ramp_up, "ramp_up_ms"),
                (self.spin_plateau, "plateau_ms"),
                (self.spin_ramp_down, "ramp_down_ms"),
                (self.spin_total_time, "total_ms"),
                (self.spin_dt, "dt_ms"),
            ):
                widget.setValue(float(payload.get(key, widget.value())))

            self._current_stimulus_path = path
        finally:
            self._loading = False

        self._update_motion_dependencies(from_speed=False)
        self._update_library_buttons()
        self.library_status.setText(f"Loaded stimulus from {path}")
        self._request_preview()

        grid_sig = payload.get("grid")
        if grid_sig and self.grid_manager is not None:
            current_sig = self._grid_signature()
            mismatch = any(
                abs(grid_sig.get(key, current_sig[key]) - current_sig[key]) > 1e-6
                for key in ("rows", "cols", "spacing", "center_x", "center_y")
            )
            if mismatch:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Grid mismatch",
                    "Loaded stimulus was created for a different grid. "
                    "Preview may differ.",
                )

    def _delete_selected_stimulus(self) -> None:
        if self._current_stimulus_path is None:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete stimulus",
            f"Delete '{self._current_stimulus_path.name}'?",
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            self._current_stimulus_path.unlink()
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Delete failed",
                f"Unable to delete stimulus:\n{exc}",
            )
            return
        self._current_stimulus_path = None
        self._refresh_stimulus_library()

    # ------------------------------------------------------------------
    # Preview generation
    # ------------------------------------------------------------------
    def _request_preview(self) -> None:
        if self._loading:
            return
        self._preview_timer.start(self.PREVIEW_THROTTLE_MS)

    def _update_preview(self) -> None:
        if self.generator is None:
            self._reset_animation_controls()
            return

        config = self._collect_config()
        frames, time_axis, amplitude = self._build_stimulus_frames(config)
        if frames is None or time_axis is None or amplitude is None:
            self._reset_animation_controls()
            self.image_view.clear()
            self.time_plot.clear()
            self.lbl_preview_status.setText("Unable to generate preview.")
            return

        data = frames.detach().cpu().numpy()
        times = time_axis.detach().cpu().numpy()
        amps = amplitude.detach().cpu().numpy()

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        display_data = data

        frame_count = data.shape[0]
        if frame_count == 0:
            self._reset_animation_controls()
            self.image_view.clear()
            self.time_plot.clear()
            self.lbl_preview_status.setText("No frames generated.")
            return

        times = np.asarray(times, dtype=float).reshape(-1)
        if times.size != frame_count:
            times = np.linspace(
                0.0,
                config.dt_ms * (frame_count - 1),
                frame_count,
            )

        amps = np.asarray(amps, dtype=float).reshape(-1)
        if amps.size != frame_count:
            amps = np.resize(amps, frame_count)
        image_kwargs: Dict[str, object] = {}
        view_bounds: Optional[Tuple[float, float, float, float]] = None
        if self.grid_manager is not None:
            try:
                xx = self.grid_manager.xx.detach().cpu().numpy()
                yy = self.grid_manager.yy.detach().cpu().numpy()
            except Exception:  # pragma: no cover - fallback if tensors missing
                xx = yy = None

            if xx is not None and yy is not None:
                x_min = float(np.min(xx))
                x_max = float(np.max(xx))
                y_min = float(np.min(yy))
                y_max = float(np.max(yy))
                width_mm = max(x_max - x_min, 1e-6)
                height_mm = max(y_max - y_min, 1e-6)
                width_px = max(float(data.shape[-1]), 1.0)
                height_px = max(float(data.shape[-2]), 1.0)
                scale_x = width_mm / width_px
                scale_y = height_mm / height_px
                image_kwargs["pos"] = (x_min, y_min)
                image_kwargs["scale"] = (scale_x, scale_y)
                view_bounds = (x_min, y_min, width_mm, height_mm)

        self.image_view.setImage(display_data, xvals=times, **image_kwargs)
        self.image_view.setCurrentIndex(0)

        axis_bottom = self.image_view.view.getAxis("bottom")
        axis_left = self.image_view.view.getAxis("left")
        if axis_bottom is not None:
            axis_bottom.setLabel("X (mm)")
        if axis_left is not None:
            axis_left.setLabel("Y (mm)")

        if view_bounds is not None:
            x_min, y_min, width_mm, height_mm = view_bounds
            rect = QtCore.QRectF(x_min, y_min, width_mm, height_mm)
            self.image_view.view.setRange(rect, padding=0.02)

        self.time_plot.clear()
        pen = pg.mkPen(QtGui.QColor("#1f77b4"), width=2.0)
        self.time_plot.plot(times, amps, pen=pen)
        amp_max = float(np.max(amps)) if amps.size else 0.0
        self.time_plot.setYRange(0.0, max(amp_max * 1.1, 1.0), padding=0.05)

        self._animation_timer.stop()
        self._playback_active = False
        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(False)
        self.btn_play.setText("Play")
        self.btn_play.setEnabled(frame_count > 1)
        self.btn_play.blockSignals(False)

        self.frame_slider.blockSignals(True)
        self.frame_slider.setEnabled(frame_count > 1)
        self.frame_slider.setRange(0, max(frame_count - 1, 0))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self._preview_frame_count = frame_count
        self._preview_times = times
        self._preview_dt_ms = max(float(config.dt_ms), MIN_TIME_STEP_MS)
        self._update_frame_label(0)

        self.lbl_time_steps.setText(
            "1 frame" if frame_count == 1 else f"{frame_count} frames"
        )
        self.lbl_preview_status.setText(
            f"Preview generated ({frame_count} frames, " f"dt={config.dt_ms:.2f} ms)"
        )

    def _update_frame_label(self, frame_index: int) -> None:
        if self._preview_frame_count == 0:
            self.lbl_frame_info.setText("Frame 0 / 0 (0.0 ms)")
            return
        idx = max(0, min(frame_index, self._preview_frame_count - 1))
        if idx != frame_index:
            frame_index = idx
        if frame_index >= len(self._preview_times):
            time_ms = 0.0
        else:
            time_ms = float(self._preview_times[frame_index])
        self.lbl_frame_info.setText(
            f"Frame {frame_index + 1} / {self._preview_frame_count} "
            f"({time_ms:.1f} ms)"
        )

    def _on_frame_changed(self, value: int) -> None:
        if self._preview_frame_count == 0:
            return
        index = max(0, min(int(value), self._preview_frame_count - 1))
        if index != value:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(index)
            self.frame_slider.blockSignals(False)
        self.image_view.setCurrentIndex(index)
        self._update_frame_label(index)
        if self._playback_active and self._preview_frame_count > 1:
            self._schedule_next_frame(index)

    def _toggle_animation(self, playing: bool) -> None:
        if self._preview_frame_count <= 1:
            if playing:
                self.btn_play.blockSignals(True)
                self.btn_play.setChecked(False)
                self.btn_play.blockSignals(False)
            self.btn_play.setText("Play")
            self._playback_active = False
            self._animation_timer.stop()
            return

        self._playback_active = playing
        if playing:
            self.btn_play.setText("Pause")
            current = self.frame_slider.value()
            if current >= self._preview_frame_count - 1:
                self.frame_slider.setValue(0)
                current = 0
            self._schedule_next_frame(int(current))
        else:
            self.btn_play.setText("Play")
            self._animation_timer.stop()

    def _schedule_next_frame(self, index: int) -> None:
        if self._preview_frame_count <= 1:
            return
        clamped = max(0, min(int(index), self._preview_frame_count - 1))
        next_index = (clamped + 1) % self._preview_frame_count
        times = self._preview_times
        current_time = float(times[clamped]) if clamped < len(times) else 0.0
        if next_index > clamped and next_index < len(times):
            interval = float(times[next_index]) - current_time
        else:
            interval = self._preview_dt_ms
        if interval <= 0.0:
            interval = self._preview_dt_ms
        interval_ms = max(int(math.ceil(interval)), 1)
        self._animation_timer.stop()
        self._animation_timer.start(interval_ms)

    def _advance_frame(self) -> None:
        if not self._playback_active or self._preview_frame_count <= 1:
            return
        next_index = (self.frame_slider.value() + 1) % self._preview_frame_count
        self.frame_slider.setValue(next_index)

    def _collect_config(self) -> StimulusConfig:
        name = self.txt_stimulus_name.text().strip() or "Stimulus"
        motion = "moving" if self.radio_moving.isChecked() else "static"
        config = StimulusConfig(
            name=name,
            stimulus_type=self._selected_type,
            motion=motion,
            start=(self.spin_start_x.value(), self.spin_start_y.value()),
            end=(self.spin_end_x.value(), self.spin_end_y.value()),
            spread=self.dbl_spread.value(),
            orientation_deg=self.dbl_orientation.value(),
            amplitude=self.dbl_amplitude.value(),
            ramp_up_ms=self.spin_ramp_up.value(),
            plateau_ms=self.spin_plateau.value(),
            ramp_down_ms=self.spin_ramp_down.value(),
            total_ms=self.spin_total_time.value(),
            dt_ms=self.spin_dt.value(),
            speed_mm_s=self.spin_speed.value(),
        )
        return config

    def _build_stimulus_frames(
        self,
        config: StimulusConfig,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],]:
        if self.generator is None or self.grid_manager is None:
            return None, None, None

        device = self.generator.xx.device
        dt = max(config.dt_ms, MIN_TIME_STEP_MS)
        total_time = max(config.total_ms, dt)
        time_axis = torch.arange(0.0, total_time + 0.5 * dt, dt, device=device)

        amplitude_profile = self._amplitude_profile(time_axis, config)
        peak = config.amplitude
        if peak <= 0.0:
            amplitude_profile = torch.zeros_like(amplitude_profile)
        else:
            amplitude_profile = amplitude_profile * peak

        xx = self.generator.xx
        yy = self.generator.yy
        frames = torch.zeros(
            (time_axis.numel(),) + xx.shape, device=device, dtype=xx.dtype
        )

        start_x, start_y = config.start
        end_x, end_y = config.end
        distance = math.hypot(end_x - start_x, end_y - start_y)
        plateau = config.plateau_ms
        ramp_up = config.ramp_up_ms

        for idx, t_val in enumerate(time_axis):
            t = float(t_val.item())
            if config.motion == "moving" and plateau > 0.0 and distance > 1e-6:
                if t <= ramp_up:
                    alpha = 0.0
                elif t >= ramp_up + plateau:
                    alpha = 1.0
                else:
                    alpha = (t - ramp_up) / plateau
                cx = start_x + alpha * (end_x - start_x)
                cy = start_y + alpha * (end_y - start_y)
            else:
                cx, cy = start_x, start_y

            if config.stimulus_type == "gaussian":
                frame = gaussian_pressure_torch(
                    xx,
                    yy,
                    cx,
                    cy,
                    amplitude=1.0,
                    sigma=max(config.spread, 1e-6),
                )
            elif config.stimulus_type == "point":
                frame = point_pressure_torch(
                    xx,
                    yy,
                    cx,
                    cy,
                    amplitude=1.0,
                    diameter_mm=max(config.spread, 1e-6),
                )
            else:
                theta = torch.tensor(
                    math.radians(config.orientation_deg),
                    device=device,
                    dtype=xx.dtype,
                )
                frame = edge_stimulus_torch(
                    xx - cx,
                    yy - cy,
                    theta=theta,
                    w=max(config.spread, 1e-6),
                    amplitude=1.0,
                )

            frames[idx] = frame * amplitude_profile[idx]

        return frames, time_axis, amplitude_profile

    def _amplitude_profile(
        self,
        time_axis: torch.Tensor,
        config: StimulusConfig,
    ) -> torch.Tensor:
        ramp_up = max(config.ramp_up_ms, 0.0)
        ramp_down = max(config.ramp_down_ms, 0.0)
        plateau = max(config.plateau_ms, 0.0)
        down_start = ramp_up + plateau
        total_duration = ramp_up + plateau + ramp_down

        profile = torch.zeros_like(time_axis)
        if ramp_up > 0.0:
            up_mask = time_axis < ramp_up
            profile[up_mask] = time_axis[up_mask] / ramp_up
        else:
            profile[time_axis < ramp_up + 1e-6] = 1.0

        hold_mask = (time_axis >= ramp_up) & (time_axis < down_start)
        profile[hold_mask] = 1.0

        if ramp_down > 0.0:
            down_mask = (time_axis >= down_start) & (
                time_axis <= down_start + ramp_down
            )
            profile[down_mask] = torch.clamp(
                1.0 - (time_axis[down_mask] - down_start) / ramp_down,
                min=0.0,
                max=1.0,
            )

        profile = torch.where(
            time_axis > total_duration,
            torch.zeros_like(profile),
            profile,
        )
        return torch.clamp(profile, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_name(name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
        return sanitized or "stimulus"

    def _update_spread_label(self) -> None:
        if self._selected_type == "edge":
            self.lbl_spread.setText("Edge width (mm):")
        elif self._selected_type == "point":
            self.lbl_spread.setText("Diameter (mm):")
        else:
            self.lbl_spread.setText("Sigma (mm):")

    def _set_spin_values(
        self,
        updates: Tuple[Tuple[QtWidgets.QDoubleSpinBox, float], ...],
    ) -> None:
        for widget, value in updates:
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

    def _select_library_item(self, path: Path) -> None:
        for index in range(self.stimulus_list.count()):
            item = self.stimulus_list.item(index)
            if Path(item.data(QtCore.Qt.UserRole)) == path:
                self.stimulus_list.setCurrentItem(item)
                break

    def _update_library_buttons(self) -> None:
        has_project = self._library_dir is not None
        has_selection = self._current_stimulus_path is not None
        self.btn_save_stimulus.setEnabled(has_project)
        self.btn_save_stimulus_as.setEnabled(has_project)
        self.btn_load_stimulus.setEnabled(has_selection)
        self.btn_delete_stimulus.setEnabled(has_selection)

    def _update_motion_dependencies(self, from_speed: bool) -> None:
        if self._loading:
            return
        moving = self.radio_moving.isChecked()
        if not moving:
            return
        distance = math.hypot(
            self.spin_end_x.value() - self.spin_start_x.value(),
            self.spin_end_y.value() - self.spin_start_y.value(),
        )
        if distance < 1e-6:
            self.spin_speed.blockSignals(True)
            self.spin_speed.setValue(0.0)
            self.spin_speed.blockSignals(False)
            return

        if from_speed:
            if self._speed_syncing:
                return
            speed = max(self.spin_speed.value(), 1e-6)
            plateau_ms = distance / speed * 1000.0
            plateau_ms = max(plateau_ms, self.spin_dt.value())
            self._speed_syncing = True
            self.spin_plateau.blockSignals(True)
            self.spin_plateau.setValue(plateau_ms)
            self.spin_plateau.blockSignals(False)
            self._speed_syncing = False
            self._on_ramp_component_changed(0.0)
        else:
            plateau_ms = max(self.spin_plateau.value(), self.spin_dt.value())
            speed = distance / (plateau_ms / 1000.0)
            self._speed_syncing = True
            self.spin_speed.blockSignals(True)
            self.spin_speed.setValue(speed)
            self.spin_speed.blockSignals(False)
            self._speed_syncing = False
        self._request_preview()

    def _grid_signature(self) -> Dict[str, float]:
        if self.grid_manager is None:
            return {
                "rows": 0,
                "cols": 0,
                "spacing": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        grid_props = self.grid_manager.get_grid_properties()
        rows, cols = self.grid_manager.grid_size
        center = grid_props.get("center", (0.0, 0.0))
        return {
            "rows": int(rows),
            "cols": int(cols),
            "spacing": float(grid_props.get("spacing", 0.0)),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
        }

    # ------------------------------------------------------------------
    # Public accessors for other tabs
    # ------------------------------------------------------------------
    def library_directory(self) -> Optional[Path]:
        """Return the directory containing saved stimulus JSON files."""

        return self._library_dir

    def list_saved_stimuli(self) -> List[Path]:
        """List all saved stimulus definitions in the current library."""

        if self._library_dir is None:
            return []
        return sorted(self._library_dir.glob("*.json"))

    def load_stimulus_payload(self, path: Path) -> Dict[str, Any]:
        """Load a saved stimulus JSON file and return its payload."""

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def generate_frames_from_payload(
        self, payload: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate stimulus frames/time/amplitude triplet from payload."""

        if self.generator is None or self.grid_manager is None:
            raise RuntimeError(
                "Stimulus generator unavailable. Configure the grid first."
            )

        start_raw = payload.get("start") or payload.get("start_position") or [0.0, 0.0]
        end_raw = payload.get("end") or payload.get("end_position") or start_raw
        stim_type = str(payload.get("type", payload.get("stimulus_type", "gaussian")))
        motion = str(payload.get("motion", "static"))
        orientation = float(
            payload.get("orientation_deg", payload.get("orientation", 0.0))
        )
        config = StimulusConfig(
            name=str(payload.get("name", "Stimulus")),
            stimulus_type=stim_type,
            motion=motion,
            start=(float(start_raw[0]), float(start_raw[1])),
            end=(float(end_raw[0]), float(end_raw[1])),
            spread=float(payload.get("spread", payload.get("sigma", 0.3))),
            orientation_deg=orientation,
            amplitude=float(payload.get("amplitude", payload.get("peak", 1.0))),
            ramp_up_ms=float(payload.get("ramp_up_ms", payload.get("ramp_up", 50.0))),
            plateau_ms=float(payload.get("plateau_ms", payload.get("plateau", 200.0))),
            ramp_down_ms=float(
                payload.get("ramp_down_ms", payload.get("ramp_down", 50.0))
            ),
            total_ms=float(payload.get("total_ms", payload.get("total_time", 300.0))),
            dt_ms=float(payload.get("dt_ms", payload.get("dt", 1.0))),
            speed_mm_s=float(payload.get("speed_mm_s", payload.get("speed", 0.0))),
        )

        frames, time_axis, amplitude = self._build_stimulus_frames(config)
        if frames is None or time_axis is None or amplitude is None:
            raise RuntimeError("Unable to generate stimulus frames from payload")
        return frames, time_axis, amplitude


__all__ = ["StimulusDesignerTab"]
