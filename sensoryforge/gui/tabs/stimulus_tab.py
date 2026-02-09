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
    
    # Texture parameters
    texture_subtype: str = "gabor"
    wavelength: float = 0.5
    phase: float = 0.0
    edge_count: int = 5
    edge_width: float = 0.05
    noise_scale: float = 1.0
    noise_kernel_size: int = 5
    
    # Moving stimulus parameters
    moving_subtype: str = "linear"
    num_steps: int = 100
    radius: float = 1.0
    start_angle: float = 0.0
    end_angle: float = 6.28318  # 2*pi
    moving_sigma: float = 0.3

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
            "texture_subtype": self.texture_subtype,
            "wavelength": float(self.wavelength),
            "phase": float(self.phase),
            "edge_count": int(self.edge_count),
            "edge_width": float(self.edge_width),
            "noise_scale": float(self.noise_scale),
            "noise_kernel_size": int(self.noise_kernel_size),
            "moving_subtype": self.moving_subtype,
            "num_steps": int(self.num_steps),
            "radius": float(self.radius),
            "start_angle": float(self.start_angle),
            "end_angle": float(self.end_angle),
            "moving_sigma": float(self.moving_sigma),
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
        self._texture_subtype = "gabor"
        self._moving_subtype = "linear"

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
        self._build_texture_subtype_section()
        self._build_moving_subtype_section()
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
            ("texture", "🎨 Texture"),
            ("moving", "➡️ Moving"),
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
    
    def _build_texture_subtype_section(self) -> None:
        """Build texture sub-type selector and parameter controls."""
        self.texture_group = QtWidgets.QGroupBox("Texture Parameters")
        layout = QtWidgets.QVBoxLayout(self.texture_group)
        
        # Sub-type selector
        subtype_row = QtWidgets.QHBoxLayout()
        subtype_row.addWidget(QtWidgets.QLabel("Texture Type:"))
        self.texture_subtype_combo = QtWidgets.QComboBox()
        self.texture_subtype_combo.addItems(["Gabor", "Edge Grating", "Noise"])
        subtype_row.addWidget(self.texture_subtype_combo)
        subtype_row.addStretch(1)
        layout.addLayout(subtype_row)
        
        # Gabor parameters
        self.gabor_params_widget = QtWidgets.QWidget()
        gabor_form = QtWidgets.QFormLayout(self.gabor_params_widget)
        gabor_form.setLabelAlignment(QtCore.Qt.AlignRight)
        
        self.spin_wavelength = QtWidgets.QDoubleSpinBox()
        self.spin_wavelength.setDecimals(3)
        self.spin_wavelength.setRange(0.1, 10.0)
        self.spin_wavelength.setSingleStep(0.1)
        self.spin_wavelength.setValue(0.5)
        
        self.spin_texture_orientation = QtWidgets.QDoubleSpinBox()
        self.spin_texture_orientation.setDecimals(1)
        self.spin_texture_orientation.setRange(0.0, 180.0)
        self.spin_texture_orientation.setSingleStep(5.0)
        self.spin_texture_orientation.setValue(0.0)
        
        self.spin_texture_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_texture_sigma.setDecimals(3)
        self.spin_texture_sigma.setRange(0.05, 5.0)
        self.spin_texture_sigma.setSingleStep(0.05)
        self.spin_texture_sigma.setValue(0.3)
        
        self.spin_phase = QtWidgets.QDoubleSpinBox()
        self.spin_phase.setDecimals(2)
        self.spin_phase.setRange(0.0, 6.28)
        self.spin_phase.setSingleStep(0.1)
        self.spin_phase.setValue(0.0)
        
        gabor_form.addRow("Wavelength (mm):", self.spin_wavelength)
        gabor_form.addRow("Orientation (°):", self.spin_texture_orientation)
        gabor_form.addRow("Sigma (mm):", self.spin_texture_sigma)
        gabor_form.addRow("Phase (rad):", self.spin_phase)
        layout.addWidget(self.gabor_params_widget)
        
        # Edge Grating parameters
        self.edge_grating_params_widget = QtWidgets.QWidget()
        edge_form = QtWidgets.QFormLayout(self.edge_grating_params_widget)
        edge_form.setLabelAlignment(QtCore.Qt.AlignRight)
        
        self.spin_edge_orientation = QtWidgets.QDoubleSpinBox()
        self.spin_edge_orientation.setDecimals(1)
        self.spin_edge_orientation.setRange(0.0, 180.0)
        self.spin_edge_orientation.setSingleStep(5.0)
        self.spin_edge_orientation.setValue(0.0)
        
        self.spin_spacing = QtWidgets.QDoubleSpinBox()
        self.spin_spacing.setDecimals(2)
        self.spin_spacing.setRange(0.1, 5.0)
        self.spin_spacing.setSingleStep(0.1)
        self.spin_spacing.setValue(0.6)
        
        self.spin_edge_count = QtWidgets.QSpinBox()
        self.spin_edge_count.setRange(1, 20)
        self.spin_edge_count.setSingleStep(1)
        self.spin_edge_count.setValue(5)
        
        self.spin_edge_width = QtWidgets.QDoubleSpinBox()
        self.spin_edge_width.setDecimals(3)
        self.spin_edge_width.setRange(0.01, 1.0)
        self.spin_edge_width.setSingleStep(0.01)
        self.spin_edge_width.setValue(0.05)
        
        edge_form.addRow("Orientation (°):", self.spin_edge_orientation)
        edge_form.addRow("Spacing (mm):", self.spin_spacing)
        edge_form.addRow("Count:", self.spin_edge_count)
        edge_form.addRow("Edge Width (mm):", self.spin_edge_width)
        layout.addWidget(self.edge_grating_params_widget)
        self.edge_grating_params_widget.setVisible(False)
        
        # Noise parameters
        self.noise_params_widget = QtWidgets.QWidget()
        noise_form = QtWidgets.QFormLayout(self.noise_params_widget)
        noise_form.setLabelAlignment(QtCore.Qt.AlignRight)
        
        self.spin_noise_scale = QtWidgets.QDoubleSpinBox()
        self.spin_noise_scale.setDecimals(2)
        self.spin_noise_scale.setRange(0.1, 10.0)
        self.spin_noise_scale.setSingleStep(0.1)
        self.spin_noise_scale.setValue(1.0)
        
        self.spin_noise_kernel = QtWidgets.QSpinBox()
        self.spin_noise_kernel.setRange(3, 21)
        self.spin_noise_kernel.setSingleStep(2)
        self.spin_noise_kernel.setValue(5)
        
        noise_form.addRow("Scale:", self.spin_noise_scale)
        noise_form.addRow("Kernel Size:", self.spin_noise_kernel)
        layout.addWidget(self.noise_params_widget)
        self.noise_params_widget.setVisible(False)
        
        self.texture_group.setVisible(False)
        self.control_layout.addWidget(self.texture_group)
    
    def _build_moving_subtype_section(self) -> None:
        """Build moving stimulus sub-type selector and parameter controls."""
        self.moving_group = QtWidgets.QGroupBox("Moving Stimulus Parameters")
        layout = QtWidgets.QVBoxLayout(self.moving_group)
        
        # Sub-type selector
        subtype_row = QtWidgets.QHBoxLayout()
        subtype_row.addWidget(QtWidgets.QLabel("Motion Type:"))
        self.moving_subtype_combo = QtWidgets.QComboBox()
        self.moving_subtype_combo.addItems(["Linear", "Circular", "Slide"])
        subtype_row.addWidget(self.moving_subtype_combo)
        subtype_row.addStretch(1)
        layout.addLayout(subtype_row)
        
        # Linear motion parameters
        self.linear_params_widget = QtWidgets.QWidget()
        linear_form = QtWidgets.QFormLayout(self.linear_params_widget)
        linear_form.setLabelAlignment(QtCore.Qt.AlignRight)
        
        self.spin_linear_start_x = QtWidgets.QDoubleSpinBox()
        self.spin_linear_start_x.setDecimals(2)
        self.spin_linear_start_x.setRange(-50.0, 50.0)
        self.spin_linear_start_x.setValue(0.0)
        
        self.spin_linear_start_y = QtWidgets.QDoubleSpinBox()
        self.spin_linear_start_y.setDecimals(2)
        self.spin_linear_start_y.setRange(-50.0, 50.0)
        self.spin_linear_start_y.setValue(0.0)
        
        self.spin_linear_end_x = QtWidgets.QDoubleSpinBox()
        self.spin_linear_end_x.setDecimals(2)
        self.spin_linear_end_x.setRange(-50.0, 50.0)
        self.spin_linear_end_x.setValue(2.0)
        
        self.spin_linear_end_y = QtWidgets.QDoubleSpinBox()
        self.spin_linear_end_y.setDecimals(2)
        self.spin_linear_end_y.setRange(-50.0, 50.0)
        self.spin_linear_end_y.setValue(0.0)
        
        self.spin_num_steps = QtWidgets.QSpinBox()
        self.spin_num_steps.setRange(10, 1000)
        self.spin_num_steps.setValue(100)
        
        self.spin_moving_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_moving_sigma.setDecimals(3)
        self.spin_moving_sigma.setRange(0.05, 2.0)
        self.spin_moving_sigma.setSingleStep(0.05)
        self.spin_moving_sigma.setValue(0.3)
        
        linear_form.addRow("Start X (mm):", self.spin_linear_start_x)
        linear_form.addRow("Start Y (mm):", self.spin_linear_start_y)
        linear_form.addRow("End X (mm):", self.spin_linear_end_x)
        linear_form.addRow("End Y (mm):", self.spin_linear_end_y)
        linear_form.addRow("Num Steps:", self.spin_num_steps)
        linear_form.addRow("Sigma (mm):", self.spin_moving_sigma)
        layout.addWidget(self.linear_params_widget)
        
        # Circular motion parameters
        self.circular_params_widget = QtWidgets.QWidget()
        circular_form = QtWidgets.QFormLayout(self.circular_params_widget)
        circular_form.setLabelAlignment(QtCore.Qt.AlignRight)
        
        self.spin_circular_center_x = QtWidgets.QDoubleSpinBox()
        self.spin_circular_center_x.setDecimals(2)
        self.spin_circular_center_x.setRange(-50.0, 50.0)
        self.spin_circular_center_x.setValue(0.0)
        
        self.spin_circular_center_y = QtWidgets.QDoubleSpinBox()
        self.spin_circular_center_y.setDecimals(2)
        self.spin_circular_center_y.setRange(-50.0, 50.0)
        self.spin_circular_center_y.setValue(0.0)
        
        self.spin_radius = QtWidgets.QDoubleSpinBox()
        self.spin_radius.setDecimals(2)
        self.spin_radius.setRange(0.1, 20.0)
        self.spin_radius.setValue(1.0)
        
        self.spin_circular_num_steps = QtWidgets.QSpinBox()
        self.spin_circular_num_steps.setRange(10, 1000)
        self.spin_circular_num_steps.setValue(100)
        
        self.spin_start_angle = QtWidgets.QDoubleSpinBox()
        self.spin_start_angle.setDecimals(2)
        self.spin_start_angle.setRange(0.0, 6.28)
        self.spin_start_angle.setValue(0.0)
        
        self.spin_end_angle = QtWidgets.QDoubleSpinBox()
        self.spin_end_angle.setDecimals(2)
        self.spin_end_angle.setRange(0.0, 6.28)
        self.spin_end_angle.setValue(6.28)
        
        self.spin_circular_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_circular_sigma.setDecimals(3)
        self.spin_circular_sigma.setRange(0.05, 2.0)
        self.spin_circular_sigma.setSingleStep(0.05)
        self.spin_circular_sigma.setValue(0.3)
        
        circular_form.addRow("Center X (mm):", self.spin_circular_center_x)
        circular_form.addRow("Center Y (mm):", self.spin_circular_center_y)
        circular_form.addRow("Radius (mm):", self.spin_radius)
        circular_form.addRow("Num Steps:", self.spin_circular_num_steps)
        circular_form.addRow("Start Angle (rad):", self.spin_start_angle)
        circular_form.addRow("End Angle (rad):", self.spin_end_angle)
        circular_form.addRow("Sigma (mm):", self.spin_circular_sigma)
        layout.addWidget(self.circular_params_widget)
        self.circular_params_widget.setVisible(False)
        
        # Slide parameters (reuses linear parameters)
        self.slide_params_widget = QtWidgets.QWidget()
        slide_form = QtWidgets.QFormLayout(self.slide_params_widget)
        slide_form.setLabelAlignment(QtCore.Qt.AlignRight)
        
        self.spin_slide_start_x = QtWidgets.QDoubleSpinBox()
        self.spin_slide_start_x.setDecimals(2)
        self.spin_slide_start_x.setRange(-50.0, 50.0)
        self.spin_slide_start_x.setValue(0.0)
        
        self.spin_slide_start_y = QtWidgets.QDoubleSpinBox()
        self.spin_slide_start_y.setDecimals(2)
        self.spin_slide_start_y.setRange(-50.0, 50.0)
        self.spin_slide_start_y.setValue(0.0)
        
        self.spin_slide_end_x = QtWidgets.QDoubleSpinBox()
        self.spin_slide_end_x.setDecimals(2)
        self.spin_slide_end_x.setRange(-50.0, 50.0)
        self.spin_slide_end_x.setValue(2.0)
        
        self.spin_slide_end_y = QtWidgets.QDoubleSpinBox()
        self.spin_slide_end_y.setDecimals(2)
        self.spin_slide_end_y.setRange(-50.0, 50.0)
        self.spin_slide_end_y.setValue(0.0)
        
        self.spin_slide_num_steps = QtWidgets.QSpinBox()
        self.spin_slide_num_steps.setRange(10, 1000)
        self.spin_slide_num_steps.setValue(100)
        
        self.spin_slide_sigma = QtWidgets.QDoubleSpinBox()
        self.spin_slide_sigma.setDecimals(3)
        self.spin_slide_sigma.setRange(0.05, 2.0)
        self.spin_slide_sigma.setSingleStep(0.05)
        self.spin_slide_sigma.setValue(0.3)
        
        slide_form.addRow("Start X (mm):", self.spin_slide_start_x)
        slide_form.addRow("Start Y (mm):", self.spin_slide_start_y)
        slide_form.addRow("End X (mm):", self.spin_slide_end_x)
        slide_form.addRow("End Y (mm):", self.spin_slide_end_y)
        slide_form.addRow("Num Steps:", self.spin_slide_num_steps)
        slide_form.addRow("Sigma (mm):", self.spin_slide_sigma)
        layout.addWidget(self.slide_params_widget)
        self.slide_params_widget.setVisible(False)
        
        self.moving_group.setVisible(False)
        self.control_layout.addWidget(self.moving_group)

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
        
        # Texture sub-type and parameter connections
        self.texture_subtype_combo.currentIndexChanged.connect(self._on_texture_subtype_changed)
        self.spin_wavelength.valueChanged.connect(self._handle_preview_request)
        self.spin_texture_orientation.valueChanged.connect(self._handle_preview_request)
        self.spin_texture_sigma.valueChanged.connect(self._handle_preview_request)
        self.spin_phase.valueChanged.connect(self._handle_preview_request)
        self.spin_edge_orientation.valueChanged.connect(self._handle_preview_request)
        self.spin_spacing.valueChanged.connect(self._handle_preview_request)
        self.spin_edge_count.valueChanged.connect(self._handle_preview_request)
        self.spin_edge_width.valueChanged.connect(self._handle_preview_request)
        self.spin_noise_scale.valueChanged.connect(self._handle_preview_request)
        self.spin_noise_kernel.valueChanged.connect(self._handle_preview_request)
        
        # Moving sub-type and parameter connections
        self.moving_subtype_combo.currentIndexChanged.connect(self._on_moving_subtype_changed)
        for spin in (
            self.spin_linear_start_x, self.spin_linear_start_y,
            self.spin_linear_end_x, self.spin_linear_end_y,
            self.spin_num_steps, self.spin_moving_sigma,
            self.spin_circular_center_x, self.spin_circular_center_y,
            self.spin_radius, self.spin_circular_num_steps,
            self.spin_start_angle, self.spin_end_angle, self.spin_circular_sigma,
            self.spin_slide_start_x, self.spin_slide_start_y,
            self.spin_slide_end_x, self.spin_slide_end_y,
            self.spin_slide_num_steps, self.spin_slide_sigma,
        ):
            spin.valueChanged.connect(self._handle_preview_request)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_type_selected(self, button: QtWidgets.QAbstractButton) -> None:
        stimulus_type = button.property("stimulusType")
        if not stimulus_type or stimulus_type == self._selected_type:
            return
        self._selected_type = str(stimulus_type)
        self._update_spread_label()
        
        # Show/hide texture group
        is_texture = self._selected_type == "texture"
        self.texture_group.setVisible(is_texture)
        
        # Show/hide moving group
        is_moving = self._selected_type == "moving"
        self.moving_group.setVisible(is_moving)
        
        # Show/hide edge orientation for edge stimulus
        is_edge = self._selected_type == "edge"
        self.dbl_orientation.setVisible(is_edge)
        if self.lbl_orientation_label is not None:
            self.lbl_orientation_label.setVisible(is_edge)
        
        self._request_preview()
    
    def _on_texture_subtype_changed(self, index: int) -> None:
        """Handle texture sub-type selection change."""
        subtype_map = {0: "gabor", 1: "edge_grating", 2: "noise"}
        self._texture_subtype = subtype_map.get(index, "gabor")
        
        # Show/hide parameter widgets based on selection
        self.gabor_params_widget.setVisible(index == 0)
        self.edge_grating_params_widget.setVisible(index == 1)
        self.noise_params_widget.setVisible(index == 2)
        
        self._request_preview()
    
    def _on_moving_subtype_changed(self, index: int) -> None:
        """Handle moving stimulus sub-type selection change."""
        subtype_map = {0: "linear", 1: "circular", 2: "slide"}
        self._moving_subtype = subtype_map.get(index, "linear")
        
        # Show/hide parameter widgets based on selection
        self.linear_params_widget.setVisible(index == 0)
        self.circular_params_widget.setVisible(index == 1)
        self.slide_params_widget.setVisible(index == 2)
        
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
            
            # Load texture parameters
            texture_subtype = payload.get("texture_subtype", "gabor")
            self._texture_subtype = texture_subtype
            texture_index_map = {"gabor": 0, "edge_grating": 1, "noise": 2}
            self.texture_subtype_combo.setCurrentIndex(texture_index_map.get(texture_subtype, 0))
            
            self.spin_wavelength.setValue(float(payload.get("wavelength", 0.5)))
            self.spin_texture_orientation.setValue(float(payload.get("texture_orientation", 0.0)))
            self.spin_texture_sigma.setValue(float(payload.get("texture_sigma", 0.3)))
            self.spin_phase.setValue(float(payload.get("phase", 0.0)))
            self.spin_edge_orientation.setValue(float(payload.get("edge_orientation", 0.0)))
            self.spin_spacing.setValue(float(payload.get("spacing", 0.6)))
            self.spin_edge_count.setValue(int(payload.get("edge_count", 5)))
            self.spin_edge_width.setValue(float(payload.get("edge_width", 0.05)))
            self.spin_noise_scale.setValue(float(payload.get("noise_scale", 1.0)))
            self.spin_noise_kernel.setValue(int(payload.get("noise_kernel_size", 5)))
            
            # Load moving parameters
            moving_subtype = payload.get("moving_subtype", "linear")
            self._moving_subtype = moving_subtype
            moving_index_map = {"linear": 0, "circular": 1, "slide": 2}
            self.moving_subtype_combo.setCurrentIndex(moving_index_map.get(moving_subtype, 0))
            
            self.spin_linear_start_x.setValue(float(payload.get("linear_start_x", 0.0)))
            self.spin_linear_start_y.setValue(float(payload.get("linear_start_y", 0.0)))
            self.spin_linear_end_x.setValue(float(payload.get("linear_end_x", 2.0)))
            self.spin_linear_end_y.setValue(float(payload.get("linear_end_y", 0.0)))
            self.spin_num_steps.setValue(int(payload.get("num_steps", 100)))
            self.spin_moving_sigma.setValue(float(payload.get("moving_sigma", 0.3)))
            
            self.spin_circular_center_x.setValue(float(payload.get("circular_center_x", 0.0)))
            self.spin_circular_center_y.setValue(float(payload.get("circular_center_y", 0.0)))
            self.spin_radius.setValue(float(payload.get("radius", 1.0)))
            self.spin_circular_num_steps.setValue(int(payload.get("circular_num_steps", 100)))
            self.spin_start_angle.setValue(float(payload.get("start_angle", 0.0)))
            self.spin_end_angle.setValue(float(payload.get("end_angle", 6.28)))
            self.spin_circular_sigma.setValue(float(payload.get("circular_sigma", 0.3)))
            
            self.spin_slide_start_x.setValue(float(payload.get("slide_start_x", 0.0)))
            self.spin_slide_start_y.setValue(float(payload.get("slide_start_y", 0.0)))
            self.spin_slide_end_x.setValue(float(payload.get("slide_end_x", 2.0)))
            self.spin_slide_end_y.setValue(float(payload.get("slide_end_y", 0.0)))
            self.spin_slide_num_steps.setValue(int(payload.get("slide_num_steps", 100)))
            self.spin_slide_sigma.setValue(float(payload.get("slide_sigma", 0.3)))

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
            texture_subtype=self._texture_subtype,
            wavelength=self.spin_wavelength.value(),
            phase=self.spin_phase.value(),
            edge_count=self.spin_edge_count.value(),
            edge_width=self.spin_edge_width.value(),
            noise_scale=self.spin_noise_scale.value(),
            noise_kernel_size=self.spin_noise_kernel.value(),
            moving_subtype=self._moving_subtype,
            num_steps=self.spin_num_steps.value() if self._moving_subtype == "linear" else self.spin_circular_num_steps.value() if self._moving_subtype == "circular" else self.spin_slide_num_steps.value(),
            radius=self.spin_radius.value(),
            start_angle=self.spin_start_angle.value(),
            end_angle=self.spin_end_angle.value(),
            moving_sigma=self.spin_moving_sigma.value() if self._moving_subtype == "linear" else self.spin_circular_sigma.value() if self._moving_subtype == "circular" else self.spin_slide_sigma.value(),
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
            elif config.stimulus_type == "edge":
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
            elif config.stimulus_type == "texture":
                frame = self._generate_texture_frame(xx, yy, cx, cy, config)
            elif config.stimulus_type == "moving":
                # Moving stimuli are generated differently - break out early
                return self._generate_moving_frames(config)
            else:
                # Unknown type, return zeros
                frame = torch.zeros_like(xx)

            frames[idx] = frame * amplitude_profile[idx]

        return frames, time_axis, amplitude_profile
    
    def _generate_texture_frame(
        self,
        xx: torch.Tensor,
        yy: torch.Tensor,
        cx: float,
        cy: float,
        config: StimulusConfig,
    ) -> torch.Tensor:
        """Generate a single texture stimulus frame."""
        from sensoryforge.stimuli.texture import gabor_texture, edge_grating, noise_texture
        
        if config.texture_subtype == "gabor":
            return gabor_texture(
                xx,
                yy,
                center_x=cx,
                center_y=cy,
                amplitude=1.0,
                sigma=max(config.spread if self._texture_subtype == "gabor" else self.spin_texture_sigma.value(), 1e-6),
                wavelength=max(config.wavelength, 0.1),
                orientation=math.radians(self.spin_texture_orientation.value()),
                phase=config.phase,
                device=xx.device,
            )
        elif config.texture_subtype == "edge_grating":
            return edge_grating(
                xx,
                yy,
                orientation=math.radians(self.spin_edge_orientation.value()),
                spacing=max(config.spread if self._texture_subtype == "edge_grating" else self.spin_spacing.value(), 0.1),
                count=config.edge_count,
                edge_width=max(config.edge_width, 0.01),
                amplitude=1.0,
                device=xx.device,
            )
        elif config.texture_subtype == "noise":
            return noise_texture(
                height=xx.shape[0],
                width=xx.shape[1],
                scale=config.noise_scale,
                kernel_size=config.noise_kernel_size,
                device=xx.device,
            )
        else:
            return torch.zeros_like(xx)
    
    def _generate_moving_frames(
        self,
        config: StimulusConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate moving stimulus frames using Phase 2 moving stimulus API."""
        from sensoryforge.stimuli.moving import (
            linear_motion,
            circular_motion,
            slide_trajectory,
            MovingStimulus,
        )
        from sensoryforge.stimuli.gaussian import gaussian_stimulus
        
        if self.generator is None or self.grid_manager is None:
            return None, None, None
        
        device = self.generator.xx.device
        xx = self.generator.xx
        yy = self.generator.yy
        
        # Create spatial stimulus generator (Gaussian blob)
        def spatial_generator(xx_grid, yy_grid, cx, cy):
            return gaussian_stimulus(
                xx_grid,
                yy_grid,
                center_x=cx,
                center_y=cy,
                amplitude=1.0,
                sigma=max(config.moving_sigma, 0.05),
                device=device,
            )
        
        # Generate trajectory based on moving subtype
        if config.moving_subtype == "linear":
            trajectory = linear_motion(
                start=(
                    self.spin_linear_start_x.value(),
                    self.spin_linear_start_y.value(),
                ),
                end=(
                    self.spin_linear_end_x.value(),
                    self.spin_linear_end_y.value(),
                ),
                num_steps=config.num_steps,
                device=device,
            )
        elif config.moving_subtype == "circular":
            trajectory = circular_motion(
                center=(
                    self.spin_circular_center_x.value(),
                    self.spin_circular_center_y.value(),
                ),
                radius=max(config.radius, 0.1),
                num_steps=config.num_steps,
                start_angle=config.start_angle,
                end_angle=config.end_angle,
                device=device,
            )
        elif config.moving_subtype == "slide":
            trajectory = slide_trajectory(
                start=(
                    self.spin_slide_start_x.value(),
                    self.spin_slide_start_y.value(),
                ),
                end=(
                    self.spin_slide_end_x.value(),
                    self.spin_slide_end_y.value(),
                ),
                num_steps=config.num_steps,
                velocity_type="constant",
                device=device,
            )
        else:
            # Fallback to linear
            trajectory = linear_motion(
                start=config.start,
                end=config.end,
                num_steps=config.num_steps,
                device=device,
            )
        
        # Create MovingStimulus
        moving_stim = MovingStimulus(
            trajectory=trajectory,
            stimulus_generator=spatial_generator,
        )
        
        # Generate frames
        frames = moving_stim(xx, yy)
        
        # Create time axis based on num_steps
        dt = max(config.dt_ms, MIN_TIME_STEP_MS)
        time_axis = torch.arange(0.0, config.num_steps * dt, dt, device=device)[:config.num_steps]
        
        # Generate amplitude profile
        amplitude_profile = self._amplitude_profile(time_axis, config)
        
        # Apply amplitude modulation and scaling
        peak = config.amplitude
        if peak <= 0.0:
            frames = torch.zeros_like(frames)
        else:
            for idx in range(frames.shape[0]):
                if idx < len(amplitude_profile):
                    frames[idx] = frames[idx] * amplitude_profile[idx] * peak
        
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

    # ------------------------------------------------------------------ #
    #  Phase B — YAML ↔ GUI bidirectional config API                      #
    # ------------------------------------------------------------------ #

    def get_config(self) -> dict:
        """Export current stimulus tab state as a plain dict for YAML.

        Returns:
            Dictionary capturing all stimulus parameters including texture
            and moving sub-type parameters, regardless of which type is
            currently selected. Supports round-trip fidelity.
        """
        cfg: dict = {
            "name": self.txt_stimulus_name.text().strip() or "Stimulus",
            "type": self._selected_type,
            "motion": "moving" if self.radio_moving.isChecked() else "static",
            "start": [self.spin_start_x.value(), self.spin_start_y.value()],
            "end": [self.spin_end_x.value(), self.spin_end_y.value()],
            "spread": self.dbl_spread.value(),
            "orientation_deg": self.dbl_orientation.value(),
            "amplitude": self.dbl_amplitude.value(),
            "speed_mm_s": self.spin_speed.value(),
            "ramp_up_ms": self.spin_ramp_up.value(),
            "plateau_ms": self.spin_plateau.value(),
            "ramp_down_ms": self.spin_ramp_down.value(),
            "total_ms": self.spin_total_time.value(),
            "dt_ms": self.spin_dt.value(),
            "texture": {
                "subtype": self._texture_subtype,
                "wavelength": self.spin_wavelength.value(),
                "orientation_deg": self.spin_texture_orientation.value(),
                "sigma": self.spin_texture_sigma.value(),
                "phase": self.spin_phase.value(),
                "edge_orientation_deg": self.spin_edge_orientation.value(),
                "spacing": self.spin_spacing.value(),
                "edge_count": self.spin_edge_count.value(),
                "edge_width": self.spin_edge_width.value(),
                "noise_scale": self.spin_noise_scale.value(),
                "noise_kernel_size": self.spin_noise_kernel.value(),
            },
            "moving": {
                "subtype": self._moving_subtype,
                "linear": {
                    "start": [self.spin_linear_start_x.value(), self.spin_linear_start_y.value()],
                    "end": [self.spin_linear_end_x.value(), self.spin_linear_end_y.value()],
                    "num_steps": self.spin_num_steps.value(),
                    "sigma": self.spin_moving_sigma.value(),
                },
                "circular": {
                    "center": [self.spin_circular_center_x.value(), self.spin_circular_center_y.value()],
                    "radius": self.spin_radius.value(),
                    "num_steps": self.spin_circular_num_steps.value(),
                    "start_angle": self.spin_start_angle.value(),
                    "end_angle": self.spin_end_angle.value(),
                    "sigma": self.spin_circular_sigma.value(),
                },
                "slide": {
                    "start": [self.spin_slide_start_x.value(), self.spin_slide_start_y.value()],
                    "end": [self.spin_slide_end_x.value(), self.spin_slide_end_y.value()],
                    "num_steps": self.spin_slide_num_steps.value(),
                    "sigma": self.spin_slide_sigma.value(),
                },
            },
        }
        return cfg

    def set_config(self, config: dict) -> None:
        """Restore stimulus tab state from a config dict.

        Args:
            config: Dictionary matching the structure returned by
                ``get_config()``. Missing keys fall back to current widget
                defaults.
        """
        # --- Metadata ---
        name = config.get("name", "Stimulus")
        self.txt_stimulus_name.setText(name)

        # --- Stimulus type ---
        stim_type = config.get("type", "gaussian")
        if stim_type in self.type_buttons:
            self.type_buttons[stim_type].setChecked(True)
        self._selected_type = stim_type

        # Visibility of type-specific groups
        self.texture_group.setVisible(stim_type == "texture")
        self.moving_group.setVisible(stim_type == "moving")

        # --- Motion ---
        motion = config.get("motion", "static")
        if motion == "moving":
            self.radio_moving.setChecked(True)
        else:
            self.radio_static.setChecked(True)

        # --- Spatial parameters ---
        start = config.get("start", [0.0, 0.0])
        end = config.get("end", [0.0, 0.0])
        _sb = self._set_spin  # helper shortcut
        _sb(self.spin_start_x, start[0])
        _sb(self.spin_start_y, start[1])
        _sb(self.spin_end_x, end[0])
        _sb(self.spin_end_y, end[1])
        _sb(self.dbl_spread, config.get("spread", 0.3))
        _sb(self.dbl_orientation, config.get("orientation_deg", 0.0))
        _sb(self.dbl_amplitude, config.get("amplitude", 1.0))
        _sb(self.spin_speed, config.get("speed_mm_s", 0.0))

        # --- Temporal parameters ---
        _sb(self.spin_ramp_up, config.get("ramp_up_ms", 50.0))
        _sb(self.spin_plateau, config.get("plateau_ms", 200.0))
        _sb(self.spin_ramp_down, config.get("ramp_down_ms", 50.0))
        _sb(self.spin_total_time, config.get("total_ms", 300.0))
        _sb(self.spin_dt, config.get("dt_ms", 1.0))

        # --- Texture parameters ---
        tex = config.get("texture", {})
        tex_sub = tex.get("subtype", "gabor")
        sub_map = {"gabor": 0, "edge_grating": 1, "noise": 2}
        idx = sub_map.get(tex_sub, 0)
        self.texture_subtype_combo.blockSignals(True)
        self.texture_subtype_combo.setCurrentIndex(idx)
        self.texture_subtype_combo.blockSignals(False)
        self._texture_subtype = tex_sub
        self.gabor_params_widget.setVisible(idx == 0)
        self.edge_grating_params_widget.setVisible(idx == 1)
        self.noise_params_widget.setVisible(idx == 2)

        _sb(self.spin_wavelength, tex.get("wavelength", 0.5))
        _sb(self.spin_texture_orientation, tex.get("orientation_deg", 0.0))
        _sb(self.spin_texture_sigma, tex.get("sigma", 0.3))
        _sb(self.spin_phase, tex.get("phase", 0.0))
        _sb(self.spin_edge_orientation, tex.get("edge_orientation_deg", 0.0))
        _sb(self.spin_spacing, tex.get("spacing", 0.6))
        _sb(self.spin_edge_count, tex.get("edge_count", 5))
        _sb(self.spin_edge_width, tex.get("edge_width", 0.05))
        _sb(self.spin_noise_scale, tex.get("noise_scale", 1.0))
        _sb(self.spin_noise_kernel, tex.get("noise_kernel_size", 5))

        # --- Moving parameters ---
        mov = config.get("moving", {})
        mov_sub = mov.get("subtype", "linear")
        mov_map = {"linear": 0, "circular": 1, "slide": 2}
        midx = mov_map.get(mov_sub, 0)
        self.moving_subtype_combo.blockSignals(True)
        self.moving_subtype_combo.setCurrentIndex(midx)
        self.moving_subtype_combo.blockSignals(False)
        self._moving_subtype = mov_sub
        self.linear_params_widget.setVisible(midx == 0)
        self.circular_params_widget.setVisible(midx == 1)
        self.slide_params_widget.setVisible(midx == 2)

        lin = mov.get("linear", {})
        lin_s = lin.get("start", [0.0, 0.0])
        lin_e = lin.get("end", [2.0, 0.0])
        _sb(self.spin_linear_start_x, lin_s[0])
        _sb(self.spin_linear_start_y, lin_s[1])
        _sb(self.spin_linear_end_x, lin_e[0])
        _sb(self.spin_linear_end_y, lin_e[1])
        _sb(self.spin_num_steps, lin.get("num_steps", 100))
        _sb(self.spin_moving_sigma, lin.get("sigma", 0.3))

        circ = mov.get("circular", {})
        circ_c = circ.get("center", [0.0, 0.0])
        _sb(self.spin_circular_center_x, circ_c[0])
        _sb(self.spin_circular_center_y, circ_c[1])
        _sb(self.spin_radius, circ.get("radius", 1.0))
        _sb(self.spin_circular_num_steps, circ.get("num_steps", 100))
        _sb(self.spin_start_angle, circ.get("start_angle", 0.0))
        _sb(self.spin_end_angle, circ.get("end_angle", 6.28))
        _sb(self.spin_circular_sigma, circ.get("sigma", 0.3))

        sl = mov.get("slide", {})
        sl_s = sl.get("start", [0.0, 0.0])
        sl_e = sl.get("end", [2.0, 0.0])
        _sb(self.spin_slide_start_x, sl_s[0])
        _sb(self.spin_slide_start_y, sl_s[1])
        _sb(self.spin_slide_end_x, sl_e[0])
        _sb(self.spin_slide_end_y, sl_e[1])
        _sb(self.spin_slide_num_steps, sl.get("num_steps", 100))
        _sb(self.spin_slide_sigma, sl.get("sigma", 0.3))

    @staticmethod
    def _set_spin(widget, value) -> None:
        """Set a spinbox value with signals blocked."""
        widget.blockSignals(True)
        widget.setValue(float(value))
        widget.blockSignals(False)


__all__ = ["StimulusDesignerTab"]
