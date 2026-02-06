import inspect
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg  # type: ignore

# Ensure repository root on sys.path for package imports when run as a script
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from GUIs.tabs.stimulus_tab import (  # noqa: E402
    STIMULUS_SCHEMA_VERSION,
    StimulusConfig,
)
from encoding.stimulus_torch import (  # noqa: E402
    StimulusGenerator,
    edge_stimulus_torch,
    gaussian_pressure_torch,
    point_pressure_torch,
)
from encoding.filters_torch import SAFilterTorch, RAFilterTorch  # noqa: E402
from neurons.izhikevich import IzhikevichNeuronTorch  # noqa: E402
from neurons.adex import AdExNeuronTorch  # noqa: E402
from neurons.mqif import MQIFNeuronTorch  # noqa: E402
from neurons.fa import FANeuronTorch  # noqa: E402
from neurons.sa import SANeuronTorch  # noqa: E402
from GUIs.filter_utils import normalize_filter_method  # noqa: E402


MODULE_SCHEMA_VERSION = "1.0.0"
MIN_TIME_STEP_MS = 0.1
DEFAULT_PARAMS_PATH = os.path.join(HERE, "..", "default_params.json")

# Will update DEFAULT_DT_MS after loading defaults
DEFAULT_DT_MS = 1.0


@dataclass
class PopulationConfig:
    """Simulation configuration for a single neuron population."""

    name: str
    neuron_type: str
    model: str = "Izhikevich"
    filter_method: str = "none"
    enabled: bool = True
    input_gain: float = 1.0
    noise_std: float = 0.0
    model_params: Dict[str, object] = field(default_factory=dict)
    filter_params: Dict[str, object] = field(default_factory=dict)
    selected_neuron: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "neuron_type": self.neuron_type,
            "model": self.model,
            "filter_method": self.filter_method,
            "enabled": self.enabled,
            "input_gain": float(self.input_gain),
            "noise_std": float(self.noise_std),
            "model_params": self.model_params,
            "filter_params": self.filter_params,
            "selected_neuron": int(self.selected_neuron),
        }

    @staticmethod
    def from_dict(payload: Dict[str, object]) -> "PopulationConfig":
        neuron_type = str(payload.get("neuron_type", "SA"))
        filter_method = normalize_filter_method(
            payload.get("filter_method", "none"),
            neuron_type,
        )
        return PopulationConfig(
            name=str(payload.get("name", "Population")),
            neuron_type=neuron_type,
            model=str(payload.get("model", "Izhikevich")),
            filter_method=filter_method,
            enabled=bool(payload.get("enabled", True)),
            input_gain=float(payload.get("input_gain", 1.0)),
            noise_std=float(payload.get("noise_std", 0.0)),
            model_params=dict(payload.get("model_params", {})),
            filter_params=dict(payload.get("filter_params", {})),
            selected_neuron=int(payload.get("selected_neuron", 0)),
        )


@dataclass
class SimulationResult:
    """Holds simulation outputs for a neuron population."""

    population_name: str
    dt_ms: float
    time_ms: np.ndarray
    v_trace: np.ndarray
    spikes: np.ndarray
    drive: np.ndarray

    @property
    def neuron_count(self) -> int:
        return self.v_trace.shape[1] if self.v_trace.ndim == 2 else 0


class CollapsibleSection(QtWidgets.QWidget):
    """Simple collapsible container with a header toggle."""

    def __init__(
        self,
        title: str,
        parent: Optional[QtWidgets.QWidget] = None,
        collapsed: bool = True,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._toggle = QtWidgets.QToolButton()
        self._toggle.setStyleSheet("QToolButton { border: none; }")
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            QtCore.Qt.RightArrow if collapsed else QtCore.Qt.DownArrow
        )
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.toggled.connect(self._on_toggled)

        self._content = QtWidgets.QWidget()
        self._content.setVisible(not collapsed)
        self._content.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._toggle)
        layout.addWidget(self._content)

    def setContentLayout(self, content_layout: QtWidgets.QLayout) -> None:
        # Ensure the previous layout is cleaned up
        old_layout = self._content.layout()
        if old_layout is not None:
            QtWidgets.QWidget().setLayout(old_layout)  # type: ignore[arg-type]
        self._content.setLayout(content_layout)

    def contentWidget(self) -> QtWidgets.QWidget:
        return self._content

    def set_collapsed(self, collapsed: bool) -> None:
        self._toggle.blockSignals(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.blockSignals(False)
        self._on_toggled(not collapsed)

    def setTitle(self, title: str) -> None:
        self._title = title
        self._toggle.setText(title)

    def setContentVisible(self, visible: bool) -> None:
        self._content.setVisible(visible)
        self._toggle.blockSignals(True)
        self._toggle.setChecked(visible)
        self._toggle.blockSignals(False)
        self._toggle.setArrowType(
            QtCore.Qt.DownArrow if visible else QtCore.Qt.RightArrow
        )

    def _on_toggled(self, expanded: bool) -> None:
        self._toggle.setArrowType(
            QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow
        )
        self._content.setVisible(expanded)


class SpikingNeuronTab(QtWidgets.QWidget):
    """Configure and simulate spiking neurons driven by saved stimuli."""

    def __init__(
        self,
        mechanoreceptor_tab,
        stimulus_tab,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True, background="w", foreground="k")

        self.default_params = self._load_default_params()
        sim_defaults = self.default_params.get("simulation", {})
        self._default_dt_ms = float(sim_defaults.get("dt", DEFAULT_DT_MS))
        self._default_device = str(sim_defaults.get("device", "cpu"))

        self.mechanoreceptor_tab = mechanoreceptor_tab
        self.stimulus_tab = stimulus_tab
        self.grid_manager = (
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
        self._stimulus_dir: Optional[Path] = None
        self._module_dir: Optional[Path] = None
        self._current_stimulus_path: Optional[Path] = None
        self._current_module_path: Optional[Path] = None

        self.population_configs: Dict[str, PopulationConfig] = {}
        self.sim_results: Dict[str, SimulationResult] = {}

        self._stimulus_frames: Optional[torch.Tensor] = None
        self._stimulus_times: np.ndarray = np.zeros(0, dtype=float)
        self._stimulus_amplitude: np.ndarray = np.zeros(0, dtype=float)
        self._stimulus_dt_ms: float = self._default_dt_ms

        self.model_param_fields: Dict[str, QtWidgets.QWidget] = {}
        self.model_param_kinds: Dict[str, str] = {}
        self.filter_param_fields: Dict[str, QtWidgets.QWidget] = {}
        self.filter_param_kinds: Dict[str, str] = {}
        self._active_neuron_centers: Optional[np.ndarray] = None

        self._setup_ui()
        self._connect_signals()
        self._populate_device_options()
        populations_attr = getattr(
            mechanoreceptor_tab,
            "populations",
            [],
        )
        self._on_populations_changed(populations_attr)
        self._on_grid_changed(self.grid_manager)
        self._on_config_dir_changed(self._current_project_dir)

    # ------------------------------------------------------------------
    # Defaults handling
    # ------------------------------------------------------------------
    def _load_default_params(self) -> Dict[str, object]:
        path = Path(DEFAULT_PARAMS_PATH).resolve()
        try:
            with path.open("r", encoding="utf-8") as fp:
                return json.load(fp)
        except (OSError, json.JSONDecodeError):
            return {
                "simulation": {"dt": DEFAULT_DT_MS, "device": "cpu"},
                "models": {},
                "filters": {},
                "stimulus": {},
            }

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

        self._build_stimulus_section()
        self._build_population_section()
        self._build_simulation_section()
        self.control_layout.addStretch(1)

        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self._build_stimulus_view(right_layout)
        self._build_membrane_plot(right_layout)
        self._build_raster_plot(right_layout)

        layout.addWidget(right_container, stretch=3)

    def _build_stimulus_section(self) -> None:
        group = QtWidgets.QGroupBox("Stimulus Library")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        self.lbl_stimulus_status = QtWidgets.QLabel("No project loaded.")
        self.lbl_stimulus_status.setWordWrap(True)
        layout.addWidget(self.lbl_stimulus_status)

        self.stimulus_list = QtWidgets.QListWidget()
        self.stimulus_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        layout.addWidget(self.stimulus_list, stretch=1)

        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(6)
        self.btn_refresh_stimuli = QtWidgets.QPushButton("Refresh")
        self.btn_open_stimuli = QtWidgets.QPushButton("Open Folder")
        button_row.addWidget(self.btn_refresh_stimuli)
        button_row.addWidget(self.btn_open_stimuli)
        layout.addLayout(button_row)

        self.control_layout.addWidget(group)

    def _build_population_section(self) -> None:
        group = QtWidgets.QGroupBox("Population Configuration")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        self.cmb_population = QtWidgets.QComboBox()
        form.addRow("Population:", self.cmb_population)

        self.chk_population_enabled = QtWidgets.QCheckBox("Enable population")
        self.chk_population_enabled.setChecked(True)
        form.addRow("Status:", self.chk_population_enabled)

        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["Izhikevich", "AdEx", "MQIF", "FA", "SA"])
        form.addRow("Model:", self.cmb_model)

        self.cmb_filter = QtWidgets.QComboBox()
        self.cmb_filter.addItem("No Filter", "none")
        self.cmb_filter.addItem("SA Filter", "sa")
        self.cmb_filter.addItem("RA Filter", "ra")
        form.addRow("Filter:", self.cmb_filter)

        self.dbl_input_gain = QtWidgets.QDoubleSpinBox()
        self.dbl_input_gain.setDecimals(4)
        self.dbl_input_gain.setRange(0.0, 1000.0)
        self.dbl_input_gain.setSingleStep(0.1)
        self.dbl_input_gain.setValue(1.0)
        form.addRow("Input gain:", self.dbl_input_gain)

        self.dbl_noise_std = QtWidgets.QDoubleSpinBox()
        self.dbl_noise_std.setDecimals(5)
        self.dbl_noise_std.setRange(0.0, 100.0)
        self.dbl_noise_std.setSingleStep(0.01)
        self.dbl_noise_std.setValue(0.0)
        form.addRow("Noise std:", self.dbl_noise_std)

        self.spin_neuron_index = QtWidgets.QSpinBox()
        self.spin_neuron_index.setRange(0, 0)
        form.addRow("Neuron index:", self.spin_neuron_index)

        self.btn_apply_population = QtWidgets.QPushButton("Apply Changes")
        form.addRow(self.btn_apply_population)

        self.control_layout.addWidget(group)
        self._build_model_parameter_section()
        self._build_filter_parameter_section()
        self._build_neuron_selector()

    def _build_model_parameter_section(self) -> None:
        self.model_params_section = CollapsibleSection(
            "Model Parameters",
            collapsed=True,
        )
        self.model_param_layout = QtWidgets.QFormLayout()
        self.model_param_layout.setLabelAlignment(QtCore.Qt.AlignRight)
        self.model_params_section.setContentLayout(self.model_param_layout)
        self.model_params_section.setVisible(False)
        self.control_layout.addWidget(self.model_params_section)

    def _build_filter_parameter_section(self) -> None:
        self.filter_params_section = CollapsibleSection(
            "Filter Parameters",
            collapsed=True,
        )
        self.filter_param_layout = QtWidgets.QFormLayout()
        self.filter_param_layout.setLabelAlignment(QtCore.Qt.AlignRight)
        self.filter_params_section.setContentLayout(self.filter_param_layout)
        self.filter_params_section.setVisible(False)
        self.control_layout.addWidget(self.filter_params_section)

    def _build_neuron_selector(self) -> None:
        section = CollapsibleSection("Neuron Map", collapsed=True)
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(6)

        self.neuron_plot = pg.PlotWidget()
        self.neuron_plot.setLabel("bottom", "X", units="mm")
        self.neuron_plot.setLabel("left", "Y", units="mm")
        self.neuron_plot.showGrid(x=True, y=True, alpha=0.1)
        self.neuron_plot.setBackground("w")
        self.neuron_scatter = pg.ScatterPlotItem(size=8)
        self.neuron_plot.addItem(self.neuron_scatter)
        self.neuron_scatter.sigClicked.connect(self._on_neuron_clicked)
        layout.addWidget(self.neuron_plot, stretch=1)

        self.lbl_neuron_info = QtWidgets.QLabel("No neuron selected.")
        layout.addWidget(self.lbl_neuron_info)

        section.setContentLayout(layout)
        self.neuron_plot.getPlotItem().setMenuEnabled(False)
        view_box = self.neuron_plot.getPlotItem().getViewBox()
        if view_box is not None:
            view_box.setMouseEnabled(False, False)
        self.neuron_section = section
        self.control_layout.addWidget(section)

    # ------------------------------------------------------------------
    # Parameter form helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clear_form_layout(layout: QtWidgets.QFormLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
                continue
            child_layout = item.layout()
            if child_layout is not None:
                if isinstance(child_layout, QtWidgets.QFormLayout):
                    SpikingNeuronTab._clear_form_layout(child_layout)
                else:
                    while child_layout.count():
                        sub_item = child_layout.takeAt(0)
                        sub_widget = sub_item.widget()
                        if sub_widget is not None:
                            sub_widget.deleteLater()
                child_layout.deleteLater()

    @staticmethod
    def _param_kind(value: Any) -> str:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return "int"
        if isinstance(value, (float, np.floating)):
            return "float"
        return "str"

    def _create_param_widget(self, value: Any, kind: str) -> QtWidgets.QWidget:
        if kind == "bool":
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(bool(value))
            return checkbox
        if kind == "int":
            spin = QtWidgets.QSpinBox()
            spin.setRange(-1_000_000_000, 1_000_000_000)
            spin.setValue(int(value))
            return spin
        if kind == "float":
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e9, 1e9)
            spin.setSingleStep(0.1)
            spin.setValue(float(value))
            return spin
        line = QtWidgets.QLineEdit()
        line.setText(str(value))
        line.setClearButtonEnabled(True)
        return line

    def _extract_param_value(
        self,
        widget: QtWidgets.QWidget,
        kind: str,
    ) -> Any:
        if kind == "bool" and isinstance(widget, QtWidgets.QCheckBox):
            return bool(widget.isChecked())
        if kind == "int":
            if isinstance(widget, QtWidgets.QSpinBox):
                return int(widget.value())
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                return int(round(widget.value()))
            text = widget.text() if isinstance(widget, QtWidgets.QLineEdit) else ""
            return int(text) if text else 0
        if kind == "float":
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                return float(widget.value())
            if isinstance(widget, QtWidgets.QSpinBox):
                return float(widget.value())
            text = widget.text() if isinstance(widget, QtWidgets.QLineEdit) else "0"
            try:
                return float(text)
            except ValueError:
                return 0.0
        if isinstance(widget, QtWidgets.QLineEdit):
            return widget.text()
        return widget.property("value") or ""

    def _populate_model_parameter_fields(
        self,
        config: PopulationConfig,
    ) -> None:
        model_name = config.model or "Izhikevich"
        model_defaults = self.default_params.get("models", {})
        defaults = dict(model_defaults.get(model_name, {}))
        merged = dict(defaults)
        merged.update(config.model_params or {})
        self._clear_form_layout(self.model_param_layout)
        self.model_param_fields.clear()
        self.model_param_kinds.clear()
        if not merged:
            self.model_params_section.setVisible(False)
            return
        self.model_params_section.setTitle(f"{model_name} Parameters")
        for key, value in merged.items():
            base_value = merged[key]
            default_value = defaults.get(key, base_value)
            kind = self._param_kind(default_value)
            widget = self._create_param_widget(base_value, kind)
            self.model_param_layout.addRow(f"{key}:", widget)
            self.model_param_fields[key] = widget
            self.model_param_kinds[key] = kind
        self.model_params_section.setVisible(True)

    def _populate_filter_parameter_fields(
        self,
        config: PopulationConfig,
    ) -> None:
        method = normalize_filter_method(
            config.filter_method,
            config.neuron_type,
        )
        defaults_map = self.default_params.get("filters", {})
        self._clear_form_layout(self.filter_param_layout)
        self.filter_param_fields.clear()
        self.filter_param_kinds.clear()
        if method == "none":
            self.filter_params_section.setVisible(False)
            return
        filter_key = "SA" if method == "sa" else "RA"
        defaults = dict(defaults_map.get(filter_key, {}))
        merged = dict(defaults)
        merged.update(config.filter_params or {})
        if not merged:
            label = QtWidgets.QLabel("No filter parameters available.")
            label.setWordWrap(True)
            self.filter_param_layout.addRow(label)
            self.filter_params_section.setTitle(f"{filter_key} Filter Parameters")
            self.filter_params_section.setVisible(True)
            return
        title = f"{filter_key} Filter Parameters"
        self.filter_params_section.setTitle(title)
        for key_name, value in merged.items():
            default_value = defaults.get(key_name, value)
            kind = self._param_kind(default_value)
            widget = self._create_param_widget(value, kind)
            self.filter_param_layout.addRow(f"{key_name}:", widget)
            self.filter_param_fields[key_name] = widget
            self.filter_param_kinds[key_name] = kind
        self.filter_params_section.setVisible(True)

    def _collect_model_parameter_overrides(
        self,
        config: PopulationConfig,
    ) -> Dict[str, Any]:
        model_name = config.model or "Izhikevich"
        defaults = dict(self.default_params.get("models", {}).get(model_name, {}))
        collected: Dict[str, Any] = {}
        for key, widget in self.model_param_fields.items():
            kind = self.model_param_kinds.get(key, "str")
            value = self._extract_param_value(widget, kind)
            if isinstance(value, str):
                value = value.strip()
            if key not in defaults or not self._values_equal(value, defaults.get(key)):
                collected[key] = value
        return collected

    def _collect_filter_parameter_overrides(
        self,
        config: PopulationConfig,
    ) -> Dict[str, Any]:
        method = normalize_filter_method(
            config.filter_method,
            config.neuron_type,
        )
        if method == "none":
            return {}
        filter_key = "SA" if method == "sa" else "RA"
        defaults = dict(self.default_params.get("filters", {}).get(filter_key, {}))
        collected: Dict[str, Any] = {}
        for key, widget in self.filter_param_fields.items():
            kind = self.filter_param_kinds.get(key, "str")
            value = self._extract_param_value(widget, kind)
            if isinstance(value, str):
                value = value.strip()
            if key not in defaults or not self._values_equal(value, defaults.get(key)):
                collected[key] = value
        return collected

    @staticmethod
    def _values_equal(a: Any, b: Any) -> bool:
        if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
            try:
                af = float(a)
                bf = float(b)
            except (TypeError, ValueError):
                return False
            return abs(af - bf) <= 1e-9 * max(1.0, abs(af), abs(bf))
        return a == b

    def _gather_model_parameters(
        self,
        config: PopulationConfig,
    ) -> Dict[str, Any]:
        model_name = config.model or "Izhikevich"
        defaults = dict(self.default_params.get("models", {}).get(model_name, {}))
        params = dict(defaults)
        params.update(config.model_params or {})
        return params

    def _gather_filter_parameters(
        self,
        config: PopulationConfig,
    ) -> Dict[str, Any]:
        method = normalize_filter_method(
            config.filter_method,
            config.neuron_type,
        )
        if method == "none":
            return {}
        key = "SA" if method == "sa" else "RA"
        defaults = dict(self.default_params.get("filters", {}).get(key, {}))
        params = dict(defaults)
        params.update(config.filter_params or {})
        return params

    @staticmethod
    def _filter_kwargs(callable_obj, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            signature = inspect.signature(callable_obj.__init__)
        except (ValueError, TypeError):
            return dict(params)
        valid: Dict[str, Any] = {}
        for name in signature.parameters:
            if name == "self":
                continue
            if name in params:
                valid[name] = params[name]
        return valid

    def _update_neuron_selector(self, config: PopulationConfig) -> None:
        population = self._find_population(config.name)
        if population is None:
            self._active_neuron_centers = None
            self.neuron_scatter.setData([], [])
            self.lbl_neuron_info.setText("Population not available.")
            self.spin_neuron_index.blockSignals(True)
            self.spin_neuron_index.setRange(0, 0)
            self.spin_neuron_index.setValue(0)
            self.spin_neuron_index.blockSignals(False)
            return
        if population.module is None and hasattr(population, "instantiate"):
            if self.grid_manager is not None:
                population.instantiate(self.grid_manager)
        if population.module is None:
            self._active_neuron_centers = None
            self.neuron_scatter.setData([], [])
            self.lbl_neuron_info.setText("Innervation module unavailable.")
            self.spin_neuron_index.blockSignals(True)
            self.spin_neuron_index.setRange(0, 0)
            self.spin_neuron_index.setValue(0)
            self.spin_neuron_index.blockSignals(False)
            return
        centers = population.module.neuron_centers.detach().cpu().numpy()
        self._active_neuron_centers = centers
        count = centers.shape[0] if centers.ndim == 2 else 0
        if count == 0:
            self.neuron_scatter.setData([], [])
            self.lbl_neuron_info.setText("No neuron centers available.")
            self.spin_neuron_index.blockSignals(True)
            self.spin_neuron_index.setRange(0, 0)
            self.spin_neuron_index.setValue(0)
            self.spin_neuron_index.blockSignals(False)
            return
        clamped = int(np.clip(config.selected_neuron, 0, count - 1))
        config.selected_neuron = clamped
        self.spin_neuron_index.blockSignals(True)
        self.spin_neuron_index.setRange(0, max(count - 1, 0))
        self.spin_neuron_index.setValue(clamped)
        self.spin_neuron_index.blockSignals(False)
        self._display_neuron_selection(clamped)

    def _display_neuron_selection(self, selected_index: int) -> None:
        centers = self._active_neuron_centers
        if centers is None or centers.size == 0:
            self.neuron_scatter.setData([], [])
            self.lbl_neuron_info.setText("No neuron centers available.")
            return
        count = centers.shape[0]
        if count == 0:
            self.neuron_scatter.setData([], [])
            self.lbl_neuron_info.setText("No neuron centers available.")
            return
        selected_index = int(np.clip(selected_index, 0, count - 1))
        x_coords = centers[:, 0]
        y_coords = centers[:, 1]
        brushes = []
        sizes = []
        for idx in range(count):
            if idx == selected_index:
                brushes.append(pg.mkBrush("#d62728"))
                sizes.append(12.0)
            else:
                brushes.append(pg.mkBrush("#1f77b4"))
                sizes.append(8.0)
        pen = pg.mkPen("#1f77b4")
        self.neuron_scatter.setData(
            x=x_coords,
            y=y_coords,
            data=list(range(count)),
            brush=brushes,
            size=sizes,
            pen=pen,
            symbol="o",
        )
        plot_item = self.neuron_plot.getPlotItem()
        if plot_item is not None:
            plot_item.enableAutoRange(x=True, y=True)
        sel_x = float(x_coords[selected_index])
        sel_y = float(y_coords[selected_index])
        self.lbl_neuron_info.setText(
            f"Selected neuron {selected_index}: ({sel_x:.2f}, {sel_y:.2f})"
        )

    def _on_neuron_clicked(self, _plot, points) -> None:
        if not points:
            return
        index = points[0].data()
        if index is None:
            return
        value = int(index)
        self.spin_neuron_index.blockSignals(True)
        self.spin_neuron_index.setValue(value)
        self.spin_neuron_index.blockSignals(False)
        self._on_neuron_index_changed(value)

    def _on_neuron_index_changed(self, value: int) -> None:
        name = self.cmb_population.currentText()
        if not name:
            return
        config = self.population_configs.get(name)
        if config is None:
            return
        config.selected_neuron = int(value)
        self._display_neuron_selection(config.selected_neuron)
        self._update_membrane_plot()

    def _on_model_changed(self, model_name: str) -> None:
        name = self.cmb_population.currentText()
        if not name:
            return
        config = self.population_configs.get(name)
        if config is None:
            config = PopulationConfig(name=name, neuron_type="SA")
            self.population_configs[name] = config
        if config.model != model_name:
            config.model_params = {}
        config.model = model_name
        self._populate_model_parameter_fields(config)
        self._apply_population_controls()

    def _on_filter_changed(self, index: int) -> None:
        if index < 0:
            return
        name = self.cmb_population.currentText()
        if not name:
            return
        config = self.population_configs.get(name)
        if config is None:
            config = PopulationConfig(name=name, neuron_type="SA")
            self.population_configs[name] = config
        selected = self.cmb_filter.itemData(index)
        filter_method = normalize_filter_method(
            selected,
            config.neuron_type,
        )
        if config.filter_method != filter_method:
            config.filter_params = {}
        config.filter_method = filter_method
        self._populate_filter_parameter_fields(config)
        self._apply_population_controls()

    def _build_simulation_section(self) -> None:
        group = QtWidgets.QGroupBox("Simulation & Persistence")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        device_row = QtWidgets.QHBoxLayout()
        device_row.addWidget(QtWidgets.QLabel("Device:"))
        self.cmb_device = QtWidgets.QComboBox()
        device_row.addWidget(self.cmb_device, stretch=1)
        layout.addLayout(device_row)

        self.btn_run_simulation = QtWidgets.QPushButton("Run Simulation")
        layout.addWidget(self.btn_run_simulation)

        save_row = QtWidgets.QHBoxLayout()
        save_row.setSpacing(6)
        self.btn_save_module = QtWidgets.QPushButton("Save (Update)")
        self.btn_save_as_module = QtWidgets.QPushButton("Save As...")
        self.btn_load_module = QtWidgets.QPushButton("Load...")
        save_row.addWidget(self.btn_save_module)
        save_row.addWidget(self.btn_save_as_module)
        save_row.addWidget(self.btn_load_module)
        layout.addLayout(save_row)

        self.lbl_module_status = QtWidgets.QLabel("No module loaded.")
        self.lbl_module_status.setWordWrap(True)
        layout.addWidget(self.lbl_module_status)

        self.control_layout.addWidget(group)

    def _build_stimulus_view(
        self,
        parent_layout: QtWidgets.QVBoxLayout,
    ) -> None:
        container = QtWidgets.QGroupBox("Stimulus Overview")
        container.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Maximum,
        )
        layout = QtWidgets.QVBoxLayout(container)
        layout.setSpacing(6)

        self.amplitude_plot = pg.PlotWidget()
        self.amplitude_plot.setBackground("w")
        self.amplitude_plot.setLabel("bottom", "Time", units="ms")
        self.amplitude_plot.setLabel("left", "Amplitude")
        self.amplitude_plot.showGrid(x=True, y=True, alpha=0.2)
        self.amplitude_plot.setMinimumHeight(100)
        self.amplitude_plot.setMaximumHeight(180)
        self.amplitude_curve = self.amplitude_plot.plot(
            [], [], pen=pg.mkPen("#2ca02c", width=2.0)
        )
        layout.addWidget(self.amplitude_plot, stretch=1)

        self.lbl_stimulus_summary = QtWidgets.QLabel(
            "Select a stimulus to view amplitude profile."
        )
        self.lbl_stimulus_summary.setWordWrap(True)
        layout.addWidget(self.lbl_stimulus_summary)

        parent_layout.addWidget(container, stretch=1)

    def _build_membrane_plot(
        self,
        parent_layout: QtWidgets.QVBoxLayout,
    ) -> None:
        container = QtWidgets.QGroupBox("Membrane Potential")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setSpacing(4)

        self.membrane_plot = pg.PlotWidget()
        self.membrane_plot.setLabel("bottom", "Time", units="ms")
        self.membrane_plot.setLabel("left", "V", units="mV")
        self.membrane_plot.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.membrane_plot, stretch=1)

        parent_layout.addWidget(container, stretch=3)

    def _build_raster_plot(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        container = QtWidgets.QGroupBox("Spike Raster")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setSpacing(4)

        self.raster_plot = pg.PlotWidget()
        self.raster_plot.setLabel("bottom", "Time", units="ms")
        self.raster_plot.setLabel("left", "Neuron")
        self.raster_plot.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.raster_plot, stretch=1)
        parent_layout.addWidget(container, stretch=3)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        if hasattr(self.mechanoreceptor_tab, "grid_changed"):
            self.mechanoreceptor_tab.grid_changed.connect(self._on_grid_changed)
        if hasattr(
            self.mechanoreceptor_tab,
            "configuration_directory_changed",
        ):
            config_signal = self.mechanoreceptor_tab.configuration_directory_changed
            config_signal.connect(self._on_config_dir_changed)
        if hasattr(self.mechanoreceptor_tab, "populations_changed"):
            self.mechanoreceptor_tab.populations_changed.connect(
                self._on_populations_changed
            )

        self.btn_refresh_stimuli.clicked.connect(self._refresh_stimulus_library)
        self.btn_open_stimuli.clicked.connect(self._open_stimulus_folder)
        self.stimulus_list.itemSelectionChanged.connect(
            self._on_stimulus_selection_changed
        )

        self.cmb_population.currentTextChanged.connect(
            self._on_population_selection_changed
        )
        self.chk_population_enabled.stateChanged.connect(
            lambda _: self._apply_population_controls()
        )
        self.cmb_model.currentTextChanged.connect(self._on_model_changed)
        self.cmb_filter.currentIndexChanged.connect(  # type: ignore[arg-type]
            self._on_filter_changed
        )
        self.dbl_input_gain.valueChanged.connect(
            lambda _: self._apply_population_controls()
        )
        self.dbl_noise_std.valueChanged.connect(
            lambda _: self._apply_population_controls()
        )
        self.btn_apply_population.clicked.connect(self._apply_population_controls)

        self.spin_neuron_index.valueChanged.connect(self._on_neuron_index_changed)

        self.btn_run_simulation.clicked.connect(self._run_simulation)
        self.btn_save_module.clicked.connect(self._save_module_update)
        self.btn_save_as_module.clicked.connect(self._save_module_as)
        self.btn_load_module.clicked.connect(self._load_module)

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------
    def _populate_device_options(self) -> None:
        self.cmb_device.clear()
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            devices.append("mps")
        for dev in devices:
            self.cmb_device.addItem(dev)
        self.cmb_device.setCurrentText("cpu")

    def _current_device(self) -> torch.device:
        choice = self.cmb_device.currentText().strip() or "cpu"
        if choice == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if (
            choice == "mps"
            and torch.backends.mps.is_available()  # type: ignore[attr-defined]
        ):
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Mechanoreceptor integration
    # ------------------------------------------------------------------
    def _on_grid_changed(self, grid_manager) -> None:
        self.grid_manager = grid_manager
        if grid_manager is None:
            self.generator = None
            self._clear_stimulus_preview()
            return
        self.generator = StimulusGenerator(grid_manager)
        if self._stimulus_frames is not None:
            self._load_stimulus_preview(self._current_stimulus_path)

    def _on_config_dir_changed(self, directory: Optional[Path]) -> None:
        self._current_project_dir = directory
        if directory is None:
            self._stimulus_dir = None
            self._module_dir = None
            self.lbl_stimulus_status.setText(
                "No project loaded. Save a configuration first."
            )
            self.lbl_module_status.setText("No module loaded.")
            self.stimulus_list.clear()
            self._current_stimulus_path = None
            self._current_module_path = None
            self.btn_save_module.setEnabled(False)
            self.btn_save_as_module.setEnabled(False)
            self.btn_load_module.setEnabled(False)
            return

        self._stimulus_dir = directory / "stimuli"
        self._stimulus_dir.mkdir(parents=True, exist_ok=True)
        self._module_dir = directory / "neuron_modules"
        self._module_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_stimulus_status.setText(f"Stimuli folder: {self._stimulus_dir}")
        self.lbl_module_status.setText("Ready to save neuron modules.")
        self.btn_save_module.setEnabled(True)
        self.btn_save_as_module.setEnabled(True)
        self.btn_load_module.setEnabled(True)
        self._refresh_stimulus_library()

    def _on_populations_changed(self, populations: Sequence) -> None:
        population_map: Dict[str, Tuple[str, object]] = {}
        if populations:
            for population in populations:
                if population is None:
                    continue
                name = getattr(population, "name", "Population")
                neuron_type = getattr(population, "neuron_type", "SA")
                population_map[name] = (neuron_type, population)
                if name not in self.population_configs:
                    self.population_configs[name] = PopulationConfig(
                        name=name,
                        neuron_type=neuron_type,
                    )
                else:
                    self.population_configs[name].neuron_type = neuron_type
        # Remove configs for populations that no longer exist
        obsolete = set(self.population_configs.keys()) - set(population_map.keys())
        for key in obsolete:
            self.population_configs.pop(key, None)
            self.sim_results.pop(key, None)
        # Update combo box
        current_text = self.cmb_population.currentText()
        self.cmb_population.blockSignals(True)
        self.cmb_population.clear()
        for name in sorted(population_map.keys()):
            self.cmb_population.addItem(name)
        self.cmb_population.blockSignals(False)
        if current_text in population_map:
            index = self.cmb_population.findText(current_text)
            if index >= 0:
                self.cmb_population.setCurrentIndex(index)
        elif self.cmb_population.count() > 0:
            self.cmb_population.setCurrentIndex(0)
        current = self.cmb_population.currentText()
        self._on_population_selection_changed(current)

    # ------------------------------------------------------------------
    # Stimulus library operations
    # ------------------------------------------------------------------
    def _refresh_stimulus_library(self) -> None:
        self.stimulus_list.blockSignals(True)
        self.stimulus_list.clear()
        if self._stimulus_dir is None:
            self.stimulus_list.blockSignals(False)
            return
        entries: List[Tuple[str, Path]] = []
        for json_path in sorted(self._stimulus_dir.glob("*.json")):
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
        if self._current_stimulus_path is not None:
            self._select_stimulus_item(self._current_stimulus_path)

    def _select_stimulus_item(self, path: Path) -> None:
        for index in range(self.stimulus_list.count()):
            item = self.stimulus_list.item(index)
            stored = item.data(QtCore.Qt.UserRole)
            if stored and Path(stored) == path:
                self.stimulus_list.setCurrentItem(item)
                break

    def _open_stimulus_folder(self) -> None:
        if self._stimulus_dir is None:
            return
        url = QtCore.QUrl.fromLocalFile(str(self._stimulus_dir))
        QtGui.QDesktopServices.openUrl(url)

    def _on_stimulus_selection_changed(self) -> None:
        item = self.stimulus_list.currentItem()
        if item is None:
            self._current_stimulus_path = None
            self._clear_stimulus_preview()
            return
        path_str = item.data(QtCore.Qt.UserRole)
        self._current_stimulus_path = Path(path_str) if path_str else None
        self._load_stimulus_preview(self._current_stimulus_path)

    def _clear_stimulus_preview(self) -> None:
        self._stimulus_frames = None
        self._stimulus_times = np.zeros(0, dtype=float)
        self._stimulus_amplitude = np.zeros(0, dtype=float)
        self._stimulus_dt_ms = self._default_dt_ms
        self.amplitude_curve.setData([], [])
        amp_item = self.amplitude_plot.getPlotItem()
        if amp_item is not None:
            amp_item.enableAutoRange(x=True, y=True)
        self.lbl_stimulus_summary.setText("No stimulus selected.")

    def _load_stimulus_preview(self, path: Optional[Path]) -> None:
        if path is None or self.generator is None:
            self._clear_stimulus_preview()
            return
        try:
            with path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Stimulus load failed",
                f"Unable to load stimulus:\n{exc}",
            )
            self._clear_stimulus_preview()
            return
        if payload.get("schema_version") != STIMULUS_SCHEMA_VERSION:
            QtWidgets.QMessageBox.warning(
                self,
                "Schema mismatch",
                "Stimulus schema version differs; attempting to load anyway.",
            )
        config = self._config_from_payload(payload, path)
        frames, time_axis, amplitude_profile = self._build_stimulus_frames(config)
        if frames is None or time_axis is None:
            self._clear_stimulus_preview()
            QtWidgets.QMessageBox.warning(
                self,
                "Stimulus incompatibility",
                "Stimulus could not be generated for the current grid.",
            )
            return
        self._stimulus_frames = frames
        self._stimulus_dt_ms = max(float(config.dt_ms), MIN_TIME_STEP_MS)
        time_np = time_axis.detach().cpu().numpy().reshape(-1)
        self._stimulus_times = time_np
        if amplitude_profile is not None:
            amplitude_np = amplitude_profile.detach().cpu().numpy().reshape(-1)
        else:
            amplitude_np = np.zeros_like(time_np)
        self._stimulus_amplitude = amplitude_np
        self.amplitude_curve.setData(time_np, amplitude_np)
        amp_item = self.amplitude_plot.getPlotItem()
        if amp_item is not None:
            amp_item.enableAutoRange(x=True, y=True)

        duration_ms = float(time_np[-1]) if time_np.size else float(config.total_ms)
        peak_amp = float(amplitude_np.max()) if amplitude_np.size else 0.0
        summary = (
            f"{config.stimulus_type} / {config.motion} stimulus — "
            f"duration {duration_ms:.1f} ms, "
            f"dt {self._stimulus_dt_ms:.2f} ms, "
            f"peak amplitude {peak_amp:.3f}"
        )
        self.lbl_stimulus_summary.setText(summary)

    def _config_from_payload(
        self,
        payload: Dict[str, object],
        path: Path,
    ) -> StimulusConfig:
        def _tuple(values, default):
            if not isinstance(values, (list, tuple)):
                return default
            if len(values) != 2:
                return default
            return float(values[0]), float(values[1])

        start = payload.get("start", [0.0, 0.0])
        end = payload.get("end", start)
        config = StimulusConfig(
            name=payload.get("name", path.stem),
            stimulus_type=payload.get("type", "gaussian"),
            motion=payload.get("motion", "static"),
            start=_tuple(start, (0.0, 0.0)),
            end=_tuple(end, (0.0, 0.0)),
            spread=float(payload.get("spread", 0.3)),
            orientation_deg=float(payload.get("orientation_deg", 0.0)),
            amplitude=float(payload.get("amplitude", 1.0)),
            ramp_up_ms=float(payload.get("ramp_up_ms", 50.0)),
            plateau_ms=float(payload.get("plateau_ms", 200.0)),
            ramp_down_ms=float(payload.get("ramp_down_ms", 50.0)),
            total_ms=float(payload.get("total_ms", 300.0)),
            dt_ms=float(payload.get("dt_ms", 1.0)),
            speed_mm_s=float(payload.get("speed_mm_s", 0.0)),
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
        peak = max(config.amplitude, 0.0)
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
    # Population configuration management
    # ------------------------------------------------------------------
    def _on_population_selection_changed(self, name: str) -> None:
        if not name:
            return
        config = self.population_configs.get(name)
        if config is None:
            config = PopulationConfig(name=name, neuron_type="SA")
            self.population_configs[name] = config
        self._load_population_into_controls(config)
        self._update_membrane_plot()
        self._update_raster_plot()

    def _load_population_into_controls(self, config: PopulationConfig) -> None:
        self.chk_population_enabled.blockSignals(True)
        self.chk_population_enabled.setChecked(config.enabled)
        self.chk_population_enabled.blockSignals(False)
        index = self.cmb_model.findText(config.model)
        self.cmb_model.blockSignals(True)
        if index >= 0:
            self.cmb_model.setCurrentIndex(index)
        else:
            self.cmb_model.setCurrentText(config.model)
        self.cmb_model.blockSignals(False)
        method = normalize_filter_method(
            config.filter_method,
            config.neuron_type,
        )
        config.filter_method = method
        index = self.cmb_filter.findData(method)
        self.cmb_filter.blockSignals(True)
        if index >= 0:
            self.cmb_filter.setCurrentIndex(index)
        else:
            self.cmb_filter.setCurrentIndex(0)
        self.cmb_filter.blockSignals(False)
        self.dbl_input_gain.blockSignals(True)
        self.dbl_input_gain.setValue(config.input_gain)
        self.dbl_input_gain.blockSignals(False)
        self.dbl_noise_std.blockSignals(True)
        self.dbl_noise_std.setValue(config.noise_std)
        self.dbl_noise_std.blockSignals(False)
        self._populate_model_parameter_fields(config)
        self._populate_filter_parameter_fields(config)
        self._update_neuron_selector(config)

    def _apply_population_controls(self) -> None:
        name = self.cmb_population.currentText()
        if not name:
            return
        config = self.population_configs.get(name)
        if config is None:
            config = PopulationConfig(name=name, neuron_type="SA")
            self.population_configs[name] = config
        config.enabled = self.chk_population_enabled.isChecked()
        config.model = self.cmb_model.currentText() or config.model
        filter_choice = self.cmb_filter.currentData()
        config.filter_method = normalize_filter_method(
            filter_choice,
            config.neuron_type,
        )
        config.input_gain = self.dbl_input_gain.value()
        config.noise_std = self.dbl_noise_std.value()
        config.model_params = self._collect_model_parameter_overrides(config)
        config.filter_params = self._collect_filter_parameter_overrides(config)
        config.selected_neuron = self.spin_neuron_index.value()
        self._display_neuron_selection(config.selected_neuron)
        self._update_membrane_plot()
        self._update_raster_plot()

    # ------------------------------------------------------------------
    # Simulation handling
    # ------------------------------------------------------------------
    def _run_simulation(self) -> None:
        if self.generator is None or self.grid_manager is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Grid required",
                (
                    "Generate or load a mechanoreceptor grid before running "
                    "simulations."
                ),
            )
            return
        if self._stimulus_frames is None or self._stimulus_times.size == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Stimulus required",
                ("Select a stimulus from the library before running " "simulations."),
            )
            return
        self._apply_population_controls()
        active_configs = [
            cfg for cfg in self.population_configs.values() if cfg.enabled
        ]
        if not active_configs:
            QtWidgets.QMessageBox.information(
                self,
                "No populations",
                "Enable at least one population for simulation.",
            )
            return
        device = self._current_device()
        frames = self._stimulus_frames.to(device=device)
        results: Dict[str, SimulationResult] = {}
        errors: List[str] = []
        dt_ms = float(self._stimulus_dt_ms)
        torch.set_grad_enabled(False)
        for cfg in active_configs:
            population = self._find_population(cfg.name)
            if population is None:
                errors.append(f"Population '{cfg.name}' unavailable.")
                continue
            try:
                result = self._simulate_population(
                    population, cfg, frames, dt_ms, device
                )
                results[cfg.name] = result
            except RuntimeError as exc:
                errors.append(f"{cfg.name}: {exc}")
        torch.set_grad_enabled(True)
        if results:
            self.sim_results.update(results)
            # Update neuron index range for selected population
            selected = self.cmb_population.currentText()
            selected_result = self.sim_results.get(selected)
            if selected_result is not None:
                max_index = max(selected_result.neuron_count - 1, 0)
                self.spin_neuron_index.blockSignals(True)
                self.spin_neuron_index.setRange(0, max_index)
                selected_config = self.population_configs.get(selected)
                current_idx = min(self.spin_neuron_index.value(), max_index)
                self.spin_neuron_index.setValue(current_idx)
                self.spin_neuron_index.blockSignals(False)
                if selected_config is not None:
                    selected_config.selected_neuron = current_idx
                self._display_neuron_selection(current_idx)
            self._update_membrane_plot()
            self._update_raster_plot()
            if errors:
                self.lbl_module_status.setText(
                    ("Simulation completed with warnings. See dialog " "for details.")
                )
            else:
                self.lbl_module_status.setText("Simulation completed successfully.")
        if errors:
            QtWidgets.QMessageBox.warning(
                self,
                "Simulation issues",
                "\n".join(errors),
            )

    def _simulate_population(
        self,
        population,
        config: PopulationConfig,
        frames: torch.Tensor,
        dt_ms: float,
        device: torch.device,
    ) -> SimulationResult:
        module = getattr(population, "module", None)
        if module is None:
            if hasattr(population, "instantiate"):
                # type: ignore[arg-type]
                population.instantiate(self.grid_manager)
                module = population.module
        if module is None:
            raise RuntimeError("Innervation module unavailable.")
        module = module.to(device)
        if frames.ndim == 3:
            stimuli = frames.unsqueeze(0)
        elif frames.ndim == 4:
            stimuli = frames
        else:
            raise RuntimeError("Unexpected stimulus dimensions.")
        neuron_drive = module(stimuli)
        if neuron_drive.ndim == 2:
            neuron_drive = neuron_drive.unsqueeze(1)
        filtered = self._apply_filter(
            neuron_drive, population.neuron_type, config, dt_ms, device
        )
        drive = filtered * config.input_gain
        if config.noise_std > 0.0:
            drive = drive + torch.randn_like(drive) * config.noise_std
        drive = drive.float()
        neuron_model = self._create_neuron_model(config, dt_ms, device)
        v_trace, spikes = neuron_model(drive)
        drive_np = drive.detach().cpu().numpy()[0]
        steps = drive_np.shape[0]
        v_np = v_trace.detach().cpu().numpy()[0]
        if v_np.shape[0] > steps:
            v_np = v_np[:steps]
        elif v_np.shape[0] < steps:
            steps = v_np.shape[0]
            drive_np = drive_np[:steps]
        spikes_np = spikes.detach().cpu().numpy()[0]
        if spikes_np.shape[0] > steps:
            spikes_np = spikes_np[:steps]
        elif spikes_np.shape[0] < steps:
            steps = spikes_np.shape[0]
            v_np = v_np[:steps]
            drive_np = drive_np[:steps]
        time_ms = np.arange(steps, dtype=float) * dt_ms
        return SimulationResult(
            population_name=config.name,
            dt_ms=dt_ms,
            time_ms=time_ms,
            v_trace=v_np,
            spikes=spikes_np,
            drive=drive_np,
        )

    def _apply_filter(
        self,
        inputs: torch.Tensor,
        neuron_type: str,
        config: PopulationConfig,
        dt_ms: float,
        device: torch.device,
    ) -> torch.Tensor:
        method = normalize_filter_method(
            config.filter_method,
            neuron_type,
        )
        if method == "none":
            return inputs
        filter_params = self._gather_filter_parameters(config)
        if method == "sa":
            sa_kwargs = dict(filter_params)
            sa_kwargs = self._filter_kwargs(SAFilterTorch, sa_kwargs)
            sa_kwargs["dt"] = dt_ms
            filter_module = SAFilterTorch(**sa_kwargs).to(device)
            if inputs.ndim == 3:
                return filter_module(inputs, reset_states=True)
            filtered = filter_module(
                inputs.squeeze(1),
                reset_states=True,
            )
            return filtered.unsqueeze(1)
        if method == "ra":
            ra_kwargs = dict(filter_params)
            ra_kwargs = self._filter_kwargs(RAFilterTorch, ra_kwargs)
            ra_kwargs["dt"] = dt_ms
            filter_module = RAFilterTorch(**ra_kwargs).to(device)
            if inputs.ndim == 3:
                return filter_module(inputs, reset_states=True)
            result = filter_module(
                inputs.squeeze(1),
                reset_states=True,
            )
            return result.unsqueeze(1)
        return inputs

    def _create_neuron_model(
        self,
        config: PopulationConfig,
        dt_ms: float,
        device: torch.device,
    ):
        dt_value = max(dt_ms, 0.05)
        model_name = (config.model or "Izhikevich").lower()
        params = self._gather_model_parameters(config)

        def instantiate(model_cls):
            kwargs = self._filter_kwargs(model_cls, params)
            try:
                signature = inspect.signature(model_cls.__init__)
            except (ValueError, TypeError):
                signature = None
            if signature is None or "dt" in getattr(signature, "parameters", {}):
                kwargs.setdefault("dt", dt_value)
            if signature is None or "noise_std" in getattr(signature, "parameters", {}):
                kwargs.setdefault("noise_std", config.noise_std)
            return model_cls(**kwargs).to(device)

        if model_name == "adex":
            return instantiate(AdExNeuronTorch)
        if model_name == "mqif":
            return instantiate(MQIFNeuronTorch)
        if model_name == "fa":
            return instantiate(FANeuronTorch)
        if model_name == "sa":
            return instantiate(SANeuronTorch)
        return instantiate(IzhikevichNeuronTorch)

    def _find_population(self, name: str):
        populations = getattr(self.mechanoreceptor_tab, "populations", [])
        for population in populations:
            if getattr(population, "name", None) == name:
                return population
        return None

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------
    def _update_membrane_plot(self) -> None:
        self.membrane_plot.clear()
        name = self.cmb_population.currentText()
        if not name:
            return
        result = self.sim_results.get(name)
        if result is None:
            return
        idx = self.spin_neuron_index.value()
        idx = max(0, min(idx, result.neuron_count - 1))
        trace = result.v_trace[:, idx]
        pen = pg.mkPen(QtGui.QColor("#1f77b4"), width=2.0)
        self.membrane_plot.plot(result.time_ms, trace, pen=pen)
        self.membrane_plot.setTitle(f"{name} — Neuron {idx}")

    def _update_raster_plot(self) -> None:
        self.raster_plot.clear()
        name = self.cmb_population.currentText()
        if not name:
            return
        result = self.sim_results.get(name)
        if result is None:
            return
        spikes = result.spikes
        if spikes.ndim != 2:
            return
        spike_indices = np.argwhere(spikes > 0)
        if spike_indices.size == 0:
            return
        times = spike_indices[:, 0].astype(float) * result.dt_ms
        neurons = spike_indices[:, 1].astype(float)
        scatter = pg.ScatterPlotItem(
            x=times,
            y=neurons,
            size=4,
            brush=pg.mkBrush(0, 0, 0),
            pen=None,
            symbol="o",
        )
        self.raster_plot.addItem(scatter)
        self.raster_plot.setTitle(f"{name} — Spike Raster")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _save_module_update(self) -> None:
        if self._current_module_path is None:
            self._save_module_as()
            return
        self._write_module_bundle(self._current_module_path)

    def _save_module_as(self) -> None:
        if self._module_dir is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Project required",
                "Save a mechanoreceptor configuration before storing modules.",
            )
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Module name",
            "Enter a module name:",
        )
        if not ok or not name.strip():
            return
        sanitized = self._sanitize_name(name)
        target_path = self._module_dir / f"{sanitized}.json"
        if target_path.exists():
            reply = QtWidgets.QMessageBox.question(
                self,
                "Overwrite?",
                f"Module '{sanitized}' already exists. Overwrite it?",
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        self._current_module_path = target_path
        self._write_module_bundle(target_path)

    def _write_module_bundle(self, path: Path) -> None:
        bundle = self._collect_module_bundle()
        try:
            with path.open("w", encoding="utf-8") as fp:
                json.dump(bundle, fp, indent=2)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Save failed",
                f"Unable to write module file:\n{exc}",
            )
            return
        self.lbl_module_status.setText(f"Saved module to {path}")

    def _collect_module_bundle(self) -> Dict[str, object]:
        stimulus_rel = None
        if self._current_stimulus_path is not None and self._stimulus_dir is not None:
            try:
                stimulus_rel = str(
                    self._current_stimulus_path.relative_to(self._stimulus_dir)
                )
            except ValueError:
                stimulus_rel = str(self._current_stimulus_path)
        population_entries = [cfg.to_dict() for cfg in self.population_configs.values()]
        return {
            "schema_version": MODULE_SCHEMA_VERSION,
            "kind": "neuron_module",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "stimulus": stimulus_rel,
            "device": self.cmb_device.currentText(),
            "population_configs": population_entries,
        }

    def _load_module(self) -> None:
        if self._module_dir is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Project required",
                "Load a mechanoreceptor configuration before loading modules.",
            )
            return
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Neuron Module",
            str(self._module_dir),
            "Neuron Modules (*.json)",
        )
        if not file_path:
            return
        manifest_path = Path(file_path)
        try:
            with manifest_path.open("r", encoding="utf-8") as fp:
                bundle = json.load(fp)
        except (OSError, json.JSONDecodeError) as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Load failed",
                f"Unable to read module:\n{exc}",
            )
            return
        if bundle.get("schema_version") != MODULE_SCHEMA_VERSION:
            QtWidgets.QMessageBox.warning(
                self,
                "Schema mismatch",
                "Module schema version differs; attempting to load anyway.",
            )
        self._current_module_path = manifest_path
        self._apply_module_bundle(bundle, manifest_path.parent)

    def _apply_module_bundle(
        self,
        bundle: Dict[str, object],
        base_dir: Path,
    ) -> None:
        population_entries = bundle.get("population_configs", [])
        if isinstance(population_entries, list):
            for entry in population_entries:
                if not isinstance(entry, dict):
                    continue
                config = PopulationConfig.from_dict(entry)
                self.population_configs[config.name] = config
        stimulus_entry = bundle.get("stimulus")
        if stimulus_entry and self._stimulus_dir is not None:
            stimulus_path = Path(stimulus_entry)
            if not stimulus_path.is_absolute():
                candidate = self._stimulus_dir / stimulus_path
            else:
                candidate = stimulus_path
            if candidate.exists():
                self._current_stimulus_path = candidate
                self._select_stimulus_item(candidate)
                self._load_stimulus_preview(candidate)
        device = bundle.get("device", "cpu")
        index = self.cmb_device.findText(str(device))
        if index >= 0:
            self.cmb_device.setCurrentIndex(index)
        self.lbl_module_status.setText(
            "Module loaded. Configure populations and run simulation."
        )
        self._on_populations_changed(
            getattr(self.mechanoreceptor_tab, "populations", [])
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_name(name: str) -> str:
        import re

        sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
        return sanitized or "module"

    def current_dt_ms(self) -> float:
        """Return the active stimulus sampling interval in milliseconds."""
        return float(max(self._stimulus_dt_ms, MIN_TIME_STEP_MS))

    def current_device(self) -> torch.device:
        """Return the selected compute device for simulations."""
        return self._current_device()


__all__ = ["SpikingNeuronTab", "PopulationConfig", "SimulationResult"]
