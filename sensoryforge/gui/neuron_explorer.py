"""Standalone PyQt5 tool for exploring tactile neuron model responses."""

import json
import math
import os
from typing import Dict, Any, Tuple

import numpy as np
import torch

from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)

# Add repo root to sys.path for imports
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Local imports from the repository
from neurons import (  # noqa: E402
    IzhikevichNeuronTorch,
    AdExNeuronTorch,
    MQIFNeuronTorch,
    FANeuronTorch,
    SANeuronTorch,
)
from encoding.filters_torch import SAFilterTorch, RAFilterTorch  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_PARAMS_PATH = os.path.join(HERE, "default_params.json")


def ensure_default_params_file() -> None:
    """Guarantee the default parameter JSON mirrors all supported models."""
    default = {
        "simulation": {"dt": 0.1, "device": "cpu", "seed": 42},
        "models": {
            "Izhikevich": {
                "a": 0.02,
                "b": 0.2,
                "c": -65.0,
                "d": 8.0,
                "threshold": 30.0,
                "noise_std": 0.0,
            },
            "AdEx": {
                "EL": -70.0,
                "VT": -50.0,
                "DeltaT": 2.0,
                "tau_m": 20.0,
                "tau_w": 100.0,
                "a": 2.0,
                "b": 0.0,
                "v_reset": -58.0,
                "v_spike": 20.0,
                "R": 1.0,
                "noise_std": 0.0,
            },
            "MQIF": {
                "a": 0.04,
                "b": 0.2,
                "vr": -60.0,
                "vt": -40.0,
                "v_reset": -60.0,
                "v_peak": 30.0,
                "d": 2.0,
                "tau_m": 10.0,
                "tau_u": 100.0,
                "noise_std": 0.0,
            },
            "FA": {
                "vb": 0.0,
                "A": 1.0,
                "theta": 1.0,
                "tau_ref": 2.0,
                "baseline_mode": "sequence",
                "tau_dc": 50.0,
                "input_gain": 0.1,
                "noise_std": 0.0,
            },
            "SA": {
                "I_tau": 25e-12,
                "I_th": 8.3e-9,
                "I_tau_ahp": 20e-12,
                "I_th_ahp": 16.6e-12,
                "I_tau_refractory": 1.6e-9,
                "C_mem": 100e-15,
                "C_adap": 250e-15,
                "C_refractory": 200e-15,
                "U_T": 26e-3,
                "kappa": 0.7,
                "Ia_frac": 0.8,
                "z_reset": 0.0,
                "I_in_op": 500e-12,
                "current_scale": 1e-12,
                "noise_std": 0.0,
            },
        },
        "filters": {
            "SA": {"tau_r": 5, "tau_d": 30, "k1": 0.05, "k2": 3.0},
            "RA": {"tau_RA": 30, "k3": 2.0},
        },
        "stimulus": {
            "shape": "step",
            "amplitude": 5.0,
            "frequency": 5.0,
            "duration_ms": 1000,
            "pre_ms": 100,
            "post_ms": 100,
            "ramp_ms": 50,
        },
    }
    if not os.path.exists(DEFAULT_PARAMS_PATH):
        with open(DEFAULT_PARAMS_PATH, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)
        return
    # Merge missing keys into existing file
    try:
        with open(DEFAULT_PARAMS_PATH, "r", encoding="utf-8") as f:
            current = json.load(f)
    except Exception:
        current = {}
    # Ensure top-level sections
    for key in ["simulation", "models", "filters", "stimulus"]:
        current.setdefault(key, {})
    # Ensure models
    for mkey, mval in default["models"].items():
        if mkey not in current["models"]:
            current["models"][mkey] = mval
        else:
            # Add missing fields only
            for p, pv in mval.items():
                current["models"][mkey].setdefault(p, pv)
    # Ensure filters
    for fkey, fval in default["filters"].items():
        if fkey not in current["filters"]:
            current["filters"][fkey] = fval
        else:
            for p, pv in fval.items():
                current["filters"][fkey].setdefault(p, pv)
    # Ensure stimulus defaults
    for skey, sval in default["stimulus"].items():
        current["stimulus"].setdefault(skey, sval)
    with open(DEFAULT_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)


class JsonEditorDialog(QtWidgets.QDialog):
    """Lightweight JSON editor dialog for tweaking default neuron configs."""

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Default Parameters JSON")
        self.path = path
        self.resize(800, 600)
        layout = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QPlainTextEdit(self)
        layout.addWidget(self.text)
        btns = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        btns.addStretch(1)
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_cancel.clicked.connect(self.reject)
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.text.setPlainText(f.read())
        except Exception:
            self.text.setPlainText("{}")

    def on_save(self) -> None:
        """Validate the current JSON payload and persist edits to disk."""
        try:
            data = json.loads(self.text.toPlainText())
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Invalid JSON", f"Error parsing JSON: {e}"
            )
            return
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Write Error", f"Error writing file: {e}"
            )
            return
        self.accept()


class MplCanvas(FigureCanvas):
    """Canvas bundling the common plots displayed by the explorer window."""

    def __init__(self, width=7, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # Create four vertically-stacked subplots sharing the x-axis
        gs = fig.add_gridspec(4, 1, hspace=0.35, height_ratios=[1, 1.2, 1.0, 0.8])
        self.ax_input = fig.add_subplot(gs[0])
        self.ax_mem = fig.add_subplot(gs[1], sharex=self.ax_input)
        self.ax_raster = fig.add_subplot(gs[2], sharex=self.ax_input)
        self.ax_psth = fig.add_subplot(gs[3], sharex=self.ax_input)
        # Titles inside the figure
        self.ax_input.set_title("Input Trace")
        self.ax_mem.set_title("Membrane Potential")
        self.ax_raster.set_title("Raster (Population)")
        self.ax_psth.set_title("PSTH (Population Rate)")
        super().__init__(fig)


class NeuronExplorer(QtWidgets.QMainWindow):
    """Interactive window for configuring neurons and visualizing outputs."""

    def __init__(self):
        super().__init__()
        ensure_default_params_file()
        self.params = self._load_params()
        self.setWindowTitle("Neuron Model Explorer")
        self.resize(1200, 800)

        # Central layout
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QGridLayout(central)

        # Controls panel
        controls = QtWidgets.QGroupBox("Controls")
        controls_layout = QtWidgets.QGridLayout(controls)
        row = 0

        # Model selection
        controls_layout.addWidget(QtWidgets.QLabel("Model:"), row, 0)
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["Izhikevich", "AdEx", "MQIF", "FA", "SA"])
        controls_layout.addWidget(self.cmb_model, row, 1)

        # Neuron count
        row += 1
        controls_layout.addWidget(QtWidgets.QLabel("Neurons:"), row, 0)
        self.spin_neurons = QtWidgets.QSpinBox()
        self.spin_neurons.setRange(1, 2048)
        self.spin_neurons.setValue(64)
        controls_layout.addWidget(self.spin_neurons, row, 1)

        # Variability std (applied where supported)
        row += 1
        controls_layout.addWidget(QtWidgets.QLabel("Variability std:"), row, 0)
        self.dbl_variability = QtWidgets.QDoubleSpinBox()
        self.dbl_variability.setRange(0.0, 1000.0)
        self.dbl_variability.setDecimals(4)
        self.dbl_variability.setSingleStep(0.1)
        self.dbl_variability.setValue(0.0)
        controls_layout.addWidget(self.dbl_variability, row, 1)

        # Filter selection (None, SA, FA)
        row += 1
        controls_layout.addWidget(QtWidgets.QLabel("Filter:"), row, 0)
        self.cmb_filter = QtWidgets.QComboBox()
        # Expose FA and SA names in the UI; internally map FA->RAFilterTorch
        self.cmb_filter.addItems(["None", "SA", "FA"])
        controls_layout.addWidget(self.cmb_filter, row, 1)

        # Stimulus controls
        stim_box = QtWidgets.QGroupBox("Stimulus")
        stim_layout = QtWidgets.QGridLayout(stim_box)
        srow = 0
        stim_layout.addWidget(QtWidgets.QLabel("Shape:"), srow, 0)
        self.cmb_shape = QtWidgets.QComboBox()
        self.cmb_shape.addItems(
            [
                "step",
                "sine",
                "ramp",
                "trapezoid",
                "periodic_trapezoid",
                "sawtooth",
            ]
        )
        stim_layout.addWidget(self.cmb_shape, srow, 1)
        srow += 1
        stim_layout.addWidget(QtWidgets.QLabel("Amplitude:"), srow, 0)
        self.dbl_amp = QtWidgets.QDoubleSpinBox()
        self.dbl_amp.setRange(0.0, 1e6)
        self.dbl_amp.setDecimals(4)
        self.dbl_amp.setValue(self.params["stimulus"].get("amplitude", 5.0))
        stim_layout.addWidget(self.dbl_amp, srow, 1)
        srow += 1
        stim_layout.addWidget(QtWidgets.QLabel("Frequency (Hz):"), srow, 0)
        self.dbl_freq = QtWidgets.QDoubleSpinBox()
        self.dbl_freq.setRange(0.0, 1000.0)
        self.dbl_freq.setDecimals(3)
        self.dbl_freq.setValue(self.params["stimulus"].get("frequency", 5.0))
        stim_layout.addWidget(self.dbl_freq, srow, 1)
        srow += 1
        stim_layout.addWidget(QtWidgets.QLabel("Duration (ms):"), srow, 0)
        self.spin_duration = QtWidgets.QSpinBox()
        self.spin_duration.setRange(1, 600000)
        self.spin_duration.setValue(self.params["stimulus"].get("duration_ms", 1000))
        stim_layout.addWidget(self.spin_duration, srow, 1)
        srow += 1
        stim_layout.addWidget(QtWidgets.QLabel("Pre (ms):"), srow, 0)
        self.spin_pre = QtWidgets.QSpinBox()
        self.spin_pre.setRange(0, 600000)
        self.spin_pre.setValue(self.params["stimulus"].get("pre_ms", 100))
        stim_layout.addWidget(self.spin_pre, srow, 1)
        srow += 1
        stim_layout.addWidget(QtWidgets.QLabel("Post (ms):"), srow, 0)
        self.spin_post = QtWidgets.QSpinBox()
        self.spin_post.setRange(0, 600000)
        self.spin_post.setValue(self.params["stimulus"].get("post_ms", 100))
        stim_layout.addWidget(self.spin_post, srow, 1)
        srow += 1
        stim_layout.addWidget(QtWidgets.QLabel("Ramp (ms):"), srow, 0)
        self.spin_ramp = QtWidgets.QSpinBox()
        self.spin_ramp.setRange(0, 600000)
        self.spin_ramp.setValue(self.params["stimulus"].get("ramp_ms", 50))
        stim_layout.addWidget(self.spin_ramp, srow, 1)
        srow += 1
        # Duty (%)
        stim_layout.addWidget(QtWidgets.QLabel("Duty (%):"), srow, 0)
        self.dbl_duty = QtWidgets.QDoubleSpinBox()
        self.dbl_duty.setRange(0.1, 99.9)
        self.dbl_duty.setDecimals(1)
        self.dbl_duty.setSingleStep(1.0)
        self.dbl_duty.setValue(self.params["stimulus"].get("duty_percent", 50.0))
        stim_layout.addWidget(self.dbl_duty, srow, 1)
        srow += 1
        # Rise/Fall (ms) for trapezoids
        stim_layout.addWidget(QtWidgets.QLabel("Rise (ms):"), srow, 0)
        self.spin_rise = QtWidgets.QSpinBox()
        self.spin_rise.setRange(0, 600000)
        self.spin_rise.setValue(self.params["stimulus"].get("rise_ms", 0))
        stim_layout.addWidget(self.spin_rise, srow, 1)
        srow += 1
        stim_layout.addWidget(QtWidgets.QLabel("Fall (ms):"), srow, 0)
        self.spin_fall = QtWidgets.QSpinBox()
        self.spin_fall.setRange(0, 600000)
        self.spin_fall.setValue(self.params["stimulus"].get("fall_ms", 0))
        stim_layout.addWidget(self.spin_fall, srow, 1)
        srow += 1
        # Plateau (ms)
        stim_layout.addWidget(QtWidgets.QLabel("Plateau (ms):"), srow, 0)
        self.spin_plateau = QtWidgets.QSpinBox()
        self.spin_plateau.setRange(0, 600000)
        self.spin_plateau.setValue(self.params["stimulus"].get("plateau_ms", 0))
        stim_layout.addWidget(self.spin_plateau, srow, 1)

        # Analysis controls (PSTH bin, SDF sigma)
        srow += 1
        analysis_box = QtWidgets.QGroupBox("Analysis")
        analysis_layout = QtWidgets.QGridLayout(analysis_box)
        arow = 0
        analysis_layout.addWidget(QtWidgets.QLabel("PSTH bin (ms):"), arow, 0)
        self.dbl_psth_bin = QtWidgets.QDoubleSpinBox()
        self.dbl_psth_bin.setRange(0.5, 5000.0)
        self.dbl_psth_bin.setDecimals(1)
        self.dbl_psth_bin.setSingleStep(1.0)
        self.dbl_psth_bin.setValue(10.0)
        analysis_layout.addWidget(self.dbl_psth_bin, arow, 1)
        arow += 1
        analysis_layout.addWidget(QtWidgets.QLabel("SDF sigma (ms):"), arow, 0)
        self.dbl_sdf_sigma = QtWidgets.QDoubleSpinBox()
        self.dbl_sdf_sigma.setRange(1.0, 1000.0)
        self.dbl_sdf_sigma.setDecimals(1)
        self.dbl_sdf_sigma.setSingleStep(1.0)
        self.dbl_sdf_sigma.setValue(10.0)
        analysis_layout.addWidget(self.dbl_sdf_sigma, arow, 1)
        arow += 1
        # Intrinsic noise control
        analysis_layout.addWidget(QtWidgets.QLabel("Noise std:"), arow, 0)
        self.dbl_noise_std = QtWidgets.QDoubleSpinBox()
        self.dbl_noise_std.setRange(0.0, 1000.0)
        self.dbl_noise_std.setDecimals(6)
        self.dbl_noise_std.setSingleStep(0.001)
        # load default from currently selected model
        try:
            model_key = self.cmb_model.currentText()
            self.dbl_noise_std.setValue(
                float(self.params["models"][model_key].get("noise_std", 0.0))
            )
        except Exception:
            self.dbl_noise_std.setValue(0.0)
        analysis_layout.addWidget(self.dbl_noise_std, arow, 1)
        # Attach analysis controls under stimulus box
        stim_layout.addWidget(analysis_box, srow, 0, 1, 2)

        # Buttons
        row += 1
        controls_layout.addWidget(stim_box, row, 0, 1, 2)
        row += 1
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_edit_params = QtWidgets.QPushButton("Edit Params JSON")
        controls_layout.addWidget(self.btn_run, row, 0)
        controls_layout.addWidget(self.btn_edit_params, row, 1)

        layout.addWidget(controls, 0, 0, 2, 1)

        # Single canvas with 3 subplots (shared x) and one toolbar
        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        # Make toolbar compact
        small = QSize(12, 12)
        self.toolbar.setIconSize(small)
        self.toolbar.setStyleSheet("QToolButton { margin: 0px; padding: 0px; }")

        # Dropdown to select neuron index for membrane plot
        neuron_sel_box = QtWidgets.QGroupBox("Membrane trace neuron index")
        h = QtWidgets.QHBoxLayout(neuron_sel_box)
        self.cmb_neuron_idx = QtWidgets.QComboBox()
        self.cmb_neuron_idx.addItems(["0"])  # will update after run
        h.addWidget(self.cmb_neuron_idx)

        plots_container = QtWidgets.QGroupBox("Visualization")
        plots_vlayout = QtWidgets.QVBoxLayout(plots_container)
        plots_vlayout.addWidget(self.toolbar)
        plots_vlayout.addWidget(self.canvas)
        plots_vlayout.addWidget(neuron_sel_box)
        layout.addWidget(plots_container, 0, 1, 2, 1)

        # With shared x-axis subplots, zoom/pan is naturally linked

        # Wire signals
        self.btn_run.clicked.connect(self.on_run)
        self.btn_edit_params.clicked.connect(self.on_edit_params)
        self.cmb_neuron_idx.currentIndexChanged.connect(self.on_update_membrane_plot)
        self.cmb_model.currentTextChanged.connect(self._on_model_change)
        # Live updates for analysis controls
        self.dbl_psth_bin.valueChanged.connect(self._plot_psth)
        self.dbl_sdf_sigma.valueChanged.connect(self.on_update_membrane_plot)
        self.dbl_noise_std.valueChanged.connect(self.on_update_membrane_plot)

        # Internal buffers for plots
        self._last_time = None
        self._last_input = None
        self._last_vtrace = None
        self._last_spikes = None
        self._last_model = None
        # For FA visualization: store pre-threshold amplifier output
        # if available
        self._last_va_cand = None
        # Store the post-filter input actually fed to the model for
        # accurate plots
        self._last_input_filtered = None
        # Cache last PSTH for speed
        self._last_psth = None
        self._last_psth_edges = None

    def _load_params(self) -> Dict[str, Any]:
        try:
            with open(DEFAULT_PARAMS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def on_edit_params(self):
        dlg = JsonEditorDialog(DEFAULT_PARAMS_PATH, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.params = self._load_params()

    # ---------- Stimulus Generation ----------
    def _generate_time(self, total_ms: int, dt_ms: float) -> np.ndarray:
        steps = int(math.ceil(total_ms / dt_ms))
        return np.arange(steps) * dt_ms

    def _make_stimulus(self, shape: str, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        amp = float(self.dbl_amp.value())
        freq = float(self.dbl_freq.value())
        duration = int(self.spin_duration.value())
        pre = int(self.spin_pre.value())
        post = int(self.spin_post.value())
        ramp = int(self.spin_ramp.value())
        duty_percent = (
            float(self.dbl_duty.value()) if hasattr(self, "dbl_duty") else 50.0
        )
        rise_ms_ctrl = int(self.spin_rise.value()) if hasattr(self, "spin_rise") else 0
        fall_ms_ctrl = int(self.spin_fall.value()) if hasattr(self, "spin_fall") else 0
        plateau_ms_ctrl = (
            int(self.spin_plateau.value()) if hasattr(self, "spin_plateau") else 0
        )

        total_ms = pre + duration + post
        t = self._generate_time(total_ms, dt)
        stim = np.zeros_like(t, dtype=np.float32)

        # Helper index ranges
        pre_steps = int(pre / dt)
        dur_steps = int(duration / dt)
        # post_steps computed implicitly by total length

        if shape == "step":
            stim[pre_steps : pre_steps + dur_steps] = amp
        elif shape == "sine":
            # Sine during duration
            tt = np.arange(dur_steps) * dt / 1000.0
            s = amp * np.sin(2 * np.pi * freq * tt)
            stim[pre_steps : pre_steps + dur_steps] = s
        elif shape == "ramp":
            s = np.linspace(0.0, amp, dur_steps, dtype=np.float32)
            stim[pre_steps : pre_steps + dur_steps] = s
        elif shape == "trapezoid":
            # Use rise/fall if provided (>0), else fall back to symmetric ramp
            rise_ms = rise_ms_ctrl if rise_ms_ctrl > 0 else ramp
            fall_ms = fall_ms_ctrl if fall_ms_ctrl > 0 else ramp
            rise_steps = int(rise_ms / dt)
            fall_steps = int(fall_ms / dt)
            if plateau_ms_ctrl > 0:
                plateau_steps = min(
                    int(plateau_ms_ctrl / dt),
                    max(0, dur_steps - rise_steps - fall_steps),
                )
            else:
                plateau_steps = max(0, dur_steps - rise_steps - fall_steps)
            s = np.concatenate(
                [
                    np.linspace(0.0, amp, max(1, rise_steps), dtype=np.float32),
                    np.full(plateau_steps, amp, dtype=np.float32),
                    np.linspace(amp, 0.0, max(1, fall_steps), dtype=np.float32),
                ]
            )
            s = s[:dur_steps]
            stim[pre_steps : pre_steps + dur_steps] = s
        elif shape == "periodic_trapezoid":
            # Duty-controlled periodic trapezoid with separate rise/fall and
            # plateau
            period_ms = 1000.0 / max(freq, 1e-6)
            duty = max(0.001, min(0.999, duty_percent / 100.0))
            on_ms = duty * period_ms
            rise_ms = rise_ms_ctrl if rise_ms_ctrl > 0 else ramp
            fall_ms = fall_ms_ctrl if fall_ms_ctrl > 0 else ramp
            # Ensure rise/fall do not exceed on_ms; plateau fills the remainder
            max_edge = max(1e-6, on_ms)
            rise_ms = min(rise_ms, max_edge)
            fall_ms = min(fall_ms, max_edge)
            # Plateau: use control if provided, else fill remainder
            plateau_ms = (
                plateau_ms_ctrl
                if plateau_ms_ctrl > 0
                else max(0.0, on_ms - rise_ms - fall_ms)
            )
            # Clamp plateau not to exceed available on_ms minus edges
            plateau_ms = min(plateau_ms, max(0.0, on_ms - rise_ms - fall_ms))
            # Compute phase within cycle for each duration sample (ms)
            tt_ms = np.arange(dur_steps) * dt
            phase = np.mod(tt_ms, period_ms)
            s = np.zeros(dur_steps, dtype=np.float32)
            # Rise region
            mask_rise = phase < rise_ms
            s[mask_rise] = (phase[mask_rise] / max(rise_ms, 1e-6)) * amp
            # Plateau region
            mask_plateau = (phase >= rise_ms) & (phase < rise_ms + plateau_ms)
            s[mask_plateau] = amp
            # Fall region
            mask_fall = (phase >= rise_ms + plateau_ms) & (phase < on_ms)
            s[mask_fall] = ((on_ms - phase[mask_fall]) / max(fall_ms, 1e-6)) * amp
            # Off window remains zero
            stim[pre_steps : pre_steps + dur_steps] = s
        elif shape == "sawtooth":
            tt = np.arange(dur_steps) * dt / 1000.0
            period = 1.0 / max(freq, 1e-6)
            frac = np.mod(tt, period) / period
            s = amp * frac
            stim[pre_steps : pre_steps + dur_steps] = s
        else:
            raise ValueError(f"Unknown shape: {shape}")

        return t, stim

    # ---------- Simulation ----------
    def _apply_filter(
        self, x_t: torch.Tensor, filter_name: str, dt: float
    ) -> torch.Tensor:
        if filter_name == "None":
            return x_t
        if filter_name == "SA":
            sa = SAFilterTorch(dt=dt)
            return sa(x_t, reset_states=True)
        if filter_name == "FA":
            ra = RAFilterTorch(dt=dt)
            return ra(x_t, reset_states=True)
        raise ValueError(f"Unknown filter: {filter_name}")

    def _run_model(
        self, model_name: str, I_bt: torch.Tensor, var_std: float, dt: float
    ):
        B, T, F = I_bt.shape
        device = I_bt.device
        # dtype = I_bt.dtype

        if model_name == "Izhikevich":
            cfg = self.params["models"]["Izhikevich"]
            # Construct with scalar params to avoid tuple handling in __init__
            # (u_init calculation uses b; variability handled in forward)
            neuron = IzhikevichNeuronTorch(
                a=cfg["a"],
                b=cfg["b"],
                c=cfg["c"],
                d=cfg["d"],
                threshold=cfg["threshold"],
                dt=dt,
                noise_std=float(self.dbl_noise_std.value()),
            ).to(device)
            if var_std > 0:
                # Pass variability tuples to forward so sampling happens there
                v_trace, spikes = neuron(
                    I_bt,
                    # a=(cfg["a"], var_std),
                    # b=(cfg["b"], var_std),
                    # c=(cfg["c"], var_std),
                    d=(cfg["d"], var_std),
                    threshold=(cfg["threshold"], var_std),
                )
            else:
                v_trace, spikes = neuron(I_bt)
            return v_trace, spikes

        if model_name == "AdEx":
            cfg = self.params["models"]["AdEx"]
            neuron = AdExNeuronTorch(
                EL=cfg["EL"],
                VT=cfg["VT"],
                DeltaT=cfg["DeltaT"],
                tau_m=cfg["tau_m"],
                tau_w=cfg["tau_w"],
                a=cfg["a"],
                b=cfg["b"],
                v_reset=cfg["v_reset"],
                v_spike=cfg["v_spike"],
                R=cfg["R"],
                dt=dt,
                noise_std=float(self.dbl_noise_std.value()),
            ).to(device)
            v_trace, spikes = neuron(I_bt)
            return v_trace, spikes

        if model_name == "MQIF":
            cfg = self.params["models"]["MQIF"]
            neuron = MQIFNeuronTorch(
                a=cfg["a"],
                b=cfg["b"],
                vr=cfg["vr"],
                vt=cfg["vt"],
                v_reset=cfg["v_reset"],
                v_peak=cfg["v_peak"],
                d=cfg["d"],
                tau_m=cfg["tau_m"],
                tau_u=cfg["tau_u"],
                dt=dt,
                noise_std=float(self.dbl_noise_std.value()),
            ).to(device)
            v_trace, spikes = neuron(I_bt)
            return v_trace, spikes

        if model_name == "FA":
            cfg = self.params["models"]["FA"]
            neuron = FANeuronTorch(
                # Force FA baseline to 0 and use EMA baseline mode
                vb=0.0,
                A=(cfg["A"], var_std) if var_std > 0 else cfg["A"],
                theta=(cfg["theta"], var_std) if var_std > 0 else cfg["theta"],
                tau_ref=(cfg["tau_ref"], var_std) if var_std > 0 else cfg["tau_ref"],
                dt=dt,
                baseline_mode="ema",
                input_gain=cfg.get("input_gain", 0.1),
                tau_dc=cfg.get("tau_dc", 50.0),
                noise_std=float(self.dbl_noise_std.value()),
            ).to(device)
            v_trace, spikes = neuron(I_bt)
            return v_trace, spikes

        if model_name == "SA":
            cfg = self.params["models"]["SA"]
            neuron = SANeuronTorch(
                I_tau=cfg.get("I_tau", 25e-12),
                I_th=cfg.get("I_th", 8.3e-9),
                I_tau_ahp=cfg.get("I_tau_ahp", 20e-12),
                I_th_ahp=cfg.get("I_th_ahp", 16.6e-12),
                I_tau_refractory=cfg.get("I_tau_refractory", 1.6e-9),
                C_mem=cfg.get("C_mem", 100e-15),
                C_adap=cfg.get("C_adap", 250e-15),
                C_refractory=cfg.get("C_refractory", 200e-15),
                U_T=cfg.get("U_T", 26e-3),
                kappa=cfg.get("kappa", 0.7),
                Ia_frac=cfg.get("Ia_frac", 0.8),
                dt=self.params.get("simulation", {}).get("dt", None),
                z_reset=cfg.get("z_reset", 0.0),
                I_in_op=cfg.get("I_in_op", 500e-12),
                current_scale=cfg.get("current_scale", 1e-12),
                noise_std=float(self.dbl_noise_std.value()),
            ).to(device)
            v_trace, spikes = neuron(I_bt)
            return v_trace, spikes

        raise ValueError(f"Unknown model: {model_name}")

    def on_run(self):
        # Load params
        dt = float(self.params.get("simulation", {}).get("dt", 0.1))
        device = self.params.get("simulation", {}).get("device", "cpu")
        torch_device = torch.device(device)

        # Stimulus
        t_ms, stim_np = self._make_stimulus(self.cmb_shape.currentText(), dt)
        neurons = int(self.spin_neurons.value())
        B = 1
        T = len(t_ms)
        F = neurons
        x = np.tile(stim_np.reshape(1, T, 1), (B, 1, F)).astype(np.float32)
        x_t = torch.from_numpy(x).to(torch_device)

        # Optional filter
        filt_name = self.cmb_filter.currentText()
        if filt_name != "None":
            x_t = self._apply_filter(x_t, filt_name, dt)
        # Cache filtered input for plotting (use CPU numpy)
        self._last_input_filtered = x_t.detach().cpu().numpy()

        # Run model
        var_std = float(self.dbl_variability.value())
        model_name = self.cmb_model.currentText()
        v_trace, spikes = self._run_model(model_name, x_t, var_std, dt)

        # Cache for plotting
        self._last_time = t_ms
        self._last_input = x
        self._last_vtrace = v_trace.detach().cpu().numpy()
        self._last_spikes = spikes.detach().cpu().numpy()
        self._last_model = model_name
        self._last_va_cand = None  # will compute for FA in plotting

        # Update neuron index dropdown
        self.cmb_neuron_idx.blockSignals(True)
        self.cmb_neuron_idx.clear()
        self.cmb_neuron_idx.addItems([str(i) for i in range(F)])
        self.cmb_neuron_idx.blockSignals(False)

        # Draw plots
        self._plot_input()
        self.on_update_membrane_plot()
        self._plot_raster()
        self._plot_psth()

    def _on_model_change(self):
        # Update noise std control to reflect model-specific default
        try:
            model_key = self.cmb_model.currentText()
            default_val = float(self.params["models"][model_key].get("noise_std", 0.0))
            self.dbl_noise_std.blockSignals(True)
            self.dbl_noise_std.setValue(default_val)
            self.dbl_noise_std.blockSignals(False)
        except Exception:
            pass

    def _plot_input(self):
        ax = self.canvas.ax_input
        ax.clear()
        if self._last_time is None or self._last_input is None:
            self.canvas.draw()
            return
        # Always show the pre-filter input, even if a filter was applied
        ax.plot(self._last_time, self._last_input[0, :, 0], color="C0")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Input amplitude (pre-filter)")
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def on_update_membrane_plot(self):
        ax = self.canvas.ax_mem
        ax.clear()
        if self._last_time is None or self._last_vtrace is None:
            self.canvas.draw()
            return
        idx = max(0, self.cmb_neuron_idx.currentIndex())
        # For FA, show the pre-threshold amplifier output to avoid flat look
        if self._last_model == "FA":
            # Compute EMA baseline to derive va_cand = vb - A*(x - m_t)
            # Use defaults from JSON (forced to vb=0, baseline_mode='ema')
            cfg = self.params["models"]["FA"]
            vb = float(cfg.get("vb", 0.0))
            A = float(cfg.get("A", 1.0))
            tau_dc = float(cfg.get("tau_dc", 50.0))
            dt = float(self.params.get("simulation", {}).get("dt", 0.1))
            # Use the filtered input if available
            x_series = (
                self._last_input_filtered[0, :, idx]
                if self._last_input_filtered is not None
                else self._last_input[0, :, idx]
            ).astype(np.float32)
            alpha = dt / max(tau_dc, 1e-6)
            m = np.zeros_like(x_series)
            if len(m) > 0:
                m[0] = x_series[0]
                for t_i in range(1, len(x_series)):
                    m[t_i] = m[t_i - 1] + alpha * (x_series[t_i] - m[t_i - 1])
            va_cand = vb - A * (x_series - m)
            self._last_va_cand = va_cand
            ax.plot(self._last_time, va_cand, color="C1", label="va (amp)")
        else:
            # v_trace is [B,T+1,F], align with time by skipping initial state
            ax.plot(self._last_time, self._last_vtrace[0, 1:, idx], color="C1")
        # Overlay selected neuron's spikes using a twin axis on right
        if self._last_spikes is not None:
            S = self._last_spikes[0, 1:, :]  # [T, F]
            T_steps, F = S.shape
            ax_r = ax.twinx()
            # Make raster occupy a compact band [0, 1] on the right axis
            # Highlight selected neuron's spikes higher in the band
            sel_times = self._last_time[S[:, idx]]
            if sel_times.size > 0:
                ax_r.scatter(
                    sel_times,
                    np.full_like(sel_times, 0.8, dtype=float),
                    s=18,
                    c="C3",
                    marker="|",
                    linewidths=1.5,
                    label=f"Cell {idx} spikes",
                )
            ax_r.set_ylim(0.0, 1.0)
            ax_r.set_yticks([])
            ax_r.grid(False)
        # Spike Density Function (SDF) for selected neuron over membrane
        if self._last_spikes is not None:
            S = self._last_spikes[0, 1:, :]  # [T, F]
            t_ms = self._last_time
            bin_ms = float(self.params.get("simulation", {}).get("dt", 0.1))
            # Build delta train for the selected neuron
            # (1 at spike indices, else 0)
            delta = S[:, idx].astype(np.float32)
            # Gaussian kernel with sigma from control (in ms)
            sigma_ms = float(self.dbl_sdf_sigma.value())
            sigma_steps = max(1, int(round(sigma_ms / max(bin_ms, 1e-6))))
            # Truncate kernel at 4 sigma on each side
            half = 4 * sigma_steps
            kx = np.arange(-half, half + 1)
            kernel = np.exp(-0.5 * (kx / max(sigma_steps, 1e-6)) ** 2)
            kernel = kernel / (kernel.sum() * (bin_ms / 1000.0))  # scale to Hz
            sdf = np.convolve(delta, kernel, mode="same")
            ax.plot(t_ms, sdf, color="C4", label="SDF (Hz)", alpha=0.9)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Imem (A)")
        ax.grid(True, alpha=0.3)
        if self._last_model == "FA":
            ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw()

    # Raster will be overlaid in the membrane plot
    def _plot_raster(self):
        ax = self.canvas.ax_raster
        ax.clear()
        if self._last_time is None or self._last_spikes is None:
            self.canvas.draw()
            return
        S = self._last_spikes[0, 1:, :]  # [T, F]
        T_steps, F = S.shape
        event_t = []
        event_n = []
        for n in range(F):
            times = self._last_time[S[:, n]]
            if times.size == 0:
                continue
            event_t.append(times)
            event_n.append(np.full_like(times, n, dtype=float))
        if len(event_t) > 0:
            ax.scatter(
                np.concatenate(event_t),
                np.concatenate(event_n),
                s=5,
                c="k",
                alpha=0.6,
                marker="|",
            )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")
        ax.set_ylim(-1, F)
        ax.grid(True, alpha=0.2)
        self.canvas.draw()

    def _plot_psth(self):
        ax = self.canvas.ax_psth
        ax.clear()
        if self._last_time is None or self._last_spikes is None:
            self.canvas.draw()
            return
        # spikes: [B,T+1,F] -> consider times 1..T
        S = self._last_spikes[0, 1:, :]  # [T, F]
        T_steps, F = S.shape
        # Convert spike boolean matrix to spike times (ms)
        t_ms = self._last_time
        # Bin width from control (ms)
        bin_w = float(self.dbl_psth_bin.value())
        edges = np.arange(t_ms[0], t_ms[-1] + bin_w, bin_w)
        # Population rate: count spikes across all neurons per bin
        # Note: S shape [T,F] corresponds to t_ms length T
        all_spike_times = t_ms[np.where(S)[0]]
        counts, edges = np.histogram(all_spike_times, bins=edges)
        # Convert to rate: spikes per second per neuron (Hz)
        bin_sec = bin_w / 1000.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = counts.astype(np.float32) / (F * bin_sec)
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax.step(centers, rate, where="mid", color="C2", label="PSTH pop (Hz)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Rate (Hz)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw()

    # No explicit axis-linking needed due to shared x-axis subplots


def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = NeuronExplorer()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
