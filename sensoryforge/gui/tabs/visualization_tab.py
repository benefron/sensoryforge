"""Visualization tab — synchronized, modular panel dashboard.

Receives simulation results from the Spiking Neurons tab and provides a
time-synchronized, multi-panel view of:

    • Stimulus pressure field       (StimulusPanel)
    • Receptor drive scatter        (ReceptorPanel)
    • Neuron activity scatter       (NeuronPanel)
    • Spike raster                  (RasterPanel)
    • Population firing rate        (FiringRatePanel)
    • Single-neuron voltage trace   (VoltagePanel)

Layout is controlled by a preset-layout toolbar.  Individual panel settings
are accessible via the ⚙ button in each panel header.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sensoryforge.gui.visualization.base_panel import VisualizationPanel, VisData
from sensoryforge.gui.visualization.playback_bar import PlaybackController
from sensoryforge.gui.visualization.stimulus_panel import StimulusPanel
from sensoryforge.gui.visualization.receptor_panel import ReceptorPanel
from sensoryforge.gui.visualization.neuron_panel import NeuronPanel
from sensoryforge.gui.visualization.raster_panel import RasterPanel
from sensoryforge.gui.visualization.firing_rate_panel import FiringRatePanel
from sensoryforge.gui.visualization.voltage_panel import VoltagePanel


# ---------------------------------------------------------------------------
# Preset layout definitions
# ---------------------------------------------------------------------------

# Each preset is a list of (row, col, rowspan, colspan, PanelClass) tuples.
_PANEL_CLASSES: Dict[str, Type[VisualizationPanel]] = {
    "Stimulus":      StimulusPanel,
    "Receptor Drive": ReceptorPanel,
    "Neuron Activity": NeuronPanel,
    "Spike Raster":  RasterPanel,
    "Firing Rate":   FiringRatePanel,
    "Voltage Trace": VoltagePanel,
}

_PRESET_ICON = {
    "1 Panel":   "▣",
    "2 – Side by Side": "▣▣",
    "2 – Stacked":      "▣\n▣",
    "2×2 Grid":  "▣▣\n▣▣",
    "3 Wide":    "▣▣▣",
    "Focus + Details": "▣▣\n▣▣",
}

# (row, col, rowspan, colspan, PanelClass)
_PRESETS: Dict[str, List[Tuple[int, int, int, int, Type[VisualizationPanel]]]] = {
    "1 Panel": [
        (0, 0, 1, 1, RasterPanel),
    ],
    "2 – Side by Side": [
        (0, 0, 1, 1, StimulusPanel),
        (0, 1, 1, 1, RasterPanel),
    ],
    "2 – Stacked": [
        (0, 0, 1, 1, RasterPanel),
        (1, 0, 1, 1, FiringRatePanel),
    ],
    "2×2 Grid": [
        (0, 0, 1, 1, StimulusPanel),
        (0, 1, 1, 1, RasterPanel),
        (1, 0, 1, 1, NeuronPanel),
        (1, 1, 1, 1, FiringRatePanel),
    ],
    "3 Wide": [
        (0, 0, 1, 1, StimulusPanel),
        (0, 1, 1, 1, RasterPanel),
        (0, 2, 1, 1, FiringRatePanel),
    ],
    "Focus + Details": [
        (0, 0, 2, 1, RasterPanel),
        (0, 1, 1, 1, StimulusPanel),
        (1, 1, 1, 1, FiringRatePanel),
    ],
}

_DEFAULT_PRESET = "2×2 Grid"


# ---------------------------------------------------------------------------
# PanelSlot: one cell in the canvas grid
# ---------------------------------------------------------------------------

class _PanelSlot(QtWidgets.QWidget):
    """Container cell that holds one VisualizationPanel and an 'Add Panel' placeholder."""

    panel_settings_requested = QtCore.pyqtSignal(object)  # VisualizationPanel

    def __init__(
        self,
        panel: Optional[VisualizationPanel] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._panel: Optional[VisualizationPanel] = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        self._layout = layout
        self.setMinimumSize(160, 120)

        if panel is not None:
            self.set_panel(panel)

    def set_panel(self, panel: VisualizationPanel) -> None:
        self._clear()
        self._panel = panel
        self._layout.addWidget(panel)
        panel.close_requested.connect(self._on_panel_close_requested)
        panel._settings_btn.clicked.connect(
            lambda: self.panel_settings_requested.emit(panel)
        )

    def _clear(self) -> None:
        if self._panel is not None:
            self._layout.removeWidget(self._panel)
            self._panel.setParent(None)
            self._panel = None

    def _on_panel_close_requested(self, panel: VisualizationPanel) -> None:
        self._clear()
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        ph = _PlaceholderWidget(self)
        self._layout.addWidget(ph)

    @property
    def panel(self) -> Optional[VisualizationPanel]:
        return self._panel


class _PlaceholderWidget(QtWidgets.QWidget):
    """Grey placeholder shown in an empty panel slot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "background: #f0f0f0; border: 2px dashed #cccccc; border-radius: 4px;"
        )
        layout = QtWidgets.QVBoxLayout(self)
        lbl = QtWidgets.QLabel("No panel")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet("color: #aaaaaa; font-size: 12px; font-style: italic;")
        layout.addWidget(lbl)


# ---------------------------------------------------------------------------
# Canvas: the grid of PanelSlots
# ---------------------------------------------------------------------------

class _Canvas(QtWidgets.QWidget):
    """Hosts the panel grid using nested QSplitters for resizable cells."""

    panel_settings_requested = QtCore.pyqtSignal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        self._slots: List[_PanelSlot] = []
        self._splitter_rows: Optional[QtWidgets.QSplitter] = None

    def apply_preset(
        self,
        preset: List[Tuple[int, int, int, int, Type[VisualizationPanel]]],
        data: Optional[VisData],
    ) -> List[VisualizationPanel]:
        """Tear down current layout, build preset, return new panels."""
        self._clear()
        self._slots.clear()

        # Build a logical grid: find dims
        max_row = max(r + rs for r, c, rs, cs, _ in preset)
        max_col = max(c + cs for r, c, rs, cs, _ in preset)

        # Use a vertical splitter of horizontal splitters
        v_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        v_splitter.setChildrenCollapsible(False)
        self._splitter_rows = v_splitter
        self._layout.addWidget(v_splitter)

        # Grid of slots: grid[row][col] = _PanelSlot or None
        grid: List[List[Optional[_PanelSlot]]] = [
            [None] * max_col for _ in range(max_row)
        ]

        panels: List[VisualizationPanel] = []

        for row, col, rowspan, colspan, PanelClass in preset:
            panel = PanelClass()
            if data is not None:
                panel.set_data(data)
            slot = _PanelSlot(panel)
            slot.panel_settings_requested.connect(self.panel_settings_requested)
            self._slots.append(slot)
            panels.append(panel)
            for r in range(row, row + rowspan):
                for c in range(col, col + colspan):
                    if r < max_row and c < max_col:
                        grid[r][c] = slot

        # Build row splitters
        row_widgets: List[QtWidgets.QWidget] = []
        placed: set = set()
        for r in range(max_row):
            h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            h_splitter.setChildrenCollapsible(False)
            for c in range(max_col):
                slot = grid[r][c]
                if slot is None:
                    ph = _PlaceholderWidget()
                    h_splitter.addWidget(ph)
                elif id(slot) not in placed:
                    h_splitter.addWidget(slot)
                    placed.add(id(slot))
            row_widgets.append(h_splitter)
            v_splitter.addWidget(h_splitter)

        # Equal sizes
        row_h = [1] * max_row
        v_splitter.setSizes(row_h)

        return panels

    def _clear(self) -> None:
        if self._splitter_rows is not None:
            self._layout.removeWidget(self._splitter_rows)
            self._splitter_rows.setParent(None)
            self._splitter_rows = None
        for slot in self._slots:
            slot.setParent(None)

    def all_panels(self) -> List[VisualizationPanel]:
        return [s.panel for s in self._slots if s.panel is not None]


# ---------------------------------------------------------------------------
# Settings sidebar
# ---------------------------------------------------------------------------

class _SettingsSidebar(QtWidgets.QWidget):
    """Right-side panel that shows the settings widget for the active panel."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet(
            "QWidget { background: #f5f5f5; border-left: 1px solid #cccccc; }"
        )
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._title_bar = QtWidgets.QWidget()
        self._title_bar.setFixedHeight(32)
        self._title_bar.setStyleSheet(
            "QWidget { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #e0e0e0,stop:1 #d0d0d0); border-bottom: 1px solid #bbb; }"
        )
        tb_layout = QtWidgets.QHBoxLayout(self._title_bar)
        tb_layout.setContentsMargins(8, 0, 4, 0)
        self._title_lbl = QtWidgets.QLabel("Settings")
        self._title_lbl.setStyleSheet(
            "font-weight: bold; font-size: 11px; border: none; background: transparent;"
        )
        tb_layout.addWidget(self._title_lbl)
        tb_layout.addStretch()
        close_btn = QtWidgets.QToolButton()
        close_btn.setText("✕")
        close_btn.setStyleSheet(
            "QToolButton { border: none; font-size: 11px; color: #999; background: transparent;}"
            "QToolButton:hover { color: #cc3333; }"
        )
        close_btn.clicked.connect(self.hide)
        tb_layout.addWidget(close_btn)
        outer.addWidget(self._title_bar)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._content = QtWidgets.QWidget()
        self._content.setStyleSheet("background: transparent;")
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll)

        self._current_widget: Optional[QtWidgets.QWidget] = None

    def show_panel_settings(self, panel: VisualizationPanel) -> None:
        if self._current_widget is not None:
            self._content_layout.removeWidget(self._current_widget)
            self._current_widget.setParent(None)
            self._current_widget = None

        self._title_lbl.setText(f"{panel.PANEL_DISPLAY_NAME} Settings")
        settings = panel.build_settings_widget()
        if settings is not None:
            self._current_widget = settings
            self._content_layout.addWidget(settings)
            self._content_layout.addStretch()
        self.show()


# ---------------------------------------------------------------------------
# Toolbar: preset chooser + Add Panel
# ---------------------------------------------------------------------------

class _Toolbar(QtWidgets.QWidget):
    """Top bar: preset selector, add-panel menu, data status."""

    preset_selected = QtCore.pyqtSignal(str)
    add_panel_requested = QtCore.pyqtSignal(str)   # panel display name

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #eaeaea, stop:1 #d8d8d8);
                border-bottom: 1px solid #bbbbbb;
            }
            QPushButton, QToolButton {
                padding: 3px 10px;
                border: 1px solid #aaaaaa;
                border-radius: 3px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #f5f5f5, stop:1 #e8e8e8);
                font-size: 11px;
            }
            QPushButton:hover { background: #eeeeee; border-color: #888; }
            QPushButton:pressed { background: #d8d8d8; }
        """)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(8)

        # Layout presets
        lbl = QtWidgets.QLabel("Layout:")
        lbl.setStyleSheet("font-size: 11px; color: #444; background: transparent; border: none;")
        layout.addWidget(lbl)

        self._preset_cmb = QtWidgets.QComboBox()
        self._preset_cmb.addItems(list(_PRESETS.keys()))
        self._preset_cmb.setCurrentText(_DEFAULT_PRESET)
        self._preset_cmb.setFixedWidth(160)
        self._preset_cmb.currentTextChanged.connect(self.preset_selected)
        layout.addWidget(self._preset_cmb)

        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.setFixedWidth(56)
        apply_btn.clicked.connect(
            lambda: self.preset_selected.emit(self._preset_cmb.currentText())
        )
        layout.addWidget(apply_btn)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep)

        # Add Panel button with dropdown
        add_menu = QtWidgets.QMenu()
        add_menu.setStyleSheet(
            "QMenu { border: 1px solid #aaa; background: #fff; }"
            "QMenu::item { padding: 4px 16px; font-size: 11px; }"
            "QMenu::item:selected { background: #4287f5; color: #fff; }"
        )
        for name in _PANEL_CLASSES:
            action = add_menu.addAction(name)
            action.triggered.connect(
                lambda checked, n=name: self.add_panel_requested.emit(n)
            )

        add_btn = QtWidgets.QPushButton("+ Add Panel ▾")
        add_btn.setFixedWidth(110)
        add_btn.clicked.connect(
            lambda: add_menu.exec_(add_btn.mapToGlobal(add_btn.rect().bottomLeft()))
        )
        layout.addWidget(add_btn)

        layout.addStretch()

        # Data status
        self._status_lbl = QtWidgets.QLabel("No simulation data loaded")
        self._status_lbl.setStyleSheet(
            "font-size: 11px; color: #888; background: transparent; border: none;"
        )
        layout.addWidget(self._status_lbl)

    def set_status(self, msg: str) -> None:
        self._status_lbl.setText(msg)


# ---------------------------------------------------------------------------
# Main VisualizationTab
# ---------------------------------------------------------------------------

class VisualizationTab(QtWidgets.QWidget):
    """Multi-panel, time-synchronized simulation visualization tab.

    Wire-up in main.py::

        spiking_tab.simulation_finished.connect(viz_tab.set_simulation_results)
        mech_tab.populations_changed.connect(viz_tab.set_populations)
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True, background="w", foreground="k")

        self._data: Optional[VisData] = None
        self._panels: List[VisualizationPanel] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_simulation_results(
        self,
        sim_results: dict,
        stimulus_frames: Optional[np.ndarray],
        time_ms: np.ndarray,
        dt_ms: float,
        stimulus_xlim: tuple = (-5.0, 5.0),
        stimulus_ylim: tuple = (-5.0, 5.0),
    ) -> None:
        """Receive simulation results from the Spiking Neurons tab.

        Args:
            sim_results: Dict[str, SimulationResult] keyed by population name.
            stimulus_frames: [T, H, W] ndarray or None.
            time_ms: [T] time axis in ms.
            dt_ms: Simulation time step in ms.
            stimulus_xlim: (x_min, x_max) spatial extent in mm.
            stimulus_ylim: (y_min, y_max) spatial extent in mm.
        """
        population_results: Dict[str, Dict[str, np.ndarray]] = {}
        population_colors: Dict[str, QtGui.QColor] = {}

        for name, result in sim_results.items():
            population_results[name] = {
                "drive":   result.drive,
                "v_trace": result.v_trace,
                "spikes":  result.spikes,
            }

        # Preserve neuron positions and receptor positions if already set
        neuron_positions = (
            self._data.neuron_positions if self._data is not None else {}
        )
        receptor_positions = (
            self._data.receptor_positions if self._data is not None else None
        )
        # Carry over colors if set
        if self._data is not None:
            population_colors = self._data.population_colors

        self._data = VisData(
            time_ms=time_ms,
            dt_ms=dt_ms,
            stimulus_frames=stimulus_frames,
            stimulus_xlim=stimulus_xlim,
            stimulus_ylim=stimulus_ylim,
            population_results=population_results,
            neuron_positions=neuron_positions,
            receptor_positions=receptor_positions,
            population_colors=population_colors,
        )

        n_pops = len(population_results)
        total_ms = float(time_ms[-1]) if time_ms.size else 0.0
        self._toolbar.set_status(
            f"{n_pops} population{'s' if n_pops != 1 else ''} · "
            f"{len(time_ms)} steps · {total_ms:.1f} ms"
        )

        self._playback.set_length(len(time_ms), dt_ms)
        self._push_data_to_panels()

    def set_populations(self, populations: list) -> None:
        """Receive population list from MechanoreceptorTab.populations_changed.

        Extracts neuron positions and receptor positions for spatial panels.
        """
        neuron_positions: Dict[str, np.ndarray] = {}
        receptor_positions: Optional[np.ndarray] = None
        population_colors: Dict[str, QtGui.QColor] = {}

        for pop in populations:
            centers = None
            if hasattr(pop, "neuron_centers") and pop.neuron_centers is not None:
                try:
                    centers = pop.neuron_centers.detach().cpu().numpy()
                except Exception:
                    pass
            if centers is not None:
                neuron_positions[pop.name] = centers
            if hasattr(pop, "color") and isinstance(pop.color, QtGui.QColor):
                population_colors[pop.name] = pop.color

        # Receptor positions: use first population's module
        if receptor_positions is None:
            for pop in populations:
                if hasattr(pop, "flat_module") and pop.flat_module is not None:
                    try:
                        coords = pop.flat_module.receptor_coords.detach().cpu().numpy()
                        receptor_positions = coords
                        break
                    except Exception:
                        pass
                if receptor_positions is None and hasattr(pop, "module") and pop.module is not None:
                    try:
                        gm = getattr(pop, "_grid_manager_ref", None)
                        if gm is not None and gm.xx is not None:
                            xx, yy = gm.xx.detach().cpu().numpy(), gm.yy.detach().cpu().numpy()
                            receptor_positions = np.stack(
                                [xx.ravel(), yy.ravel()], axis=1
                            )
                            break
                    except Exception:
                        pass

        if self._data is not None:
            self._data.neuron_positions = neuron_positions
            self._data.receptor_positions = receptor_positions
            self._data.population_colors = population_colors
            self._push_data_to_panels()
        else:
            # Store for when simulation data arrives
            self._pending_neuron_positions = neuron_positions
            self._pending_receptor_positions = receptor_positions
            self._pending_population_colors = population_colors

    def set_receptor_positions(self, positions: np.ndarray) -> None:
        """Directly set receptor (x, y) positions [M, 2] in mm."""
        if self._data is not None:
            self._data.receptor_positions = positions
            self._push_data_to_panels()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Toolbar
        self._toolbar = _Toolbar()
        self._toolbar.preset_selected.connect(self._on_preset_selected)
        self._toolbar.add_panel_requested.connect(self._on_add_panel_requested)
        outer.addWidget(self._toolbar)

        # Main area: canvas + settings sidebar
        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._canvas = _Canvas()
        self._canvas.panel_settings_requested.connect(self._on_panel_settings_requested)
        body.addWidget(self._canvas, stretch=1)

        self._sidebar = _SettingsSidebar()
        self._sidebar.hide()
        body.addWidget(self._sidebar)

        body_container = QtWidgets.QWidget()
        body_container.setLayout(body)
        outer.addWidget(body_container, stretch=1)

        # Playback bar
        self._playback = PlaybackController()
        self._playback.seek_requested.connect(self._on_seek)
        outer.addWidget(self._playback)

        # Apply default preset (no data yet)
        self._apply_preset(_DEFAULT_PRESET)

        # Empty-state overlay
        self._empty_label = QtWidgets.QLabel(
            "Run a simulation in the Spiking Neurons tab to populate this view."
        )
        self._empty_label.setAlignment(QtCore.Qt.AlignCenter)
        self._empty_label.setStyleSheet(
            "QLabel { color: #aaaaaa; font-size: 14px; font-style: italic; "
            "background: transparent; }"
        )
        self._empty_label.setParent(self._canvas)
        self._empty_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._empty_label.setGeometry(self._canvas.rect())

        self._canvas.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if obj is self._canvas and event.type() == QtCore.QEvent.Resize:
            self._empty_label.setGeometry(self._canvas.rect())
        return False

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_seek(self, t_idx: int) -> None:
        for panel in self._panels:
            panel.seek(t_idx)

    def _on_preset_selected(self, name: str) -> None:
        if name in _PRESETS:
            self._apply_preset(name)

    def _on_add_panel_requested(self, panel_name: str) -> None:
        """Add a floating panel (simple: add to a new row in a 1-column layout)."""
        PanelClass = _PANEL_CLASSES.get(panel_name)
        if PanelClass is None:
            return
        panel = PanelClass()
        if self._data is not None:
            panel.set_data(self._data)
        # For simplicity, apply "2 – Stacked" after appending — in a full
        # implementation this would dynamically add a row.  For V1 we show a
        # dialog asking the user to choose a preset instead.
        QtWidgets.QMessageBox.information(
            self,
            "Add Panel",
            f"'{panel_name}' panel added.\n\n"
            "Tip: Use the Layout preset selector to arrange panels in a grid.",
        )
        self._panels.append(panel)

    def _on_panel_settings_requested(self, panel: VisualizationPanel) -> None:
        self._sidebar.show_panel_settings(panel)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _apply_preset(self, preset_name: str) -> None:
        self._playback.stop()
        preset = _PRESETS.get(preset_name, _PRESETS[_DEFAULT_PRESET])
        self._panels = self._canvas.apply_preset(preset, self._data)
        for panel in self._panels:
            panel._settings_btn.clicked.connect(
                lambda _, p=panel: self._on_panel_settings_requested(p)
            )
        # Hide empty label once data arrives
        if self._data is not None:
            self._empty_label.hide()

    def _push_data_to_panels(self) -> None:
        """Push current VisData to all active panels."""
        if self._data is None:
            return
        # Merge any pending spatial data
        if hasattr(self, "_pending_neuron_positions"):
            self._data.neuron_positions = self._pending_neuron_positions
            del self._pending_neuron_positions
        if hasattr(self, "_pending_receptor_positions"):
            self._data.receptor_positions = self._pending_receptor_positions
            del self._pending_receptor_positions
        if hasattr(self, "_pending_population_colors"):
            self._data.population_colors = self._pending_population_colors
            del self._pending_population_colors

        import traceback
        for panel in self._panels:
            try:
                panel.set_data(self._data)
            except Exception:
                print(f"[VisualizationTab] Error in {panel.PANEL_DISPLAY_NAME}.set_data:")
                traceback.print_exc()

        if self._data.n_steps > 0:
            self._empty_label.hide()
