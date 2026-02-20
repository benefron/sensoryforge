"""Base classes for visualization panels.

All panels share a common interface so the tab can drive them uniformly
through :meth:`VisualizationPanel.seek`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg  # type: ignore


# ---------------------------------------------------------------------------
# Data container passed to every panel
# ---------------------------------------------------------------------------

@dataclass
class VisData:
    """Bundle of simulation outputs shared across all panels.

    All arrays are indexed [time_step, ...] so ``seek(t)`` simply selects
    row ``t`` from each array.

    Attributes:
        time_ms: 1-D time axis [T] in milliseconds.
        dt_ms: Simulation time step in milliseconds.
        stimulus_frames: Stimulus pressure grid [T, H, W].
        stimulus_xlim: (x_min, x_max) spatial extent in mm.
        stimulus_ylim: (y_min, y_max) spatial extent in mm.
        population_results: Mapping from population name to per-population
            arrays. Each value is a dict with keys:
            - ``drive``     : [T, N_neurons]  filtered drive (after SA/RA) (mA)
            - ``raw_drive`` : [T, N_neurons]  drive before temporal filter (mA)
            - ``v_trace``   : [T, N_neurons]  membrane voltage (mV)
            - ``spikes``    : [T, N_neurons]  binary spike indicator
        neuron_positions: Mapping from population name to [N, 2] (x, y) mm.
        receptor_positions: [M, 2] receptor (x, y) positions in mm.
        population_colors: Mapping from population name to QColor.
        innervation_weights: Mapping from population name to weight array.
            Grid module: [N_neurons, grid_h, grid_w]. Flat: [N_neurons, N_receptors].
    """

    time_ms: np.ndarray = field(default_factory=lambda: np.array([]))
    dt_ms: float = 1.0
    stimulus_frames: Optional[np.ndarray] = None        # [T, H, W]
    stimulus_xlim: tuple = (-5.0, 5.0)
    stimulus_ylim: tuple = (-5.0, 5.0)
    population_results: Dict[str, Dict[str, np.ndarray]] = field(
        default_factory=dict
    )
    neuron_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    receptor_positions: Optional[np.ndarray] = None     # [M, 2]
    population_colors: Dict[str, QtGui.QColor] = field(default_factory=dict)
    innervation_weights: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.time_ms)

    @property
    def population_names(self) -> List[str]:
        return list(self.population_results.keys())

    def has_spikes(self, name: str) -> bool:
        return name in self.population_results and "spikes" in self.population_results[name]


# ---------------------------------------------------------------------------
# Shared colormaps
# ---------------------------------------------------------------------------

def make_colormap(name: str = "viridis") -> pg.ColorMap:
    """Return a named pyqtgraph colormap, falling back to a grey gradient."""
    try:
        return pg.colormap.get(name, source="matplotlib")
    except Exception:
        return pg.colormap.get("grey")


# ---------------------------------------------------------------------------
# Base panel
# ---------------------------------------------------------------------------

class VisualizationPanel(QtWidgets.QWidget):
    """Abstract base for all visualization panels.

    Subclasses must implement :meth:`set_data` and :meth:`_render_frame`.
    The :meth:`seek` method calls :meth:`_render_frame` after updating the
    current time index.

    Signals:
        title_changed: emitted when the panel title changes.
        close_requested: emitted when the user clicks the panel close button.
    """

    title_changed = QtCore.pyqtSignal(str)
    close_requested = QtCore.pyqtSignal(object)   # passes self
    replace_with_requested = QtCore.pyqtSignal(str)  # panel type name

    PANEL_DISPLAY_NAME: str = "Panel"  # override in subclasses

    # Shared style constants
    HEADER_HEIGHT = 28
    HEADER_STYLE = """
        QWidget#PanelHeader {{
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #e0e0e0, stop:1 #d0d0d0
            );
            border-bottom: 1px solid #aaaaaa;
        }}
    """
    PANEL_STYLE = """
        QWidget#PanelFrame {{
            border: 1px solid #cccccc;
            border-radius: 4px;
            background: #ffffff;
        }}
    """

    def __init__(
        self,
        title: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("PanelFrame")
        self.setStyleSheet(self.PANEL_STYLE)

        self._data: Optional[VisData] = None
        self._t_idx: int = 0
        self._title: str = title or self.PANEL_DISPLAY_NAME
        self._panel_type_options: List[str] = []  # set via set_panel_type_options

        # Outer layout: header + content
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = self._build_header()
        outer.addWidget(self._header)

        self._content = QtWidgets.QWidget()
        self._content.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(2, 2, 2, 2)
        outer.addWidget(self._content)

        self._build_content(self._content_layout)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _build_header(self) -> QtWidgets.QWidget:
        header = QtWidgets.QWidget()
        header.setObjectName("PanelHeader")
        header.setFixedHeight(self.HEADER_HEIGHT)
        header.setStyleSheet(self.HEADER_STYLE)
        h_layout = QtWidgets.QHBoxLayout(header)
        h_layout.setContentsMargins(8, 0, 4, 0)
        h_layout.setSpacing(4)

        self._title_label = QtWidgets.QLabel(self._title)
        self._title_label.setStyleSheet(
            "QLabel { font-weight: bold; font-size: 11px; color: #333333; }"
        )
        h_layout.addWidget(self._title_label)
        h_layout.addStretch()

        self._settings_btn = QtWidgets.QToolButton()
        self._settings_btn.setText("⚙")
        self._settings_btn.setToolTip("Panel settings")
        self._settings_btn.setFixedSize(20, 20)
        self._settings_btn.setStyleSheet(
            "QToolButton { border: none; font-size: 12px; color: #666; }"
            "QToolButton:hover { color: #222; }"
        )
        self._settings_btn.clicked.connect(self._on_settings_clicked)
        h_layout.addWidget(self._settings_btn)

        self._change_type_btn = QtWidgets.QToolButton()
        self._change_type_btn.setText("▾")
        self._change_type_btn.setToolTip("Change panel type — click to swap this panel for another view")
        self._change_type_btn.setFixedSize(20, 20)
        self._change_type_btn.setStyleSheet(
            "QToolButton { border: none; font-size: 11px; color: #666; }"
            "QToolButton:hover { color: #222; }"
        )
        self._change_type_btn.clicked.connect(self._on_change_type_clicked)
        h_layout.addWidget(self._change_type_btn)

        close_btn = QtWidgets.QToolButton()
        close_btn.setText("✕")
        close_btn.setToolTip("Remove panel")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            "QToolButton { border: none; font-size: 11px; color: #999; }"
            "QToolButton:hover { color: #cc3333; }"
        )
        close_btn.clicked.connect(lambda: self.close_requested.emit(self))
        h_layout.addWidget(close_btn)
        return header

    def set_title(self, title: str) -> None:
        self._title = title
        self._title_label.setText(title)
        self.title_changed.emit(title)

    def set_panel_type_options(self, names: List[str]) -> None:
        """Set the list of panel type names for the 'Change type' menu."""
        self._panel_type_options = list(names)

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _build_content(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Build the panel-specific plot/widget inside ``layout``."""
        raise NotImplementedError

    def _render_frame(self, t_idx: int) -> None:
        """Render the frame at time index ``t_idx``.  Called from :meth:`seek`."""
        raise NotImplementedError

    def set_data(self, data: VisData) -> None:
        """Accept a new data bundle and reset to t=0."""
        self._data = data
        self._t_idx = 0
        self._on_data_set()

    def _on_data_set(self) -> None:
        """Hook called after :meth:`set_data` updates ``_data``.  Override to
        rebuild axes, colormaps, etc., before the first :meth:`seek` call."""
        pass

    def seek(self, t_idx: int) -> None:
        """Jump to time step ``t_idx`` and redraw."""
        if self._data is None or self._data.n_steps == 0:
            return
        self._t_idx = max(0, min(t_idx, self._data.n_steps - 1))
        self._render_frame(self._t_idx)

    def build_settings_widget(self) -> Optional[QtWidgets.QWidget]:
        """Return a widget shown in the settings sidebar, or None."""
        return None

    def _on_settings_clicked(self) -> None:
        """Emit a signal so the tab can show the settings sidebar."""
        # The tab connects to this via the parent VisualizationTab
        pass

    def _on_change_type_clicked(self) -> None:
        """Show menu of panel types and emit replace_with_requested when chosen."""
        if not self._panel_type_options:
            return
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            "QMenu { border: 1px solid #aaa; background: #fff; }"
            "QMenu::item { padding: 4px 16px; font-size: 11px; }"
            "QMenu::item:selected { background: #4287f5; color: #fff; }"
        )
        current = self.PANEL_DISPLAY_NAME
        for name in self._panel_type_options:
            if name == current:
                continue
            action = menu.addAction(name)
            action.triggered.connect(
                lambda checked, n=name: self.replace_with_requested.emit(n)
            )
        if menu.actions():
            menu.exec_(self._change_type_btn.mapToGlobal(
                self._change_type_btn.rect().bottomLeft()
            ))

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _make_plot(
        background: str = "w",
        **kwargs,
    ) -> pg.PlotWidget:
        pw = pg.PlotWidget(background=background, **kwargs)
        pw.showGrid(x=True, y=True, alpha=0.15)
        return pw

    @staticmethod
    def _colormap_lut(cmap_name: str = "viridis", n: int = 256) -> np.ndarray:
        """Return an (n, 4) uint8 RGBA lookup table."""
        cmap = make_colormap(cmap_name)
        return cmap.getLookupTable(nPts=n, alpha=True)

    def _no_data_label(self, layout: QtWidgets.QVBoxLayout, msg: str) -> None:
        lbl = QtWidgets.QLabel(msg)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet(
            "QLabel { color: #aaaaaa; font-size: 13px; font-style: italic; }"
        )
        layout.addWidget(lbl)
