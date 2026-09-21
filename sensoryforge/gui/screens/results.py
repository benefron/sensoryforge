"""Run & Results screen: a fixed panel grid, one shared time cursor.

:class:`ResultsScreen` shows a :class:`~sensoryforge.gui.screens.results_data.ResultsView`
built either from the session's live :class:`~sensoryforge.gui.session.RunResult`
or from a bundle opened with **Open bundle...**. Every panel
(:mod:`results_stimulus_panel`, :mod:`results_raster_panel`,
:mod:`results_rate_panel`, :mod:`results_trace_panel`,
:mod:`results_map_panel`) reads only that view -- see
``results_data.py`` for why.

Layout: a fixed 2x3 grid (stimulus, raster, rate on the top row; trace, map
and the visibility picker's column on the bottom), a stale banner above it,
and a shared :class:`~sensoryforge.gui.screens.results_playback.PlaybackBar`
below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PyQt5 import QtWidgets

from sensoryforge.gui import theme
from sensoryforge.gui.screens import results_data
from sensoryforge.gui.screens.results_bundle_browser import OpenBundleDialog
from sensoryforge.gui.screens.results_map_panel import NeuronMapPanel
from sensoryforge.gui.screens.results_playback import PlaybackBar
from sensoryforge.gui.screens.results_raster_panel import RasterPanel
from sensoryforge.gui.screens.results_rate_panel import RatePanel
from sensoryforge.gui.screens.results_stimulus_panel import StimulusFramePanel
from sensoryforge.gui.screens.results_trace_panel import TraceNeuronPanel
from sensoryforge.gui.session import Session
from sensoryforge.gui.settings import gui_settings
from sensoryforge.io.bundle import load_bundle

#: Panel keys, in the order the visibility picker shows them.
PANEL_KEYS = ("stimulus", "raster", "rate", "trace", "map")
PANEL_TITLES = {
    "stimulus": "Stimulus",
    "raster": "Raster",
    "rate": "Rate",
    "trace": "Neuron trace",
    "map": "Neuron map",
}

_SETTINGS_PREFIX = "gui/results_screen/panel_visible/"


class ResultsScreen(QtWidgets.QWidget):
    """The Run & Results stage: live or bundled, one shared time cursor."""

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._view: Optional[results_data.ResultsView] = None
        #: Set while a saved bundle is being viewed instead of live results.
        self._viewing_bundle = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.stale_banner = QtWidgets.QLabel(
            "Results are from before your last edits — Run again"
        )
        self.stale_banner.setStyleSheet(
            f"background: {theme.PALETTE['warning']}; color: white; padding: 6px;"
        )
        self.stale_banner.setVisible(False)
        root.addWidget(self.stale_banner)

        self.bundle_banner = QtWidgets.QWidget()
        bundle_layout = QtWidgets.QHBoxLayout(self.bundle_banner)
        bundle_layout.setContentsMargins(6, 4, 6, 4)
        self.bundle_banner_label = QtWidgets.QLabel("")
        bundle_layout.addWidget(self.bundle_banner_label)
        bundle_layout.addStretch(1)
        self.back_to_live_button = QtWidgets.QPushButton("Back to live results")
        self.back_to_live_button.clicked.connect(self.show_live_results)
        bundle_layout.addWidget(self.back_to_live_button)
        self.bundle_banner.setStyleSheet(
            f"background: {theme.PALETTE['accent_subtle']};"
        )
        self.bundle_banner.setVisible(False)
        root.addWidget(self.bundle_banner)

        self.error_banner = QtWidgets.QLabel("")
        self.error_banner.setStyleSheet(
            f"background: {theme.PALETTE['error']}; color: white; padding: 6px;"
        )
        self.error_banner.setVisible(False)
        root.addWidget(self.error_banner)

        toolbar = QtWidgets.QHBoxLayout()
        self.open_bundle_button = QtWidgets.QPushButton("Open bundle...")
        self.open_bundle_button.clicked.connect(self._on_open_bundle_clicked)
        toolbar.addWidget(self.open_bundle_button)
        toolbar.addStretch(1)
        toolbar_widget = QtWidgets.QWidget()
        toolbar_widget.setLayout(toolbar)
        root.addWidget(toolbar_widget)

        self.empty_state = QtWidgets.QLabel(
            "No results yet. Set up sensors, stimulus and populations, then press Run."
        )
        self.empty_state.setAlignment(theme.QtCore.Qt.AlignCenter)
        self.empty_state.setStyleSheet(
            f"color: {theme.PALETTE['text_secondary']}; padding: 24px;"
        )
        root.addWidget(self.empty_state)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QHBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)

        self.stimulus_panel = StimulusFramePanel()
        self.raster_panel = RasterPanel()
        self.rate_panel = RatePanel()
        self.trace_panel = TraceNeuronPanel()
        self.map_panel = NeuronMapPanel()
        self.map_panel.neuronClicked.connect(self.trace_panel.select_neuron)

        self._panel_widgets: Dict[str, QtWidgets.QWidget] = {
            "stimulus": self.stimulus_panel,
            "raster": self.raster_panel,
            "rate": self.rate_panel,
            "trace": self.trace_panel,
            "map": self.map_panel,
        }

        self.grid = QtWidgets.QGridLayout()
        content_layout.addLayout(self.grid, stretch=1)

        picker_box = QtWidgets.QGroupBox("Panels")
        picker_layout = QtWidgets.QVBoxLayout(picker_box)
        self._checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        settings = gui_settings()
        for key in PANEL_KEYS:
            checkbox = QtWidgets.QCheckBox(PANEL_TITLES[key])
            visible = settings.value(_SETTINGS_PREFIX + key, True, type=bool)
            checkbox.setChecked(bool(visible))
            checkbox.toggled.connect(
                lambda on, k=key: self._on_visibility_toggled(k, on)
            )
            picker_layout.addWidget(checkbox)
            self._checkboxes[key] = checkbox
        picker_layout.addStretch(1)
        content_layout.addWidget(picker_box)

        root.addWidget(content, stretch=1)

        self.playback = PlaybackBar()
        self.playback.frameChanged.connect(self._on_frame_changed)
        root.addWidget(self.playback)

        self._relayout_panels()

        session.resultsChanged.connect(self._on_results_changed)
        session.staleChanged.connect(self._on_stale_changed)
        session.configReplaced.connect(self._on_config_replaced)

        self.stale_banner.setVisible(session.stale)
        if session.last_results is not None:
            self._set_view(results_data.from_run_result(session.last_results))

    # ----------------------------------------------------------------- layout

    def _relayout_panels(self) -> None:
        """Place every visible panel in the fixed grid, hidden ones excluded."""
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)

        # Space on the left, time on the right: the two square spatial panels
        # share a column, and the three time-series panels (which share the
        # playback cursor) stack in a wider one. (row, col, row span).
        positions = {
            "stimulus": (0, 0, 3),
            "map": (3, 0, 3),
            "raster": (0, 1, 2),
            "rate": (2, 1, 2),
            "trace": (4, 1, 2),
        }
        for grid_row in range(6):
            self.grid.setRowStretch(grid_row, 1)
        self.grid.setColumnStretch(0, 2)
        self.grid.setColumnStretch(1, 3)
        for key in PANEL_KEYS:
            widget = self._panel_widgets[key]
            visible = self._checkboxes[key].isChecked()
            widget.setVisible(visible)
            if visible:
                row, col, row_span = positions[key]
                self.grid.addWidget(widget, row, col, row_span, 1)

    def _on_visibility_toggled(self, key: str, on: bool) -> None:
        gui_settings().setValue(_SETTINGS_PREFIX + key, on)
        self._relayout_panels()

    # ------------------------------------------------------------------ data

    def _on_results_changed(self, result) -> None:
        if self._viewing_bundle:
            return
        if result is None:
            self._set_view(None)
            return
        self._set_view(results_data.from_run_result(result))

    def _on_stale_changed(self, stale: bool) -> None:
        if not self._viewing_bundle:
            self.stale_banner.setVisible(stale)

    def _on_config_replaced(self) -> None:
        if self._viewing_bundle:
            self.show_live_results()

    def _set_view(self, view: Optional[results_data.ResultsView]) -> None:
        self._view = view
        self.empty_state.setVisible(view is None)
        self.error_banner.setVisible(False)
        for widget in (
            self.stimulus_panel,
            self.raster_panel,
            self.rate_panel,
            self.trace_panel,
            self.map_panel,
        ):
            widget.set_view(view)
        if view is None:
            self.playback.set_range(0, 1.0)
            return
        time_ms = view.time_ms
        dt_ms = float(time_ms[1] - time_ms[0]) if len(time_ms) > 1 else 1.0
        self.playback.set_range(len(time_ms), dt_ms)

    def _on_frame_changed(self, index: int) -> None:
        if self._view is None or index >= len(self._view.time_ms):
            return
        self.stimulus_panel.set_frame(index)
        time_value = float(self._view.time_ms[index])
        self.raster_panel.set_cursor(time_value)
        self.rate_panel.set_cursor(time_value)
        self.trace_panel.set_cursor(time_value)

    # --------------------------------------------------------------- bundles

    def _on_open_bundle_clicked(self) -> None:
        run_dirs = self._session.project.list_runs() if self._session.project else []
        dialog = OpenBundleDialog(run_dirs, self)
        if (
            dialog.exec_() == QtWidgets.QDialog.Accepted
            and dialog.selected_path is not None
        ):
            self.open_bundle(dialog.selected_path)

    def open_bundle(self, path: Path) -> None:
        """Load and show the bundle at ``path``, or show its loader error.

        Args:
            path: A bundle directory.
        """
        try:
            bundle = load_bundle(path)
        except (ValueError, FileNotFoundError, ImportError) as exc:
            self.error_banner.setText(f"Could not open {path}: {exc}")
            self.error_banner.setVisible(True)
            return
        self.error_banner.setVisible(False)
        view = results_data.from_bundle(
            bundle, label=f"Viewing saved run {Path(path).name}"
        )
        self._viewing_bundle = True
        self.stale_banner.setVisible(False)
        self.bundle_banner_label.setText(view.label)
        self.bundle_banner.setVisible(True)
        self._set_view(view)

    def show_live_results(self) -> None:
        """Stop viewing a saved bundle and go back to the session's own results."""
        self._viewing_bundle = False
        self.bundle_banner.setVisible(False)
        result = self._session.last_results
        self._set_view(
            results_data.from_run_result(result) if result is not None else None
        )
        self.stale_banner.setVisible(self._session.stale)
