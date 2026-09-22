"""The GUI v2 app shell: one window, one ``Session``, no tabs and no graph.

``SensoryForgeApp`` is the window every screen plugs into: a menu, a toolbar
with an Advanced toggle, a left stage-navigation list driving a
``QStackedWidget``, a ``PipelineStrip`` above it, and a ``RunBar`` below.
Screens are placeholders (``sensoryforge.gui.screens.SCREEN_FACTORIES``)
until Phase 2 replaces them one at a time.

Run with ``python -m sensoryforge.gui.app``.
"""

from __future__ import annotations

import os
import sys
import traceback
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
import torch
from PyQt5 import QtCore, QtGui, QtWidgets

from sensoryforge.gui.execution.threads import wait_all as wait_for_worker_threads
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.gui import theme
from sensoryforge.gui.execution.run_controller import RunController
from sensoryforge.gui.project import ProjectHandle
from sensoryforge.gui.screens import SCREEN_FACTORIES
from sensoryforge.gui.session import Session
from sensoryforge.gui.settings import gui_settings
from sensoryforge.gui.widgets.pipeline_strip import PipelineStrip
from sensoryforge.gui.widgets.run_bar import RunBar

#: Documentation site (Help > Documentation).
DOCS_URL = "https://benefron.github.io/sensoryforge/"

#: Bundled presets directory (``sensoryforge/presets/*.yml``).
PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"

#: The GUI's own fallback experiment when there is no last project: the
#: pressure-simulation recipe, so the window is never empty.
DEFAULT_PRESET = PRESETS_DIR / "tactile_sa1_ra1.yml"

#: Left-hand stage navigation, in display order.
STAGE_ORDER = ["sensors", "stimulus", "populations", "results", "batch"]

STAGE_LABELS: Dict[str, str] = {
    "sensors": "Sensors",
    "stimulus": "Stimulus",
    "populations": "Populations",
    "results": "Run & Results",
    "batch": "Batch",
}

#: Which stage in the nav a pipeline-strip chip's stage belongs to.
CHIP_STAGE_TO_NAV_STAGE: Dict[str, str] = {
    "sensors": "sensors",
    "sensor_array": "populations",
    "receptive_field": "populations",
    "combine": "populations",
    "filter": "populations",
    "neuron": "populations",
    "readout": "populations",
}

_GEOMETRY_KEY = "gui/window_geometry"
_LAST_PROJECT_KEY = "gui/last_project"
_ADVANCED_KEY = "gui/advanced"


class SensoryForgeApp(QtWidgets.QMainWindow):
    """The GUI v2 main window.

    Signals:
        advancedChanged(bool): The Advanced toggle changed; screens hide or
            show advanced parameter rows in response.
        stageSelected(str, int): A chip in the pipeline strip was clicked and
            the stage nav followed it -- the nav stage name and, for a
            population chip, the population index (for a sensors chip, the
            grid index).

    Args:
        session: The one experiment this window edits and runs.
        parent: Qt parent.
    """

    advancedChanged = QtCore.pyqtSignal(bool)
    stageSelected = QtCore.pyqtSignal(str, int)

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._session = session
        #: Guard so tests can disable ``QMessageBox`` dialogs.
        self.show_dialogs = True

        self.setObjectName("AppRoot")
        self.setWindowTitle("SensoryForge")

        self._build_menu()
        self._build_toolbar()
        self._build_central()

        session.projectChanged.connect(self._on_project_changed)
        self._on_project_changed(session.project)

        self._restore_geometry()

    # --------------------------------------------------------------- menu

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        file_menu.addAction("New project…", self._on_new_project)
        file_menu.addAction("Open project…", self._on_open_project)
        file_menu.addAction("Open config (YAML)…", self._on_open_config)

        preset_menu = file_menu.addMenu("Open preset")
        for preset_path in sorted(PRESETS_DIR.glob("*.yml")):
            preset_menu.addAction(
                preset_path.stem, partial(self._load_config_file, str(preset_path))
            )

        save_action = file_menu.addAction("Save config", self._on_save_config)
        save_action.setShortcut(QtGui.QKeySequence.Save)
        file_menu.addAction("Save config as…", self._on_save_config_as)

        export_menu = file_menu.addMenu("Export")
        export_menu.addAction("YAML…", self._on_export_yaml_dialog)

        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("About", self._on_about)
        help_menu.addAction("Documentation", self._on_open_docs)

    # ------------------------------------------------------------- toolbar

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)

        self._project_label = QtWidgets.QLabel("")
        toolbar.addWidget(self._project_label)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        toolbar.addWidget(spacer)

        self.advanced_check = QtWidgets.QCheckBox("Advanced")
        self.advanced_check.setChecked(
            bool(gui_settings().value(_ADVANCED_KEY, False, type=bool))
        )
        self.advanced_check.toggled.connect(self._on_advanced_toggled)
        toolbar.addWidget(self.advanced_check)

    # ------------------------------------------------------------- central

    def _build_central(self) -> None:
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.pipeline_strip = PipelineStrip(self._session)
        self.pipeline_strip.chipClicked.connect(self._on_chip_clicked)
        outer.addWidget(self.pipeline_strip)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.stage_list = QtWidgets.QListWidget()
        for stage in STAGE_ORDER:
            self.stage_list.addItem(STAGE_LABELS[stage])
        self.stage_list.currentRowChanged.connect(self._on_stage_row_changed)
        # Five short labels: a fixed narrow column, so the screens get the room.
        self.stage_list.setFixedWidth(168)
        splitter.addWidget(self.stage_list)

        self.stack = QtWidgets.QStackedWidget()
        self._screens: Dict[str, QtWidgets.QWidget] = {}
        for stage in STAGE_ORDER:
            widget = SCREEN_FACTORIES[stage](self._session, None)
            self._screens[stage] = widget
            self.stack.addWidget(widget)
            # The Advanced convention (see ``screens/__init__.py``).
            set_advanced = getattr(widget, "set_advanced", None)
            if callable(set_advanced):
                set_advanced(self.advanced_check.isChecked())
                self.advancedChanged.connect(set_advanced)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)

        self.run_bar = RunBar(self._session)
        # One controller per window: every run the shell performs (the run
        # bar's, and Phase 2's per-population Quick run) goes through it, so
        # the GUI can never acquire a second execution path that drifts from
        # ``SimulationEngine.run`` (the guard is
        # ``tests/integration/test_gui_engine_equality.py``).
        self.run_controller = RunController(self._session, self)
        self.run_bar.set_controller(self.run_controller)
        outer.addWidget(self.run_bar)

        self.setCentralWidget(central)
        self.stage_list.setCurrentRow(0)

    # ---------------------------------------------------------- navigation

    def _on_stage_row_changed(self, row: int) -> None:
        if 0 <= row < self.stack.count():
            self.stack.setCurrentIndex(row)

    def _on_chip_clicked(self, stage: str, index: int) -> None:
        nav_stage = CHIP_STAGE_TO_NAV_STAGE.get(stage, "populations")
        row = STAGE_ORDER.index(nav_stage)
        self.stage_list.setCurrentRow(row)
        self.stageSelected.emit(nav_stage, index)

    # --------------------------------------------------------------- advanced

    def _on_advanced_toggled(self, checked: bool) -> None:
        gui_settings().setValue(_ADVANCED_KEY, checked)
        self.advancedChanged.emit(checked)

    # ---------------------------------------------------------------- config

    def _load_config_file(self, path: str) -> None:
        """Load a YAML config (or preset) and put it in the session.

        Args:
            path: Path to a YAML file readable by
                :meth:`SensoryForgeConfig.from_yaml_file`.
        """
        try:
            config = SensoryForgeConfig.from_yaml_file(Path(path))
        except _LOAD_ERRORS as exc:
            if self.show_dialogs:
                QtWidgets.QMessageBox.critical(self, "Could not load config", str(exc))
            return
        self._session.replace_config(config)
        self.pipeline_strip.mark_saved()

    def _on_open_config(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open config", "", "YAML (*.yml *.yaml)"
        )
        if path:
            self._load_config_file(path)

    # --------------------------------------------------------------- project

    def _new_project(self, root: str) -> None:
        """Create a project at ``root`` from the current config (File > New).

        Args:
            root: Directory to create. Must not already hold a
                ``config.yml`` -- see :meth:`ProjectHandle.create`.
        """
        project = ProjectHandle.create(Path(root), self._session.config)
        self._session.set_project(project)
        gui_settings().setValue(_LAST_PROJECT_KEY, str(project.root))

    def _on_new_project(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "New project")
        if not path:
            return
        try:
            self._new_project(path)
        except ValueError as exc:
            if self.show_dialogs:
                QtWidgets.QMessageBox.critical(
                    self, "Could not create project", str(exc)
                )

    def _open_project(self, root: str) -> None:
        """Open an existing project at ``root`` and load its config.

        Args:
            root: Directory holding a ``config.yml``.
        """
        project = ProjectHandle.open(Path(root))
        self._session.set_project(project)
        self._session.replace_config(project.load_config())
        self.pipeline_strip.mark_saved()
        gui_settings().setValue(_LAST_PROJECT_KEY, str(project.root))

    def _on_open_project(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open project")
        if not path:
            return
        try:
            self._open_project(path)
        except _LOAD_ERRORS as exc:
            if self.show_dialogs:
                QtWidgets.QMessageBox.critical(self, "Could not open project", str(exc))

    # ----------------------------------------------------------- save config

    def _on_save_config(self) -> None:
        if self._session.project is not None:
            self._session.project.save_config(self._session.config)
            self.pipeline_strip.mark_saved()
        else:
            self._on_save_config_as()

    def _save_config_as(self, root: str) -> None:
        """Save the current config into ``root``, as a project.

        Args:
            root: A directory -- either a fresh one, or an existing project
                (its ``config.yml`` is overwritten and it becomes the open
                project).
        """
        root_path = Path(root)
        if (root_path / "config.yml").is_file():
            project = ProjectHandle.open(root_path)
            project.save_config(self._session.config)
        else:
            project = ProjectHandle.create(root_path, self._session.config)
        self._session.set_project(project)
        self.pipeline_strip.mark_saved()
        gui_settings().setValue(_LAST_PROJECT_KEY, str(project.root))

    def _on_save_config_as(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Save config as")
        if path:
            self._save_config_as(path)

    # ------------------------------------------------------------ export yaml

    def _export_yaml(self, path: str) -> None:
        """Write ``session.config.to_yaml()`` to ``path``.

        Args:
            path: File to write (not necessarily inside a project).
        """
        Path(path).write_text(self._session.config.to_yaml(), encoding="utf-8")

    def _on_export_yaml_dialog(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export YAML", "", "YAML (*.yml)"
        )
        if path:
            self._export_yaml(path)

    # -------------------------------------------------------------- project ui

    def _on_project_changed(self, project: Optional[ProjectHandle]) -> None:
        self._project_label.setText(project.root.name if project else "(no project)")

    # ------------------------------------------------------------------- help

    def _on_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About SensoryForge",
            "SensoryForge -- sensory encoding simulation workbench.",
        )

    def _on_open_docs(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(DOCS_URL))

    # ---------------------------------------------------------------- window

    def _restore_geometry(self) -> None:
        geometry = gui_settings().value(_GEOMETRY_KEY)
        if geometry is not None:
            self.restoreGeometry(geometry)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Persist window geometry, stop a run, and join worker threads."""
        gui_settings().setValue(_GEOMETRY_KEY, self.saveGeometry())
        self.run_controller.cancel()
        wait_for_worker_threads()
        super().closeEvent(event)


def _excepthook(exc_type: type, exc_value: BaseException, exc_tb: object) -> None:
    """Show unhandled exceptions in a QMessageBox instead of aborting (F-044).

    PyQt5 aborts the process by default when an exception escapes a Qt slot;
    installed as ``sys.excepthook`` so anything a specific handler did not
    already catch is at least visible and recoverable.
    """
    traceback.print_exception(exc_type, exc_value, exc_tb)
    QtWidgets.QMessageBox.critical(
        None, "Unexpected error", f"{exc_type.__name__}: {exc_value}"
    )


#: What reading a config can raise: a bad file, bad YAML, or a config the
#: schema rejects.
_LOAD_ERRORS = (ValueError, OSError, TypeError, KeyError, yaml.YAMLError)


def _initial_config() -> Tuple[SensoryForgeConfig, Optional[ProjectHandle]]:
    """The config (and project, if any) to start the window with.

    Reopens ``gui/last_project`` if it still exists on disk; otherwise loads
    the ``tactile_sa1_ra1`` preset, so the window is never empty.
    """
    last_project = gui_settings().value(_LAST_PROJECT_KEY, "", type=str)
    if last_project:
        try:
            project = ProjectHandle.open(last_project)
            return project.load_config(), project
        except _LOAD_ERRORS as exc:
            # A broken last project must not stop the app from starting.
            print(f"Could not reopen {last_project}: {exc}", file=sys.stderr)
    return SensoryForgeConfig.from_yaml_file(DEFAULT_PRESET), None


def main() -> None:
    """Launch the SensoryForge GUI v2 application."""
    sys.excepthook = _excepthook
    app = QtWidgets.QApplication(sys.argv)
    theme.apply(app)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    config, project = _initial_config()
    session = Session(config)
    if project is not None:
        session.set_project(project)

    window = SensoryForgeApp(session)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
