"""The Stimulus screen: one ``StimulusConfig`` form + a live render preview.

``StimulusScreen`` is GUI v2's Task 2.2 screen (``docs_root/gui_audit/10_plan.md``,
Phase 2). Left (~60%): stimulus type (from ``STIMULUS_REGISTRY`` plus the
legacy names :mod:`sensoryforge.stimuli.render` still accepts), name, target
grid, target channel, and the selected type's own parameter form
(:class:`~sensoryforge.gui.screens.stimulus_paramform.StimulusParamForm`).
Right (~40%): :class:`~sensoryforge.gui.screens.stimulus_preview.StimulusPreview`,
rendered through :func:`sensoryforge.gui.execution.render.render_for_config`
-- the same renderer a run uses, so the preview is what will actually run.

The one rule that matters here (see the task brief): what the screen shows
must be what runs. ``StimulusConfig.to_dict()`` is sparse -- a field the user
never touched is not in it, and the selected stimulus type's own constructor
default applies at render time, not the schema's default. The parameter form
shows that distinction; this module only wires the type/name/grid/channel
selectors and hands everything else to
:class:`~sensoryforge.gui.screens.stimulus_paramform.StimulusParamForm`.
"""

from __future__ import annotations

from typing import List, Optional

from PyQt5 import QtWidgets

from sensoryforge.gui.screens.stimulus_layers import LayerEditor
from sensoryforge.gui.screens.stimulus_substimuli import SubStimulusEditor
from sensoryforge.gui.widgets.problem_list import ProblemList
from sensoryforge.gui.screens.stimulus_paramform import (
    LEGACY_STIMULUS_TYPES,
    StimulusParamForm,
)
from sensoryforge.gui.screens.stimulus_preview import StimulusPreview
from sensoryforge.gui.session import Session
from sensoryforge.gui.settings import gui_settings
from sensoryforge.gui.widgets.collapsible import CollapsibleGroupBox
from sensoryforge.registry import STIMULUS_REGISTRY

_ADVANCED_KEY = "gui/advanced"

#: The one channel name a grid with no explicit ``channels:`` block carries
#: (``GridConfig.channels`` default). Shown when a grid has only this one
#: channel -- there is nothing meaningful to pick.
_DEFAULT_CHANNEL = "value"


class StimulusScreen(QtWidgets.QWidget):
    """The Stimulus stage of the GUI v2 stage navigation.

    Args:
        session: The one experiment this screen edits.
        parent: Qt parent.
    """

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._session = session

        outer = QtWidgets.QHBoxLayout(self)
        splitter = QtWidgets.QSplitter()
        outer.addWidget(splitter)

        editor = QtWidgets.QWidget()
        editor_layout = QtWidgets.QVBoxLayout(editor)
        self.problems = ProblemList(session, "stimulus")
        editor_layout.addWidget(self.problems)

        selector_box = CollapsibleGroupBox("Stimulus", start_expanded=True)
        editor_layout.addWidget(selector_box)

        self.type_combo = QtWidgets.QComboBox()
        self._populate_type_combo()
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        selector_box.addRow("Type", self.type_combo)

        self.name_edit = QtWidgets.QLineEdit(session.config.stimulus.name)
        self.name_edit.editingFinished.connect(self._on_name_edited)
        selector_box.addRow("Name", self.name_edit)

        self.grid_combo = QtWidgets.QComboBox()
        self.grid_combo.currentIndexChanged.connect(self._on_grid_changed)
        selector_box.addRow("Target grid", self.grid_combo)

        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        selector_box.addRow("Target channel", self.channel_combo)

        self._refresh_grid_combo()

        # Composite and timeline stimuli are built from sub-stimuli.
        self.sub_stimuli = SubStimulusEditor(session)
        editor_layout.addWidget(self.sub_stimuli)

        # Layered stimuli are built from a stack of layers.
        self.layer_editor = LayerEditor(session)
        editor_layout.addWidget(self.layer_editor, 1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.param_form = StimulusParamForm(
            session,
            advanced=bool(gui_settings().value(_ADVANCED_KEY, False, type=bool)),
        )
        scroll.setWidget(self.param_form)
        editor_layout.addWidget(scroll, 1)
        self._param_scroll = scroll
        self._show_editor_for_type()

        splitter.addWidget(editor)

        self.preview = StimulusPreview(session)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        session.configChanged.connect(self._on_config_changed)
        session.configReplaced.connect(self._on_config_replaced)

        self.preview.render_now()

    # ------------------------------------------------------------- type combo

    def _populate_type_combo(self) -> None:
        current = self._session.config.stimulus.type
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        # "layered" first: it is the preferred, general way to build a
        # stimulus (a stack of shape/pattern/motion/timing layers) -- every
        # other registered name is a fixed special case of it.
        registered = STIMULUS_REGISTRY.list_registered()
        ordered = ([n for n in registered if n == "layered"]) + [
            name for name in registered if name != "layered"
        ]
        for name in ordered:
            self.type_combo.addItem(name, name)
        self.type_combo.insertSeparator(self.type_combo.count())
        legacy_start = self.type_combo.count()
        for name in LEGACY_STIMULUS_TYPES:
            self.type_combo.addItem(f"{name} (legacy)", name)
        # A separator item has no userData; findData would never match it,
        # so it is otherwise inert -- but disable it too, so it cannot be
        # arrow-keyed onto.
        model = self.type_combo.model()
        separator_item = model.item(legacy_start - 1)
        if separator_item is not None:
            separator_item.setEnabled(False)
        idx = self.type_combo.findData(current)
        self.type_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.type_combo.blockSignals(False)

    def _on_type_changed(self, _index: int) -> None:
        new_type = self.type_combo.currentData()
        stim = self._session.config.stimulus
        if new_type is None or new_type == stim.type:
            return
        # `stimulus.params` holds the OLD type's own parameters (a Braille
        # `v_mms`, an edge_grating `count`, ...); the new type's constructor
        # either rejects an unknown key outright or, worse, silently accepts
        # a same-named key with a different meaning. Clear it so switching
        # types never carries stale parameters across.
        if stim.params:
            stim.params.clear()
            self._session.notify("stimulus.params")
        self._session.set_by_path("stimulus.type", new_type)

    # ------------------------------------------------------------- name field

    def _on_name_edited(self) -> None:
        text = self.name_edit.text()
        if text == self._session.config.stimulus.name:
            return
        self._session.set_by_path("stimulus.name", text)

    # ------------------------------------------------------------- grid combo

    def _refresh_grid_combo(self) -> None:
        stim = self._session.config.stimulus
        names: List[str] = [g.name for g in self._session.config.grids]
        self.grid_combo.blockSignals(True)
        self.grid_combo.clear()
        for name in names:
            self.grid_combo.addItem(name, name)
        target = (
            stim.target_layer
            if stim.target_layer in names
            else (names[0] if names else None)
        )
        idx = self.grid_combo.findData(target) if target is not None else -1
        self.grid_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.grid_combo.blockSignals(False)
        self._refresh_channel_combo()

    def _on_grid_changed(self, _index: int) -> None:
        name = self.grid_combo.currentData()
        if name is None or name == self._session.config.stimulus.target_layer:
            self._refresh_channel_combo()
            return
        self._session.set_by_path("stimulus.target_layer", name)
        self._refresh_channel_combo()

    # ---------------------------------------------------------- channel combo

    def _current_grid_channels(self) -> List[str]:
        name = self.grid_combo.currentData()
        for grid in self._session.config.grids:
            if grid.name == name:
                return list(grid.channels)
        return [_DEFAULT_CHANNEL]

    def _refresh_channel_combo(self) -> None:
        stim = self._session.config.stimulus
        channels = self._current_grid_channels()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        for name in channels:
            self.channel_combo.addItem(name, name)
        self.channel_combo.setEnabled(len(channels) > 1)
        target = (
            stim.channel
            if stim.channel in channels
            else (channels[0] if channels else None)
        )
        idx = self.channel_combo.findData(target) if target is not None else -1
        self.channel_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.channel_combo.blockSignals(False)

    def _on_channel_changed(self, _index: int) -> None:
        name = self.channel_combo.currentData()
        if name is None or name == self._session.config.stimulus.channel:
            return
        self._session.set_by_path("stimulus.channel", name)

    # -------------------------------------------------------------- session

    def _show_editor_for_type(self) -> None:
        """A layered stimulus is edited in the layer editor, which takes the
        column; every other type uses the parameter form."""
        layered = self._session.config.stimulus.type == "layered"
        self._param_scroll.setVisible(not layered)

    def _on_config_changed(self, path: str) -> None:
        if path == "stimulus.type":
            self._show_editor_for_type()
            idx = self.type_combo.findData(self._session.config.stimulus.type)
            self.type_combo.blockSignals(True)
            self.type_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.type_combo.blockSignals(False)
        elif path == "stimulus.name":
            if self.name_edit.text() != self._session.config.stimulus.name:
                self.name_edit.setText(self._session.config.stimulus.name)
        elif path == "stimulus.target_layer":
            self._refresh_grid_combo()
        elif path == "stimulus.channel":
            self._refresh_channel_combo()
        elif path.startswith("grids."):
            self._refresh_grid_combo()

    def _on_config_replaced(self) -> None:
        self._show_editor_for_type()
        self._populate_type_combo()
        self.name_edit.setText(self._session.config.stimulus.name)
        self._refresh_grid_combo()

    # -------------------------------------------------------------- public

    def set_advanced(self, on: bool) -> None:
        """Show or hide advanced parameter rows (session-wide Advanced toggle)."""
        self.param_form.set_advanced(on)
