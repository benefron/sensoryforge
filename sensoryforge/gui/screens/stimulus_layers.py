"""Layer editor for layered stimuli (``stimulus.type == "layered"``).

A layered stimulus (:mod:`sensoryforge.stimuli.layered`) is a list of layer
dicts plus a ``combine`` mode. :class:`LayerEditor` is the GUI for that list:
a layer list (add/duplicate/remove/move/enable) on the left, and four
sections for the selected layer's **shape**, **pattern**, **motion** and
**timing** on the right, each a kind combo (shape/pattern/motion only) plus a
:class:`~sensoryforge.gui.widgets.param_form.ParamForm` built from that
kind's :data:`~sensoryforge.stimuli.layered.SHAPES`/``PATTERNS``/``MOTIONS``
spec list, bound straight to the layer's own dict at
``stimulus.layers.<i>.<part>``.

Mirrors :class:`~sensoryforge.gui.screens.stimulus_substimuli.SubStimulusEditor`:
hidden for every other stimulus type, every structural edit (add, remove,
move, change a part's kind) rewrites the whole ``layers`` list through
:meth:`~sensoryforge.gui.session.Session.set_by_path`, and a change made
elsewhere (undo, a loaded config) is picked up through
``session.configChanged``/``configReplaced`` and reloads the widget.
"""

from __future__ import annotations

import copy
import functools
from typing import Any, Dict, List, Optional

from PyQt5 import QtCore, QtWidgets

from sensoryforge.config.defaults import resolve_duration_ms
from sensoryforge.gui.session import Session
from sensoryforge.gui.widgets.collapsible import CollapsibleGroupBox
from sensoryforge.gui.widgets.param_form import ParamForm
from sensoryforge.stimuli.layered import (
    COMBINE_MODES,
    MOTIONS,
    PATTERNS,
    SHAPES,
    TIMING_SPECS,
    default_layer,
    defaults,
)
from sensoryforge.stimuli.presets import PRESETS, preset

#: Part name -> its kind -> ParamSpec list.
_PART_SPECS = {"shape": SHAPES, "pattern": PATTERNS, "motion": MOTIONS}

#: Part name -> default kind used when a layer is missing that part.
_PART_DEFAULT_KIND = {"shape": "gaussian", "pattern": "single", "motion": "none"}


def _fmt_ms(value: Any) -> str:
    return "auto" if value is None else f"{float(value):g}"


def layer_summary(layer: Dict[str, Any]) -> str:
    """A one-line description of a layer, for the layer list row.

    Args:
        layer: A layer dict (``shape``/``pattern``/``motion``/``timing``).

    Returns:
        e.g. ``"disc x braille 'hello', linear, 50/auto/50 ms"``.
    """
    shape = layer.get("shape") or {}
    pattern = layer.get("pattern") or {}
    motion = layer.get("motion") or {}
    timing = {**defaults(TIMING_SPECS), **(layer.get("timing") or {})}
    bits = [f"{shape.get('kind', '?')} x {pattern.get('kind', '?')}"]
    if pattern.get("kind") == "braille" and pattern.get("text"):
        bits[0] += f" {pattern['text']!r}"
    bits.append(motion.get("kind", "none"))
    bits.append(
        f"{_fmt_ms(timing['ramp_up_ms'])}/{_fmt_ms(timing['hold_ms'])}/"
        f"{_fmt_ms(timing['ramp_down_ms'])} ms"
    )
    return ", ".join(bits)


class LayerEditor(QtWidgets.QGroupBox):
    """Editor for the session's layered stimulus (``stimulus.layers``/``combine``).

    Hidden for every other stimulus type.

    Args:
        session: The experiment.
        parent: Qt parent.
    """

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__("Layers", parent)
        self._session = session
        self._loading = False
        # True while this editor writes; a rebuild mid-signal-delivery would
        # delete the widget whose signal is being handled (F-035 discipline,
        # mirrors SubStimulusEditor._writing).
        self._writing = False
        self._current_index = 0
        self._part_forms: Dict[str, Optional[ParamForm]] = {
            "shape": None,
            "pattern": None,
            "motion": None,
            "timing": None,
        }

        outer = QtWidgets.QVBoxLayout(self)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Combine overlapping as"))
        self.combine_combo = QtWidgets.QComboBox()
        self.combine_combo.addItems(list(COMBINE_MODES))
        self.combine_combo.currentTextChanged.connect(self._on_combine_changed)
        top_row.addWidget(self.combine_combo)
        top_row.addSpacing(16)
        top_row.addWidget(QtWidgets.QLabel("Start from preset…"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("(choose a preset)", None)
        for name in sorted(PRESETS):
            self.preset_combo.addItem(name, name)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_chosen)
        top_row.addWidget(self.preset_combo)
        top_row.addStretch(1)
        outer.addLayout(top_row)

        # The layer list on top, the selected layer's forms below at the full
        # width of the column.
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        outer.addWidget(splitter, 1)

        # ---------------------------------------------------------- list

        list_panel = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout(list_panel)
        list_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_list = QtWidgets.QListWidget()
        self.layer_list.currentRowChanged.connect(self._on_row_selected)
        self.layer_list.itemChanged.connect(self._on_item_changed)
        list_layout.addWidget(self.layer_list, 1)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_add.clicked.connect(self._on_add)
        self.btn_duplicate = QtWidgets.QPushButton("Duplicate")
        self.btn_duplicate.clicked.connect(self._on_duplicate)
        self.btn_remove = QtWidgets.QPushButton("Remove")
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_up = QtWidgets.QPushButton("Move up")
        self.btn_up.clicked.connect(self._on_move_up)
        self.btn_down = QtWidgets.QPushButton("Move down")
        self.btn_down.clicked.connect(self._on_move_down)
        for button in (
            self.btn_add,
            self.btn_duplicate,
            self.btn_remove,
            self.btn_up,
            self.btn_down,
        ):
            buttons.addWidget(button)
        list_layout.addLayout(buttons)
        splitter.addWidget(list_panel)

        # ------------------------------------------------------- part editor

        detail_scroll = QtWidgets.QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_panel = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(detail_panel)

        self._kind_combos: Dict[str, QtWidgets.QComboBox] = {}
        self._part_placeholders: Dict[str, QtWidgets.QWidget] = {}
        self._part_boxes: Dict[str, CollapsibleGroupBox] = {}
        for part in ("shape", "pattern", "motion"):
            box = CollapsibleGroupBox(part.capitalize(), start_expanded=True)
            combo = QtWidgets.QComboBox()
            combo.addItems(sorted(_PART_SPECS[part]))
            combo.currentTextChanged.connect(functools.partial(self._on_kind, part))
            box.addRow("Kind", combo)
            self._kind_combos[part] = combo
            # A placeholder row whose widget is swapped for a fresh
            # ParamForm on every layer/kind change (QFormLayout has no
            # "replace row" call, so the row itself is created once here).
            placeholder = QtWidgets.QWidget()
            QtWidgets.QVBoxLayout(placeholder).setContentsMargins(0, 0, 0, 0)
            box.addRow(placeholder)
            self._part_placeholders[part] = placeholder
            self._part_boxes[part] = box
            detail_layout.addWidget(box)

        timing_box = CollapsibleGroupBox("Timing", start_expanded=True)
        timing_placeholder = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(timing_placeholder).setContentsMargins(0, 0, 0, 0)
        timing_box.addRow(timing_placeholder)
        self._part_placeholders["timing"] = timing_placeholder
        self._part_boxes["timing"] = timing_box
        detail_layout.addWidget(timing_box)

        detail_layout.addStretch(1)
        detail_scroll.setWidget(detail_panel)
        splitter.addWidget(detail_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        session.configChanged.connect(self._on_config_changed)
        session.configReplaced.connect(self.reload)
        self.reload()

    # ------------------------------------------------------------------ model

    def _applicable(self) -> bool:
        return self._session.config.stimulus.type == "layered"

    def layers(self) -> List[Dict[str, Any]]:
        """The current layers (a deep copy, safe to mutate and write back)."""
        return copy.deepcopy(self._session.config.stimulus.layers)

    def _write_layers(self, layers: List[Dict[str, Any]], *, keep_index=True) -> None:
        self._writing = True
        try:
            self._session.set_by_path("stimulus.layers", layers)
        finally:
            self._writing = False
        self.reload(keep_index=keep_index)

    # ------------------------------------------------------------------ view

    def reload(self, *_args: object, keep_index: bool = True) -> None:
        """Rebuild the whole editor (list + part forms) from the config."""
        applicable = self._applicable()
        self.setVisible(applicable)
        if not applicable:
            return
        self._loading = True
        try:
            self.combine_combo.setCurrentText(self._session.config.stimulus.combine)
            layers = self._session.config.stimulus.layers
            self.layer_list.clear()
            for layer in layers:
                item = QtWidgets.QListWidgetItem(layer_summary(layer))
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.Checked
                    if layer.get("enabled", True)
                    else QtCore.Qt.Unchecked
                )
                self.layer_list.addItem(item)
            self.btn_remove.setEnabled(len(layers) > 1)
            index = (
                min(self._current_index, len(layers) - 1)
                if keep_index and layers
                else (0 if layers else -1)
            )
            if layers and index >= 0:
                self.layer_list.setCurrentRow(index)
                self._current_index = index
                self._rebuild_part_forms(index)
        finally:
            self._loading = False

    def _rebuild_part_forms(self, index: int) -> None:
        layers = self._session.config.stimulus.layers
        if not 0 <= index < len(layers):
            return
        layer = layers[index]
        for part in ("shape", "pattern", "motion"):
            self._rebuild_one_part(part, index, layer)
        self._rebuild_timing(index, layer)

    def _rebuild_one_part(self, part: str, index: int, layer: Dict[str, Any]) -> None:
        part_dict = layer.get(part) or {"kind": _PART_DEFAULT_KIND[part]}
        kind = part_dict.get("kind", _PART_DEFAULT_KIND[part])
        combo = self._kind_combos[part]
        combo.blockSignals(True)
        idx = combo.findText(kind)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

        placeholder = self._part_placeholders[part]
        old = self._part_forms[part]
        if old is not None:
            placeholder.layout().removeWidget(old)
            old.setVisible(False)
            old.setParent(None)
            old.deleteLater()
        specs = _PART_SPECS[part].get(kind, [])
        form = ParamForm(
            specs,
            part_dict,
            self._session,
            f"stimulus.layers.{index}.{part}",
        )
        placeholder.layout().addWidget(form)
        self._part_forms[part] = form

    def _rebuild_timing(self, index: int, layer: Dict[str, Any]) -> None:
        timing = layer.get("timing")
        if timing is None:
            timing = defaults(TIMING_SPECS)
            layer["timing"] = timing
        placeholder = self._part_placeholders["timing"]
        old = self._part_forms["timing"]
        if old is not None:
            placeholder.layout().removeWidget(old)
            old.setVisible(False)
            old.setParent(None)
            old.deleteLater()
        form = ParamForm(
            list(TIMING_SPECS),
            timing,
            self._session,
            f"stimulus.layers.{index}.timing",
        )
        placeholder.layout().addWidget(form)
        self._part_forms["timing"] = form

    # --------------------------------------------------------------- editing

    def _on_row_selected(self, row: int) -> None:
        if self._loading or row < 0:
            return
        self._current_index = row
        self._rebuild_part_forms(row)

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._loading:
            return
        row = self.layer_list.row(item)
        layers = self.layers()
        if not 0 <= row < len(layers):
            return
        enabled = item.checkState() == QtCore.Qt.Checked
        if layers[row].get("enabled", True) == enabled:
            return
        layers[row]["enabled"] = enabled
        self._write_layers(layers)

    def _on_kind(self, part: str, kind: str) -> None:
        if self._loading or not kind:
            return
        layers = self.layers()
        index = self._current_index
        if not 0 <= index < len(layers):
            return
        layer = layers[index]
        current_kind = (layer.get(part) or {}).get("kind")
        if current_kind == kind:
            return
        new_part: Dict[str, Any] = {"kind": kind}
        new_part.update(defaults(_PART_SPECS[part][kind]))
        layer[part] = new_part
        self._write_layers(layers)

    def _on_combine_changed(self, mode: str) -> None:
        if self._loading or not self._applicable():
            return
        self._writing = True
        try:
            self._session.set_by_path("stimulus.combine", mode)
        finally:
            self._writing = False

    def _on_preset_chosen(self, _index: int) -> None:
        if self._loading:
            return
        name = self.preset_combo.currentData()
        if not name:
            return
        duration_ms = resolve_duration_ms(self._session.config.simulation.duration_ms)
        loaded = preset(name, duration_ms)
        self._loading = True
        try:
            self.preset_combo.setCurrentIndex(0)
        finally:
            self._loading = False
        self._writing = True
        try:
            self._session.set_by_path("stimulus.combine", loaded["combine"])
        finally:
            self._writing = False
        self._current_index = 0
        self._write_layers(loaded["layers"], keep_index=False)

    def _on_add(self) -> None:
        layers = self.layers()
        layers.append(default_layer())
        self._current_index = len(layers) - 1
        self._write_layers(layers, keep_index=False)

    def _on_duplicate(self) -> None:
        layers = self.layers()
        index = self._current_index
        if not 0 <= index < len(layers):
            return
        copy_ = copy.deepcopy(layers[index])
        layers.insert(index + 1, copy_)
        self._current_index = index + 1
        self._write_layers(layers, keep_index=False)

    def _on_remove(self) -> None:
        layers = self.layers()
        if len(layers) <= 1:
            return
        index = self._current_index
        if not 0 <= index < len(layers):
            index = len(layers) - 1
        del layers[index]
        self._current_index = min(index, len(layers) - 1)
        self._write_layers(layers, keep_index=False)

    def _on_move_up(self) -> None:
        layers = self.layers()
        index = self._current_index
        if index <= 0 or index >= len(layers):
            return
        layers[index - 1], layers[index] = layers[index], layers[index - 1]
        self._current_index = index - 1
        self._write_layers(layers, keep_index=False)

    def _on_move_down(self) -> None:
        layers = self.layers()
        index = self._current_index
        if index < 0 or index >= len(layers) - 1:
            return
        layers[index + 1], layers[index] = layers[index], layers[index + 1]
        self._current_index = index + 1
        self._write_layers(layers, keep_index=False)

    # -------------------------------------------------------------- session

    def _on_config_changed(self, path: str) -> None:
        if self._writing:
            return
        if path == "stimulus.type":
            stim = self._session.config.stimulus
            if not self._applicable():
                # Leaving "layered": drop its fields, mirroring
                # SubStimulusEditor's _OWNED_FIELDS cleanup, so a stray
                # layers list never lingers as an explicit field of an
                # unrelated stimulus type.
                for name in ("layers", "combine"):
                    if name in stim.explicit_fields():
                        stim.unset(name)
                        self._session.notify(f"stimulus.{name}")
                self.reload()
                return
            self.reload()
            if not self._session.config.stimulus.layers:
                self._writing = True
                try:
                    self._session.set_by_path("stimulus.layers", [default_layer()])
                finally:
                    self._writing = False
                self.reload()
            return
        if not self._applicable():
            return
        if path in ("stimulus", "stimulus.layers", "stimulus.combine"):
            self.reload()
        elif path.startswith("stimulus.layers."):
            # A field inside the currently-open forms already reflects the
            # edit (ParamForm wrote it); only the list's summary label needs
            # refreshing, and only structural changes (kind swaps go through
            # _write_layers -> reload already) need more. Refresh the label.
            self._refresh_current_label()

    def _refresh_current_label(self) -> None:
        layers = self._session.config.stimulus.layers
        index = self._current_index
        if not 0 <= index < len(layers):
            return
        item = self.layer_list.item(index)
        if item is None:
            return
        self._loading = True
        try:
            item.setText(layer_summary(layers[index]))
        finally:
            self._loading = False
