"""Sub-stimulus editor for ``composite`` and ``timeline`` stimuli.

A composite stimulus sums (or takes the max/mean/product of) several static
sub-stimuli; a timeline shows each one during its own ``[onset, onset +
duration)`` window. Both are stored in the config exactly as the renderer
reads them, in ``stimulus.params``:

- ``composite``: the ``stimulus.stimuli`` field (list) and
  ``stimulus.params["mode"]``
- ``timeline``: ``stimulus.params["sub_stimuli"]`` (list of ``{"stimulus":
  ..., "onset_ms": ..., "duration_ms": ...}``) and the
  ``stimulus.composition_mode`` field

(``StimulusConfig`` has named fields for two of the four, and
``stimulus.params`` may not repeat a named field.)

Each sub-stimulus is ``{"class": "StaticStimulus", "stim_type": kind,
"params": {...}}`` with ``kind`` one of :data:`KINDS`.
"""

from __future__ import annotations

import copy
import functools
from typing import Any, Dict, List, Optional

from PyQt5 import QtCore, QtWidgets

from sensoryforge.gui.session import Session

#: Sub-stimulus kinds StaticStimulus draws, and the "size" parameter of each.
KINDS: Dict[str, str] = {
    "gaussian": "sigma",
    "point": "diameter_mm",
    "edge": "width",
    "gabor": "sigma",
}

#: Composition modes shared by CompositeStimulus and TimelineStimulus.
MODES = ["add", "max", "mean", "multiply"]

#: Stimulus types this editor applies to: (list path, mode path) below
#: ``stimulus.``.
_LAYOUT = {
    "composite": ("stimuli", "params.mode"),
    "timeline": ("params.sub_stimuli", "composition_mode"),
}

#: Named StimulusConfig fields a composed type sets; unset when leaving it,
#: so they are not forwarded to the next type's constructor.
_OWNED_FIELDS = {"composite": ("stimuli",), "timeline": ("composition_mode",)}


def _get(stimulus, path: str, default: Any) -> Any:
    if path.startswith("params."):
        return stimulus.params.get(path[len("params.") :], default)
    value = getattr(stimulus, path)
    return value if value else default


def _has(stimulus, path: str) -> bool:
    if path.startswith("params."):
        return path[len("params.") :] in stimulus.params
    return path in stimulus.explicit_fields() and bool(getattr(stimulus, path))


_COLUMNS_COMMON = ["Kind", "Amplitude", "Size", "X", "Y"]
_COLUMNS_TIMELINE = ["From", "For"]
_UNITS = "Amplitude in mA; size and position in mm."
_UNITS_TIMELINE = _UNITS + " Each is shown from a time, for a duration (ms)."


def default_sub_stimulus(center_x: float = 0.0) -> Dict[str, Any]:
    """A Gaussian sub-stimulus: amplitude 30 mA, sigma 1 mm, at ``center_x``."""
    return {
        "class": "StaticStimulus",
        "stim_type": "gaussian",
        "params": {
            "amplitude": 30.0,
            "sigma": 1.0,
            "center_x": center_x,
            "center_y": 0.0,
        },
    }


def is_composed(stimulus_type: Optional[str]) -> bool:
    """Whether ``stimulus_type`` is built from sub-stimuli."""
    return stimulus_type in _LAYOUT


class SubStimulusEditor(QtWidgets.QGroupBox):
    """Table of sub-stimuli for the session's composite or timeline stimulus.

    Hidden for every other stimulus type. Every edit writes the whole list
    back through :meth:`Session.set_by_path`.

    Args:
        session: The experiment.
        parent: Qt parent.
    """

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__("Sub-stimuli", parent)
        self._session = session
        self._loading = False
        # True while this editor writes; its own value edits need no rebuild
        # (the table already shows them), and a rebuild would delete the
        # widget whose signal is being delivered.
        self._writing = False

        layout = QtWidgets.QVBoxLayout(self)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Combine overlapping as"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(MODES)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self.units_label = QtWidgets.QLabel(_UNITS)
        self.units_label.setObjectName("Caption")
        self.units_label.setWordWrap(True)
        layout.addWidget(self.units_label)

        self.table = QtWidgets.QTableWidget()
        self.table.verticalHeader().setVisible(False)
        # Every column visible without scrolling: they share the width.
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.table.horizontalHeader().setMinimumSectionSize(48)
        # The Kind combo is a cell widget, which ResizeToContents ignores;
        # fix the column to the widest kind name.
        probe = QtWidgets.QComboBox()
        probe.addItems(list(KINDS))
        self._kind_width = probe.sizeHint().width()
        probe.deleteLater()
        layout.addWidget(self.table)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_add.clicked.connect(self._on_add)
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_remove.clicked.connect(self._on_remove)
        buttons.addWidget(self.btn_add)
        buttons.addWidget(self.btn_remove)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        session.configChanged.connect(self._on_config_changed)
        session.configReplaced.connect(self.reload)
        self.reload()

    # ----------------------------------------------------------------- model

    def _layout(self):
        return _LAYOUT.get(self._session.config.stimulus.type)

    def entries(self) -> List[Dict[str, Any]]:
        """The sub-stimuli as the renderer reads them (a copy)."""
        layout = self._layout()
        if layout is None:
            return []
        return copy.deepcopy(_get(self._session.config.stimulus, layout[0], []))

    def _write(self, entries: List[Dict[str, Any]], *, rebuild: bool = False) -> None:
        list_key, _ = self._layout()
        self._writing = True
        try:
            self._session.set_by_path(f"stimulus.{list_key}", entries)
        finally:
            self._writing = False
        if rebuild:
            # Deferred: the sender may be a widget in the table.
            QtCore.QTimer.singleShot(0, self.reload)

    # ------------------------------------------------------------------ view

    def reload(self, *_args: object) -> None:
        """Rebuild the table from the config."""
        layout = self._layout()
        self.setVisible(layout is not None)
        if layout is None:
            return
        stimulus = self._session.config.stimulus
        self._loading = True
        try:
            self.mode_combo.setCurrentText(_get(stimulus, layout[1], "add"))
            timeline = self._session.config.stimulus.type == "timeline"
            columns = _COLUMNS_COMMON + (_COLUMNS_TIMELINE if timeline else [])
            self.units_label.setText(_UNITS_TIMELINE if timeline else _UNITS)
            self.table.clear()
            self.table.setColumnCount(len(columns))
            self.table.setHorizontalHeaderLabels(columns)
            entries = self.entries()
            self.table.setRowCount(len(entries))
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
            self.table.setColumnWidth(0, self._kind_width)
            for row, entry in enumerate(entries):
                self._fill_row(row, entry, timeline)
            self.btn_remove.setEnabled(len(entries) > 1)
        finally:
            self._loading = False

    def _fill_row(self, row: int, entry: Dict[str, Any], timeline: bool) -> None:
        sub = entry["stimulus"] if timeline else entry
        kind = sub.get("stim_type", "gaussian")
        p = sub.get("params", {})

        kind_combo = QtWidgets.QComboBox()
        kind_combo.addItems(list(KINDS))
        kind_combo.setCurrentText(kind)
        kind_combo.currentTextChanged.connect(functools.partial(self._on_kind, row))
        self.table.setCellWidget(row, 0, kind_combo)

        values = [
            ("amplitude", p.get("amplitude", 1.0), 0.0, 1000.0),
            (
                KINDS.get(kind, "sigma"),
                p.get(KINDS.get(kind, "sigma"), 1.0),
                0.001,
                50.0,
            ),
            ("center_x", p.get("center_x", 0.0), -100.0, 100.0),
            ("center_y", p.get("center_y", 0.0), -100.0, 100.0),
        ]
        for column, (name, value, low, high) in enumerate(values, start=1):
            self.table.setCellWidget(
                row, column, self._spin(value, low, high, row, ("params", name))
            )
        if timeline:
            for column, name in ((5, "onset_ms"), (6, "duration_ms")):
                value = entry.get(name, 0.0)
                self.table.setCellWidget(
                    row, column, self._spin(value, 0.0, 1.0e6, row, (name,))
                )

    def _spin(self, value, low, high, row, key) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(2)
        # Typed, not stepped: arrows would take a third of each narrow cell.
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        spin.setFrame(False)
        spin.setRange(low, high)
        spin.setValue(float(value))
        spin.editingFinished.connect(functools.partial(self._on_value, row, key, spin))
        return spin

    # --------------------------------------------------------------- editing

    def _sub(self, entries, row):
        timeline = self._session.config.stimulus.type == "timeline"
        return entries[row]["stimulus"] if timeline else entries[row]

    def _on_value(self, row: int, key: tuple, spin: QtWidgets.QDoubleSpinBox) -> None:
        if self._loading:
            return
        entries = self.entries()
        if key[0] == "params":
            self._sub(entries, row).setdefault("params", {})[key[1]] = spin.value()
        else:
            entries[row][key[0]] = spin.value()
        if entries != self.entries():
            self._write(entries)

    def _on_kind(self, row: int, kind: str) -> None:
        if self._loading:
            return
        entries = self.entries()
        sub = self._sub(entries, row)
        old_size_key = KINDS.get(sub.get("stim_type", "gaussian"), "sigma")
        params = sub.setdefault("params", {})
        size = params.pop(old_size_key, 1.0)
        sub["stim_type"] = kind
        params[KINDS[kind]] = size
        self._write(entries, rebuild=True)

    def _on_mode_changed(self, mode: str) -> None:
        if self._loading or self._layout() is None:
            return
        _, mode_key = self._layout()
        self._writing = True
        try:
            self._session.set_by_path(f"stimulus.{mode_key}", mode)
        finally:
            self._writing = False

    def _new_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """``entries`` plus one default sub-stimulus, placed after the last."""
        timeline = self._session.config.stimulus.type == "timeline"
        center_x = float(len(entries))  # 1 mm apart, so a new one is visible
        sub = default_sub_stimulus(center_x)
        if not timeline:
            return entries + [sub]
        onset = max(
            (e.get("onset_ms", 0.0) + e.get("duration_ms", 0.0) for e in entries),
            default=0.0,
        )
        return entries + [{"stimulus": sub, "onset_ms": onset, "duration_ms": 100.0}]

    def _on_add(self) -> None:
        self._write(self._new_entries(self.entries()), rebuild=True)

    def _on_remove(self) -> None:
        entries = self.entries()
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if not rows:
            rows = [len(entries) - 1]
        kept = [e for i, e in enumerate(entries) if i not in rows]
        if kept:
            self._write(kept, rebuild=True)

    def _on_config_changed(self, path: str) -> None:
        if self._writing:
            return
        layout = self._layout()
        stimulus = self._session.config.stimulus
        if path == "stimulus.type":
            for stim_type, fields in _OWNED_FIELDS.items():
                if stim_type == stimulus.type:
                    continue
                for name in fields:
                    if name in stimulus.explicit_fields():
                        stimulus.unset(name)
                        self._session.notify(f"stimulus.{name}")
        if (
            path == "stimulus.type"
            and layout is not None
            and not _has(stimulus, layout[0])
        ):
            # Switched to a composed type: one visible, editable sub-stimulus
            # instead of a stimulus that cannot render. (Only on a switch --
            # a loaded config is shown as it is, and validated.)
            self._session.set_by_path(f"stimulus.{layout[0]}", self._new_entries([]))
            return  # that write reloads the table
        if path in (
            "stimulus",
            "stimulus.type",
            "stimulus.params",
            "stimulus.stimuli",
            "stimulus.composition_mode",
        ) or path.startswith("stimulus.params."):
            self.reload()
