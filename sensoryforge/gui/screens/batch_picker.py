"""The "Add parameter" picker: every sweepable path, grouped and with its value.

Lists :func:`~sensoryforge.gui.execution.sweep_controller.sweep_paths` for
the session's current config, grouped by top-level config section (grids,
populations, stimulus, simulation) with each path's current value shown next
to it, so picking one is "which field, and what is it now" rather than
memorising dotted-path spelling.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt5 import QtWidgets

from sensoryforge.gui.execution.sweep_controller import sweep_paths
from sensoryforge.gui.session import Session
from sensoryforge.stimuli.base import ParamSpec

#: Order the top-level sections are shown in; anything else (there is
#: nothing else today) is appended after, alphabetically.
_SECTION_ORDER = ("grids", "populations", "stimulus", "simulation")


def _section_of(path: str) -> str:
    """The top-level config section a dotted path belongs to."""
    return path.split(".", 1)[0]


class ParameterPickerDialog(QtWidgets.QDialog):
    """Pick one sweepable dotted path from ``session.config``.

    Args:
        session: The experiment whose :func:`sweep_paths` are offered.
        parent: Qt parent.

    Example:
        >>> dialog = ParameterPickerDialog(session)     # doctest: +SKIP
        >>> if dialog.exec_() == QtWidgets.QDialog.Accepted:
        ...     path, spec = dialog.selected()           # doctest: +SKIP
    """

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add sweep parameter")
        self._session = session
        self._paths = sweep_paths(session.config)
        self._by_path = dict(self._paths)

        layout = QtWidgets.QVBoxLayout(self)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Field", "Current value"])
        self.tree.setColumnCount(2)
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        layout.addWidget(self.tree, 1)

        self._populate()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate(self) -> None:
        sections: dict = {}
        for path, _spec in self._paths:
            sections.setdefault(_section_of(path), []).append(path)

        ordered = [s for s in _SECTION_ORDER if s in sections]
        ordered += sorted(s for s in sections if s not in _SECTION_ORDER)

        for section in ordered:
            section_item = QtWidgets.QTreeWidgetItem([section])
            self.tree.addTopLevelItem(section_item)
            for path in sections[section]:
                try:
                    value = self._session.get_by_path(path)
                except ValueError:
                    value = "?"
                leaf = QtWidgets.QTreeWidgetItem([path, str(value)])
                leaf.setData(0, 100, path)
                section_item.addChild(leaf)
            section_item.setExpanded(True)

    def _on_double_clicked(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        if item.data(0, 100) is not None:
            self.accept()

    def selected(self) -> Optional[Tuple[str, Optional[ParamSpec]]]:
        """The chosen ``(path, spec)`` pair.

        Returns:
            The pair for the currently selected leaf item, or ``None`` if no
            leaf is selected (a section header, or nothing).
        """
        items = self.tree.selectedItems()
        if not items:
            return None
        path = items[0].data(0, 100)
        if path is None:
            return None
        return path, self._by_path.get(path)

    def all_paths(self) -> List[Tuple[str, Optional[ParamSpec]]]:
        """Every offered ``(path, spec)`` pair, for tests that skip the UI."""
        return list(self._paths)
