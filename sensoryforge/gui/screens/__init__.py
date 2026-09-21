"""Placeholder screens for the GUI v2 stage navigation.

``SCREEN_FACTORIES`` maps a stage name to a factory ``(session, parent) ->
QWidget``. Phase 1 ships a centred label for each; Phase 2 replaces these one
at a time with the real screen for that stage.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from PyQt5 import QtCore, QtWidgets

from sensoryforge.gui.session import Session


def _make_placeholder_factory(
    label_text: str,
) -> Callable[[Session, Optional[QtWidgets.QWidget]], QtWidgets.QWidget]:
    """Build a factory returning a widget with one centred label.

    Args:
        label_text: What the placeholder names itself.

    Returns:
        A ``(session, parent) -> QWidget`` factory matching every real
        screen's constructor signature, so ``app.py`` does not special-case
        placeholders.
    """

    def factory(
        session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> QtWidgets.QWidget:
        del session  # placeholder does not read the session yet
        widget = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(widget)
        label = QtWidgets.QLabel(label_text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)
        return widget

    return factory


#: One factory per stage in the left-hand navigation
#: (``sensoryforge/gui/app.py::STAGE_ORDER``).
SCREEN_FACTORIES: Dict[
    str, Callable[[Session, Optional[QtWidgets.QWidget]], QtWidgets.QWidget]
] = {
    "sensors": _make_placeholder_factory("Sensors"),
    "stimulus": _make_placeholder_factory("Stimulus"),
    "populations": _make_placeholder_factory("Populations"),
    "results": _make_placeholder_factory("Run & Results"),
    "batch": _make_placeholder_factory("Batch"),
}

__all__ = ["SCREEN_FACTORIES"]
