"""The screens behind the GUI v2 stage navigation.

``SCREEN_FACTORIES`` maps a stage name to a factory ``(session, parent) ->
QWidget``. A stage whose screen is not built yet gets a centred label.

A screen that has rows hidden in Basic mode defines ``set_advanced(on: bool)``;
the shell calls it once after construction and again whenever the toolbar's
Advanced toggle changes. That is the whole convention -- a screen never reads
the preference itself.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from PyQt5 import QtCore, QtWidgets

from sensoryforge.gui.screens.batch import BatchScreen
from sensoryforge.gui.screens.results import ResultsScreen
from sensoryforge.gui.screens.sensors import SensorsScreen
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
    "sensors": SensorsScreen,
    "stimulus": _make_placeholder_factory("Stimulus"),
    "populations": _make_placeholder_factory("Populations"),
    "results": ResultsScreen,
    "batch": BatchScreen,
}

__all__ = ["SCREEN_FACTORIES"]
