"""A red list of the config problems that concern one screen.

Reads :attr:`sensoryforge.gui.session.Session.errors` (the shared
:func:`sensoryforge.gui.validation.validate` result) and shows the messages
under one key prefix, such as ``"grids.0"`` on the Sensors screen or
``"populations.1"`` on the Populations screen. Hidden when there are none.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from PyQt5 import QtCore, QtWidgets

from sensoryforge.gui import theme
from sensoryforge.gui.session import Session
from sensoryforge.gui.validation import errors_under

Prefix = Union[str, Callable[[], Optional[str]]]


class ProblemList(QtWidgets.QLabel):
    """Shows ``errors_under(session.errors, prefix)``, one message per line.

    Args:
        session: The experiment whose problems are shown.
        prefix: A key prefix, or a function returning the current one (for a
            screen whose selection changes); ``None`` from it shows nothing.
        parent: Qt parent.
    """

    def __init__(
        self,
        session: Session,
        prefix: Prefix,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._prefix = prefix
        self.setObjectName("ProblemList")
        self.setWordWrap(True)
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.setStyleSheet(
            f"color: {theme.PALETTE['error']}; padding: 4px 6px; "
            f"border: 1px solid {theme.PALETTE['error']}; border-radius: 4px;"
        )
        session.validationChanged.connect(self.refresh)
        session.configReplaced.connect(self.refresh)
        self.refresh()

    def messages(self) -> list:
        """The messages currently shown (empty when hidden)."""
        prefix = self._prefix() if callable(self._prefix) else self._prefix
        if prefix is None:
            return []
        return errors_under(self._session.errors, prefix)

    def refresh(self, *_args: object) -> None:
        """Re-read the session's problems for the current prefix."""
        messages = self.messages()
        self.setText("\n".join(f"⚠ {message}" for message in messages))
        self.setVisible(bool(messages))
