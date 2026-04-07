"""Collapsible group box widget for SensoryForge GUI panels.

Extracted from mechanoreceptor_tab.py so it can be shared across all tabs.
"""

from typing import Optional

from PyQt5 import QtWidgets


class CollapsibleGroupBox(QtWidgets.QWidget):
    """Collapsible section with toggle button, normal content styling.

    Provides a collapsible panel with a clickable header that expands or
    collapses the contained content. Supports both flat (main section) and
    nested (sub-section) visual styles.

    Args:
        title: Text displayed on the toggle button header.
        parent: Optional parent widget.
        start_expanded: Whether the section starts in the expanded state.
        nested: If True, uses a more compact nested style.

    Example:
        >>> group = CollapsibleGroupBox("Spatial Parameters", start_expanded=True)
        >>> group.addRow("X (mm):", spin_x)
        >>> group.addRow("Y (mm):", spin_y)
        >>> layout.addWidget(group)
    """

    def __init__(
        self,
        title: str,
        parent: Optional[QtWidgets.QWidget] = None,
        start_expanded: bool = False,
        nested: bool = False,
    ):
        super().__init__(parent)
        self._title = title
        self._is_expanded = start_expanded
        self._nested = nested

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 4)
        main_layout.setSpacing(2)

        # Toggle button — subtle for main sections, minimal for nested
        self._toggle_btn = QtWidgets.QPushButton()
        if nested:
            self._toggle_btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 3px 6px;
                    border: 1px solid #b0b0b0;
                    border-radius: 2px;
                    background: #e0e0e0;
                }
                QPushButton:hover {
                    background: #d8d8d8;
                    border: 1px solid #909090;
                }
            """)
        else:
            self._toggle_btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 5px 8px;
                    border: 1px solid #a0a0a0;
                    border-radius: 3px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #e0e0e0, stop:1 #d4d4d4);
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #e8e8e8, stop:1 #dcdcdc);
                    border: 1px solid #909090;
                }
            """)
        self._toggle_btn.clicked.connect(self._on_toggle)
        main_layout.addWidget(self._toggle_btn)

        # Content widget — no special styling, inherits normal UI look
        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QFormLayout(self._content)
        self._content_layout.setContentsMargins(8, 6, 8, 6)
        self._content_layout.setSpacing(6)
        main_layout.addWidget(self._content)

        self._update_button_text()
        self._content.setVisible(self._is_expanded)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_button_text(self) -> None:
        """Update header button text with expand/collapse indicator."""
        arrow = "▼" if self._is_expanded else "▶"
        self._toggle_btn.setText(f"{arrow}  {self._title}")

    def _on_toggle(self) -> None:
        """Toggle the collapsed/expanded state."""
        self._is_expanded = not self._is_expanded
        self._content.setVisible(self._is_expanded)
        self._update_button_text()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setChecked(self, checked: bool) -> None:
        """Open or close the section.

        Args:
            checked: True to expand, False to collapse.
        """
        self._is_expanded = checked
        self._content.setVisible(self._is_expanded)
        self._update_button_text()

    def isChecked(self) -> bool:
        """Return True if the section is currently expanded."""
        return self._is_expanded

    def layout(self) -> QtWidgets.QFormLayout:  # type: ignore[override]
        """Return the inner form layout for adding labelled rows.

        Returns:
            The QFormLayout inside the collapsible content area.
        """
        return self._content_layout

    def addRow(self, *args) -> None:
        """Add a labelled row (or spanning widget) to the content area.

        Delegates to :py:meth:`QFormLayout.addRow`, so all overloads work::

            group.addRow("Label:", widget)
            group.addRow(widget)          # spanning widget
        """
        self._content_layout.addRow(*args)
