"""The Batch screen's right-hand preview pane: sweep + single-run YAML/command.

Two read-only panes, each with its literal YAML and the exact command that
runs it, and a Copy button on every field -- the sweep's first combination
(kept live as the table changes) and the session's own config (the plan's
"Export" view, CLAUDE.md/phase2-common "Report").
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtGui, QtWidgets


def _section_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setObjectName("SectionTitle")
    return label


def _read_only_text() -> QtWidgets.QPlainTextEdit:
    widget = QtWidgets.QPlainTextEdit()
    widget.setReadOnly(True)
    widget.setFont(QtGui.QFont("Monospace"))
    widget.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
    return widget


class SweepPreviewPane(QtWidgets.QWidget):
    """The literal YAML and command for a sweep's first job, and a single run.

    Args:
        parent: Qt parent.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(_section_label("First combination (config.yml)"))
        self.sweep_yaml = _read_only_text()
        layout.addWidget(self.sweep_yaml, 2)
        self._add_copy_button(layout, self.sweep_yaml.toPlainText, "Copy YAML")

        layout.addWidget(_section_label("Command"))
        self.sweep_command = QtWidgets.QLineEdit()
        self.sweep_command.setReadOnly(True)
        layout.addWidget(self.sweep_command)
        self._add_copy_button(layout, self.sweep_command.text, "Copy command")

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        layout.addWidget(line)

        layout.addWidget(_section_label("Export - this config (config.yml)"))
        self.export_yaml = _read_only_text()
        layout.addWidget(self.export_yaml, 2)
        self._add_copy_button(layout, self.export_yaml.toPlainText, "Copy YAML")

        layout.addWidget(_section_label("Command (single run)"))
        self.export_command = QtWidgets.QLineEdit()
        self.export_command.setReadOnly(True)
        layout.addWidget(self.export_command)
        self._add_copy_button(layout, self.export_command.text, "Copy command")

    def _add_copy_button(
        self, layout: QtWidgets.QVBoxLayout, source, text: str
    ) -> None:
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(lambda: self._copy(source()))
        layout.addWidget(button)

    def _copy(self, text: str) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)

    # -------------------------------------------------------------- updates

    def set_sweep_preview(self, yaml_text: str, command_text: str) -> None:
        """Show the first combination's config and the command that runs it."""
        self.sweep_yaml.setPlainText(yaml_text)
        self.sweep_yaml.moveCursor(self.sweep_yaml.textCursor().Start)
        self.sweep_command.setText(command_text)
        self.sweep_command.setCursorPosition(0)

    def set_export(self, yaml_text: str, command_text: str) -> None:
        """Show the session config's own YAML and its single-run command."""
        self.export_yaml.setPlainText(yaml_text)
        self.export_yaml.moveCursor(self.export_yaml.textCursor().Start)
        self.export_command.setText(command_text)
        self.export_command.setCursorPosition(0)
