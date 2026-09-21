"""The Sensors screen: the grid list and the selected grid's editor + preview.

``SensorsScreen`` is the GUI v2 view onto ``session.config.grids``
(``SensoryForgeConfig.grids``, a list of
:class:`~sensoryforge.config.schema.GridConfig`). Left: the grid list with
Add/Duplicate/Remove; a form for the selected grid built from
:func:`~sensoryforge.config.schema.grid_config_param_specs` through
:class:`~sensoryforge.gui.widgets.param_form.ParamForm`, plus hand-built rows
for ``channels`` and ``coords_file`` (list/path fields ``ParamForm`` does not
cover) and a read-only table for ``layers`` on a composite grid. Right:
:class:`~sensoryforge.gui.widgets.grid_preview.GridPreview` plus a caption
(receptor count, extent in mm) computed from the grid's real coordinates via
:func:`sensoryforge.core.simulation_engine.build_grid` -- the exact
construction the engine uses.

Every edit goes through ``session.set_by_path`` (via ``ParamForm``) or
:meth:`~sensoryforge.gui.session.Session.notify`, never a private copy of the
config; the screen rebuilds itself from ``session.configChanged``/
``configReplaced``, filtering by the ``"grids."`` path prefix.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from sensoryforge.gui.widgets.problem_list import ProblemList
from sensoryforge.config.schema import GridConfig, grid_config_param_specs
from sensoryforge.core.simulation_engine import build_grid
from sensoryforge.gui import theme
from sensoryforge.gui.session import Session
from sensoryforge.gui.widgets.grid_preview import GridPreview
from sensoryforge.gui.widgets.param_form import ParamForm

#: Debounce for the preview rebuild after an edit (phase2-common.md /
#: this screen's brief: typing in a spin box must not rebuild per keystroke).
_PREVIEW_DEBOUNCE_MS = 150

#: Which GridConfig fields the ``grid`` arrangement actually reads
#: (core/grid.py: the "grid"/"jittered_grid" branch builds a mesh from
#: grid_size/spacing/center; density and seed are unused for "grid").
# Every arrangement is sized by rows x cols x spacing (core.simulation_engine.
# build_grid): rows/cols set the extent and the receptor count for hex,
# poisson and blue noise as much as for a lattice, so they stay editable.
_SEED_ARRANGEMENTS = {"jittered_grid", "blue_noise", "poisson"}

_COMPOSITE_DISABLED_TOOLTIP = "A composite array is sized by its layers."
_SEED_DISABLED_TOOLTIP = "This arrangement has no randomness to seed."


def _unique_name(base: str, existing: List[str]) -> str:
    """The first ``base``/``base_2``/``base_3``/... not already in ``existing``."""
    if base not in existing:
        return base
    n = 2
    while f"{base}_{n}" in existing:
        n += 1
    return f"{base}_{n}"


def _grids_in_use(config: Any) -> dict:
    """Map grid name -> list of population names that target it.

    A population targets a grid through ``target_grid`` (the single-input
    sugar form) or through ``inputs[*].grid`` (multi-input populations).
    """
    used: dict = {}
    for population in config.populations:
        names = set()
        if population.inputs:
            for inp in population.inputs:
                names.add(inp.grid)
        elif population.target_grid:
            names.add(population.target_grid)
        for name in names:
            used.setdefault(name, []).append(population.name)
    return used


class SensorsScreen(QtWidgets.QWidget):
    """Editor + live preview for ``session.config.grids``.

    Args:
        session: The session this screen edits.
        parent: Qt parent.
    """

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._selected_index: Optional[int] = None
        self._advanced = False
        self._param_form: Optional[ParamForm] = None
        self._highlight_item: Optional[Any] = None
        self._suspend_name_edit = False

        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(_PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._update_preview)

        self._build_ui()

        self._session.configChanged.connect(self._on_config_changed)
        self._session.configReplaced.connect(self._on_config_replaced)

        self._refresh_list(select_index=0 if session.config.grids else None)

    # --------------------------------------------------------------- build

    def _build_ui(self) -> None:
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(splitter)

        splitter.addWidget(self._build_editor())
        splitter.addWidget(self._build_preview())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    def _build_editor(self) -> QtWidgets.QWidget:
        editor = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)

        self.problems = ProblemList(self._session, self._problem_prefix)
        layout.addWidget(self.problems)

        list_label = QtWidgets.QLabel("Grids")
        list_label.setObjectName("SectionTitle")
        layout.addWidget(list_label)

        self.grid_list = QtWidgets.QListWidget()
        self.grid_list.setMaximumHeight(140)
        self.grid_list.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self.grid_list)

        button_row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_add.clicked.connect(self._on_add_clicked)
        self.btn_duplicate = QtWidgets.QPushButton("Duplicate")
        self.btn_duplicate.clicked.connect(self._on_duplicate_clicked)
        self.btn_remove = QtWidgets.QPushButton("Remove")
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        for btn in (self.btn_add, self.btn_duplicate, self.btn_remove):
            button_row.addWidget(btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.list_message = QtWidgets.QLabel("")
        self.list_message.setWordWrap(True)
        self.list_message.setStyleSheet(f"color: {theme.PALETTE['error']};")
        self.list_message.setVisible(False)
        layout.addWidget(self.list_message)

        form_label = QtWidgets.QLabel("Selected grid")
        form_label.setObjectName("SectionTitle")
        layout.addWidget(form_label)

        name_row = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.editingFinished.connect(self._on_name_edited)
        name_row.addRow("Name", self.name_edit)
        layout.addLayout(name_row)

        self.form_container = QtWidgets.QVBoxLayout()
        form_host = QtWidgets.QWidget()
        form_host.setLayout(self.form_container)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_host)
        layout.addWidget(scroll, 1)

        channels_row = QtWidgets.QFormLayout()
        self.channels_edit = QtWidgets.QLineEdit()
        self.channels_edit.setToolTip(
            "Comma-separated channel/plane names (e.g. 'value' or 'pressure,shear')."
        )
        self.channels_edit.editingFinished.connect(self._on_channels_edited)
        channels_row.addRow("Channels", self.channels_edit)
        layout.addLayout(channels_row)
        self.channels_error = QtWidgets.QLabel("")
        self.channels_error.setWordWrap(True)
        self.channels_error.setStyleSheet(f"color: {theme.PALETTE['error']};")
        self.channels_error.setVisible(False)
        layout.addWidget(self.channels_error)

        coords_row = QtWidgets.QHBoxLayout()
        coords_label = QtWidgets.QLabel("Coords file")
        self.coords_file_edit = QtWidgets.QLineEdit()
        self.coords_file_edit.editingFinished.connect(self._on_coords_file_edited)
        self.btn_browse_coords = QtWidgets.QPushButton("Browse…")
        self.btn_browse_coords.clicked.connect(self._on_browse_coords)
        coords_row.addWidget(coords_label)
        coords_row.addWidget(self.coords_file_edit, 1)
        coords_row.addWidget(self.btn_browse_coords)
        layout.addLayout(coords_row)

        self.layers_label = QtWidgets.QLabel(
            "Composite layers (read-only here -- edit them in the YAML for now)"
        )
        self.layers_label.setWordWrap(True)
        layout.addWidget(self.layers_label)
        self.layers_table = QtWidgets.QTableWidget(0, 3)
        self.layers_table.setHorizontalHeaderLabels(["name", "density", "arrangement"])
        self.layers_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.layers_table.setMaximumHeight(120)
        layout.addWidget(self.layers_table)

        return editor

    def _build_preview(self) -> QtWidgets.QWidget:
        preview_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(preview_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview = GridPreview()
        layout.addWidget(self.preview, 1)

        self.caption = QtWidgets.QLabel("")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)

        return preview_widget

    # ----------------------------------------------------------- selection

    def _grids(self) -> List[GridConfig]:
        return self._session.config.grids

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._grids()):
            self._selected_index = None
            self._clear_form()
            return
        self._select_grid(row)

    def _select_grid(self, index: int) -> None:
        self._selected_index = index
        if hasattr(self, "problems"):
            self.problems.refresh()
        grid_cfg = self._grids()[index]

        self._suspend_name_edit = True
        self.name_edit.setText(grid_cfg.name)
        self._suspend_name_edit = False

        self.channels_edit.blockSignals(True)
        self.channels_edit.setText(", ".join(grid_cfg.channels))
        self.channels_edit.blockSignals(False)
        self.channels_error.setVisible(False)

        self.coords_file_edit.blockSignals(True)
        self.coords_file_edit.setText(grid_cfg.coords_file or "")
        self.coords_file_edit.blockSignals(False)

        self._rebuild_param_form(index, grid_cfg)
        self._update_field_enablement(grid_cfg.arrangement)
        self._update_layers_table(grid_cfg)
        self._schedule_preview_update()

    def _teardown_param_form(self) -> None:
        """Disconnect and dispose the current ``ParamForm``.

        ``deleteLater`` only schedules destruction for the next event-loop
        turn; the form's own ``session.configChanged``/``configReplaced``
        connections stay live until then. Without an explicit disconnect
        here, a second structural change (e.g. two ``replace_config`` calls
        in the same turn) would still invoke the discarded form's slots
        against a grid it no longer owns.
        """
        if self._param_form is None:
            return
        try:
            self._session.configChanged.disconnect(self._param_form._on_config_changed)
        except TypeError:
            pass
        try:
            self._session.configReplaced.disconnect(
                self._param_form._on_config_replaced
            )
        except TypeError:
            pass
        self.form_container.removeWidget(self._param_form)
        self._param_form.setParent(None)
        self._param_form.deleteLater()
        self._param_form = None

    def _clear_form(self) -> None:
        self._teardown_param_form()
        self.name_edit.clear()
        self.channels_edit.clear()
        self.coords_file_edit.clear()
        self.layers_table.setRowCount(0)
        self._update_preview()

    def _rebuild_param_form(self, index: int, grid_cfg: GridConfig) -> None:
        self._teardown_param_form()
        self._param_form = ParamForm(
            grid_config_param_specs(),
            grid_cfg,
            self._session,
            f"grids.{index}",
            advanced=self._advanced,
        )
        self.form_container.addWidget(self._param_form)

    def _problem_prefix(self) -> Optional[str]:
        """The selected grid's validation key, for the problem list."""
        if self._selected_index is None:
            return None
        return f"grids.{self._selected_index}"

    def set_advanced(self, on: bool) -> None:
        """Show or hide the advanced rows (called by the shell's toggle).

        Args:
            on: Whether Advanced mode is on.
        """
        self._advanced = bool(on)
        if self._param_form is not None:
            self._param_form.set_advanced(self._advanced)

    # -------------------------------------------------------------- list UI

    def _refresh_list(self, *, select_index: Optional[int]) -> None:
        grids = self._grids()
        self.grid_list.blockSignals(True)
        self.grid_list.clear()
        for grid_cfg in grids:
            self.grid_list.addItem(self._list_item_text(grid_cfg))
        self.grid_list.blockSignals(False)

        self.btn_remove.setEnabled(len(grids) > 1)

        if not grids:
            self._selected_index = None
            self._clear_form()
            return

        if select_index is None:
            select_index = (
                self._selected_index if self._selected_index is not None else 0
            )
        select_index = max(0, min(select_index, len(grids) - 1))
        # The list was just cleared (currentRow() is -1), so this always
        # changes and fires currentRowChanged -> _select_grid.
        self.grid_list.setCurrentRow(select_index)

    def _list_item_text(self, grid_cfg: GridConfig) -> str:
        try:
            count = build_grid(grid_cfg, device="cpu").get_all_coordinates().shape[0]
            count_text = str(count)
        except (ValueError, RuntimeError, OSError):
            count_text = "?"
        return f"{grid_cfg.name} — {grid_cfg.arrangement}, {count_text} receptors"

    def _refresh_selected_list_item(self) -> None:
        if self._selected_index is None or self._selected_index >= len(self._grids()):
            return
        item = self.grid_list.item(self._selected_index)
        if item is None:
            return
        item.setText(self._list_item_text(self._grids()[self._selected_index]))

    # ------------------------------------------------------------- actions

    def _on_add_clicked(self) -> None:
        existing = [g.name for g in self._grids()]
        name = _unique_name("grid", existing)
        new_grid = GridConfig(name=name, rows=20, cols=20, spacing=0.15)
        self._session.config.grids.append(new_grid)
        new_index = len(self._session.config.grids) - 1
        self._session.notify(f"grids.{new_index}")
        self._refresh_list(select_index=new_index)

    def _on_duplicate_clicked(self) -> None:
        if self._selected_index is None:
            return
        existing = [g.name for g in self._grids()]
        source = self._grids()[self._selected_index]
        new_name = _unique_name(source.name, existing)
        new_grid = dataclasses.replace(source, name=new_name)
        insert_index = self._selected_index + 1
        self._session.config.grids.insert(insert_index, new_grid)
        self._session.notify(f"grids.{insert_index}")
        self._refresh_list(select_index=insert_index)

    def _on_remove_clicked(self) -> None:
        if self._selected_index is None:
            return
        grids = self._grids()
        if len(grids) <= 1:
            return
        index = self._selected_index
        grid_cfg = grids[index]
        used_by = _grids_in_use(self._session.config).get(grid_cfg.name, [])
        if used_by:
            names = ", ".join(used_by)
            self.list_message.setText(
                f"Cannot remove {grid_cfg.name!r}: used by population(s) {names}."
            )
            self.list_message.setVisible(True)
            return
        self.list_message.setVisible(False)
        del grids[index]
        new_index = max(0, index - 1)
        self._session.notify(f"grids.{new_index}")
        self._refresh_list(select_index=new_index)

    def _on_name_edited(self) -> None:
        if self._suspend_name_edit or self._selected_index is None:
            return
        grids = self._grids()
        index = self._selected_index
        new_name = self.name_edit.text().strip()
        current = grids[index].name
        if new_name == current:
            return
        others = [g.name for i, g in enumerate(grids) if i != index]
        if not new_name or new_name in others:
            self.list_message.setText(
                f"Grid names must be unique and non-empty; {new_name!r} is taken."
            )
            self.list_message.setVisible(True)
            self._suspend_name_edit = True
            self.name_edit.setText(current)
            self._suspend_name_edit = False
            return
        self.list_message.setVisible(False)
        self._session.set_by_path(f"grids.{index}.name", new_name)
        self._refresh_selected_list_item()

    def _on_channels_edited(self) -> None:
        if self._selected_index is None:
            return
        index = self._selected_index
        grid_cfg = self._grids()[index]
        text = self.channels_edit.text().strip()
        parsed = (
            [c.strip() for c in text.split(",") if c.strip()] if text else ["value"]
        )
        try:
            dataclasses.replace(grid_cfg, channels=parsed)
        except ValueError as exc:
            self.channels_error.setText(str(exc))
            self.channels_error.setVisible(True)
            return
        self.channels_error.setVisible(False)
        self._session.set_by_path(f"grids.{index}.channels", parsed)

    def _on_coords_file_edited(self) -> None:
        if self._selected_index is None:
            return
        text = self.coords_file_edit.text().strip()
        self._session.set_by_path(
            f"grids.{self._selected_index}.coords_file", text or None
        )

    def _on_browse_coords(self) -> None:
        if self._selected_index is None:
            return
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose receptor coordinates",
            "",
            "CSV/Tensor (*.csv *.pt);;All files (*)",
        )
        if not path:
            return
        self.coords_file_edit.setText(path)
        self._session.set_by_path(f"grids.{self._selected_index}.coords_file", path)

    # -------------------------------------------------------- field states

    def _update_field_enablement(self, arrangement: str) -> None:
        if self._param_form is None:
            return
        seed_enabled = arrangement in _SEED_ARRANGEMENTS
        is_composite = arrangement == "composite"

        for name, enabled, tooltip in (
            ("rows", not is_composite, _COMPOSITE_DISABLED_TOOLTIP),
            ("cols", not is_composite, _COMPOSITE_DISABLED_TOOLTIP),
            ("seed", seed_enabled and not is_composite, _SEED_DISABLED_TOOLTIP),
        ):
            try:
                widget = self._param_form.widget_for(name)
            except KeyError:
                continue
            widget.setEnabled(enabled)
            widget.setToolTip(widget.toolTip() if enabled else tooltip)

        self.layers_table.setVisible(is_composite)
        self.layers_label.setVisible(is_composite)
        self.coords_file_edit.setEnabled(not is_composite)
        self.btn_browse_coords.setEnabled(not is_composite)

    def _update_layers_table(self, grid_cfg: GridConfig) -> None:
        layers = grid_cfg.layers or []
        self.layers_table.setRowCount(len(layers))
        for row, layer in enumerate(layers):
            for col, key in enumerate(("name", "density", "arrangement")):
                value = layer.get(key, "")
                self.layers_table.setItem(
                    row, col, QtWidgets.QTableWidgetItem(str(value))
                )

    # -------------------------------------------------------------- preview

    def _schedule_preview_update(self) -> None:
        self._preview_timer.start()

    def _update_preview(self) -> None:
        """Rebuild the preview -- never lets a bad grid raise out of this slot.

        Every grid is pre-built with :func:`build_grid` before touching
        ``self.preview`` at all: ``GridPreview.set_grids`` clears its old
        scatters up front and then builds each grid in turn, so calling it
        with a list containing one bad grid would destroy the last good
        drawing before raising partway through. Pre-validating means a bad
        grid never reaches it -- the widget is only ever updated with a set
        that is known to build cleanly, so a failure leaves the previous
        drawing exactly as it was.
        """
        grids = self._grids()

        coords_by_index = {}
        error_by_index = {}
        for index, grid_cfg in enumerate(grids):
            try:
                built = build_grid(grid_cfg, device="cpu")
                coords_by_index[index] = (
                    built.get_all_coordinates().detach().cpu().numpy()
                )
            except (ValueError, RuntimeError, OSError) as exc:
                error_by_index[index] = str(exc)

        if not error_by_index:
            self.preview.set_grids(grids)

        self._remove_highlight()

        if self._selected_index is None or not grids:
            self.caption.setText("No grid selected.")
            self.caption.setStyleSheet(f"color: {theme.PALETTE['text_secondary']};")
            return

        if self._selected_index in error_by_index:
            self.caption.setText(error_by_index[self._selected_index])
            self.caption.setStyleSheet(f"color: {theme.PALETTE['error']};")
            return

        self.caption.setStyleSheet(f"color: {theme.PALETTE['text_secondary']};")
        grid_cfg = grids[self._selected_index]
        coords = coords_by_index[self._selected_index]
        if coords.shape[0] == 0:
            self.caption.setText(f"{grid_cfg.name}: 0 receptors.")
            return

        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
        self.caption.setText(
            f"{grid_cfg.name}: {coords.shape[0]} receptors, "
            f"x [{x_min:.2f}, {x_max:.2f}] mm, y [{y_min:.2f}, {y_max:.2f}] mm"
        )
        if not error_by_index:
            self._add_highlight(coords)

    def _add_highlight(self, coords) -> None:
        from sensoryforge.gui.widgets import plot_factory

        color = theme.PALETTE["accent"]
        scatter = plot_factory.make_scatter(size=6.0, color=color)
        scatter.setData(x=coords[:, 0], y=coords[:, 1])
        scatter.setZValue(1)
        self.preview.plot.addItem(scatter)
        self._highlight_item = scatter

    def _remove_highlight(self) -> None:
        if self._highlight_item is not None:
            self.preview.plot.removeItem(self._highlight_item)
            self._highlight_item = None

    # ------------------------------------------------------------- signals

    def _on_config_changed(self, path: str) -> None:
        if not path.startswith("grids"):
            return
        self._refresh_selected_list_item()
        segments = path.split(".")
        changed_index: Optional[int] = None
        if len(segments) >= 2 and segments[1].isdigit():
            changed_index = int(segments[1])
        if (
            changed_index is not None
            and changed_index == self._selected_index
            and self._selected_index < len(self._grids())
        ):
            grid_cfg = self._grids()[self._selected_index]
            if len(segments) >= 3 and segments[2] == "arrangement":
                self._update_field_enablement(grid_cfg.arrangement)
            if len(segments) >= 3 and segments[2] == "layers":
                self._update_layers_table(grid_cfg)
            self._schedule_preview_update()

    def _on_config_replaced(self) -> None:
        self._selected_index = None
        self._refresh_list(select_index=0 if self._session.config.grids else None)

    # -------------------------------------------------------------- teardown

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt override)
        self._preview_timer.stop()
        super().closeEvent(event)
