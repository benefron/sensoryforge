"""The pipeline strip: one row per population, chips for each build stage.

:class:`PipelineStrip` gives every screen the same at-a-glance picture of
the config: for each population, a chip for the sensor array it reads from,
one chip per receptive-field input, a combine chip when there is more than
one input, then filter, neuron and readout. A row above the populations
summarises the grids. Clicking a chip announces the stage and index so
``app.py`` can route the user to the right screen.

Chips are rebuilt only for the row (or the sensors row) a
``Session.configChanged`` path touches -- not on every keystroke across the
whole config -- by keeping ``PipelineStrip._pop_rows`` indexed by population.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtWidgets

from sensoryforge.config.schema import GridConfig, PopulationConfig, PopulationInput
from sensoryforge.gui import theme
from sensoryforge.gui.session import Session
from sensoryforge.gui.validation import validate


class _Row(QtWidgets.QWidget):
    """One strip row: a population's (or the sensors') chips.

    Attributes:
        chips_by_stage: Every chip on this row, keyed by stage name (e.g.
            ``"receptive_field"`` may hold more than one chip).
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.chips_by_stage: Dict[str, List[QtWidgets.QToolButton]] = {}
        self._layout = QtWidgets.QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        # Kept alive on the row so PyQt does not garbage-collect it once
        # __init__ returns (QButtonGroup is not reparented automatically).
        self.button_group = QtWidgets.QButtonGroup(self)
        self.button_group.setExclusive(True)

    def add_widget(self, widget: QtWidgets.QWidget) -> None:
        self._layout.addWidget(widget)

    def add_stretch(self) -> None:
        self._layout.addStretch(1)

    def add_chip(self, stage: str, chip: QtWidgets.QToolButton) -> None:
        self.chips_by_stage.setdefault(stage, []).append(chip)
        self.button_group.addButton(chip)
        self.add_widget(chip)


def _grid_summary(grid: Optional[GridConfig]) -> str:
    """A one-line summary of a grid for its chip's second line."""
    if grid is None:
        return "no grid"
    if grid.rows and grid.cols:
        return f"{grid.rows}×{grid.cols} @{grid.spacing} mm"
    return grid.arrangement


def _rf_value(
    population: PopulationConfig, population_input: PopulationInput
) -> Tuple[str, object]:
    """The one number worth showing on a receptive-field chip: d or sigma.

    Explicit ``inputs`` carry the value in ``rf.params``; the single-input
    sugar form (``PopulationConfig.effective_inputs``) leaves ``rf.params``
    empty and the value lives on the population itself instead.
    """
    params = population_input.rf.params
    if "resolvable_distance_mm" in params:
        return "d", params["resolvable_distance_mm"]
    if "sigma_d_mm" in params:
        return "σ", params["sigma_d_mm"]
    if not population.inputs:
        if population.resolvable_distance_mm is not None:
            return "d", population.resolvable_distance_mm
        return "σ", population.sigma_d_mm
    return "d", None


def _dot_status(path: str, errors: Dict[str, str], is_default: bool) -> Tuple[str, str]:
    """Colour and tooltip message for one chip's status dot.

    Args:
        path: The exact ``validate()`` key this chip is responsible for
            (``""`` if none).
        errors: The current ``validate(config)`` result.
        is_default: Whether the field(s) this chip shows are all at their
            schema default -- shown grey when true and no error, green
            otherwise.

    Returns:
        ``(hex_color, tooltip_message)``.
    """
    message = errors.get(path, "") if path else ""
    if message:
        return theme.PALETTE["error"], message
    if is_default:
        return theme.PALETTE["border_strong"], ""
    return theme.PALETTE["success"], ""


class PipelineStrip(QtWidgets.QWidget):
    """One row per population, plus a sensors row, chips per build stage.

    Signals:
        chipClicked(str, int): A chip was clicked -- the stage name
            (``"sensors"``, ``"sensor_array"``, ``"receptive_field"``,
            ``"combine"``, ``"filter"``, ``"neuron"``, ``"readout"``) and,
            for a population chip, the population index (for a sensors
            chip, the grid index).

    Example:
        >>> strip = PipelineStrip(session)                # doctest: +SKIP
        >>> strip.chipClicked.connect(on_chip_clicked)     # doctest: +SKIP
    """

    chipClicked = QtCore.pyqtSignal(str, int)

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._dirty = False
        self._pop_rows: Dict[int, _Row] = {}
        self._sensor_row: Optional[_Row] = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        header = QtWidgets.QHBoxLayout()
        header.addStretch(1)
        self._status_label = QtWidgets.QLabel("saved")
        header.addWidget(self._status_label)
        outer.addLayout(header)

        self._rows_layout = QtWidgets.QVBoxLayout()
        self._rows_layout.setSpacing(4)
        outer.addLayout(self._rows_layout)

        session.configReplaced.connect(partial(self._rebuild))
        session.configChanged.connect(self._on_config_changed)
        session.staleChanged.connect(self._update_status_label)

        self._rebuild()

    # ------------------------------------------------------------- chip build

    def _make_chip(
        self,
        stage: str,
        index: int,
        text: str,
        tooltip: str,
        color: str,
        row: _Row,
    ) -> QtWidgets.QToolButton:
        chip = QtWidgets.QToolButton()
        chip.setObjectName("Chip")
        chip.setCheckable(True)
        chip.setText(text)
        chip.setToolTip(tooltip)

        dot = QtWidgets.QLabel(chip)
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(
            f"background: {color}; border-radius: 5px; "
            "border: 1px solid rgba(0, 0, 0, 40);"
        )
        dot.setToolTip(tooltip)
        dot_layout = QtWidgets.QHBoxLayout(chip)
        dot_layout.setContentsMargins(4, 2, 4, 2)
        dot_layout.addStretch(1)
        dot_layout.addWidget(dot, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)
        chip.status_dot = dot  # kept for tests/introspection

        chip.clicked.connect(partial(self._emit_chip_clicked, stage, index))
        row.add_chip(stage, chip)
        return chip

    def _emit_chip_clicked(self, stage: str, index: int, checked: bool = False) -> None:
        self.chipClicked.emit(stage, index)

    # ------------------------------------------------------------------ build

    def _rebuild(self) -> None:
        """Rebuild every row from scratch (config replaced, or a grid changed)."""
        self._dirty = False
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._pop_rows = {}

        self._sensor_row = self._build_sensor_row()
        self._rows_layout.addWidget(self._sensor_row)
        for index in range(len(self._session.config.populations)):
            row = self._build_population_row(index)
            self._pop_rows[index] = row
            self._rows_layout.addWidget(row)
        self._update_status_label()

    def _build_sensor_row(self) -> _Row:
        row = _Row()
        label = QtWidgets.QLabel("Sensors")
        label.setObjectName("SectionTitle")
        row.add_widget(label)
        errors = validate(self._session.config)
        for index, grid in enumerate(self._session.config.grids):
            text = f"{grid.name}\n{_grid_summary(grid)}"
            color, message = _dot_status("", errors, is_default=False)
            self._make_chip("sensors", index, text, message, color, row)
        row.add_stretch()
        return row

    def _build_population_row(self, index: int) -> _Row:
        population = self._session.config.populations[index]
        errors = validate(self._session.config)
        row = _Row()

        name_label = QtWidgets.QLabel(population.name)
        qcolor = theme.population_color(index, population.neuron_type)
        name_label.setStyleSheet(f"color: {qcolor.name()}; font-weight: 600;")
        row.add_widget(name_label)

        grid = self._grid_by_name(population.target_grid)
        color, message = _dot_status(
            f"populations.{index}.target_grid",
            errors,
            is_default=population.target_grid is None,
        )
        self._make_chip(
            "sensor_array",
            index,
            f"Sensor array\n{_grid_summary(grid)}",
            message,
            color,
            row,
        )

        inputs = population.effective_inputs()
        rf_is_default = (
            not population.inputs and population.innervation_method == "gaussian"
        )
        for population_input in inputs:
            symbol, value = _rf_value(population, population_input)
            text = f"Receptive field\n{population_input.rf.method} {symbol}={value}"
            color, message = _dot_status(
                f"populations.{index}.target_grid", errors, is_default=rf_is_default
            )
            self._make_chip("receptive_field", index, text, message, color, row)

        if len(inputs) > 1:
            color, message = _dot_status(
                "", errors, is_default=population.combine == "sum"
            )
            self._make_chip(
                "combine", index, f"Combine\n{population.combine}", message, color, row
            )

        color, message = _dot_status(
            "", errors, is_default=population.filter_method == "none"
        )
        self._make_chip(
            "filter", index, f"Filter\n{population.filter_method}", message, color, row
        )

        preset = population.model_params.get("preset")
        neuron_text = f"Neuron\n{population.neuron_model}"
        if preset:
            neuron_text += f" ({preset})"
        color, message = _dot_status(
            "", errors, is_default=population.neuron_model == "Izhikevich"
        )
        self._make_chip("neuron", index, neuron_text, message, color, row)

        color, message = _dot_status(
            "", errors, is_default=population.readout == "auto"
        )
        self._make_chip(
            "readout", index, f"Readout\n{population.readout}", message, color, row
        )

        row.add_stretch()
        return row

    def _grid_by_name(self, name: Optional[str]) -> Optional[GridConfig]:
        if name is None:
            return None
        for grid in self._session.config.grids:
            if grid.name == name:
                return grid
        return None

    # ------------------------------------------------------------- listeners

    def _on_config_changed(self, path: str) -> None:
        self._dirty = True
        if path.startswith("grids."):
            self._rebuild()
            return
        if path.startswith("populations."):
            segments = path.split(".")
            if len(segments) >= 2:
                try:
                    index = int(segments[1])
                except ValueError:
                    index = None
                if index is not None and index in self._pop_rows:
                    self._replace_population_row(index)
                    self._update_status_label()
                    return
        self._update_status_label()

    def _replace_population_row(self, index: int) -> None:
        old_row = self._pop_rows.get(index)
        new_row = self._build_population_row(index)
        if old_row is not None:
            position = self._rows_layout.indexOf(old_row)
            self._rows_layout.removeWidget(old_row)
            old_row.setParent(None)
            old_row.deleteLater()
            self._rows_layout.insertWidget(position, new_row)
        else:
            self._rows_layout.addWidget(new_row)
        self._pop_rows[index] = new_row

    def _update_status_label(self, *_args: object) -> None:
        if self._session.stale:
            text, color_key = "● edited since last run", "warning"
        elif self._dirty:
            text, color_key = "● edited", "warning"
        else:
            text, color_key = "saved", "text_secondary"
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {theme.PALETTE[color_key]};")
