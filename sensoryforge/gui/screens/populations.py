"""The Populations screen (Task 2.3): edit every population's pipeline.

Layout: far left a population list (add/duplicate/remove, enabled
checkbox); centre a fixed chain of cards
(:mod:`~sensoryforge.gui.screens.populations_cards`) for the selected
population's inputs, neuron layout, filter, neuron model and readout; right
a column of bench tests (:mod:`sensoryforge.gui.bench`) that preview one
piece of the pipeline without a full simulation, plus a **Quick run** button
that runs just this population for 100 ms through
:class:`~sensoryforge.gui.execution.run_controller.RunController`.

The screen owns no config state of its own: every edit goes through
:meth:`~sensoryforge.gui.session.Session.set_by_path` (directly, or via the
cards' :class:`~sensoryforge.gui.widgets.param_form.ParamForm`\\ s), and every
view here reacts to :attr:`~sensoryforge.gui.session.Session.configChanged`.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from sensoryforge.gui.widgets.problem_list import ProblemList
from sensoryforge.config.schema import PopulationConfig
from sensoryforge.gui import theme
from sensoryforge.gui.bench import find_population, population_index
from sensoryforge.gui.bench.filter_step import FilterStepBench
from sensoryforge.gui.bench.neuron_trace import NeuronTraceBench
from sensoryforge.gui.bench.rf_footprint import RfFootprintBench
from sensoryforge.gui.execution.run_controller import RunController
from sensoryforge.gui.screens.dsl_editor import DslEditorDialog
from sensoryforge.gui.screens.populations_cards import (
    FilterCard,
    InputsCard,
    NeuronCard,
    NeuronLayoutCard,
    ReadoutCard,
)
from sensoryforge.gui.session import Session
from sensoryforge.gui.widgets import plot_factory

#: Bench recompute debounce (ms) -- CLAUDE.md/brief: "debounced 250 ms".
_BENCH_DEBOUNCE_MS = 250
#: Quick run duration, capped by RunController's own QUICK_DURATION_MS too.
_QUICK_RUN_MS = 100.0


class PopulationsScreen(QtWidgets.QWidget):
    """The Populations screen.

    Args:
        session: The experiment this screen edits.
        parent: Qt parent.
    """

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._selected: Optional[str] = None
        self._run_controller = RunController(session, self)
        self._run_controller.finished.connect(self._on_quick_finished)
        self._run_controller.failed.connect(self._on_quick_failed)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        splitter.addWidget(self._build_list_panel())
        splitter.addWidget(self._build_cards_panel())
        splitter.addWidget(self._build_bench_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([220, 620, 460])

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_BENCH_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._refresh_benches)

        session.configChanged.connect(self._on_config_changed)
        session.configReplaced.connect(self._on_config_replaced)

        self._refresh_list()

    # ---------------------------------------------------------------- left

    def _build_list_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(self._section_title("Populations"))

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.list_widget.itemDoubleClicked.connect(self._on_rename)
        layout.addWidget(self.list_widget, 1)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_add.clicked.connect(self._on_add)
        self.btn_duplicate = QtWidgets.QPushButton("Duplicate")
        self.btn_duplicate.clicked.connect(self._on_duplicate)
        self.btn_remove = QtWidgets.QPushButton("Remove")
        self.btn_remove.clicked.connect(self._on_remove)
        for btn in (self.btn_add, self.btn_duplicate, self.btn_remove):
            buttons.addWidget(btn)
        layout.addLayout(buttons)
        return panel

    # -------------------------------------------------------------- centre

    def _build_cards_panel(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        self.problems = ProblemList(self._session, self._problem_prefix)
        layout.addWidget(self.problems)

        self.inputs_card = InputsCard(self._session)
        self.layout_card = NeuronLayoutCard(self._session)
        self.filter_card = FilterCard(self._session)
        self.neuron_card = NeuronCard(self._session)
        self.neuron_card.dslEditRequested.connect(self._on_dsl_edit_requested)
        self.readout_card = ReadoutCard(self._session)

        self._cards = [
            self.inputs_card,
            self.layout_card,
            self.filter_card,
            self.neuron_card,
            self.readout_card,
        ]
        for card in self._cards:
            layout.addWidget(card)
        layout.addStretch(1)

        scroll.setWidget(content)
        return scroll

    # --------------------------------------------------------------- right

    def _build_bench_panel(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.addWidget(self._section_title("Bench tests"))

        rf_box = QtWidgets.QGroupBox("RF footprint")
        rf_layout = QtWidgets.QVBoxLayout(rf_box)
        self.rf_bench = RfFootprintBench(self._session)
        rf_layout.addWidget(self.rf_bench)
        layout.addWidget(rf_box)

        filter_box = QtWidgets.QGroupBox("Filter step response")
        filter_layout = QtWidgets.QVBoxLayout(filter_box)
        self.filter_bench = FilterStepBench(self._session)
        filter_layout.addWidget(self.filter_bench)
        layout.addWidget(filter_box)

        neuron_box = QtWidgets.QGroupBox("Neuron trace / f-I")
        neuron_layout = QtWidgets.QVBoxLayout(neuron_box)
        self.neuron_bench = NeuronTraceBench(self._session)
        neuron_layout.addWidget(self.neuron_bench)
        layout.addWidget(neuron_box)

        quick_box = QtWidgets.QGroupBox("Quick run (100 ms, this population only)")
        quick_layout = QtWidgets.QVBoxLayout(quick_box)
        self.btn_quick_run = QtWidgets.QPushButton("Quick run")
        self.btn_quick_run.clicked.connect(self._on_quick_run)
        quick_layout.addWidget(self.btn_quick_run)
        self.quick_plot = plot_factory.make_plot(
            "Quick run raster", "Time", "Neuron", x_unit="ms"
        )
        self.quick_raster = plot_factory.make_raster_item(theme.PALETTE["accent"])
        self.quick_plot.addItem(self.quick_raster)
        quick_layout.addWidget(self.quick_plot)
        self.quick_error = QtWidgets.QLabel("")
        self.quick_error.setStyleSheet(f"color: {theme.PALETTE['error']};")
        self.quick_error.setWordWrap(True)
        self.quick_error.setVisible(False)
        quick_layout.addWidget(self.quick_error)
        layout.addWidget(quick_box)

        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _section_title(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    # ------------------------------------------------------------ list ops

    def _current_name(self) -> Optional[str]:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        return item.data(QtCore.Qt.UserRole)

    def _select(self, name: Optional[str]) -> None:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item.data(QtCore.Qt.UserRole) == name:
                self.list_widget.setCurrentRow(row)
                return

    def _refresh_list(self) -> None:
        previous = self._current_name() or self._selected
        self.list_widget.blockSignals(True)
        try:
            self.list_widget.clear()
            for index, pop in enumerate(self._session.config.populations):
                item = QtWidgets.QListWidgetItem(f"{pop.name}  ({pop.neuron_type})")
                item.setData(QtCore.Qt.UserRole, pop.name)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.Checked if pop.enabled else QtCore.Qt.Unchecked
                )
                item.setForeground(theme.population_color(index, pop.neuron_type))
                self.list_widget.addItem(item)
            names = [p.name for p in self._session.config.populations]
            if previous in names:
                self.list_widget.setCurrentRow(names.index(previous))
            elif names:
                self.list_widget.setCurrentRow(0)
        finally:
            self.list_widget.blockSignals(False)
        self._on_selection_changed()

    def _unique_name(self, base: str) -> str:
        existing = {p.name for p in self._session.config.populations}
        if base not in existing:
            return base
        n = 2
        while f"{base} {n}" in existing:
            n += 1
        return f"{base} {n}"

    def _on_add(self) -> None:
        config = self._session.config
        name = self._unique_name("Population")
        grid_name = config.grids[0].name if config.grids else None
        new_pop = PopulationConfig(name=name, target_grid=grid_name)
        updated = list(config.populations) + [new_pop]
        self._session.set_by_path("populations", updated)
        self._refresh_list()
        self._select(name)

    def _on_duplicate(self) -> None:
        name = self._current_name()
        pop_cfg = find_population(self._session.config, name)
        if pop_cfg is None:
            return
        new_cfg = copy.deepcopy(pop_cfg)
        new_cfg.name = self._unique_name(f"{pop_cfg.name} copy")
        updated = list(self._session.config.populations) + [new_cfg]
        self._session.set_by_path("populations", updated)
        self._refresh_list()
        self._select(new_cfg.name)

    def _on_remove(self) -> None:
        name = self._current_name()
        if name is None:
            return
        updated = [p for p in self._session.config.populations if p.name != name]
        self._session.set_by_path("populations", updated)
        self._refresh_list()

    def _on_rename(self, item: QtWidgets.QListWidgetItem) -> None:
        name = item.data(QtCore.Qt.UserRole)
        pop_cfg = find_population(self._session.config, name)
        if pop_cfg is None:
            return
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename population", "Name:", text=name
        )
        new_name = new_name.strip()
        if not ok or not new_name or new_name == name:
            return
        existing = {p.name for p in self._session.config.populations} - {name}
        if new_name in existing:
            QtWidgets.QMessageBox.warning(
                self, "Rename population", f"{new_name!r} is already in use."
            )
            return
        index = population_index(self._session.config, name)
        self._session.set_by_path(f"populations.{index}.name", new_name)
        self._refresh_list()
        self._select(new_name)

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        name = item.data(QtCore.Qt.UserRole)
        pop_cfg = find_population(self._session.config, name)
        if pop_cfg is None:
            return
        enabled = item.checkState() == QtCore.Qt.Checked
        if pop_cfg.enabled == enabled:
            return
        index = population_index(self._session.config, name)
        self._session.set_by_path(f"populations.{index}.enabled", enabled)

    # ------------------------------------------------------------- binding

    def _problem_prefix(self) -> Optional[str]:
        """The selected population's validation key, for the problem list."""
        name = getattr(self, "_selected", None)
        if name is None:
            return None
        try:
            return f"populations.{population_index(self._session.config, name)}"
        except ValueError:
            return None

    def _on_selection_changed(self) -> None:
        self._selected = self._current_name()
        self.problems.refresh()
        for card in self._cards:
            card.set_population(self._selected)
        self.rf_bench.set_population(self._selected)
        self.filter_bench.set_population(self._selected)
        self.neuron_bench.set_population(self._selected)

    def _on_config_replaced(self) -> None:
        self._refresh_list()

    def _on_config_changed(self, path: str) -> None:
        if path in ("populations", ""):
            self._refresh_list()
            return
        if self._selected is None:
            return
        try:
            index = population_index(self._session.config, self._selected)
        except ValueError:
            return
        prefix = f"populations.{index}."
        relevant = path.startswith(prefix) or path in (
            "simulation.dt_ms",
            "simulation.integrate_dt_ms",
        )
        if relevant:
            self._debounce.start()
        # The list row's label (name, neuron_type, enabled) can change from
        # a card edit too (e.g. neuron_type isn't card-editable today, but
        # enabled/name changes elsewhere should still be reflected).
        if path == f"{prefix}enabled" or path == f"{prefix}name":
            self._refresh_list()

    def _refresh_benches(self) -> None:
        self.rf_bench.refresh()
        self.filter_bench.refresh()
        self.neuron_bench.refresh()

    def _on_dsl_edit_requested(self, population_name: str) -> None:
        dialog = DslEditorDialog(self._session, population_name, self)
        dialog.exec_()

    # ---------------------------------------------------------------- run

    def _on_quick_run(self) -> None:
        if self._selected is None or self._run_controller.running:
            return
        self.quick_error.setVisible(False)
        try:
            self._run_controller.run(
                duration_ms=_QUICK_RUN_MS, bundle=False, quick_population=self._selected
            )
        except (RuntimeError, ValueError) as exc:
            self.quick_error.setText(str(exc))
            self.quick_error.setVisible(True)

    def _on_quick_finished(self, result: Any) -> None:
        pop_results: Optional[Dict[str, Any]] = result.results.get(self._selected)
        if pop_results is None or "spikes" not in pop_results:
            self.quick_raster.setData([], [])
            return
        counts = pop_results["spikes"][0].detach().cpu().numpy()
        time_idx, neuron_idx = np.nonzero(counts > 0)
        dt_ms = result.config_snapshot.simulation.dt_ms
        self.quick_raster.setData(x=time_idx * dt_ms, y=neuron_idx)

    def _on_quick_failed(self, message: str) -> None:
        self.quick_error.setText(message)
        self.quick_error.setVisible(True)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        plot_factory.teardown(self.quick_plot)
        super().closeEvent(event)
