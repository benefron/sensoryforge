"""RF footprint bench: one population's receptive-field bank, built exactly

as :class:`~sensoryforge.core.simulation_engine.SimulationEngine` builds it,
previewed with a neuron-index spin box and click-to-select.

Faithfulness (not re-derivation) is the whole point here: rather than
re-implementing :meth:`SimulationEngine._build_input_bank`'s grid/channel/
builder-parameter logic, :func:`build_population_bank_for_config` runs a real
:class:`SimulationEngine` over a config reduced to just this one population
(force-enabled) and reads its ``bank`` back out. That is bit-identical to
what a full run over the whole config would build for this population,
since bank construction has no cross-population state.
"""

from __future__ import annotations

from pathlib import Path
import copy
from typing import Optional

import numpy as np
from PyQt5 import QtWidgets

from sensoryforge.core.rf_builders.imported import write_csv_folder
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.rf_bank import ReceptiveFieldBank
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.gui import theme
from sensoryforge.gui.bench import find_population
from sensoryforge.gui.session import Session
from sensoryforge.gui.widgets.grid_preview import GridPreview


def build_population_bank_for_config(
    config: SensoryForgeConfig, population_name: str
) -> ReceptiveFieldBank:
    """Build ``population_name``'s bank the way :class:`SimulationEngine` does.

    Args:
        config: The full config (every grid is kept -- an input may name
            any of them).
        population_name: The population to build.

    Returns:
        That population's combined :class:`ReceptiveFieldBank`.

    Raises:
        ValueError: If no population has that name, or building it produces
            nothing (e.g. every input names an unknown grid).
    """
    snapshot = copy.deepcopy(config)
    target = find_population(snapshot, population_name)
    if target is None:
        known = [p.name for p in config.populations]
        raise ValueError(
            f"no population named {population_name!r}; the config has {known}"
        )
    target.enabled = True
    snapshot.populations = [target]
    engine = SimulationEngine(snapshot, device="cpu")
    if not engine.populations:
        raise ValueError(
            f"population {population_name!r} built no bank -- check its inputs"
        )
    return engine.populations[0]["bank"]


class RfFootprintBench(QtWidgets.QWidget):
    """RF footprint preview: a :class:`GridPreview` plus a neuron picker.

    Args:
        session: The experiment the previewed population belongs to.
        parent: Qt parent.
    """

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._population_name: Optional[str] = None
        self._bank: Optional[ReceptiveFieldBank] = None
        self._neuron_index: Optional[int] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Neuron index:"))
        self.spin_neuron = QtWidgets.QSpinBox()
        self.spin_neuron.setRange(0, 0)
        self.spin_neuron.valueChanged.connect(self._on_index_changed)
        header.addWidget(self.spin_neuron)
        header.addStretch(1)
        self.export_rf_button = QtWidgets.QToolButton()
        self.export_rf_button.setText("Export receptive fields…")
        self.export_rf_button.setToolTip(
            "Write this population's receptive fields as a CSV folder "
            "(neuron positions, weights, bank.pt, manifest) -- the format the "
            "'imported' RF builder reads."
        )
        self.export_rf_button.clicked.connect(self._on_export_rf_clicked)
        header.addWidget(self.export_rf_button)
        #: ``(parent, title) -> str`` returning the chosen folder ("" to
        #: cancel); a directory dialog by default, replaceable in tests.
        self.choose_folder = _ask_for_folder
        layout.addLayout(header)

        self.preview = GridPreview()
        self.preview.neuronClicked.connect(self._on_neuron_clicked)
        layout.addWidget(self.preview, 1)

        self.caption = QtWidgets.QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setObjectName("SectionTitle")
        layout.addWidget(self.caption)

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setStyleSheet(f"color: {theme.PALETTE['error']};")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

    def export_receptive_fields(self, folder) -> Path:
        """Write the current bank as an ``imported``-builder CSV folder.

        Args:
            folder: Destination directory (created if needed).

        Returns:
            The folder written.

        Raises:
            ValueError: If no bank has been built (see :attr:`bank`).
        """
        # Rebuild from the config as it is now, so an edit still waiting on
        # the preview's debounce is never exported stale.
        self.refresh()
        if self._bank is None:
            raise ValueError("no receptive fields to export: the bank did not build")
        return write_csv_folder(self._bank, Path(folder))

    def _on_export_rf_clicked(self, *_args: object) -> None:
        folder = self.choose_folder(self, "Export receptive fields to folder")
        if not folder:
            return
        try:
            written = self.export_receptive_fields(folder)
        except (ValueError, OSError) as exc:
            self.error_label.setText(f"Export failed: {exc}")
            self.error_label.setVisible(True)
            return
        self.export_rf_button.setToolTip(f"Last export: {written}")

    @property
    def neuron_index(self) -> int:
        """The neuron whose footprint is shown."""
        return self.spin_neuron.value()

    @property
    def bank(self) -> Optional[ReceptiveFieldBank]:
        """The last successfully built bank, or ``None``."""
        return self._bank

    def set_population(self, population_name: Optional[str]) -> None:
        """Point the bench at a different population and recompute.

        The bench starts on the neuron nearest the array centre: neuron 0 sits
        in a corner, where its footprint is clipped and hard to see.
        """
        self._population_name = population_name
        self._neuron_index = None
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the bank from the session's current config and redraw."""
        if self._population_name is None:
            self._bank = None
            self.preview.clear_populations()
            self.caption.setText("")
            self.error_label.setVisible(False)
            return
        try:
            bank = build_population_bank_for_config(
                self._session.config, self._population_name
            )
        except (ValueError, KeyError, RuntimeError, TypeError) as exc:
            self._bank = None
            self.preview.clear_populations()
            self.error_label.setText(f"RF footprint unavailable: {exc}")
            self.error_label.setVisible(True)
            self.caption.setText("")
            return

        self.error_label.setVisible(False)
        self._bank = bank

        pop_cfg = find_population(self._session.config, self._population_name)
        color = theme.population_color(0, pop_cfg.neuron_type if pop_cfg else None)

        self.preview.set_grids(self._session.config.grids)
        self.preview.clear_populations()
        self.preview.set_population(self._population_name, bank, color)

        n_neurons = bank.num_neurons
        self.spin_neuron.blockSignals(True)
        self.spin_neuron.setRange(0, max(0, n_neurons - 1))
        if self._neuron_index is None or self._neuron_index >= n_neurons:
            self._neuron_index = _central_neuron(bank)
        self.spin_neuron.setValue(self._neuron_index)
        self.spin_neuron.blockSignals(False)

        self._show_footprint()
        self._update_caption()

    def _on_index_changed(self, value: int) -> None:
        self._neuron_index = value
        self._show_footprint()

    def _on_neuron_clicked(self, name: str, index: int) -> None:
        if name != self._population_name:
            return
        self.spin_neuron.setValue(index)

    def _show_footprint(self) -> None:
        if self._bank is None:
            return
        self.preview.show_rf_footprint(self._population_name, self._neuron_index)

    def _update_caption(self) -> None:
        bank = self._bank
        if bank is None:
            self.caption.setText("")
            return
        weights = bank.weights.detach().cpu().numpy()
        counts = (weights != 0.0).sum(axis=1)
        if counts.size:
            lo, med, hi = int(counts.min()), int(np.median(counts)), int(counts.max())
        else:
            lo = med = hi = 0
        self.caption.setText(
            f"{bank.num_neurons} neurons — receptors/neuron min {lo}, "
            f"median {med}, max {hi}"
        )


def _central_neuron(bank: ReceptiveFieldBank) -> int:
    """Index of the neuron whose centre is nearest the receptors' centroid."""
    if bank.num_neurons == 0:
        return 0
    centroid = bank.receptor_coords.mean(dim=0)
    return int(((bank.neuron_centers - centroid) ** 2).sum(dim=1).argmin())


def _ask_for_folder(parent: QtWidgets.QWidget, title: str) -> str:
    return QtWidgets.QFileDialog.getExistingDirectory(parent, title)
