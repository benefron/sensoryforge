"""CSV import/export of neuron populations through the Mechanoreceptor tab.

Phase 2 (I7): a population's receptive fields are a ``ReceptiveFieldBank``;
CSV import goes through the ``imported`` builder (so the population can be
simulated) and a receptor-count mismatch raises instead of zero-filling.
The ``_CSVPopulationModule`` stub is gone.

Covers:
- Export writes neuron_positions.csv, innervation_weights.csv, bank.pt,
  manifest.json with correct content
- Export -> import gives bit-identical weights and centres
- Imported population is marked (csv_folder, innervation_method "imported")
- Receptor mismatch raises ValueError naming both counts
- CSV populations are preserved through _generate_populations
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`


_APP = None  # keep the QApplication alive for the module (F-016 pattern)


def _qt_app():
    global _APP
    try:
        from PyQt5 import QtWidgets

        _APP = QtWidgets.QApplication.instance()
        if _APP is None:
            _APP = QtWidgets.QApplication(sys.argv[:1])
        return _APP
    except ImportError:
        pytest.skip("PyQt5 not available")


def _tab_with_grid(rows=8, cols=8, spacing=0.15):
    """A MechanoreceptorTab with one generated regular grid and one population."""
    _qt_app()
    from PyQt5 import QtGui
    from sensoryforge.gui.tabs.mechanoreceptor_tab import GridEntry, MechanoreceptorTab

    tab = MechanoreceptorTab()
    entry = GridEntry(
        name="G", rows=rows, cols=cols, spacing=spacing, color=QtGui.QColor(90, 90, 200)
    )
    tab._add_grid_entry(entry)
    tab._on_add_population()
    tab._generate_populations()
    return tab


def test_stub_module_is_gone():
    _qt_app()
    import sensoryforge.gui.tabs.mechanoreceptor_tab as mod

    assert not hasattr(mod, "_CSVPopulationModule")


def test_export_writes_four_files_with_correct_manifest(tmp_path):
    tab = _tab_with_grid()
    pop = tab.populations[-1]
    target = tmp_path / "pop_csv"
    tab.export_population_csv(pop, target)
    for name in (
        "neuron_positions.csv",
        "innervation_weights.csv",
        "bank.pt",
        "manifest.json",
    ):
        assert (target / name).exists(), name
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["num_neurons"] == pop.num_neurons
    assert manifest["num_receptors"] == 64
    assert manifest["positions_file"] == "neuron_positions.csv"
    assert manifest["weights_file"] == "innervation_weights.csv"
    assert manifest["bank_file"] == "bank.pt"
    weights = np.loadtxt(target / "innervation_weights.csv", delimiter=",")
    assert weights.shape == (pop.num_neurons, 64)


def test_export_then_import_is_bit_identical(tmp_path):
    tab = _tab_with_grid()
    source = tab.populations[-1]
    target = tmp_path / "pop_csv"
    tab.export_population_csv(source, target)
    original_weights = source.bank.weights.clone()
    original_centers = source.bank.neuron_centers.clone()

    tab._on_add_population()
    imported = tab.populations[-1]
    tab.import_population_csv(imported, target)

    assert torch.equal(imported.bank.weights, original_weights)
    assert torch.equal(imported.bank.neuron_centers, original_centers)
    assert torch.equal(imported.bank.receptor_coords, source.bank.receptor_coords)
    assert imported.csv_folder == str(target)
    assert imported.innervation_method == "imported"
    assert imported.innervation_params["path"] == str(target.resolve())
    assert imported.bank.provenance["builder"] == "imported"
    assert imported.grid_shape == (8, 8)
    assert imported.innervation_weights.shape == (imported.num_neurons, 8, 8)
    assert "imported" in tab.lbl_population_info.text()


def test_export_requires_generated_population(tmp_path):
    _qt_app()
    from PyQt5 import QtGui
    from sensoryforge.gui.tabs.mechanoreceptor_tab import (
        MechanoreceptorTab,
        NeuronPopulation,
    )

    tab = MechanoreceptorTab()
    pop = NeuronPopulation(
        name="p",
        neuron_type="SA",
        color=QtGui.QColor(1, 2, 3),
        neurons_per_row=2,
        connections_per_neuron=5.0,
        sigma_d_mm=0.3,
        weight_min=0.1,
        weight_max=1.0,
    )
    with pytest.raises(ValueError, match="Generate"):
        tab.export_population_csv(pop, tmp_path / "x")


def test_receptor_mismatch_raises_naming_both_counts(tmp_path):
    tab = _tab_with_grid(rows=8, cols=8)
    tab.export_population_csv(tab.populations[-1], tmp_path / "pop_csv")

    other = _tab_with_grid(rows=6, cols=6)
    pop = other.populations[-1]
    before = pop.bank.weights.clone()
    with pytest.raises(ValueError, match=r"64.*36|36.*64"):
        other.import_population_csv(pop, tmp_path / "pop_csv")
    # the population is untouched by a failed import
    assert torch.equal(pop.bank.weights, before)
    assert pop.csv_folder is None


def test_import_requires_a_grid(tmp_path):
    tab = _tab_with_grid()
    tab.export_population_csv(tab.populations[-1], tmp_path / "pop_csv")
    _qt_app()
    from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab

    empty = MechanoreceptorTab()
    empty._on_add_population()
    with pytest.raises(RuntimeError, match="grid"):
        empty.import_population_csv(empty.populations[-1], tmp_path / "pop_csv")


def test_csv_population_preserved_through_generate(tmp_path):
    tab = _tab_with_grid()
    tab.export_population_csv(tab.populations[-1], tmp_path / "pop_csv")
    tab._on_add_population()
    pop = tab.populations[-1]
    tab.import_population_csv(pop, tmp_path / "pop_csv")
    bank = pop.bank

    tab._generate_populations()

    assert pop.csv_folder == str(tmp_path / "pop_csv")
    assert pop.bank is bank, "imported bank must not be rebuilt by generate"


def test_get_config_marks_imported_population(tmp_path):
    tab = _tab_with_grid()
    tab.export_population_csv(tab.populations[-1], tmp_path / "pop_csv")
    tab._on_add_population()
    pop = tab.populations[-1]
    tab.import_population_csv(pop, tmp_path / "pop_csv")
    cfg = tab.get_config()["populations"][-1]
    assert cfg["innervation_method"] == "imported"
    assert cfg["innervation_params"]["path"] == str((tmp_path / "pop_csv").resolve())
    # the exported config rebuilds the same bank through the engine
    from sensoryforge.config.schema import PopulationConfig
    from sensoryforge.core.simulation_engine import SimulationEngine

    pc = PopulationConfig.from_dict(cfg)
    params = SimulationEngine.builder_params(pc)
    assert params["path"] == cfg["innervation_params"]["path"]
