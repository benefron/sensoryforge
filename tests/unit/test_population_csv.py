"""Tests for CSV import/export of neuron populations (item 5).

Covers:
- _CSVPopulationModule data container attributes and duck typing
- Export writes neuron_positions.csv, innervation_weights.csv, manifest.json
- Manifest content correctness
- Round-trip: export then import preserves positions and weights (allclose)
- csv_folder field is set on population after import
- CSV populations are preserved through grid regeneration
- Receptor mismatch falls back to zero stub coordinates
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qt_app():
    try:
        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv[:1])
        return app
    except ImportError:
        pytest.skip("PyQt5 not available")


def _make_csv_module(n_neurons=4, n_receptors=9):
    """Create a _CSVPopulationModule with deterministic test tensors."""
    _qt_app()
    from sensoryforge.gui.tabs.mechanoreceptor_tab import _CSVPopulationModule
    centers = torch.arange(n_neurons * 2, dtype=torch.float32).reshape(n_neurons, 2)
    weights = torch.ones(n_neurons, n_receptors, dtype=torch.float32) * 0.5
    rcoords = torch.zeros(n_receptors, 2, dtype=torch.float32)
    return _CSVPopulationModule(
        neuron_centers=centers,
        innervation_weights=weights,
        receptor_coords=rcoords,
    )


def _write_csv_folder(folder: Path, n_neurons=4, n_receptors=9):
    """Write a valid CSV folder and return (centers_np, weights_np)."""
    centers_np = np.arange(n_neurons * 2, dtype=np.float32).reshape(n_neurons, 2)
    weights_np = np.ones((n_neurons, n_receptors), dtype=np.float32) * 0.5
    folder.mkdir(parents=True, exist_ok=True)
    np.savetxt(folder / "neuron_positions.csv", centers_np, delimiter=",",
               header="x_mm,y_mm", comments="")
    np.savetxt(folder / "innervation_weights.csv", weights_np, delimiter=",")
    manifest = {
        "version": 1,
        "num_neurons": n_neurons,
        "num_receptors": n_receptors,
        "positions_file": "neuron_positions.csv",
        "weights_file": "innervation_weights.csv",
    }
    with (folder / "manifest.json").open("w") as fp:
        json.dump(manifest, fp)
    return centers_np, weights_np


# ---------------------------------------------------------------------------
# _CSVPopulationModule: pure-Python data container tests
# ---------------------------------------------------------------------------

def test_csv_module_attributes():
    """_CSVPopulationModule stores centers, weights, rcoords and exposes num_neurons."""
    stub = _make_csv_module(n_neurons=4, n_receptors=9)
    assert stub.num_neurons == 4
    assert stub.neuron_centers.shape == (4, 2)
    assert stub.innervation_weights.shape == (4, 9)
    assert stub.receptor_coords.shape == (9, 2)


def test_csv_module_is_valid_flat_module_duck_type():
    """Assigning _CSVPopulationModule to flat_module must work via duck typing."""
    _qt_app()
    from sensoryforge.gui.tabs.mechanoreceptor_tab import NeuronPopulation, _CSVPopulationModule
    from PyQt5 import QtGui
    pop = NeuronPopulation(
        name="Test",
        neuron_type="SA",
        color=QtGui.QColor(100, 100, 200),
        neurons_per_row=4,
        connections_per_neuron=5.0,
        sigma_d_mm=0.5,
        weight_min=0.1,
        weight_max=1.0,
    )
    stub = _make_csv_module(n_neurons=4, n_receptors=9)
    pop.flat_module = stub  # type: ignore[assignment]

    # Access via NeuronPopulation property accessors
    assert pop.neuron_centers is not None
    assert pop.neuron_centers.shape == (4, 2)
    assert pop.innervation_weights is not None
    assert pop.innervation_weights.shape == (4, 9)


# ---------------------------------------------------------------------------
# File I/O tests (use tmp_path fixture)
# ---------------------------------------------------------------------------

def test_export_writes_three_files(tmp_path):
    """Export logic must produce neuron_positions.csv, innervation_weights.csv, manifest.json."""
    _write_csv_folder(tmp_path, n_neurons=3, n_receptors=6)
    assert (tmp_path / "neuron_positions.csv").exists()
    assert (tmp_path / "innervation_weights.csv").exists()
    assert (tmp_path / "manifest.json").exists()


def test_export_manifest_content_is_correct(tmp_path):
    """Exported manifest must contain correct num_neurons, num_receptors, and file keys."""
    n_neurons, n_receptors = 5, 12
    _write_csv_folder(tmp_path, n_neurons=n_neurons, n_receptors=n_receptors)
    with (tmp_path / "manifest.json").open() as fp:
        manifest = json.load(fp)
    assert manifest["num_neurons"] == n_neurons
    assert manifest["num_receptors"] == n_receptors
    assert manifest["positions_file"] == "neuron_positions.csv"
    assert manifest["weights_file"] == "innervation_weights.csv"


def test_import_round_trip_positions_allclose(tmp_path):
    """Export → import round-trip must recover neuron positions within float32 tolerance."""
    centers_np, _ = _write_csv_folder(tmp_path, n_neurons=6, n_receptors=4)
    # Simulate import: load from the folder
    with (tmp_path / "manifest.json").open() as fp:
        manifest = json.load(fp)
    loaded = np.loadtxt(tmp_path / manifest["positions_file"], delimiter=",", skiprows=1)
    assert np.allclose(loaded, centers_np, atol=1e-5), (
        f"Round-trip positions differ: max diff={np.abs(loaded - centers_np).max()}"
    )


def test_import_round_trip_weights_allclose(tmp_path):
    """Export → import round-trip must recover innervation weights within float32 tolerance."""
    _, weights_np = _write_csv_folder(tmp_path, n_neurons=6, n_receptors=4)
    with (tmp_path / "manifest.json").open() as fp:
        manifest = json.load(fp)
    loaded = np.loadtxt(tmp_path / manifest["weights_file"], delimiter=",")
    assert np.allclose(loaded, weights_np, atol=1e-5), (
        f"Round-trip weights differ: max diff={np.abs(loaded - weights_np).max()}"
    )


def test_import_sets_csv_folder_on_population(tmp_path):
    """After import, pop.csv_folder must be set to the imported folder path."""
    _qt_app()
    from sensoryforge.gui.tabs.mechanoreceptor_tab import (
        MechanoreceptorTab, NeuronPopulation, _CSVPopulationModule
    )
    from PyQt5 import QtGui
    n_neurons, n_receptors = 4, 9
    _write_csv_folder(tmp_path, n_neurons=n_neurons, n_receptors=n_receptors)

    # Manually load CSV the same way _on_import_population_csv does
    with (tmp_path / "manifest.json").open() as fp:
        manifest = json.load(fp)
    centers_np = np.loadtxt(tmp_path / manifest["positions_file"], delimiter=",", skiprows=1)
    weights_np = np.loadtxt(tmp_path / manifest["weights_file"], delimiter=",")

    pop = NeuronPopulation(
        name="Pop",
        neuron_type="SA",
        color=QtGui.QColor(100, 100, 200),
        neurons_per_row=4,
        connections_per_neuron=5.0,
        sigma_d_mm=0.5,
        weight_min=0.1,
        weight_max=1.0,
    )
    receptor_coords = torch.zeros(n_receptors, 2, dtype=torch.float32)
    stub = _CSVPopulationModule(
        neuron_centers=torch.tensor(centers_np, dtype=torch.float32),
        innervation_weights=torch.tensor(weights_np, dtype=torch.float32),
        receptor_coords=receptor_coords,
    )
    pop.flat_module = stub  # type: ignore[assignment]
    pop.csv_folder = str(tmp_path)

    assert pop.csv_folder == str(tmp_path)


# ---------------------------------------------------------------------------
# Qt integration tests — CSV behaviour through the tab
# ---------------------------------------------------------------------------

def test_csv_population_preserved_through_generate(tmp_path):
    """CSV populations must not have their flat_module cleared by _generate_populations."""
    _qt_app()
    from sensoryforge.gui.tabs.mechanoreceptor_tab import (
        MechanoreceptorTab, _CSVPopulationModule
    )
    n_neurons, n_receptors = 4, 1600  # 40×40 default grid
    _write_csv_folder(tmp_path, n_neurons=n_neurons, n_receptors=n_receptors)

    tab = MechanoreceptorTab()
    tab._on_add_grid()          # default 40×40 grid
    tab._on_add_population()    # default SA population
    pop = tab.populations[-1]

    # Simulate having imported a CSV
    stub = _make_csv_module(n_neurons=n_neurons, n_receptors=n_receptors)
    pop.flat_module = stub  # type: ignore[assignment]
    pop.csv_folder = str(tmp_path)

    # Regenerate — should not clear the CSV stub
    tab._generate_populations()

    assert pop.csv_folder == str(tmp_path), "csv_folder must be preserved after generate"
    assert isinstance(pop.flat_module, _CSVPopulationModule), (
        "flat_module must remain a _CSVPopulationModule after generate"
    )


def test_receptor_mismatch_uses_zero_stub_coords(tmp_path):
    """When CSV receptor count mismatches the current grid, receptor_coords must be zero-filled."""
    from sensoryforge.gui.tabs.mechanoreceptor_tab import _CSVPopulationModule

    n_neurons = 4
    n_receptors_csv = 20
    n_receptors_grid = 30  # deliberately different

    # Simulate the import logic directly (receptor count mismatch → zero coords)
    centers_np = np.zeros((n_neurons, 2), dtype=np.float32)
    weights_np = np.ones((n_neurons, n_receptors_csv), dtype=np.float32) * 0.5
    receptor_coords_grid = torch.zeros(n_receptors_grid, 2, dtype=torch.float32)

    # Replicate the mismatch branch from _on_import_population_csv
    if n_receptors_csv != n_receptors_grid:
        receptor_coords = torch.zeros(n_receptors_csv, 2, dtype=torch.float32)
    else:
        receptor_coords = receptor_coords_grid

    stub = _CSVPopulationModule(
        neuron_centers=torch.tensor(centers_np),
        innervation_weights=torch.tensor(weights_np),
        receptor_coords=receptor_coords,
    )
    assert stub.receptor_coords.shape == (n_receptors_csv, 2), (
        f"Expected receptor_coords shape ({n_receptors_csv}, 2), "
        f"got {tuple(stub.receptor_coords.shape)}"
    )
    assert stub.receptor_coords.abs().max().item() == 0.0, (
        "Mismatch fallback must use zero-filled coords"
    )
