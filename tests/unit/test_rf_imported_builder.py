"""Tests for the ``imported`` receptive-field builder (Phase 2, I5).

Three sources: (a) the GUI's CSV export folder, (b) a ``.pt`` bank file,
(c) an ``.npz`` shaped like pressure-simulation's ``ConstructedRF`` whose
centres are ``[y, x]`` and whose ``H`` columns follow its ``[y, x]``
row-major grid ordering. Coordinates are ``(x, y)`` in mm inside
SensoryForge, so (c) is converted at the boundary (Phase 2 guardrail 4).
"""

import hashlib
import json

import numpy as np
import pytest
import torch

from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.core.rf_bank import ReceptiveFieldBank
from sensoryforge.core.rf_builders.imported import ImportedRFBuilder
from sensoryforge.register_components import register_all
from sensoryforge.registry import INNERVATION_REGISTRY
from sensoryforge.testing.contracts import check_component

register_all()

N_X, N_Y, SPACING = 4, 6, 0.15  # asymmetric on purpose


@pytest.fixture(scope="module")
def coords():
    return ReceptorGrid(
        grid_size=(N_X, N_Y), spacing=SPACING
    ).get_receptor_coordinates()


@pytest.fixture(scope="module")
def source_bank(coords):
    torch.manual_seed(0)
    weights = torch.rand(5, coords.shape[0])
    centers = torch.rand(5, 2)
    return ReceptiveFieldBank(
        weights, centers, coords, provenance={"builder": "gaussian", "seed": 1}
    )


def _write_csv_folder(folder, bank):
    folder.mkdir()
    np.savetxt(
        folder / "neuron_positions.csv",
        bank.neuron_centers.numpy(),
        delimiter=",",
        header="x_mm,y_mm",
        comments="",
    )
    np.savetxt(folder / "innervation_weights.csv", bank.weights.numpy(), delimiter=",")
    manifest = {
        "version": 1,
        "num_neurons": bank.num_neurons,
        "num_receptors": bank.num_receptors,
        "positions_file": "neuron_positions.csv",
        "weights_file": "innervation_weights.csv",
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def test_registered_under_imported():
    assert INNERVATION_REGISTRY.get_class("imported") is ImportedRFBuilder


def test_csv_folder_round_trip(tmp_path, coords, source_bank):
    folder = _write_csv_folder(tmp_path / "pop_csv", source_bank)
    b = ImportedRFBuilder(coords, path=str(folder))
    bank = b.build()
    # CSV text round trip is exact for float32 written with %.18e
    assert torch.allclose(bank.weights, source_bank.weights, atol=0, rtol=0)
    assert torch.allclose(bank.neuron_centers, source_bank.neuron_centers)
    assert torch.equal(bank.receptor_coords, coords)
    prov = bank.provenance
    assert prov["builder"] == "imported"
    assert prov["source_path"] == str(folder.resolve())
    assert prov["source_format"] == "csv_folder"
    for name in ("manifest.json", "neuron_positions.csv", "innervation_weights.csv"):
        expected = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        assert prov["source_files"][name] == expected
    assert len(prov["source_sha256"]) == 64


def test_pt_bank_round_trip(tmp_path, coords, source_bank):
    path = tmp_path / "bank.pt"
    source_bank.save(path)
    bank = ImportedRFBuilder(coords, path=str(path)).build()
    assert torch.equal(bank.weights, source_bank.weights)
    assert torch.equal(bank.neuron_centers, source_bank.neuron_centers)
    assert bank.provenance["source_format"] == "pt"
    assert (
        bank.provenance["source_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    # the original bank's provenance is kept under the import record
    assert bank.provenance["source_provenance"]["builder"] == "gaussian"


def test_pt_pressure_simulation_population_file(tmp_path, coords):
    # [N, H, W] weights and no receptor_coords: the target grid supplies them.
    n = 3
    weights = torch.rand(n, N_X, N_Y)
    torch.save(
        {"innervation_weights": weights, "neuron_centers": torch.rand(n, 2)},
        tmp_path / "population_01_SA_3.pt",
    )
    bank = ImportedRFBuilder(
        coords, path=str(tmp_path / "population_01_SA_3.pt")
    ).build()
    assert torch.equal(bank.weights, weights.reshape(n, -1))
    assert torch.equal(bank.receptor_coords, coords)


def _ps_grid_coords_yx(n_y, n_x, spacing):
    """pressure-simulation's _grid_coords: (y, x) columns, y-slow row-major."""
    y = torch.linspace(-(n_y - 1) * spacing / 2, (n_y - 1) * spacing / 2, n_y)
    x = torch.linspace(-(n_x - 1) * spacing / 2, (n_x - 1) * spacing / 2, n_x)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([yy.flatten(), xx.flatten()], dim=1)


def test_npz_converts_yx_centres_and_column_order(tmp_path, coords):
    # Two neurons at physical (x, y) = (0.225, -0.15) and (-0.075, 0.375):
    # swapping x and y moves them to different receptors on this 4x6 grid.
    centers_xy = torch.tensor([[0.225, -0.15], [-0.075, 0.375]])
    sigma = 0.1
    ps_coords = _ps_grid_coords_yx(N_Y, N_X, SPACING)  # (y, x), y-slow
    centers_yx = centers_xy[:, [1, 0]]
    d2 = ((centers_yx.unsqueeze(1) - ps_coords.unsqueeze(0)) ** 2).sum(-1)
    H = torch.exp(-d2 / (2 * sigma**2))
    H = H / H.norm(dim=1, keepdim=True)
    path = tmp_path / "constructed_rf.npz"
    np.savez(
        path,
        H=H.numpy(),
        centers=centers_yx.numpy(),
        sigma=np.float64(sigma),
        pitch=np.float64(0.3),
    )

    bank = ImportedRFBuilder(coords, path=str(path)).build()
    assert torch.allclose(bank.neuron_centers, centers_xy)
    # each neuron's strongest receptor is the SensoryForge receptor nearest
    # its (x, y) centre -- wrong if either the centre or the column order
    # were left in [y, x]
    for n in range(2):
        nearest = ((coords - centers_xy[n]) ** 2).sum(-1).argmin()
        assert bank.weights[n].argmax() == nearest
    # weights are a pure re-indexing of H: same sorted values per row
    assert torch.allclose(
        torch.sort(bank.weights, dim=1).values, torch.sort(H, dim=1).values
    )
    assert bank.provenance["source_format"] == "npz"
    assert bank.provenance["source_sigma_mm"] == pytest.approx(sigma)
    assert bank.provenance["source_pitch_mm"] == pytest.approx(0.3)
    assert (
        bank.provenance["source_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_npz_without_conversion_would_be_wrong(tmp_path, coords):
    # Sanity check on the test above: a naive import (no centre swap, no
    # column permutation) puts the peak on the wrong receptor.
    centers_xy = torch.tensor([[0.225, -0.15]])
    ps_coords = _ps_grid_coords_yx(N_Y, N_X, SPACING)
    d2 = ((centers_xy[:, [1, 0]].unsqueeze(1) - ps_coords.unsqueeze(0)) ** 2).sum(-1)
    H = torch.exp(-d2 / (2 * 0.1**2))
    nearest = ((coords - centers_xy[0]) ** 2).sum(-1).argmin()
    assert H[0].argmax() != nearest


def test_receptor_count_mismatch_raises_naming_both(tmp_path, coords, source_bank):
    folder = _write_csv_folder(tmp_path / "pop_csv", source_bank)
    small = ReceptorGrid(grid_size=(3, 3), spacing=0.15).get_receptor_coordinates()
    with pytest.raises(ValueError, match=r"24.*9|9.*24"):
        ImportedRFBuilder(small, path=str(folder))


def test_pt_with_mismatched_receptor_coords_raises(tmp_path, coords, source_bank):
    path = tmp_path / "bank.pt"
    source_bank.save(path)
    small = ReceptorGrid(grid_size=(3, 3), spacing=0.15).get_receptor_coordinates()
    with pytest.raises(ValueError, match=r"24.*9|9.*24"):
        ImportedRFBuilder(small, path=str(path))


def test_missing_path_raises(tmp_path, coords):
    with pytest.raises(ValueError, match="path"):
        ImportedRFBuilder(coords)
    with pytest.raises(FileNotFoundError):
        ImportedRFBuilder(coords, path=str(tmp_path / "nope.pt"))


def test_unknown_format_raises(tmp_path, coords):
    bad = tmp_path / "weights.txt"
    bad.write_text("1,2,3")
    with pytest.raises(ValueError, match="csv folder|\\.pt|\\.npz"):
        ImportedRFBuilder(coords, path=str(bad))


def test_round_trip_and_contract(tmp_path, coords, source_bank):
    path = tmp_path / "bank.pt"
    source_bank.save(path)
    b = ImportedRFBuilder(coords, path=str(path))
    d = b.to_dict()
    assert d["method"] == "imported" and d["path"] == str(path)
    rebuilt = ImportedRFBuilder.from_config({**d, "receptor_coords": coords})
    assert torch.equal(rebuilt.compute_weights(), b.compute_weights())
    check_component("innervation", ImportedRFBuilder, b)


def test_neuron_centers_argument_is_ignored_with_warning(tmp_path, coords, source_bank):
    path = tmp_path / "bank.pt"
    source_bank.save(path)
    with pytest.warns(UserWarning, match="neuron_centers"):
        b = ImportedRFBuilder(coords, torch.zeros(2, 2), path=str(path))
    assert b.num_neurons == 5
