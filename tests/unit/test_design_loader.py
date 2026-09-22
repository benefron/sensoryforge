"""Tests for :mod:`sensoryforge.io.design` (Phase 2a, T1).

``tests/fixtures/design_8x8/`` is pressure-simulation's own
``tests/fixtures/design_8x8`` fixture (copied verbatim, not regenerated) --
a ``design.json`` plus ``sa.npz``/``ra.npz`` written by that repo's
``design.export.write_design``. These tests check that
:func:`sensoryforge.io.design.load_design` turns it into a
:class:`~sensoryforge.config.schema.SensoryForgeConfig` whose ``imported``
population(s) build receptive fields identical (up to the documented
y-slow -> x-slow re-index) to the ``.npz`` contents, and that missing
required keys raise a ``ValueError`` naming the key.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sensoryforge.config.defaults import resolve_filter_params, resolve_neuron_params
from sensoryforge.config.schema import PopulationConfig, SensoryForgeConfig
from sensoryforge.core.innervation import build_population_bank
from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.io.design import load_design, read_manifest

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "design_8x8"


@pytest.fixture()
def manifest():
    return read_manifest(FIXTURE_DIR)


def test_load_design_builds_valid_config():
    config = load_design(FIXTURE_DIR)
    assert isinstance(config, SensoryForgeConfig)
    assert len(config.grids) == 1
    grid = config.grids[0]
    assert (grid.rows, grid.cols) == (8, 8)
    assert grid.spacing == pytest.approx(0.15)
    # round trip through to_dict/from_dict should not raise
    SensoryForgeConfig.from_dict(config.to_dict())


def test_population_count_and_N_matches_manifest(manifest):
    config = load_design(FIXTURE_DIR)
    grid = config.grids[0]
    n_grid = grid.rows * grid.cols

    by_name = {p.name: p for p in config.populations}
    assert set(by_name) == {prec["name"] for prec in manifest["populations"]}

    for prec in manifest["populations"]:
        pop = by_name[prec["name"]]
        assert pop.innervation_method == "imported"
        npz_path = Path(pop.innervation_params["path"])
        assert npz_path.is_absolute()
        with np.load(npz_path) as data:
            H = data["H"]
        assert H.shape[1] == n_grid
        assert H.shape[0] == prec["N"]
        assert pop.filter_method == prec["filter_method"]
        assert pop.filter_params == prec["filter_params"]
        assert pop.neuron_model == prec["neuron_model"]
        assert pop.model_params == prec["model_params"]
        assert pop.input_gain == pytest.approx(prec["input_gain"])


def test_imported_weights_match_npz_up_to_documented_reindex(manifest):
    """The bank ImportedRFBuilder builds from each population's npz path
    must carry exactly the values in that npz's ``H`` -- ``ImportedRFBuilder``
    guarantees only that its weights are a *re-indexing* of ``H``'s columns
    (pressure-simulation's y-slow grid order -> SensoryForge's x-slow order,
    see ``imported.py``'s module docstring), not that the column order is
    unchanged. So this checks the per-row *set* of values is identical
    (``torch.sort`` on each axis), the same contract
    ``test_rf_imported_builder.py::test_npz_converts_yx_centres_and_column_order``
    pins for the builder itself.
    """
    config = load_design(FIXTURE_DIR)
    grid = config.grids[0]
    receptor_coords = ReceptorGrid(
        grid_size=(grid.cols, grid.rows), spacing=grid.spacing
    ).get_receptor_coordinates()

    for pop in config.populations:
        npz_path = Path(pop.innervation_params["path"])
        with np.load(npz_path) as data:
            H = torch.from_numpy(np.ascontiguousarray(data["H"], dtype=np.float32))

        bank = build_population_bank(
            receptor_coords=receptor_coords,
            innervation_method=pop.innervation_method,
            path=str(npz_path),
        )
        assert bank.weights.shape == H.shape
        assert torch.allclose(
            torch.sort(bank.weights, dim=1).values,
            torch.sort(H, dim=1).values,
        )


def test_neuron_type_derived_from_filter_method_drives_resolvers(manifest):
    """``neuron_type`` must be "SA"/"RA" (not the dataclass default "SA" for
    every population), because ``resolve_neuron_params``/``resolve_filter_params``
    key their presets off ``neuron_type``/``filter_method`` respectively --
    not asserting hard-coded preset numbers, just that the loaded config
    resolves identically to an equivalent hand-written ``PopulationConfig``.
    """
    config = load_design(FIXTURE_DIR)
    by_name = {p.name: p for p in config.populations}

    assert by_name["sa"].neuron_type == "SA"
    assert by_name["ra"].neuron_type == "RA"

    for prec in manifest["populations"]:
        pop = by_name[prec["name"]]
        expected_type = {"sa": "SA", "ra": "RA"}[prec["filter_method"]]
        hand_written = PopulationConfig(
            name=prec["name"],
            neuron_type=expected_type,
            neuron_model=prec["neuron_model"],
            model_params=prec["model_params"],
            filter_method=prec["filter_method"],
            filter_params=prec["filter_params"],
        )
        assert resolve_neuron_params(
            pop.neuron_model, pop.neuron_type, pop.model_params
        ) == resolve_neuron_params(
            hand_written.neuron_model,
            hand_written.neuron_type,
            hand_written.model_params,
        )
        assert resolve_filter_params(
            pop.filter_method, pop.filter_params
        ) == resolve_filter_params(
            hand_written.filter_method, hand_written.filter_params
        )


def test_missing_decisions_grid_names_key(tmp_path):
    broken = tmp_path / "design_8x8_broken"
    broken.mkdir()
    manifest = json.loads((FIXTURE_DIR / "design.json").read_text())
    del manifest["decisions"]["grid"]
    (broken / "design.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="grid"):
        load_design(broken)


def test_missing_population_key_names_key(tmp_path):
    broken = tmp_path / "design_8x8_broken2"
    broken.mkdir()
    manifest = json.loads((FIXTURE_DIR / "design.json").read_text())
    del manifest["populations"][0]["filter_params"]
    (broken / "design.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="filter_params"):
        load_design(broken)


def test_missing_design_json_raises_file_not_found(tmp_path):
    empty = tmp_path / "empty_design"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_design(empty)


def test_missing_npz_raises_file_not_found(tmp_path):
    broken = tmp_path / "design_8x8_no_npz"
    broken.mkdir()
    (broken / "design.json").write_text((FIXTURE_DIR / "design.json").read_text())
    with pytest.raises(FileNotFoundError):
        load_design(broken)
