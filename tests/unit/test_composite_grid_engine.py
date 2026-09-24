"""Composite grids in SimulationEngine (Phase 2, Wave L4, F-010).

Before this wave, ``SimulationEngine._build_grids`` raised
``NotImplementedError`` for ``arrangement == "composite"`` even though
``_build_populations`` already had a working ``CompositeReceptorGrid``
branch (Fact L-b) -- only the grid half was missing. This closes it: a
``GridConfig`` with ``arrangement="composite"`` builds its layers from
``layers:`` in declaration order.
"""

from __future__ import annotations

import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
)
from sensoryforge.core.composite_grid import CompositeReceptorGrid
from sensoryforge.core.grid import load_receptor_coords_file
from sensoryforge.core.simulation_engine import SimulationEngine


def _two_layer_config(density_b: float = 30.0):
    return SensoryForgeConfig(
        grids=[
            GridConfig(
                name="composite_g",
                arrangement="composite",
                rows=10,
                cols=10,
                spacing=0.2,
                layers=[
                    {
                        "name": "layer_a",
                        "density": 40.0,
                        "arrangement": "grid",
                        "seed": 1,
                    },
                    {"name": "layer_b", "density": density_b, "arrangement": "hex"},
                ],
            )
        ],
        populations=[
            PopulationConfig(
                name="P",
                neurons_per_row=2,
                seed=7,
                filter_method="none",
                innervation_method="gaussian",
            )
        ],
    )


class TestCompositeGridBuilds:
    def test_no_longer_raises_not_implemented(self):
        cfg = _two_layer_config()
        engine = SimulationEngine(cfg)
        grid = engine.grid_names["composite_g"]
        assert isinstance(grid, CompositeReceptorGrid)
        assert grid.list_layers() == ["layer_a", "layer_b"]

    def test_missing_layers_raises_value_error(self):
        cfg = SensoryForgeConfig(
            grids=[GridConfig(name="g", arrangement="composite", rows=5, cols=5)],
            populations=[PopulationConfig(name="P", neurons_per_row=2)],
        )
        try:
            SimulationEngine(cfg)
        except ValueError as exc:
            assert "layers" in str(exc)
        else:
            raise AssertionError("expected ValueError for empty layers list")

    def test_provenance_records_layer_order_and_counts(self):
        cfg = _two_layer_config()
        engine = SimulationEngine(cfg)
        grid = engine.grid_names["composite_g"]
        prov = grid.provenance
        names = [entry["name"] for entry in prov["layers"]]
        assert names == ["layer_a", "layer_b"]
        for entry in prov["layers"]:
            assert entry["count"] == grid.get_layer_count(entry["name"])

    def test_two_layer_composite_runs_end_to_end(self):
        cfg = _two_layer_config()
        engine = SimulationEngine(cfg)
        bank = engine.populations[0]["bank"]
        # A composite grid has no fixed pixel raster of its own -- the
        # stimulus is a spatial field over the grid's bounding box, sampled
        # at each receptor's (x, y) position (Wave L3), same as any other
        # non-"grid" arrangement. Any resolution works; 25x25 here.
        h, w = 25, 25
        stim = torch.ones(1, 5, h, w)  # [batch, time, H, W]
        out = engine.run(stim, return_intermediates=True)["P"]
        assert torch.isfinite(out["drive"]).all()
        assert out["drive"].shape == (1, 5, bank.num_neurons)
        # A constant field of 1.0 sampled anywhere gives a constant drive
        # equal to each neuron's row sum of weights.
        expected = bank.weights.sum(dim=1)  # [N]
        assert torch.allclose(out["drive"][0, 0], expected, atol=1e-5)

    def test_layer_a_bank_unaffected_by_layer_b_density_change(self):
        engine1 = SimulationEngine(_two_layer_config(density_b=30.0))
        engine2 = SimulationEngine(_two_layer_config(density_b=90.0))
        grid1 = engine1.grid_names["composite_g"]
        grid2 = engine2.grid_names["composite_g"]
        assert torch.equal(
            grid1.get_layer_coordinates("layer_a"),
            grid2.get_layer_coordinates("layer_a"),
        )
        bank1 = engine1.populations[0]["bank"]
        bank2 = engine2.populations[0]["bank"]
        # Both populations innervate the whole composite grid by default,
        # so layer B's larger receptor count does change the overall bank
        # shape -- but layer A's own coordinates (and hence the columns
        # of the weight matrix that map to it) are identical either way.
        n_a = grid1.get_layer_count("layer_a")
        assert torch.equal(bank1.receptor_coords[:n_a], bank2.receptor_coords[:n_a])


class TestTargetLayers:
    def test_population_can_target_a_layer_subset(self):
        cfg = _two_layer_config()
        cfg.populations[0].target_layers = ["layer_a"]
        engine = SimulationEngine(cfg)
        grid = engine.grid_names["composite_g"]
        bank = engine.populations[0]["bank"]
        n_a = grid.get_layer_count("layer_a")
        assert bank.num_receptors == n_a
        assert torch.equal(bank.receptor_coords, grid.get_layer_coordinates("layer_a"))


class TestCoordsFile:
    def test_coords_file_builds_single_layer_composite(self, tmp_path):
        coords = torch.tensor([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5], [0.3, 0.9]])
        path = tmp_path / "coords.pt"
        torch.save(coords, path)

        cfg = SensoryForgeConfig(
            grids=[GridConfig(name="imported_g", coords_file=str(path))],
            populations=[
                PopulationConfig(
                    name="P",
                    neurons_per_row=1,
                    seed=1,
                    filter_method="none",
                    innervation_method="one_to_one",
                )
            ],
        )
        engine = SimulationEngine(cfg)
        grid = engine.grid_names["imported_g"]
        assert isinstance(grid, CompositeReceptorGrid)
        loaded = grid.get_all_coordinates()
        assert torch.equal(loaded.cpu(), coords)

    def test_csv_round_trip(self, tmp_path):
        coords = torch.tensor([[0.0, 0.0], [2.0, -1.0], [0.5, 0.5]])
        csv_path = tmp_path / "coords.csv"
        with open(csv_path, "w") as f:
            f.write("x,y\n")
            for row in coords.tolist():
                f.write(f"{row[0]},{row[1]}\n")
        loaded = load_receptor_coords_file(str(csv_path))
        assert torch.allclose(loaded, coords)
