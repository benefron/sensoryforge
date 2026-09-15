"""Seeded, reproducible receptor grids (F-050, Phase 2 task I1).

Before I1, ``ReceptorGrid``/``CompositeReceptorGrid`` took no seed and the
``jittered_grid``/``blue_noise``/``poisson`` arrangements drew from the
global RNG, so two identical builds differed, building a grid advanced
``torch.get_rng_state()``, and ``from_config(to_dict())`` could not
reproduce a random layout.
"""

import pytest
import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
)
from sensoryforge.core.composite_grid import CompositeReceptorGrid
from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.core.grid_arrangements import (
    BlueNoiseArrangement,
    JitteredGridArrangement,
    PoissonArrangement,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.testing.contracts import check_component

RANDOM_ARRANGEMENTS = ["jittered_grid", "blue_noise", "poisson"]
ARRANGEMENT_CLASSES = {
    "jittered_grid": JitteredGridArrangement,
    "blue_noise": BlueNoiseArrangement,
    "poisson": PoissonArrangement,
}


def _grid(arrangement, seed, **kw):
    return ReceptorGrid(
        grid_size=(6, 5), spacing=0.2, arrangement=arrangement, seed=seed, **kw
    )


@pytest.mark.parametrize("arrangement", RANDOM_ARRANGEMENTS)
def test_same_seed_gives_identical_coordinates(arrangement):
    a = _grid(arrangement, seed=11).get_all_coordinates()
    b = _grid(arrangement, seed=11).get_all_coordinates()
    assert torch.equal(a, b)


@pytest.mark.parametrize("arrangement", RANDOM_ARRANGEMENTS)
def test_different_seeds_differ(arrangement):
    a = _grid(arrangement, seed=11).get_all_coordinates()
    b = _grid(arrangement, seed=12).get_all_coordinates()
    assert a.shape == b.shape
    assert not torch.equal(a, b)


@pytest.mark.parametrize("arrangement", RANDOM_ARRANGEMENTS)
def test_seeded_build_leaves_global_rng_untouched(arrangement):
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    _grid(arrangement, seed=7)
    after = torch.get_rng_state()
    assert torch.equal(before, after)


@pytest.mark.parametrize("arrangement", RANDOM_ARRANGEMENTS)
def test_receptor_grid_to_dict_carries_seed_and_round_trips(arrangement):
    grid = _grid(arrangement, seed=3)
    d = grid.to_dict()
    assert d["seed"] == 3
    rebuilt = ReceptorGrid.from_config(d)
    assert torch.equal(rebuilt.get_all_coordinates(), grid.get_all_coordinates())


@pytest.mark.parametrize("arrangement", RANDOM_ARRANGEMENTS)
def test_arrangement_class_round_trip_reproduces_coordinates(arrangement):
    cls = ARRANGEMENT_CLASSES[arrangement]
    grid = cls(grid_size=5, spacing=0.3, seed=21)
    rebuilt = cls.from_config(grid.to_dict())
    assert torch.equal(rebuilt.get_all_coordinates(), grid.get_all_coordinates())


@pytest.mark.parametrize("arrangement", RANDOM_ARRANGEMENTS)
def test_contract_check_covers_seed(arrangement):
    cls = ARRANGEMENT_CLASSES[arrangement]
    check_component("grid", cls, cls(grid_size=4, spacing=0.5, seed=5))
    assert any(p.name == "seed" for p in cls.get_param_spec())


@pytest.mark.parametrize("arrangement", RANDOM_ARRANGEMENTS)
def test_composite_layer_seed_is_reproducible(arrangement):
    def build(seed):
        cg = CompositeReceptorGrid(xlim=(0.0, 2.0), ylim=(0.0, 1.5))
        cg.add_layer("L", density=20.0, arrangement=arrangement, seed=seed)
        return cg

    a = build(9).get_layer_coordinates("L")
    b = build(9).get_layer_coordinates("L")
    c = build(10).get_layer_coordinates("L")
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("arrangement", RANDOM_ARRANGEMENTS)
def test_composite_layer_round_trip_reproduces_coordinates(arrangement):
    cg = CompositeReceptorGrid(xlim=(0.0, 2.0), ylim=(0.0, 1.5))
    cg.add_layer("L", density=20.0, arrangement=arrangement, seed=4)
    d = cg.to_dict()
    assert d["layers"]["L"]["seed"] == 4
    rebuilt = CompositeReceptorGrid.from_config(d)
    assert torch.equal(
        rebuilt.get_layer_coordinates("L"), cg.get_layer_coordinates("L")
    )


def test_composite_seeded_layer_leaves_global_rng_untouched():
    torch.manual_seed(99)
    before = torch.get_rng_state()
    cg = CompositeReceptorGrid(xlim=(0.0, 2.0), ylim=(0.0, 1.5))
    cg.add_layer("L", density=20.0, arrangement="poisson", seed=1)
    assert torch.equal(before, torch.get_rng_state())


def test_grid_config_seed_round_trips_through_dict():
    cfg = GridConfig(name="g", arrangement="jittered_grid", rows=5, cols=5, seed=8)
    assert GridConfig.from_dict(cfg.to_dict()).seed == 8
    assert GridConfig.from_dict({"name": "g"}).seed is None


def test_engine_passes_grid_seed_through():
    def engine(seed):
        cfg = SensoryForgeConfig(
            grids=[
                GridConfig(
                    name="g", arrangement="jittered_grid", rows=6, cols=6, seed=seed
                )
            ],
            populations=[
                PopulationConfig(name="SA", neuron_type="SA", neurons_per_row=2, seed=1)
            ],
        )
        return SimulationEngine(cfg)

    a = engine(5).grids[0].get_all_coordinates()
    b = engine(5).grids[0].get_all_coordinates()
    c = engine(6).grids[0].get_all_coordinates()
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert engine(5).grids[0].seed == 5
