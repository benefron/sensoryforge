"""Tests for G3: real grid-arrangement classes in GRID_REGISTRY.

Before this, `register_components.py` registered each arrangement name
("grid", "poisson", "hex", "jittered_grid", "blue_noise") against the
placeholder `str` type, so `GRID_REGISTRY.create(...)`, `get_param_spec()`,
and `list-components`/plugin discovery had nothing real to introspect.
"""

import pytest
import torch

from sensoryforge.register_components import register_all
from sensoryforge.registry import GRID_REGISTRY
from sensoryforge.core.grid_base import BaseGrid

register_all()


@pytest.mark.parametrize(
    "name", ["grid", "poisson", "hex", "jittered_grid", "blue_noise"]
)
def test_registered_class_is_not_the_str_placeholder(name):
    cls = GRID_REGISTRY.get_class(name)
    assert cls is not str
    assert issubclass(cls, BaseGrid)


@pytest.mark.parametrize(
    "name", ["grid", "poisson", "hex", "jittered_grid", "blue_noise"]
)
def test_arrangement_builds_real_coordinates(name):
    grid = GRID_REGISTRY.create(name, grid_size=4, spacing=0.5)
    coords = grid.get_all_coordinates()
    assert isinstance(coords, torch.Tensor)
    assert coords.shape[1] == 2
    assert coords.shape[0] > 0


@pytest.mark.parametrize(
    "name", ["grid", "poisson", "hex", "jittered_grid", "blue_noise"]
)
def test_arrangement_from_config_to_dict_round_trip(name):
    cls = GRID_REGISTRY.get_class(name)
    grid = GRID_REGISTRY.create(name, grid_size=4, spacing=0.5)
    reconstructed = cls.from_config(grid.to_dict())
    assert isinstance(reconstructed, cls)
    assert reconstructed.get_all_coordinates().shape[1] == 2
