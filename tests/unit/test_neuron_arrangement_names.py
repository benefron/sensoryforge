"""An unknown neuron arrangement is refused, never laid out as a square grid."""

import pytest
import torch

from sensoryforge.core.innervation import NEURON_ARRANGEMENTS, create_neuron_centers


@pytest.mark.parametrize("name", ["demo_spiral", "jittered", "Hex", ""])
def test_an_unknown_name_is_refused_with_the_valid_ones(name):
    with pytest.raises(ValueError, match="neuron arrangement.*jittered_grid"):
        create_neuron_centers(5, (-1.0, 1.0), (-1.0, 1.0), arrangement=name)


@pytest.mark.parametrize("name", NEURON_ARRANGEMENTS)
def test_every_listed_name_is_laid_out(name):
    centres = create_neuron_centers(5, (-1.0, 1.0), (-1.0, 1.0), arrangement=name)
    assert tuple(centres.shape) == (25, 2)


def test_a_run_with_an_unknown_name_fails_instead_of_using_a_square_grid():
    from sensoryforge.config.schema import (
        GridConfig,
        PopulationConfig,
        SensoryForgeConfig,
    )
    from sensoryforge.core.simulation_engine import SimulationEngine

    config = SensoryForgeConfig(
        grids=[GridConfig(name="g", rows=10, cols=10)],
        populations=[PopulationConfig(name="p", neuron_arrangement="demo_spiral")],
    )
    with pytest.raises(ValueError, match="demo_spiral"):
        SimulationEngine(config)


def _hex_centers(rows, cols):
    return create_neuron_centers(
        cols,
        (-2.0, 2.0),
        (-2.0, 2.0),
        rows=rows,
        cols=cols,
        arrangement="hex",
    )


def test_hex_that_cannot_fill_the_count_raises_instead_of_using_a_square_grid():
    with pytest.raises(ValueError, match=r"hex.*221"):
        _hex_centers(13, 17)


def test_hex_that_can_fill_the_count_still_returns_a_hexagonal_layout():
    hex_centers = _hex_centers(10, 10)
    square = create_neuron_centers(10, (-2.0, 2.0), (-2.0, 2.0), arrangement="grid")

    assert hex_centers.shape == (100, 2)
    assert not torch.allclose(hex_centers, square)
