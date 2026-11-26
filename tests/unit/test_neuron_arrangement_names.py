"""An unknown neuron arrangement is refused, never laid out as a square grid."""

import pytest

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
