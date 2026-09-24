"""Tests for D-88b4b41 / F-081: ``GridConfig.density`` actually sizes a grid.

Before this fix, ``core.simulation_engine.build_grid`` passed
``density=grid_cfg.density`` to ``ReceptorGrid``, but ``ReceptorGrid`` never
read it for any arrangement -- every arrangement (``grid``, ``poisson``,
``hex``, ``jittered_grid``, ``blue_noise``) was sized from
``rows x cols x spacing`` alone, so a YAML ``density`` value silently did
nothing (F-081).

The decision (D-88b4b41): ``density`` sets the receptor count of
``poisson``, ``hex`` and ``blue_noise`` layouts to ``density`` times the
``rows x cols x spacing`` extent; setting it on ``grid`` or
``jittered_grid``, where ``spacing`` already fixes the count, is an error.
"""

import pytest
import torch

from sensoryforge.config.schema import GridConfig
from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.core.simulation_engine import build_grid

_DENSITY_ARRANGEMENTS = ["poisson", "hex", "blue_noise"]
_FIXED_COUNT_ARRANGEMENTS = ["grid", "jittered_grid"]


@pytest.mark.parametrize("arrangement", _DENSITY_ARRANGEMENTS)
def test_density_scales_receptor_count_about_10x(arrangement):
    """5 vs 50 receptors/mm^2 gives about 10x the receptors (D-88b4b41)."""
    low = ReceptorGrid(
        grid_size=(20, 20),
        spacing=0.15,
        arrangement=arrangement,
        density=5.0,
        seed=42,
    )
    high = ReceptorGrid(
        grid_size=(20, 20),
        spacing=0.15,
        arrangement=arrangement,
        density=50.0,
        seed=42,
    )
    low_count = low.get_all_coordinates().shape[0]
    high_count = high.get_all_coordinates().shape[0]
    ratio = high_count / low_count
    # Measured (grid_size=(20, 20), spacing=0.15, seed=42):
    # poisson 49 -> 441 (9.0x), hex 39 -> 407 (10.44x), blue_noise 49 -> 441 (9.0x).
    assert 7.0 <= ratio <= 13.0, (arrangement, low_count, high_count, ratio)


@pytest.mark.parametrize("arrangement", _DENSITY_ARRANGEMENTS)
def test_density_unset_is_unchanged(arrangement):
    """With density=None the arrangement is sized from rows x cols x spacing.

    The same seed/geometry with density explicitly omitted must reproduce
    identical coordinates to a second, independent build -- pinning that the
    density-aware code path did not change the unset (default) behaviour,
    consistent with the pre-existing seeded-reproducibility tests in
    ``tests/unit/test_grid_seed.py`` (unaffected by this change) and the
    golden receptive-field fixtures built on top of these grids.
    """
    a = ReceptorGrid(
        grid_size=(20, 20), spacing=0.15, arrangement=arrangement, seed=42
    ).get_all_coordinates()
    b = ReceptorGrid(
        grid_size=(20, 20), spacing=0.15, arrangement=arrangement, seed=42
    ).get_all_coordinates()
    assert torch.equal(a, b)


#: Receptor counts for grid_size=(20, 20), spacing=0.15, seed=42, density=None,
#: bit-for-bit compared (torch.equal) against sensoryforge/core/grid.py as of
#: commit e621387 (pre-D-88b4b41), to pin that density=None is unaffected by
#: this change for every arrangement, not only the three density controls.
_PRE_FIX_COUNTS = {
    "grid": 400,
    "poisson": 441,
    "hex": 407,
    "jittered_grid": 400,
    "blue_noise": 400,
}


@pytest.mark.parametrize("arrangement", sorted(_PRE_FIX_COUNTS))
def test_density_unset_matches_pre_fix_receptor_count(arrangement):
    grid = ReceptorGrid(
        grid_size=(20, 20), spacing=0.15, arrangement=arrangement, seed=42
    )
    assert grid.get_all_coordinates().shape[0] == _PRE_FIX_COUNTS[arrangement]


@pytest.mark.parametrize("arrangement", _FIXED_COUNT_ARRANGEMENTS)
def test_density_on_fixed_count_arrangement_raises(arrangement):
    with pytest.raises(ValueError, match="density"):
        ReceptorGrid(
            grid_size=(10, 10), spacing=0.15, arrangement=arrangement, density=5.0
        )


@pytest.mark.parametrize(
    "arrangement", _DENSITY_ARRANGEMENTS + _FIXED_COUNT_ARRANGEMENTS
)
def test_non_positive_density_raises(arrangement):
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="density"):
            ReceptorGrid(
                grid_size=(10, 10), spacing=0.15, arrangement=arrangement, density=bad
            )


@pytest.mark.parametrize("arrangement", _FIXED_COUNT_ARRANGEMENTS)
def test_build_grid_raises_for_fixed_count_arrangement_with_density(arrangement):
    cfg = GridConfig(name="g", arrangement=arrangement, rows=10, cols=10, density=5.0)
    with pytest.raises(ValueError, match="density"):
        build_grid(cfg, device="cpu")


@pytest.mark.parametrize("arrangement", _DENSITY_ARRANGEMENTS)
def test_build_grid_honours_density_for_scalable_arrangements(arrangement):
    low_cfg = GridConfig(
        name="g", arrangement=arrangement, rows=20, cols=20, spacing=0.15, density=5.0
    )
    high_cfg = GridConfig(
        name="g", arrangement=arrangement, rows=20, cols=20, spacing=0.15, density=50.0
    )
    low_count = build_grid(low_cfg, device="cpu").get_all_coordinates().shape[0]
    high_count = build_grid(high_cfg, device="cpu").get_all_coordinates().shape[0]
    assert high_count > low_count * 5
