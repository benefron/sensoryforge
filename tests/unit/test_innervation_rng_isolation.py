"""Regression for F-006 (task E2): innervation uses a per-instance RNG.

Before this change, InnervationModule/FlatInnervationModule and the lower
BaseInnervation subclasses (GaussianInnervation, UniformInnervation,
OneToOneInnervation, DistanceWeightedInnervation) all reseeded the *global*
torch RNG via torch.manual_seed(seed) and then drew from the global
generator. That meant: (1) building innervation mutated global RNG state
that unrelated code (or a later, unseeded innervation build) could observe,
and (2) two builds with the same seed could still see different unrelated
global draws in between and be unaffected -- accidentally correct only
because manual_seed fully resets the global stream each time -- but any
global draw *inside* the same call between the reseed and the random ops
would still leak into the result.

Now every random draw uses a local torch.Generator (sensoryforge.core.
innervation._seeded_generator), following the pattern in
sensoryforge.filters.noise. This test asserts the global RNG state is left
completely untouched by construction, and that unrelated global draws
between two same-seed builds don't affect the result.
"""

import torch

from sensoryforge.core.grid import GridManager
from sensoryforge.core.innervation import InnervationModule


def _make_grid_manager() -> GridManager:
    return GridManager(grid_size=20, spacing=0.15, center=(0.0, 0.0))


def test_global_rng_state_unchanged_by_construction():
    torch.manual_seed(123)
    _ = torch.rand(5)
    state_before = torch.get_rng_state().clone()

    InnervationModule(
        neuron_type="SA",
        grid_manager=_make_grid_manager(),
        neurons_per_row=3,
        connections_per_neuron=10,
        sigma_d_mm=0.5,
        seed=42,
    )

    state_after = torch.get_rng_state()
    assert torch.equal(state_before, state_after), (
        "constructing InnervationModule must not consume or reseed the "
        "global torch RNG state"
    )


def test_same_seed_gives_identical_weights_across_unrelated_global_draws():
    grid_manager = _make_grid_manager()

    module1 = InnervationModule(
        neuron_type="SA",
        grid_manager=grid_manager,
        neurons_per_row=3,
        connections_per_neuron=10,
        sigma_d_mm=0.5,
        seed=42,
    )

    # Draw unrelated global random numbers in between.
    torch.manual_seed(999)
    _ = torch.rand(100)
    _ = torch.randn(50)

    module2 = InnervationModule(
        neuron_type="SA",
        grid_manager=grid_manager,
        neurons_per_row=3,
        connections_per_neuron=10,
        sigma_d_mm=0.5,
        seed=42,
    )

    assert torch.equal(module1.innervation_weights, module2.innervation_weights), (
        "two InnervationModule instances built with the same seed must "
        "produce identical weights regardless of unrelated global RNG draws "
        "in between"
    )
