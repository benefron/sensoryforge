"""Regression for F-003 (task E1): default innervation weights are analytic Gaussian.

Before this change, InnervationModule/FlatInnervationModule (and the canonical
PopulationConfig/GridConfig schema and the GUI's NeuronPopulation dataclass)
defaulted use_distance_weights=False: the stochastic builder, where receptor
*selection* is Gaussian-weighted by distance but the resulting connection
*weight* is drawn uniformly at random from weight_range, independent of
distance. That builder remains available (pass use_distance_weights=False) as
the named "stochastic" control arm, but the default is now the analytic
Gaussian weighting: for a fixed neuron, the weight of each connected receptor
is a strictly decreasing function of that receptor's distance from the
neuron's center.
"""

import torch

from sensoryforge.core.grid import GridManager
from sensoryforge.core.innervation import InnervationModule
from sensoryforge.config.schema import PopulationConfig


def test_population_config_defaults_to_distance_weights():
    """The canonical schema must default use_distance_weights=True."""
    pop = PopulationConfig(name="p", neuron_type="SA")
    assert pop.use_distance_weights is True


def test_default_innervation_module_weights_decrease_monotonically_with_distance():
    """InnervationModule with no use_distance_weights override (the default,
    matching a canonical config with no explicit override) must produce,
    for one neuron, connection weights that decrease monotonically with the
    connected receptor's distance from that neuron's center.
    """
    torch.manual_seed(0)
    grid_manager = GridManager(grid_size=20, spacing=0.15, center=(0.0, 0.0))

    module = InnervationModule(
        neuron_type="SA",
        grid_manager=grid_manager,
        neurons_per_row=3,
        connections_per_neuron=40,
        sigma_d_mm=0.5,
        weight_range=(0.1, 1.0),
        seed=7,
    )

    xx, yy = grid_manager.get_coordinates()
    grid_coords = torch.stack([xx, yy], dim=-1)  # [H, W, 2]

    neuron_idx = 0
    center = module.neuron_centers[neuron_idx]
    weights = module.innervation_weights[neuron_idx]  # [H, W]

    distances = torch.sqrt(((grid_coords - center) ** 2).sum(-1))  # [H, W]

    connected_mask = weights > 0
    assert connected_mask.sum() > 5, "expected a nontrivial number of connections"

    connected_distances = distances[connected_mask]
    connected_weights = weights[connected_mask]

    order = torch.argsort(connected_distances)
    sorted_weights = connected_weights[order]

    # Strictly decreasing (weight is an affine function of the Gaussian
    # falloff at each receptor's distance, which is itself strictly
    # decreasing in distance), allowing for floating-point ties at equal
    # distance.
    diffs = sorted_weights[1:] - sorted_weights[:-1]
    assert torch.all(diffs <= 1e-6), (
        "connection weights must not increase with distance from the neuron "
        f"center; found an increase of {diffs.max().item():.6f}"
    )


def test_stochastic_control_arm_still_reachable():
    """use_distance_weights=False must still work (the named control arm)."""
    torch.manual_seed(0)
    grid_manager = GridManager(grid_size=20, spacing=0.15, center=(0.0, 0.0))

    module = InnervationModule(
        neuron_type="SA",
        grid_manager=grid_manager,
        neurons_per_row=3,
        connections_per_neuron=40,
        sigma_d_mm=0.5,
        weight_range=(0.1, 1.0),
        use_distance_weights=False,
        seed=7,
    )
    assert (module.innervation_weights > 0).any()
