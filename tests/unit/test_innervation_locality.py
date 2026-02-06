"""Innervation locality tests aligned with the current encoding API."""

from __future__ import annotations

import numpy as np

from sensoryforge.core import GridManager, create_ra_innervation, create_sa_innervation


def _build_grid_manager(
    grid_size: tuple[int, int],
    spacing: float = 0.15,
) -> GridManager:
    return GridManager(
        grid_size=grid_size,
        spacing=spacing,
        center=(0.0, 0.0),
        device="cpu",
    )


def _extract_receptive_field_data(
    module,
    grid_manager: GridManager,
    neuron_idx: int,
):
    weights = module.innervation_weights[neuron_idx].detach().cpu().numpy()
    xx, yy = grid_manager.get_coordinates()
    xx_np = xx.cpu().numpy()
    yy_np = yy.cpu().numpy()
    centers = module.neuron_centers.cpu().numpy()
    center = centers[neuron_idx]
    mask = weights > 0
    return weights[mask], xx_np[mask], yy_np[mask], center


def test_receptive_field_locality_and_weights():
    grid_size = (80, 80)
    n_connections = 28
    sa_sigma = 0.3
    ra_sigma = 0.39

    grid_manager = _build_grid_manager(grid_size)
    sa_module = create_sa_innervation(
        grid_manager,
        seed=123,
        connections_per_neuron=n_connections,
        sigma_d_mm=sa_sigma,
        weight_range=(0.1, 1.0),
    )
    ra_module = create_ra_innervation(
        grid_manager,
        seed=123,
        connections_per_neuron=n_connections,
        sigma_d_mm=ra_sigma,
        weight_range=(0.1, 1.0),
    )

    rng = np.random.default_rng(123)
    for module, sigma in ((sa_module, sa_sigma), (ra_module, ra_sigma)):
        neuron_idx = int(rng.integers(module.num_neurons))
        weights, xs, ys, center = _extract_receptive_field_data(
            module, grid_manager, neuron_idx
        )
        dists = np.sqrt((xs - center[0]) ** 2 + (ys - center[1]) ** 2)
        assert (
            dists < 3 * sigma
        ).mean() > 0.95, f"Neuron {neuron_idx} has non-local connections"
        assert np.all(weights > 0), "Innervation weights must remain positive"


def test_weight_independence():
    grid_size = (80, 80)
    grid_manager = _build_grid_manager(grid_size)
    sa_module = create_sa_innervation(
        grid_manager,
        seed=456,
        weight_range=(0.1, 1.0),
    )

    w1 = sa_module.innervation_weights[0].detach().cpu().numpy()
    w2 = sa_module.innervation_weights[1].detach().cpu().numpy()
    shared = (w1 > 0) & (w2 > 0)
    if np.any(shared):
        assert not np.allclose(
            w1[shared], w2[shared]
        ), "Shared mechanoreceptors should vary in weight"
