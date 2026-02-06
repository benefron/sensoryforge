"""Innervation centroid alignment tests."""

import numpy as np
import torch

from sensoryforge.core.grid import GridManager
from sensoryforge.core.innervation import (
    create_innervation_map_tensor,
    create_neuron_centers,
)


def test_innervation_center_alignment() -> None:
    """Innervation centroids remain close to their neuron centres."""
    grid_size = 20
    grid_spacing_mm = 0.4
    neurons_per_row = 4
    sigma_d_mm = 0.8
    connections_per_neuron = 10
    device = "cpu"
    max_offset = 1.5 * sigma_d_mm

    grid_manager = GridManager(
        grid_size,
        spacing=grid_spacing_mm,
        device=device,
    )
    xx, yy = grid_manager.get_coordinates()
    grid_coords = torch.stack([xx, yy], dim=-1)
    props = grid_manager.get_grid_properties()
    xlim, ylim = props["xlim"], props["ylim"]

    neuron_centers = create_neuron_centers(
        neurons_per_row,
        xlim,
        ylim,
        device=device,
    )
    innervation_map = create_innervation_map_tensor(
        grid_coords,
        neuron_centers,
        connections_per_neuron,
        sigma_d_mm,
        grid_spacing_mm,
        weight_range=(0.1, 1.0),
        seed=42,
        device=device,
    )

    num_neurons = neuron_centers.shape[0]
    errors = 0

    for neuron_idx in range(num_neurons):
        weights = innervation_map[neuron_idx].cpu().numpy()
        mask = weights > 0
        if not np.any(mask):
            continue

        coords = grid_coords[mask].cpu().numpy()
        vals = weights[mask]
        centroid = np.average(coords, axis=0, weights=vals)
        neuron_pos = neuron_centers[neuron_idx].cpu().numpy()
        dist = np.linalg.norm(centroid - neuron_pos)

        if dist > max_offset:
            errors += 1
            print(
                (
                    "Neuron {n}: center {center}, weighted centroid {pos},"
                    " dist {dist:.3f} [ERROR]"
                ).format(
                    n=neuron_idx,
                    center=neuron_pos,
                    pos=centroid,
                    dist=dist,
                )
            )

    assert errors == 0, f"{errors} neurons show large centroid offsets"
