"""
Test script to check spatial coverage and bias of neuron innervation.
Follows the same steps as the notebook, but prints innervation statistics and neuron center locations.
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from sensoryforge.core.grid import GridManager
from sensoryforge.core.innervation import create_sa_innervation, create_ra_innervation

# Configuration
GRID_SIZE = 40
SPACING = 0.15
NUM_SA_NEURONS = 49
NUM_RA_NEURONS = 100
RECEPTORS_PER_NEURON = 14
SA_SPREAD = 0.3
RA_SPREAD = 0.39
CONNECTION_STRENGTH = (0.1, 1.0)
SEED = 33
DEVICE = "cpu"


def print_centers_stats(centers, label):
    x = centers[:, 0]
    y = centers[:, 1]
    print(f"{label} neuron centers:")
    print(
        f"  X: min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}, std={x.std():.3f}"
    )
    print(
        f"  Y: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}, std={y.std():.3f}"
    )
    print(f"  X range: {x.max() - x.min():.3f}, Y range: {y.max() - y.min():.3f}")
    print(f"  First 5 centers: {centers[:5]}")
    print(f"  Last 5 centers: {centers[-5:]}")


def main():
    print("Testing innervation spatial coverage...")
    grid_manager = GridManager(grid_size=GRID_SIZE, spacing=SPACING, device=DEVICE)
    xx, yy = grid_manager.get_coordinates()
    x_1d, y_1d = grid_manager.get_1d_coordinates()
    x_min, x_max = x_1d.min().item(), x_1d.max().item()
    y_min, y_max = y_1d.min().item(), y_1d.max().item()
    print(f"Grid X: {x_min:.3f} to {x_max:.3f}, Y: {y_min:.3f} to {y_max:.3f}")

    sa_inn = create_sa_innervation(
        grid_manager,
        num_neurons=NUM_SA_NEURONS,
        connections_per_neuron=RECEPTORS_PER_NEURON,
        sigma_d_mm=SA_SPREAD,
        weight_range=CONNECTION_STRENGTH,
        seed=SEED,
    )
    ra_inn = create_ra_innervation(
        grid_manager,
        num_neurons=NUM_RA_NEURONS,
        connections_per_neuron=RECEPTORS_PER_NEURON,
        sigma_d_mm=RA_SPREAD,
        weight_range=CONNECTION_STRENGTH,
        seed=SEED,
    )

    print_centers_stats(sa_inn.neuron_centers.cpu().numpy(), "SA")
    print_centers_stats(ra_inn.neuron_centers.cpu().numpy(), "RA")

    # Print innervation map stats
    sa_map = sa_inn.innervation_weights.cpu().numpy()
    ra_map = ra_inn.innervation_weights.cpu().numpy()
    print(f"SA innervation map shape: {sa_map.shape}")
    print(f"RA innervation map shape: {ra_map.shape}")
    print(
        f"SA total innervation per grid point: min={sa_map.sum(0).min():.2f}, max={sa_map.sum(0).max():.2f}, mean={sa_map.sum(0).mean():.2f}"
    )
    print(
        f"RA total innervation per grid point: min={ra_map.sum(0).min():.2f}, max={ra_map.sum(0).max():.2f}, mean={ra_map.sum(0).mean():.2f}"
    )

    # Optionally, print a slice of the total innervation for visual inspection
    print("SA total innervation (center row):", sa_map.sum(0)[GRID_SIZE // 2])
    print("RA total innervation (center row):", ra_map.sum(0)[GRID_SIZE // 2])


if __name__ == "__main__":
    main()
