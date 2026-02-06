import torch
import numpy as np
from sensoryforge.core.grid import GridManager
from sensoryforge.core.innervation import create_sa_innervation


def test_square_neuron_grid():
    grid_size = 40
    spacing = 0.2
    device = "cpu"
    grid_manager = GridManager(grid_size, spacing=spacing, device=device)
    weight_range = (0.1, 1.0)
    connections_per_neuron = 14
    # Test for various square neuron grid sizes
    for neurons_per_row in [6, 10, 14, 18, 20, 45, 80]:
        inn = create_sa_innervation(
            grid_manager,
            neurons_per_row=neurons_per_row,
            weight_range=weight_range,
            connections_per_neuron=connections_per_neuron,
        )
        centers = inn.neuron_centers.cpu().numpy()
        n = neurons_per_row
        assert (
            centers.shape[0] == n * n
        ), f"Expected {n*n} centers, got {centers.shape[0]}"
        # Check grid shape
        x = centers[:, 0].reshape(n, n)
        y = centers[:, 1].reshape(n, n)
        # X should increase along axis 1, Y along axis 0
        assert np.all(np.diff(x, axis=1) > 0), f"X not increasing across rows for n={n}"
        assert np.all(np.diff(y, axis=0) > 0), f"Y not increasing across cols for n={n}"
        # Check coverage: min/max should match grid bounds
        x_min, x_max = grid_manager.get_grid_properties()["xlim"]
        y_min, y_max = grid_manager.get_grid_properties()["ylim"]
        assert np.isclose(
            centers[:, 0].min(), x_min, atol=1e-3
        ), f"X min mismatch for n={n}"
        assert np.isclose(
            centers[:, 0].max(), x_max, atol=1e-3
        ), f"X max mismatch for n={n}"
        assert np.isclose(
            centers[:, 1].min(), y_min, atol=1e-3
        ), f"Y min mismatch for n={n}"
        assert np.isclose(
            centers[:, 1].max(), y_max, atol=1e-3
        ), f"Y max mismatch for n={n}"
        print(f"Passed for neurons_per_row={n} (total {n*n})")


if __name__ == "__main__":
    test_square_neuron_grid()
    print("All square neuron grid tests passed.")
