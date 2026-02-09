import torch
import numpy as np
from sensoryforge.core.grid import GridManager
from sensoryforge.core.innervation import create_neuron_centers, create_innervation_map_tensor

for seed in [42, 123, 0, 7, 99, 2024]:
    grid_manager = GridManager(20, spacing=0.4, device='cpu')
    xx, yy = grid_manager.get_coordinates()
    grid_coords = torch.stack([xx, yy], dim=-1)
    props = grid_manager.get_grid_properties()
    neuron_centers = create_neuron_centers(4, props['xlim'], props['ylim'], 'cpu')
    innervation_map = create_innervation_map_tensor(grid_coords, neuron_centers, 10, 0.8, 0.4, (0.1,1.0), seed, 'cpu')
    
    errors = 0
    for i in range(neuron_centers.shape[0]):
        w = innervation_map[i].numpy()
        mask = w > 0
        if not np.any(mask):
            continue
        coords = grid_coords[mask].numpy()
        centroid = np.average(coords, axis=0, weights=w[mask])
        d = np.linalg.norm(centroid - neuron_centers[i].numpy())
        if d > 1.5 * 0.8:
            errors += 1
    print(f'seed={seed}: errors={errors}')
