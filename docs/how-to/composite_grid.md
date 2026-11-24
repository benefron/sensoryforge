# Composite Grid

## Overview

Composite grids support multiple receptor populations sharing a single coordinate system. Each population can have different densities and spatial arrangements — useful for modeling SA1/RA1/SA2 mosaics in touch, or L/M/S cone types in vision.

**Key point:** The grid manages receptor *positions* only. Filter types (SA/RA) are assigned at the sensory neuron level via innervation, not in the grid.

## Arrangement Types

| Type | Description | Use Case |
|------|-------------|----------|
| `grid` | Regular rectangular lattice | Uniform sampling |
| `hex` | Hexagonal lattice | Optimal packing, biological plausibility |
| `poisson` | Poisson disk sampling | Irregular, naturalistic |
| `jittered_grid` | Regular grid with random jitter | Controlled irregularity |

## Basic Usage

```python
from sensoryforge.core.composite_grid import CompositeReceptorGrid

grid = CompositeReceptorGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0), device='cpu')

# Add populations with different arrangements
grid.add_layer(name="SA1", density=100.0, arrangement="grid")
grid.add_layer(name="RA", density=50.0, arrangement="hex")
grid.add_layer(name="PC", density=25.0, arrangement="poisson")

# Retrieve coordinates for each layer
sa1_coords = grid.get_layer_coordinates("SA1")  # [num_receptors, 2]
ra_coords = grid.get_layer_coordinates("RA")
print(sa1_coords.shape)  # e.g. torch.Size([5000, 2])
```

## With Offset and Metadata

Use `offset` to shift layers and avoid exact overlap. Use `color` for GUI visualization.

```python
grid.add_layer(
    name="SA1",
    density=100.0,
    arrangement="grid",
    offset=(0.05, 0.0),  # Slight shift in mm
    color=(66, 135, 245, 255),  # RGBA for visualization
)
```

## YAML Configuration

```yaml
grid:
  type: composite
  xlim: [-5.0, 5.0]
  ylim: [-5.0, 5.0]
  populations:
    sa1:
      density: 100.0
      arrangement: grid
    ra:
      density: 50.0
      arrangement: hex
    pc:
      density: 25.0
      arrangement: poisson
```

**Note:** The `filter` field is deprecated. Filter associations are configured in the pipeline's filter section.

## Pipeline Integration

The generalized pipeline uses `CompositeReceptorGrid` when `grid.type: composite` is set. Receptor coordinates feed into innervation, which assigns SA/RA filters per population.

```python
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline

pipeline = GeneralizedTactileEncodingPipeline.from_yaml("config_with_composite_grid.yml")
results = pipeline.forward(duration_ms=100)
```

## See Also

- [Equation DSL](equation_dsl.md)
- [Solvers](solvers.md)
- [YAML Configuration](yaml_configuration.md)
