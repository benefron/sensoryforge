# Composite Grid

## Overview
Composite grids support multiple receptor populations sharing a coordinate system.
Each population can have different densities and spatial arrangements.

## Concepts
Populations differ by density and spatial arrangement (grid, hex, Poisson).
Filter associations are handled at the pipeline level, not in grid metadata.

## Usage

### Basic Example
```python
from sensoryforge.core.composite_grid import CompositeReceptorGrid

grid = CompositeReceptorGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0), device='cpu')
grid.add_layer(name="SA1", density=100.0, arrangement="grid")
grid.add_layer(name="RA", density=50.0, arrangement="hex")

# Access population coordinates
sa1_coords = grid.get_layer_coordinates("SA1")  # Returns [num_receptors, 2] tensor
```

## Configuration
```yaml
grid:
  type: composite
  xlim: [-5.0, 5.0]
  ylim: [-5.0, 5.0]
  populations:
    sa1:
      density: 100.0
      arrangement: grid  # 'grid', 'hex', 'poisson', 'jittered_grid'
    ra:
      density: 50.0
      arrangement: hex
```

**Note:** The `filter` field is deprecated and ignored. Filter associations are
configured separately in the pipeline's filter section.

## See Also
- [Equation DSL](equation_dsl.md)
- [Solvers](solvers.md)
- [YAML Configuration](yaml_configuration.md)
