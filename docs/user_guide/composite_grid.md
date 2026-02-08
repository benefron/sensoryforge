# Composite Grid

## Overview
Composite grids support multiple receptor populations sharing a coordinate system.

## Concepts
Populations differ by density, arrangement, and metadata (e.g., filter tags).

## Usage

### Basic Example
```python
from sensoryforge.core.composite_grid import CompositeGrid

grid = CompositeGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
grid.add_population(name="SA1", density=100.0, arrangement="grid")
grid.add_population(name="RA", density=50.0, arrangement="hex")

coords = grid.get_population_coordinates("SA1")
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
      arrangement: grid
    ra:
      density: 50.0
      arrangement: hex
```

## See Also
- [Equation DSL](equation_dsl.md)
- [Solvers](solvers.md)
