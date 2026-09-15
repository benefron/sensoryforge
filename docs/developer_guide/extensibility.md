# SensoryForge Extensibility Guide

This guide explains how to extend SensoryForge with new components, following the established architecture and patterns.

## Architecture Overview

SensoryForge uses a **registry-based architecture** for extensibility:

```
┌─────────────────────────────────────────────────────────┐
│              Component Registry System                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Neurons  │  │ Filters │  │Innervation│ │ Stimuli │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│       │            │              │             │         │
│       └────────────┴──────────────┴─────────────┘         │
│                    │                                        │
│            ComponentRegistry                                │
│         (register/create/list)                              │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Base Classes (ABC)   │
        │  - BaseNeuron         │
        │  - BaseFilter         │
        │  - BaseInnervation    │
        │  - BaseStimulus       │
        │  - BaseSolver         │
        │  - BaseGrid           │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Concrete Classes     │
        │  - IzhikevichNeuron   │
        │  - SAFilter           │
        │  - GaussianInnervation│
        │  - GaussianStimulus   │
        └───────────────────────┘
```

## Component Lifecycle

### 1. Registration

Two routes, both ending at the same registries:

- **Plugin package** (recommended for third parties): `sensoryforge
  new-component <kind> <Name> --dest DIR` scaffolds an installable package
  that registers itself via a `sensoryforge.components` entry point,
  discovered automatically at import time — no edits to this checkout.
  See `plugins.md`.
- **In-repo** (for contributing to SensoryForge itself): components are
  registered by hand in `sensoryforge/register_components.py`:

```python
from sensoryforge.registry import NEURON_REGISTRY
from sensoryforge.neurons.my_neuron import MyNeuron

def register_all():
    NEURON_REGISTRY.register("my_neuron", MyNeuron)
```

Registry lookups are case-insensitive (H1, F-046): `"MyNeuron"` and
`"myneuron"` refer to the same registration; a genuine collision between
two different classes under the same case-folded name still raises.

### 2. Instantiation

Components are created via registry lookup:

```python
from sensoryforge.registry import NEURON_REGISTRY

# Lookup by name
neuron_cls = NEURON_REGISTRY.get_class("my_neuron")
neuron = neuron_cls(**config)

# Or use create() which handles factory functions
neuron = NEURON_REGISTRY.create("my_neuron", **config)
```

### 3. Configuration

Components support config-based instantiation:

```python
# From config dict
neuron = MyNeuron.from_config({"param1": 1.0, "param2": 2.0})

# Serialize to dict
config = neuron.to_dict()
```

## Base Classes

### BaseNeuron

All neuron models inherit from `BaseNeuron`:

```python
from sensoryforge.neurons.base import BaseNeuron

class MyNeuron(BaseNeuron):
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Return spike output [batch, time, num_neurons]."""
        pass
    
    @classmethod
    def from_config(cls, config: Dict) -> 'MyNeuron':
        pass
    
    def to_dict(self) -> Dict:
        pass
```

### BaseFilter

All filters inherit from `BaseFilter`:

```python
from sensoryforge.filters.base import BaseFilter

class MyFilter(BaseFilter):
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """Return filtered current [batch, time, num_neurons]."""
        pass
```

### BaseInnervation

All innervation methods inherit from `BaseInnervation`:

```python
from sensoryforge.core.innervation import BaseInnervation

class MyInnervation(BaseInnervation):
    def compute_weights(
        self,
        receptor_coords: torch.Tensor,
        neuron_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Return weight matrix [num_neurons, num_receptors]."""
        pass
```

## Registry Pattern

The `ComponentRegistry` provides a unified interface for component lookup:

```python
class ComponentRegistry:
    def register(self, name: str, cls: Type, factory_func: Optional[Callable] = None)
    def create(self, name: str, **kwargs) -> Any
    def get_class(self, name: str) -> Type
    def is_registered(self, name: str) -> bool
    def list_registered(self) -> List[str]
```

### Benefits

1. **No hardcoded if/else chains** - all lookups go through registry
2. **Easy extensibility** - just register new components
3. **Consistent interface** - all components follow same pattern
4. **Clear errors** - registry provides helpful error messages

## Configuration Schema

SensoryForge uses a **canonical configuration schema** (`SensoryForgeConfig`) that supports:

- Multiple grid layers
- N populations (not hardcoded SA/RA/SA2)
- Per-population innervation, filter, neuron, solver config
- Stimulus definitions
- Simulation settings

### Example Config

```yaml
grids:
  - name: "Grid 1"
    arrangement: "grid"
    rows: 40
    cols: 40
    spacing: 0.15

populations:
  - name: "SA Population"
    neuron_model: "izhikevich"
    filter_method: "sa"
    innervation_method: "gaussian"
    connections_per_neuron: 28
    sigma_d_mm: 0.3

stimulus:
  type: "gaussian"
  amplitude: 10.0
  sigma: 1.0

simulation:
  device: "cpu"
  dt: 1.0
```

## Future Extension Points

The architecture supports future extensions:

### Composite Stimuli

Multiple stimuli over multiple composite grids:

```python
# Future: CompositeStimulus
composite_stimulus = CompositeStimulus(
    stimuli=[
        GaussianStimulus(...),
        TextureStimulus(...),
    ],
    grids=[grid1, grid2],
)
```

### Additional Filtering Layers

On-off cells, center-surround, lateral inhibition:

```python
# Future: ProcessingPipeline with multiple layers
pipeline = ProcessingPipeline([
    IdentityLayer(),
    CenterSurroundLayer(),
    LateralInhibitionLayer(),
])
```

### Inter-Population Connections

Inhibitory cells connecting populations:

```python
# Future: InterPopulationInnervation
inhibitory_innervation = InterPopulationInnervation(
    source_population=sa_population,
    target_population=ra_population,
    connection_type="inhibitory",
)
```

## Best Practices

### 1. Always Use Registries

```python
# ❌ BAD
if model_name == "izhikevich":
    return IzhikevichNeuronTorch(**kwargs)

# ✅ GOOD
neuron_cls = NEURON_REGISTRY.get_class(model_name)
return neuron_cls(**kwargs)
```

### 2. Document Tensor Shapes

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Process input.
    
    Args:
        x: Input tensor [batch, time, num_neurons] in mA
    
    Returns:
        Output tensor [batch, time, num_neurons] in mV
    """
```

### 3. Specify Device Explicitly

```python
# ❌ BAD
tensor = torch.zeros(10, 10)

# ✅ GOOD
tensor = torch.zeros(10, 10, device=self.device)
```

### 4. Implement Config Methods

All components must implement:

- `from_config(config: Dict) -> Self` (classmethod)
- `to_dict() -> Dict` (instance method) — must include every `__init__`
  parameter so `from_config(instance.to_dict())` round-trips to a fixed
  point (H3 completeness requirement)
- `get_param_spec() -> List[ParamSpec]` (classmethod, required on every
  component since G1 — not just stimuli)

`sensoryforge.testing.contracts.check_component(kind, cls)` runs all three
checks (plus a forward-pass shape check) in one call, and is what both the
plugin-package and in-repo scaffolds' generated tests use.

### 5. Write Tests

Every new component needs:

- Unit tests for functionality
- Registry lookup tests
- Config round-trip tests

## Examples

See the following files for reference implementations:

- **Neuron**: `sensoryforge/neurons/izhikevich.py`
- **Filter**: `sensoryforge/filters/sa_ra.py`
- **Innervation**: `sensoryforge/core/innervation.py` (GaussianInnervation)
- **Stimulus**: `sensoryforge/stimuli/gaussian.py`
- **Solver**: `sensoryforge/solvers/euler.py`

## Getting Help

- See `plugins.md` for the entry-point plugin-package route in full
- See `add_neuron.md`, `add_filter.md`, `add_stimulus.md` for step-by-step
  guides per component kind
- See `.cursor/rules/extensibility-patterns.mdc` for coding patterns
- See `.cursor/skills/add-new-component/SKILL.md` for step-by-step guide
- Check existing components for reference implementations
