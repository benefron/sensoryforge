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

### BaseInnervation (receptive-field builders)

All receptive-field builders inherit from `BaseInnervation`; `build()` (inherited) wraps
`compute_weights()` in a `ReceptiveFieldBank` with provenance. See
`add_rf_builder.md` for the full guide and `docs/examples/rf_builder_plugin.py` for a
runnable example.

```python
from sensoryforge.core.innervation import BaseInnervation

class MyInnervation(BaseInnervation):
    def __init__(self, receptor_coords, neuron_centers, my_param=1.0, device="cpu"):
        super().__init__(receptor_coords, neuron_centers, device)
        self.my_param = my_param

    def compute_weights(self, **kwargs) -> torch.Tensor:
        """Return weight matrix [num_neurons, num_receptors] from self.receptor_coords
        and self.neuron_centers."""
        ...

    def to_dict(self):
        return {**super().to_dict(), "my_param": self.my_param}
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
- `to_dict() -> Dict` (instance method) — should include every `__init__`
  parameter so `from_config(instance.to_dict())` round-trips to a fixed
  point
- `get_param_spec() -> List[ParamSpec]` (classmethod, required on every
  component since G1 — not just stimuli)

`sensoryforge.testing.contracts.check_component(kind, cls)` runs the same checks
for every kind it supports (`available_kinds()`: `neuron`, `filter`, `grid`,
`solver`, `innervation`, `stimulus`, `processing`):

1. `get_param_spec()` returns a list of `ParamSpec`.
2. One forward pass with the kind's canonical tensor shape.
3. `from_config(instance.to_dict())` returns an instance of the class.
4. **The round trip is complete** (F-049): every `__init__` parameter appears
   in `instance.to_dict()`, and `from_config(instance.to_dict()).to_dict()`
   equals `instance.to_dict()` (a fixed point). A name in the class's
   `_TO_DICT_EXCLUDE_PARAMS` is the one exception, for a parameter a bundle
   stores separately (an innervation's `receptor_coords`/`neuron_centers`) or a
   convenience that is not round-tripped (a neuron's `preset`). A processing
   layer is held to the same two rules against its `{"method": ..., "params":
   {...}}` shape; its `receptor_coords` is exempt.
5. For `innervation`, `build()` returns a `ReceptiveFieldBank` of the right shapes.

Pass an `instance` built with **non-default values** when you can: a default
instance reaches the fixed point even if `from_config` ignores a key, because
the missing key falls back to the same default. A component that leaves a
parameter out of `to_dict()` fails with
`... to_dict() is missing constructor parameters: [...]`; one whose
`from_config` drops or changes a value fails with
`... round trip is not a fixed point`. Every shipped built-in passes, and
`tests/unit/test_contract_checks.py` breaks one of each kind on purpose to show
the check refuses it.

#### World registries

Worlds (`sensoryforge/world/`) register functions and tables rather than classes,
so `check_component` does not apply. After registering a shape, pattern,
modulation, distribution or class kind (`register_shape` and friends, or a plugin
entry point), check it by name with
`sensoryforge.testing.world_contracts.check_world_component(kind, name)`; `kind`
is one of `available_world_kinds()`: `shape`, `pattern`, `modulation`,
`distribution`, `class_kind`.

```python
from sensoryforge.testing.world_contracts import check_world_component

check_world_component("shape", "ring")
check_world_component("distribution", "my_dist", example={"scale": 2.0})
check_world_component("class_kind", "my_kind", example=my_world_mapping)
```

The check calls your function the way the renderer does (batch first, numbers as
tensors that broadcast against the coordinates, strings and bools plain, and
`amplitude` taken out first) and raises `AssertionError` naming the entry and the
rule. What it holds each kind to:

| Kind | Rules |
|---|---|
| every kind | `ParamSpec` rows are well formed: unique names, none called `kind`, a `dtype` of `float`, `int`, `bool` or `str`, a default of that type inside `min_val`/`max_val` and `choices`, finite and JSON-plain. The rows rebuild from their own `to_dict()`, and `{kind: name}` normalises, through a JSON round trip, to the kind plus its defaults and then to itself (fields in `layered.ADDED_FIELDS` stay invisible until set). |
| `shape` | `amplitude` is a spec. `fn(x, y, params)` returns a finite tensor of the coordinates' shape and dtype, in float32 and float64, at the defaults and at both declared ends of every range (an end the shape's own `check` refuses is not a draw and is skipped). Draw *i* alone equals draw *i* in a batch, bit for bit (guarantee 1): a reduction over the batch, or a loop whose length depends on a draw, breaks it. The defaults satisfy the shape's own `check`. An indenter has no spec called `depth` (the renderer passes it) and returns exactly 0 at depth 0. |
| `pattern` | `x_mm` and `y_mm` are declared together. `fn(params)` returns `(positions, scales)`: two lists of equal length, positions finite `(x, y)` in mm. It is a pure function of its parameters (the kernel caches by them), and `pattern_batch` pads a group of them. |
| `modulation` | `fn(tc, params)` (or `None` for no modulation) returns a finite tensor of `tc`'s shape and dtype with values in `[0, 1]`, and draw *i* alone equals draw *i* in a batch. |
| `distribution` | `sample(u, axis)` returns a list with one finite, JSON-plain value per `u` in `[0, 1)` (including 0 and the largest float below 1), the same on every call, each depending on its own `u` only. Every value lies in `support(axis)` and every number in `bounds(axis)` when they are declared; `quantile=True` means numbers. The axis re-parses from its own `to_dict()`. Pass `example=` the options its axis takes if it needs any; a `check` hook must raise `ValueError` for options it refuses. |
| `class_kind` | The registered name is the kind's `name`, and the methods `ClassKind` leaves open are overridden. On an `example` world (the mapping `World.from_dict` takes, with a class of this kind): `normalise_layer` is a fixed point, every built-in default resolves, and each draw has a finite `end_ms >= 0` and an ordered `timeline` inside it. Rendered frames are finite, exactly 0 before 0 and from `end_ms` on, and draw *i* alone equals draw *i* in a batch. |

The checks say nothing about registering a name twice. A shipped built-in that
could not pass would be recorded as an open ledger item rather than excused by a
weaker check; none is open today, and `tests/contract/test_world_component_contracts.py`
runs every shipped shape, pattern, modulation, distribution and class kind through them.

### 5. Write Tests

Every new component needs:

- Unit tests for functionality
- Registry lookup tests
- Config round-trip tests

## Examples

See the following files for reference implementations:

- **Neuron**: `sensoryforge/neurons/izhikevich.py`
- **Filter**: `sensoryforge/filters/sa_ra.py`
- **Innervation / receptive-field builders**: `sensoryforge/core/innervation.py` (GaussianInnervation), `sensoryforge/core/rf_builders/template.py` (TemplateRFBuilder)
- **Stimulus**: `sensoryforge/stimuli/gaussian.py`
- **Solver**: `sensoryforge/solvers/euler.py`

## Getting Help

- See `plugins.md` for the entry-point plugin-package route in full
- See `add_neuron.md`, `add_filter.md`, `add_stimulus.md` for step-by-step
  guides per component kind
- See `architecture_notes.md` ("Adding a New Component") for the contract every
  component meets
- Check existing components for reference implementations
