Add a new extensible component to SensoryForge following the registry pattern.

$ARGUMENTS should describe what to add, e.g.:
- "neuron model: leaky integrate-and-fire"
- "filter: center-surround antagonism"
- "stimulus: sinusoidal grating"
- "innervation: one-to-one"

## Step 1 — Identify the base class

| Component type | Base class | Directory | Registry |
|---|---|---|---|
| Neuron model | `BaseNeuron` (`neurons/base.py`) | `sensoryforge/neurons/` | `NEURON_REGISTRY` |
| Filter | `BaseFilter` (`filters/base.py`) | `sensoryforge/filters/` | `FILTER_REGISTRY` |
| Innervation | `BaseInnervation` (`core/innervation.py`) | `sensoryforge/core/` | `INNERVATION_REGISTRY` |
| Stimulus | `BaseStimulus` (`stimuli/base.py`) | `sensoryforge/stimuli/` | `STIMULUS_REGISTRY` |
| Solver | `BaseSolver` (`solvers/base.py`) | `sensoryforge/solvers/` | `SOLVER_REGISTRY` |
| Grid | `BaseGrid` (`core/grid_base.py`) | `sensoryforge/core/` | `GRID_REGISTRY` |
| Processing layer | `BaseProcessingLayer` (`core/processing.py`) | `sensoryforge/core/` | (none) |

Read the chosen base class file before writing any code.

## Step 2 — Create the component file

Follow this template exactly:

```python
"""<One-line description of what this component does>."""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple
from sensoryforge.<subsystem>.base import Base<Type>


class <ClassName>(Base<Type>):
    """<One-line summary>.

    <Extended description. Include the scientific motivation, any paper
    reference, and when to use this component vs alternatives.>

    Args:
        param1: <description>. Units: <unit if physical>.
        param2: <description>. Default: <value>.
        dt: Time step in ms. Default: 1.0.

    References:
        Author et al. (Year). "Title". Journal. DOI.
    """

    def __init__(self, param1: float, param2: float = 1.0, dt: float = 1.0, device: str = "cpu") -> None:
        super().__init__(dt=dt)
        self.param1 = param1
        self.param2 = param2
        self.device = torch.device(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """<One-line summary>.

        Args:
            x: <description> [batch, time, num_neurons] in <units>

        Returns:
            <description> [batch, time, num_neurons] in <units>

        Example:
            >>> component = <ClassName>(param1=1.0)
            >>> out = component(torch.randn(2, 100, 64))
            >>> out.shape
            torch.Size([2, 100, 64])
        """
        # Never hand-roll loops over neurons — use vectorised tensor ops
        raise NotImplementedError

    def reset_state(self) -> None:
        """Reset any internal state buffers to initial conditions."""
        pass  # replace with actual state reset if component is stateful

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "<ClassName>":
        """Instantiate from a config dictionary (YAML-deserialisable).

        Args:
            config: Dictionary with component parameters.

        Returns:
            <ClassName> instance.
        """
        return cls(
            param1=config["param1"],
            param2=config.get("param2", 1.0),
            dt=config.get("dt", 1.0),
            device=config.get("device", "cpu"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (round-trips through from_config).

        Returns:
            Dictionary with 'type' key and all constructor parameters.
        """
        return {
            "type": "<registry_name>",
            "param1": float(self.param1),
            "param2": float(self.param2),
            "dt": float(self.dt),
        }
```

## Step 3 — Register in `register_components.py`

Add at the top of the file:
```python
from sensoryforge.<subsystem>.<filename> import <ClassName>
```

Inside `register_all()`:
```python
<REGISTRY>.register("<snake_case_name>", <ClassName>)
<REGISTRY>.register("<ClassName>", <ClassName>)  # alias for GUI compatibility
```

## Step 4 — Add to canonical schema (if user-configurable)

In `sensoryforge/config/schema.py`, add relevant fields to the appropriate `@dataclass` with type annotations and defaults.

## Step 5 — Update GUI (if user-selectable)

If the component should appear in a GUI dropdown, add it to the relevant combo box in `sensoryforge/gui/tabs/`.

## Step 6 — Write tests

Create `tests/unit/test_<component_type>.py` (or add to existing). Minimum required:

```python
def test_<name>_forward_shape():
    """Output shape matches input shape."""

def test_<name>_registry():
    """Component is registered and retrievable by name."""

def test_<name>_config_roundtrip():
    """from_config(obj.to_dict()) produces an equivalent object."""

def test_<name>_device(device):
    """Component works on cpu (and cuda if available)."""
```

Run: `pytest tests/unit/test_<component_type>.py -v`

## Step 7 — Commit

```
feat(<subsystem>): add <ComponentName> <type>
```

## Innervation special case

Innervation methods need a factory function because they require pre-computed coordinates:

```python
def create_<name>(**kwargs):
    receptor_coords = kwargs.pop("receptor_coords")
    neuron_centers = kwargs.pop("neuron_centers")
    device = kwargs.pop("device", "cpu")
    return <ClassName>(receptor_coords, neuron_centers, device=device, **kwargs)

INNERVATION_REGISTRY.register("<name>", <ClassName>, create_<name>)
```

See `sensoryforge/core/innervation.py` (`GaussianInnervation`) as the reference implementation.
