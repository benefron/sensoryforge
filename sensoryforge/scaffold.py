"""``sensoryforge new-component`` scaffold generator (G5).

Writes a starter component class, a unit test that exercises the same
checks as the G4 contract test (``tests/contract/test_component_contracts.py``),
and a docs stub, for one of the five extensible component kinds: neuron,
filter, stimulus, solver, grid.

Registration with the relevant ``*_REGISTRY`` in
``sensoryforge/register_components.py`` is deliberately left as a manual,
printed-out step (mirroring ``docs/developer_guide/add_filter.md`` etc.) --
editing that file's ``register_all()`` body by string manipulation is a
higher-risk automation than the benefit it would save for a one-line change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


def _to_snake(name: str) -> str:
    """Convert ``CamelCase`` or ``Mixed Case`` to ``snake_case``."""
    name = re.sub(r"[\s-]+", "_", name.strip())
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return name.lower().strip("_")


def _to_pascal(name: str) -> str:
    """Convert ``snake_case``/``Mixed Case`` to ``PascalCase``."""
    parts = re.split(r"[\s_-]+", name.strip())
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


@dataclass(frozen=True)
class _KindSpec:
    module_dir: str
    base_import: str
    base_class: str
    class_suffix: str
    module_template: str
    test_template: str
    docs_template: str


def _neuron_kind() -> _KindSpec:
    module_template = '''"""{{class_name}}: a new spiking neuron model.

Implements the ``BaseNeuron`` contract -- see
``docs/developer_guide/add_neuron.md`` for the full guide.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from sensoryforge.neurons.base import BaseNeuron
from sensoryforge.stimuli.base import ParamSpec


class {{class_name}}(BaseNeuron):
    """{{class_name}} neuron model.

    Args:
        tau_m: Membrane time constant in ms.
        v_thresh: Spike threshold in mV.
        v_reset: Reset potential in mV.
        dt: Integration time step in ms.
    """

    def __init__(
        self,
        tau_m: float = 20.0,
        v_thresh: float = -50.0,
        v_reset: float = -65.0,
        dt: float = 0.05,
    ) -> None:
        super().__init__(dt=dt)
        self.tau_m = tau_m
        self.v_thresh = v_thresh
        self.v_reset = v_reset

    def reset_state(self) -> None:
        """No persistent state: forward() re-initialises v each call."""

    def forward(
        self, input_current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate a leaky membrane over ``input_current`` (placeholder LIF).

        Args:
            input_current: Input currents ``[batch, steps, features]`` in mA.

        Returns:
            Tuple of ``v_trace`` ``[batch, steps+1, features]`` in mV and
            ``spikes`` ``[batch, steps+1, features]`` (bool).
        """
        batch, steps, features = input_current.shape
        device, dtype = input_current.device, input_current.dtype
        v = torch.full((batch, features), self.v_reset, device=device, dtype=dtype)
        v_trace = [v]
        spikes = [torch.zeros_like(v, dtype=torch.bool)]
        for t in range(steps):
            dv = (-(v - self.v_reset) + input_current[:, t, :]) / self.tau_m * self.dt
            v = v + dv
            spiked = v >= self.v_thresh
            v = torch.where(spiked, torch.full_like(v, self.v_reset), v)
            v_trace.append(v)
            spikes.append(spiked)
        return torch.stack(v_trace, dim=1), torch.stack(spikes, dim=1)

    def to_dict(self) -> dict:
        return {
            "tau_m": self.tau_m,
            "v_thresh": self.v_thresh,
            "v_reset": self.v_reset,
            "dt": self.dt,
        }

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("tau_m", dtype="float", default=20.0, min_val=0.1, unit="ms"),
            ParamSpec("v_thresh", dtype="float", default=-50.0, unit="mV"),
            ParamSpec("v_reset", dtype="float", default=-65.0, unit="mV"),
        ]
'''
    test_template = '''"""Unit test for {{class_name}} (mirrors tests/contract/test_component_contracts.py)."""

import torch

from {{module_path}} import {{class_name}}


def test_forward_shape():
    neuron = {{class_name}}()
    current = torch.randn(1, 5, 3)
    v_trace, spikes = neuron(current)
    assert v_trace.shape == (1, 6, 3)
    assert spikes.shape == (1, 6, 3)


def test_param_spec_returns_param_specs():
    from sensoryforge.stimuli.base import ParamSpec

    spec = {{class_name}}.get_param_spec()
    assert isinstance(spec, list)
    assert all(isinstance(p, ParamSpec) for p in spec)


def test_from_config_to_dict_round_trip():
    neuron = {{class_name}}(tau_m=15.0)
    reconstructed = {{class_name}}.from_config(neuron.to_dict())
    assert isinstance(reconstructed, {{class_name}})
    assert reconstructed.tau_m == 15.0
'''
    docs_template = """# {{class_name}}

Scaffolded with `sensoryforge new-component neuron {{name}}`.

Implements the `BaseNeuron` contract (see
`docs/developer_guide/add_neuron.md`). Edit `{{module_path}}` to replace the
placeholder leaky-integrate-and-fire dynamics with your model's equations.

## Registration

Add to `sensoryforge/register_components.py`:

```python
from {{module_path}} import {{class_name}}

NEURON_REGISTRY.register("{{registry_key}}", {{class_name}})
```
"""
    return _KindSpec(
        module_dir="sensoryforge/neurons",
        base_import="sensoryforge.neurons.base",
        base_class="BaseNeuron",
        class_suffix="NeuronTorch",
        module_template=module_template,
        test_template=test_template,
        docs_template=docs_template,
    )


def _filter_kind() -> _KindSpec:
    module_template = '''"""{{class_name}}: a new temporal filter.

Implements the ``BaseFilter`` contract -- see
``docs/developer_guide/add_filter.md`` for the full guide.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from sensoryforge.filters.base import BaseFilter
from sensoryforge.stimuli.base import ParamSpec


class {{class_name}}(BaseFilter):
    """{{class_name}} temporal filter (placeholder single-pole lowpass).

    Args:
        tau: Time constant in ms.
        dt: Integration time step in ms.
    """

    def __init__(self, tau: float = 10.0, dt: float = 0.1) -> None:
        super().__init__(dt=dt)
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau
        self._state: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        self._state = None

    def forward(
        self, x: torch.Tensor, dt: Optional[float] = None
    ) -> torch.Tensor:
        """Apply a single-pole lowpass over the time dimension.

        Args:
            x: Input drive ``[batch, time, N_neurons]`` in mA.
            dt: Optional override for the integration step (ms).

        Returns:
            Filtered current, same shape as ``x``.
        """
        step = dt if dt is not None else self.dt
        batch, steps, n = x.shape
        state = (
            self._state
            if self._state is not None
            else torch.zeros(batch, n, device=x.device, dtype=x.dtype)
        )
        outputs = []
        for t in range(steps):
            state = state + (x[:, t, :] - state) / self.tau * step
            outputs.append(state.unsqueeze(1))
        self._state = state.detach()
        return torch.cat(outputs, dim=1)

    def to_dict(self) -> dict:
        return {"tau": self.tau, "dt": self.dt}

    @classmethod
    def from_config(cls, config: dict) -> "{{class_name}}":
        return cls(tau=config.get("tau", 10.0), dt=config.get("dt", 0.1))

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("tau", dtype="float", default=10.0, min_val=0.01, unit="ms"),
        ]
'''
    test_template = '''"""Unit test for {{class_name}} (mirrors tests/contract/test_component_contracts.py)."""

import torch

from {{module_path}} import {{class_name}}


def test_forward_shape():
    filt = {{class_name}}()
    x = torch.randn(1, 5, 3)
    out = filt(x)
    assert out.shape == x.shape


def test_param_spec_returns_param_specs():
    from sensoryforge.stimuli.base import ParamSpec

    spec = {{class_name}}.get_param_spec()
    assert isinstance(spec, list)
    assert all(isinstance(p, ParamSpec) for p in spec)


def test_from_config_to_dict_round_trip():
    filt = {{class_name}}(tau=8.0)
    reconstructed = {{class_name}}.from_config(filt.to_dict())
    assert isinstance(reconstructed, {{class_name}})
    assert reconstructed.tau == 8.0
'''
    docs_template = """# {{class_name}}

Scaffolded with `sensoryforge new-component filter {{name}}`.

Implements the `BaseFilter` contract (see
`docs/developer_guide/add_filter.md`). Edit `{{module_path}}` to replace the
placeholder single-pole lowpass with your filter's dynamics.

## Registration

Add to `sensoryforge/register_components.py`:

```python
from {{module_path}} import {{class_name}}

FILTER_REGISTRY.register("{{registry_key}}", {{class_name}})
```
"""
    return _KindSpec(
        module_dir="sensoryforge/filters",
        base_import="sensoryforge.filters.base",
        base_class="BaseFilter",
        class_suffix="FilterTorch",
        module_template=module_template,
        test_template=test_template,
        docs_template=docs_template,
    )


def _stimulus_kind() -> _KindSpec:
    module_template = '''"""{{class_name}}: a new spatial stimulus.

Implements the ``BaseStimulus`` contract -- see
``docs/developer_guide/add_stimulus.md`` for the full guide.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from sensoryforge.stimuli.base import BaseStimulus, ParamSpec


class {{class_name}}(BaseStimulus):
    """{{class_name}} stimulus (placeholder flat-top disc).

    Args:
        center_x: X-coordinate of the disc center, mm.
        center_y: Y-coordinate of the disc center, mm.
        radius: Disc radius, mm.
        amplitude: Peak amplitude inside the disc.
    """

    def __init__(
        self,
        center_x: float = 0.0,
        center_y: float = 0.0,
        radius: float = 0.5,
        amplitude: float = 1.0,
    ) -> None:
        super().__init__()
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.amplitude = amplitude

    def reset_state(self) -> None:
        """Stateless: nothing to reset."""

    def forward(
        self, xx: torch.Tensor, yy: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Generate a flat-top disc on the given coordinate grid.

        Args:
            xx: X-coordinate meshgrid, mm.
            yy: Y-coordinate meshgrid, mm.

        Returns:
            Stimulus field matching ``xx``'s shape.
        """
        dist_sq = (xx - self.center_x) ** 2 + (yy - self.center_y) ** 2
        return torch.where(
            dist_sq <= self.radius**2,
            torch.full_like(xx, self.amplitude),
            torch.zeros_like(xx),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
            "amplitude": self.amplitude,
        }

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("center_x", dtype="float", default=0.0, unit="mm"),
            ParamSpec("center_y", dtype="float", default=0.0, unit="mm"),
            ParamSpec("radius", dtype="float", default=0.5, min_val=0.001, unit="mm"),
            ParamSpec("amplitude", dtype="float", default=1.0),
        ]
'''
    test_template = '''"""Unit test for {{class_name}} (mirrors tests/contract/test_component_contracts.py)."""

import torch

from {{module_path}} import {{class_name}}


def test_forward_shape():
    stim = {{class_name}}()
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 8), indexing="ij"
    )
    out = stim(xx, yy)
    assert out.shape == xx.shape


def test_param_spec_returns_param_specs():
    from sensoryforge.stimuli.base import ParamSpec

    spec = {{class_name}}.get_param_spec()
    assert isinstance(spec, list)
    assert all(isinstance(p, ParamSpec) for p in spec)


def test_from_config_to_dict_round_trip():
    stim = {{class_name}}(radius=0.7)
    reconstructed = {{class_name}}.from_config(stim.to_dict())
    assert isinstance(reconstructed, {{class_name}})
    assert reconstructed.radius == 0.7
'''
    docs_template = """# {{class_name}}

Scaffolded with `sensoryforge new-component stimulus {{name}}`.

Implements the `BaseStimulus` contract (see
`docs/developer_guide/add_stimulus.md`). Edit `{{module_path}}` to replace the
placeholder flat-top disc with your stimulus pattern.

## Registration

Add to `sensoryforge/register_components.py`:

```python
from {{module_path}} import {{class_name}}

STIMULUS_REGISTRY.register("{{registry_key}}", {{class_name}})
```
"""
    return _KindSpec(
        module_dir="sensoryforge/stimuli",
        base_import="sensoryforge.stimuli.base",
        base_class="BaseStimulus",
        class_suffix="Stimulus",
        module_template=module_template,
        test_template=test_template,
        docs_template=docs_template,
    )


def _solver_kind() -> _KindSpec:
    module_template = '''"""{{class_name}}: a new ODE solver.

Implements the ``BaseSolver`` contract -- see ``sensoryforge/solvers/base.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import torch

from sensoryforge.solvers.base import BaseSolver
from sensoryforge.stimuli.base import ParamSpec


class {{class_name}}(BaseSolver):
    """{{class_name}} solver (placeholder: forward Euler with a fixed dt)."""

    def __init__(self, dt: float = 0.05) -> None:
        super().__init__(dt=dt)

    def step(
        self,
        ode_func: Callable[[torch.Tensor, float], torch.Tensor],
        state: torch.Tensor,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        return state + dt * ode_func(state, t)

    def integrate(
        self,
        ode_func: Callable[[torch.Tensor, float], torch.Tensor],
        state: torch.Tensor,
        t_span: Tuple[float, float],
        dt: float,
    ) -> torch.Tensor:
        t0, t1 = t_span
        n_steps = max(1, int((t1 - t0) / dt))
        trajectory = [state]
        t = t0
        for _ in range(n_steps):
            state = self.step(ode_func, state, t, dt)
            trajectory.append(state)
            t += dt
        return torch.stack(trajectory, dim=1)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "{{class_name}}":
        return cls(dt=config.get("dt", 0.05))

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("dt", dtype="float", default=0.05, min_val=0.001, unit="ms"),
        ]
'''
    test_template = '''"""Unit test for {{class_name}} (mirrors tests/contract/test_component_contracts.py)."""

import torch

from {{module_path}} import {{class_name}}


def test_step_shape():
    solver = {{class_name}}()

    def decay(state, t):
        return -0.1 * state

    state = torch.randn(1, 3)
    new_state = solver.step(decay, state, t=0.0, dt=solver.dt)
    assert new_state.shape == state.shape


def test_param_spec_returns_param_specs():
    from sensoryforge.stimuli.base import ParamSpec

    spec = {{class_name}}.get_param_spec()
    assert isinstance(spec, list)
    assert all(isinstance(p, ParamSpec) for p in spec)


def test_from_config_to_dict_round_trip():
    solver = {{class_name}}(dt=0.1)
    reconstructed = {{class_name}}.from_config(solver.to_dict())
    assert isinstance(reconstructed, {{class_name}})
    assert reconstructed.dt == 0.1
'''
    docs_template = """# {{class_name}}

Scaffolded with `sensoryforge new-component solver {{name}}`.

Implements the `BaseSolver` contract (see `sensoryforge/solvers/base.py`).
Edit `{{module_path}}` to replace the placeholder fixed-step Euler
integration with your solver's scheme.

## Registration

Add to `sensoryforge/register_components.py`:

```python
from {{module_path}} import {{class_name}}

SOLVER_REGISTRY.register("{{registry_key}}", {{class_name}})
```
"""
    return _KindSpec(
        module_dir="sensoryforge/solvers",
        base_import="sensoryforge.solvers.base",
        base_class="BaseSolver",
        class_suffix="Solver",
        module_template=module_template,
        test_template=test_template,
        docs_template=docs_template,
    )


def _grid_kind() -> _KindSpec:
    module_template = '''"""{{class_name}}: a new receptor grid arrangement.

Implements the ``BaseGrid`` contract -- see ``sensoryforge/core/grid_base.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from sensoryforge.core.grid_base import BaseGrid
from sensoryforge.stimuli.base import ParamSpec


class {{class_name}}(BaseGrid):
    """{{class_name}} grid arrangement (placeholder: uniform random points).

    Args:
        num_points: Number of receptor points to generate.
        xlim: Spatial bounds along x-axis (min, max) in mm.
        ylim: Spatial bounds along y-axis (min, max) in mm.
        device: PyTorch device identifier.
    """

    def __init__(
        self,
        num_points: int = 100,
        xlim: Tuple[float, float] = (-1.0, 1.0),
        ylim: Tuple[float, float] = (-1.0, 1.0),
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(xlim=xlim, ylim=ylim, device=device)
        self.num_points = num_points
        x0, x1 = xlim
        y0, y1 = ylim
        xs = torch.rand(num_points, device=self.device) * (x1 - x0) + x0
        ys = torch.rand(num_points, device=self.device) * (y1 - y0) + y0
        self._coords = torch.stack([xs, ys], dim=1)

    def get_all_coordinates(self) -> torch.Tensor:
        return self._coords

    _CONSTRUCTOR_KEYS = frozenset({"num_points", "xlim", "ylim", "device"})

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "{{class_name}}":
        # to_dict() adds derived fields (type) that aren't constructor args;
        # filter down to the actual constructor keywords.
        filtered = {k: v for k, v in config.items() if k in cls._CONSTRUCTOR_KEYS}
        if "xlim" in filtered:
            filtered["xlim"] = tuple(filtered["xlim"])
        if "ylim" in filtered:
            filtered["ylim"] = tuple(filtered["ylim"])
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["num_points"] = self.num_points
        return result

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("num_points", dtype="int", default=100, min_val=1),
        ]
'''
    test_template = '''"""Unit test for {{class_name}} (mirrors tests/contract/test_component_contracts.py)."""

from {{module_path}} import {{class_name}}


def test_get_all_coordinates_shape():
    grid = {{class_name}}(num_points=10)
    coords = grid.get_all_coordinates()
    assert coords.shape == (10, 2)


def test_param_spec_returns_param_specs():
    from sensoryforge.stimuli.base import ParamSpec

    spec = {{class_name}}.get_param_spec()
    assert isinstance(spec, list)
    assert all(isinstance(p, ParamSpec) for p in spec)


def test_from_config_to_dict_round_trip():
    grid = {{class_name}}(num_points=5)
    reconstructed = {{class_name}}.from_config(grid.to_dict())
    assert isinstance(reconstructed, {{class_name}})
    assert reconstructed.num_points == 5
'''
    docs_template = """# {{class_name}}

Scaffolded with `sensoryforge new-component grid {{name}}`.

Implements the `BaseGrid` contract (see `sensoryforge/core/grid_base.py`).
Edit `{{module_path}}` to replace the placeholder uniform-random point
generation with your arrangement's actual coordinate math.

## Registration

Add to `sensoryforge/register_components.py`:

```python
from {{module_path}} import {{class_name}}

GRID_REGISTRY.register("{{registry_key}}", {{class_name}})
```
"""
    return _KindSpec(
        module_dir="sensoryforge/core",
        base_import="sensoryforge.core.grid_base",
        base_class="BaseGrid",
        class_suffix="Arrangement",
        module_template=module_template,
        test_template=test_template,
        docs_template=docs_template,
    )


_KINDS: Dict[str, _KindSpec] = {
    "neuron": _neuron_kind(),
    "filter": _filter_kind(),
    "stimulus": _stimulus_kind(),
    "solver": _solver_kind(),
    "grid": _grid_kind(),
}


def available_kinds() -> List[str]:
    """Return the component kinds ``new-component`` can scaffold."""
    return sorted(_KINDS)


def generate_component(
    kind: str,
    name: str,
    repo_root: Path,
) -> Dict[str, Path]:
    """Write a scaffolded component's module, unit test, and docs stub.

    Args:
        kind: One of :func:`available_kinds` (``neuron``, ``filter``,
            ``stimulus``, ``solver``, ``grid``).
        name: A human-readable component name, e.g. ``"my cool filter"`` or
            ``"MyCoolFilter"`` -- converted to ``PascalCase`` for the class
            name and ``snake_case`` for the module/test file names and
            registry key.
        repo_root: Repository root the paths below are relative to.

    Returns:
        Dict with keys ``"module"``, ``"test"``, ``"docs"`` mapping to the
        written file paths.

    Raises:
        ValueError: If ``kind`` is not one of :func:`available_kinds`.
        FileExistsError: If the target module file already exists.
    """
    if kind not in _KINDS:
        raise ValueError(
            f"Unknown component kind {kind!r}; choose one of {available_kinds()}"
        )
    spec = _KINDS[kind]

    snake = _to_snake(name)
    pascal = _to_pascal(name)
    class_name = pascal + spec.class_suffix
    module_path_dotted = f"{spec.module_dir.replace('/', '.')}.{snake}"
    registry_key = snake

    module_file = repo_root / spec.module_dir / f"{snake}.py"
    test_file = repo_root / "tests" / "unit" / f"test_{snake}_{kind}.py"
    docs_file = repo_root / "docs" / "developer_guide" / "generated" / f"{snake}.md"

    if module_file.exists():
        raise FileExistsError(f"{module_file} already exists")

    substitutions = {
        "class_name": class_name,
        "name": name,
        "module_path": module_path_dotted,
        "registry_key": registry_key,
    }

    def render(template: str) -> str:
        rendered = template
        for key, value in substitutions.items():
            rendered = rendered.replace("{{" + key + "}}", value)
        return rendered

    module_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.parent.mkdir(parents=True, exist_ok=True)
    docs_file.parent.mkdir(parents=True, exist_ok=True)

    module_file.write_text(render(spec.module_template), encoding="utf-8")
    test_file.write_text(render(spec.test_template), encoding="utf-8")
    docs_file.write_text(render(spec.docs_template), encoding="utf-8")

    return {"module": module_file, "test": test_file, "docs": docs_file}
