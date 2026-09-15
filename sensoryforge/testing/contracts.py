"""Reusable contract checks for SensoryForge component classes (H2, F-047).

Extracted from ``tests/contract/test_component_contracts.py`` (G4) so the
same three checks -- ``get_param_spec()`` shape, one forward pass with the
kind's canonical tensor shape, and a ``from_config(to_dict())`` round trip --
can be shared between the in-repo contract test sweep and the
``tests/test_contract.py`` generated for every ``sensoryforge new-component``
plugin package.

No Qt imports; only ``torch`` plus stdlib, so a generated plugin package can
depend on nothing but ``sensoryforge`` itself to run its own tests.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch

from sensoryforge.stimuli.base import ParamSpec

__all__ = ["check_component", "available_kinds"]


def _assert_param_spec(cls: type) -> None:
    """Assert ``cls.get_param_spec()`` returns a list of ``ParamSpec``."""
    spec = cls.get_param_spec()
    if not isinstance(spec, list):
        raise AssertionError(
            f"{cls.__name__}.get_param_spec() must return a list, got {type(spec)!r}"
        )
    if not all(isinstance(p, ParamSpec) for p in spec):
        raise AssertionError(
            f"{cls.__name__}.get_param_spec() must return a list of ParamSpec "
            f"instances, got {spec!r}"
        )


def _check_neuron(cls: type, instance: Any) -> None:
    """Canonical neuron shape: [batch, steps, features] -> v/spikes [batch, steps+1, features]."""
    _assert_param_spec(cls)
    current = torch.randn(1, 5, 3)
    v_trace, spikes = instance(current)
    if tuple(v_trace.shape) != (1, 6, 3):
        raise AssertionError(
            f"{cls.__name__} forward(): expected v_trace shape (1, 6, 3), "
            f"got {tuple(v_trace.shape)}"
        )
    if tuple(spikes.shape) != (1, 6, 3):
        raise AssertionError(
            f"{cls.__name__} forward(): expected spikes shape (1, 6, 3), "
            f"got {tuple(spikes.shape)}"
        )
    reconstructed = cls.from_config(instance.to_dict())
    if not isinstance(reconstructed, cls):
        raise AssertionError(
            f"{cls.__name__}.from_config(instance.to_dict()) did not return "
            f"a {cls.__name__} instance (got {type(reconstructed)!r})"
        )


def _check_filter(cls: type, instance: Any) -> None:
    """Canonical filter shape: [batch, time, neurons] -> same shape."""
    _assert_param_spec(cls)
    x = torch.randn(1, 5, 3)
    out = instance(x)
    if tuple(out.shape) != tuple(x.shape):
        raise AssertionError(
            f"{cls.__name__} forward(): expected output shape {tuple(x.shape)}, "
            f"got {tuple(out.shape)}"
        )
    reconstructed = cls.from_config(instance.to_dict())
    if not isinstance(reconstructed, cls):
        raise AssertionError(
            f"{cls.__name__}.from_config(instance.to_dict()) did not return "
            f"a {cls.__name__} instance (got {type(reconstructed)!r})"
        )


def _check_grid(cls: type, instance: Any) -> None:
    """Canonical grid arrangement forward: get_all_coordinates() -> [N, 2]."""
    _assert_param_spec(cls)
    coords = instance.get_all_coordinates()
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise AssertionError(
            f"{cls.__name__}.get_all_coordinates() must return shape [N, 2], "
            f"got {tuple(coords.shape)}"
        )
    reconstructed = cls.from_config(instance.to_dict())
    if not isinstance(reconstructed, cls):
        raise AssertionError(
            f"{cls.__name__}.from_config(instance.to_dict()) did not return "
            f"a {cls.__name__} instance (got {type(reconstructed)!r})"
        )
    if reconstructed.get_all_coordinates().shape[1] != 2:
        raise AssertionError(
            f"{cls.__name__}: round-tripped instance's get_all_coordinates() "
            "is not [N, 2]"
        )


def _check_solver(cls: type, instance: Any) -> None:
    """Canonical solver forward: one Euler-style step() -> same shape as state."""
    _assert_param_spec(cls)

    def decay(state: torch.Tensor, t: float) -> torch.Tensor:
        return -0.1 * state

    state = torch.randn(1, 3)
    new_state = instance.step(decay, state, t=0.0, dt=instance.dt)
    if tuple(new_state.shape) != tuple(state.shape):
        raise AssertionError(
            f"{cls.__name__}.step() must preserve state shape "
            f"{tuple(state.shape)}, got {tuple(new_state.shape)}"
        )
    reconstructed = cls.from_config(instance.to_dict())
    if not isinstance(reconstructed, cls):
        raise AssertionError(
            f"{cls.__name__}.from_config(instance.to_dict()) did not return "
            f"a {cls.__name__} instance (got {type(reconstructed)!r})"
        )


def _check_innervation(cls: type, instance: Any) -> None:
    """Canonical innervation forward: compute_weights() -> [neurons, receptors]."""
    _assert_param_spec(cls)
    weights = instance.compute_weights()
    expected_shape = (instance.num_neurons, instance.num_receptors)
    if tuple(weights.shape) != expected_shape:
        raise AssertionError(
            f"{cls.__name__}.compute_weights() expected shape {expected_shape}, "
            f"got {tuple(weights.shape)}"
        )
    # to_dict() deliberately drops receptor_coords/neuron_centers (too large
    # to serialise) and adds derived, non-constructor fields (method,
    # num_neurons, num_receptors) -- strip those, merge the tensors back in.
    serialised = instance.to_dict()
    for derived_key in ("method", "num_neurons", "num_receptors"):
        serialised.pop(derived_key, None)
    reconstructed = cls.from_config(
        {
            **serialised,
            "receptor_coords": instance.receptor_coords,
            "neuron_centers": instance.neuron_centers,
        }
    )
    if not isinstance(reconstructed, cls):
        raise AssertionError(
            f"{cls.__name__}.from_config(...) did not return a {cls.__name__} "
            f"instance (got {type(reconstructed)!r})"
        )


def _check_stimulus(cls: type, instance: Any) -> None:
    """Canonical stimulus forward shape: [H, W] -> [H, W]."""
    _assert_param_spec(cls)
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 8), indexing="ij"
    )
    out = instance(xx, yy)
    if tuple(out.shape) != tuple(xx.shape):
        raise AssertionError(
            f"{cls.__name__} forward(): expected shape {tuple(xx.shape)}, "
            f"got {tuple(out.shape)}"
        )
    reconstructed = cls.from_config(instance.to_dict())
    if not isinstance(reconstructed, cls):
        raise AssertionError(
            f"{cls.__name__}.from_config(instance.to_dict()) did not return "
            f"a {cls.__name__} instance (got {type(reconstructed)!r})"
        )
    out2 = reconstructed(xx, yy)
    if tuple(out2.shape) != tuple(xx.shape):
        raise AssertionError(
            f"{cls.__name__}: round-tripped instance's forward() expected "
            f"shape {tuple(xx.shape)}, got {tuple(out2.shape)}"
        )


_CHECKS: Dict[str, Callable[[type, Any], None]] = {
    "neuron": _check_neuron,
    "filter": _check_filter,
    "grid": _check_grid,
    "solver": _check_solver,
    "innervation": _check_innervation,
    "stimulus": _check_stimulus,
}


def available_kinds() -> List[str]:
    """Return the component kinds :func:`check_component` supports."""
    return sorted(_CHECKS)


def check_component(kind: str, cls: type, instance: Optional[Any] = None) -> None:
    """Run the standard SensoryForge component contract checks.

    Checks (matching ``tests/contract/test_component_contracts.py``, G4):

    1. ``cls.get_param_spec()`` returns a list of ``ParamSpec`` instances.
    2. One forward pass succeeds with ``kind``'s canonical tensor shape.
    3. ``cls.from_config(instance.to_dict())`` round-trips to a ``cls`` instance.

    Args:
        kind: One of ``"neuron"``, ``"filter"``, ``"grid"``, ``"solver"``,
            ``"innervation"``, ``"stimulus"``.
        cls: The component class under test. Must be (or subclass) the
            matching base class -- ``BaseNeuron``, ``BaseFilter``,
            ``BaseGrid``, ``BaseSolver``, ``BaseInnervation``, or
            ``BaseStimulus``.
        instance: An already-constructed instance to check against. Required
            for kinds without a parameter-free default constructor (e.g.
            ``innervation``, which always needs ``receptor_coords``/
            ``neuron_centers``) or when the default-constructed instance
            would not exercise the case under test (e.g. a composite
            stimulus needing a sub-stimulus). If omitted, ``cls()`` is used.

    Raises:
        ValueError: If ``kind`` is not one of :func:`available_kinds`.
        AssertionError: If any contract check fails, with a message naming
            the class and the check that failed.

    Example:
        >>> from my_plugin.component import MyFilter
        >>> check_component("filter", MyFilter)
    """
    if kind not in _CHECKS:
        raise ValueError(
            f"Unknown component kind {kind!r}; choose one of {available_kinds()}"
        )
    if instance is None:
        instance = cls()
    _CHECKS[kind](cls, instance)
