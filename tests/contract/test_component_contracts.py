"""Contract tests over every registered component (G4).

For each name registered in NEURON_REGISTRY, FILTER_REGISTRY, GRID_REGISTRY,
SOLVER_REGISTRY, INNERVATION_REGISTRY and STIMULUS_REGISTRY, this checks the
extensibility baseline every component is expected to satisfy (G1-G3):

1. ``from_config(instance.to_dict())`` round-trips without raising.
2. ``get_param_spec()`` returns a list of ``ParamSpec`` instances.
3. One forward pass succeeds with the kind's canonical tensor shape.

A few registered entries are intentionally excluded, each with a documented
reason (see ``_SKIP`` below): they are not standalone leaf components in the
sense this contract targets (a placeholder "no filter" entry, a DSL neuron
that needs `.compile()` before it behaves like a `BaseNeuron`, an optional
solver whose backend isn't installed in this environment).
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import pytest
import torch

from sensoryforge.register_components import register_all
from sensoryforge.registry import (
    NEURON_REGISTRY,
    FILTER_REGISTRY,
    INNERVATION_REGISTRY,
    STIMULUS_REGISTRY,
    SOLVER_REGISTRY,
    GRID_REGISTRY,
)
from sensoryforge.stimuli.base import ParamSpec
from sensoryforge.stimuli.builder import (
    StaticStimulus,
    MovingStimulus,
    CompositeStimulus,
    TimelineStimulus,
    RepeatedPatternStimulus,
)

register_all()

# Entries that are registered but excluded from this generic sweep, with why.
_SKIP: Dict[str, str] = {
    "none": "placeholder 'no filter' entry (type(None)), not a component",
    "identity": "placeholder 'no filter' alias (type(None)), not a component",
    "dsl": "DSL neuron needs equations/.compile() before behaving like a BaseNeuron; covered by test_model_dsl.py",
    "DSL (Custom)": "DSL neuron needs equations/.compile(); covered by test_model_dsl.py",
}


def _assert_param_spec(cls_or_obj: Any) -> None:
    spec = cls_or_obj.get_param_spec()
    assert isinstance(spec, list)
    assert all(isinstance(p, ParamSpec) for p in spec)


# ---------------------------------------------------------------------------
# Neurons: canonical forward shape [batch, steps, features] -> v/spikes
# [batch, steps+1, features]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(NEURON_REGISTRY.list_registered()))
def test_neuron_contract(name):
    if name in _SKIP:
        pytest.skip(_SKIP[name])
    cls = NEURON_REGISTRY.get_class(name)
    neuron = NEURON_REGISTRY.create(name)

    _assert_param_spec(cls)

    current = torch.randn(1, 5, 3)
    v_trace, spikes = neuron(current)
    assert v_trace.shape == (1, 6, 3)
    assert spikes.shape == (1, 6, 3)

    reconstructed = cls.from_config(neuron.to_dict())
    assert isinstance(reconstructed, cls)


# ---------------------------------------------------------------------------
# Filters: canonical forward shape [batch, time, neurons] -> same shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FILTER_REGISTRY.list_registered()))
def test_filter_contract(name):
    if name in _SKIP:
        pytest.skip(_SKIP[name])
    cls = FILTER_REGISTRY.get_class(name)
    filt = FILTER_REGISTRY.create(name)

    _assert_param_spec(cls)

    x = torch.randn(1, 5, 3)
    out = filt(x)
    assert out.shape == x.shape

    reconstructed = cls.from_config(filt.to_dict())
    assert isinstance(reconstructed, cls)


# ---------------------------------------------------------------------------
# Grid arrangements: canonical forward is get_all_coordinates() -> [N, 2]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GRID_REGISTRY.list_registered()))
def test_grid_contract(name):
    if name in _SKIP:
        pytest.skip(_SKIP[name])
    cls = GRID_REGISTRY.get_class(name)
    grid = GRID_REGISTRY.create(name, grid_size=4, spacing=0.5)

    _assert_param_spec(cls)

    coords = grid.get_all_coordinates()
    assert coords.ndim == 2
    assert coords.shape[1] == 2

    reconstructed = cls.from_config(grid.to_dict())
    assert isinstance(reconstructed, cls)
    assert reconstructed.get_all_coordinates().shape[1] == 2


# ---------------------------------------------------------------------------
# Solvers: canonical forward is one Euler-style step() -> same shape as state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SOLVER_REGISTRY.list_registered()))
def test_solver_contract(name):
    if name in _SKIP:
        pytest.skip(_SKIP[name])
    cls = SOLVER_REGISTRY.get_class(name)
    try:
        solver = SOLVER_REGISTRY.create(name)
    except ImportError as exc:
        pytest.skip(f"{name} backend not installed: {exc}")

    _assert_param_spec(cls)

    def decay(state, t):
        return -0.1 * state

    state = torch.randn(1, 3)
    new_state = solver.step(decay, state, t=0.0, dt=solver.dt)
    assert new_state.shape == state.shape

    reconstructed = cls.from_config(solver.to_dict())
    assert isinstance(reconstructed, cls)


# ---------------------------------------------------------------------------
# Innervation: canonical forward is compute_weights() -> [neurons, receptors]
#
# to_dict() deliberately excludes receptor_coords/neuron_centers (too large
# to serialise, see BaseInnervation.to_dict docstring), so the round-trip
# merges the same tensors back in rather than expecting to_dict() alone to
# be sufficient -- documented asymmetry, not a bug.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(INNERVATION_REGISTRY.list_registered()))
def test_innervation_contract(name):
    if name in _SKIP:
        pytest.skip(_SKIP[name])
    cls = INNERVATION_REGISTRY.get_class(name)
    receptor_coords = torch.rand(20, 2)
    neuron_centers = torch.rand(4, 2)
    innervation = INNERVATION_REGISTRY.create(
        name,
        receptor_coords=receptor_coords,
        neuron_centers=neuron_centers,
        device="cpu",
    )

    _assert_param_spec(cls)

    weights = innervation.compute_weights()
    assert weights.shape == (4, 20)

    # to_dict() deliberately drops receptor_coords/neuron_centers (too large
    # to serialise) and adds derived, non-constructor fields (method,
    # num_neurons, num_receptors) -- strip those, merge the tensors back in.
    serialised = innervation.to_dict()
    for derived_key in ("method", "num_neurons", "num_receptors"):
        serialised.pop(derived_key, None)
    reconstructed = cls.from_config(
        {
            **serialised,
            "receptor_coords": receptor_coords,
            "neuron_centers": neuron_centers,
        }
    )
    assert isinstance(reconstructed, cls)


# ---------------------------------------------------------------------------
# Stimuli: canonical forward shape [H, W] -> [H, W]
#
# Composite/wrapper stimuli (moving/composite/timeline/repeated_pattern) need
# a constituent stimulus to be constructed at all; canonical builders supply
# one minimal StaticStimulus so the same round-trip/forward checks apply.
# ---------------------------------------------------------------------------


def _static_gaussian() -> StaticStimulus:
    return StaticStimulus(
        stim_type="gaussian",
        params={"amplitude": 1.0, "sigma": 0.3, "center_x": 0.0, "center_y": 0.0},
    )


_STIMULUS_BUILDERS: Dict[str, Callable[[], Any]] = {
    "static": _static_gaussian,
    "moving": lambda: MovingStimulus(
        base_stimulus=_static_gaussian(),
        motion_type="stationary",
        motion_params={"center": (0.0, 0.0), "num_steps": 1},
    ),
    "composite": lambda: CompositeStimulus(stimuli=[_static_gaussian()]),
    "timeline": lambda: TimelineStimulus(
        sub_stimuli=[
            {"stimulus": _static_gaussian(), "onset_ms": 0.0, "duration_ms": 10.0}
        ],
        total_time_ms=10.0,
        dt_ms=1.0,
    ),
    "repeated_pattern": lambda: RepeatedPatternStimulus(
        base_stimulus=_static_gaussian(), copies_x=2, copies_y=2
    ),
}


@pytest.mark.parametrize("name", sorted(STIMULUS_REGISTRY.list_registered()))
def test_stimulus_contract(name):
    if name in _SKIP:
        pytest.skip(_SKIP[name])
    cls = STIMULUS_REGISTRY.get_class(name)

    if name in _STIMULUS_BUILDERS:
        stim = _STIMULUS_BUILDERS[name]()
    else:
        stim = STIMULUS_REGISTRY.create(name)

    _assert_param_spec(cls)

    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 8), indexing="ij"
    )
    out = stim(xx, yy)
    assert out.shape == xx.shape

    reconstructed = cls.from_config(stim.to_dict())
    assert isinstance(reconstructed, cls)
    out2 = reconstructed(xx, yy)
    assert out2.shape == xx.shape
