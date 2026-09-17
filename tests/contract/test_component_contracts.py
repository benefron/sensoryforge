"""Contract tests over every registered component (G4).

For each name registered in NEURON_REGISTRY, FILTER_REGISTRY, GRID_REGISTRY,
SOLVER_REGISTRY, INNERVATION_REGISTRY, STIMULUS_REGISTRY and
PROCESSING_REGISTRY, this checks the
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
    PROCESSING_REGISTRY,
)
from sensoryforge.stimuli.builder import (
    StaticStimulus,
    MovingStimulus,
    CompositeStimulus,
    TimelineStimulus,
    RepeatedPatternStimulus,
)
from sensoryforge.testing.contracts import check_component

register_all()

# Entries that are registered but excluded from this generic sweep, with why.
_SKIP: Dict[str, str] = {
    "none": "placeholder 'no filter' entry (type(None)), not a component",
    "identity": "placeholder 'no filter' alias (type(None)), not a component",
    "dsl": "DSL neuron needs equations/.compile() before behaving like a BaseNeuron; covered by test_model_dsl.py",
    "DSL (Custom)": "DSL neuron needs equations/.compile(); covered by test_model_dsl.py",
}


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
    check_component("neuron", cls, instance=neuron)


# ---------------------------------------------------------------------------
# Filters: canonical forward shape [batch, time, neurons] -> same shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FILTER_REGISTRY.list_registered()))
def test_filter_contract(name):
    if name in _SKIP:
        pytest.skip(_SKIP[name])
    cls = FILTER_REGISTRY.get_class(name)
    filt = FILTER_REGISTRY.create(name)
    check_component("filter", cls, instance=filt)


# ---------------------------------------------------------------------------
# Grid arrangements: canonical forward is get_all_coordinates() -> [N, 2]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GRID_REGISTRY.list_registered()))
def test_grid_contract(name):
    if name in _SKIP:
        pytest.skip(_SKIP[name])
    cls = GRID_REGISTRY.get_class(name)
    grid = GRID_REGISTRY.create(name, grid_size=4, spacing=0.5)
    check_component("grid", cls, instance=grid)


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
    check_component("solver", cls, instance=solver)


# ---------------------------------------------------------------------------
# Innervation: canonical forward is compute_weights() -> [neurons, receptors]
#
# to_dict() deliberately excludes receptor_coords/neuron_centers (too large
# to serialise, see BaseInnervation.to_dict docstring), so the round-trip
# merges the same tensors back in rather than expecting to_dict() alone to
# be sufficient -- documented asymmetry, not a bug (see
# sensoryforge.testing.contracts._check_innervation).
# ---------------------------------------------------------------------------


# Builders whose constructor needs more than the two coordinate tensors
# (Phase 2, I4/I5): each takes receptor_coords and returns the extra kwargs.
def _template_kwargs(receptor_coords, tmp_path):
    # template derives its own lattice; it requires one design parameter form.
    return {"resolvable_distance_mm": 0.4}


def _imported_kwargs(receptor_coords, tmp_path):
    # imported reads weights from a file; give it a saved bank on this grid.
    from sensoryforge.core.rf_bank import ReceptiveFieldBank

    path = tmp_path / "bank.pt"
    ReceptiveFieldBank(
        torch.rand(4, receptor_coords.shape[0]), torch.rand(4, 2), receptor_coords
    ).save(path)
    return {"path": str(path)}


_INNERVATION_KWARGS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "template": _template_kwargs,
    "imported": _imported_kwargs,
}
_DERIVES_OWN_CENTRES = {"template", "imported"}


@pytest.mark.parametrize("name", sorted(INNERVATION_REGISTRY.list_registered()))
def test_innervation_contract(name, tmp_path):
    if name in _SKIP:
        pytest.skip(_SKIP[name])
    cls = INNERVATION_REGISTRY.get_class(name)
    receptor_coords = torch.rand(20, 2)
    neuron_centers = torch.rand(4, 2)
    kwargs = (
        _INNERVATION_KWARGS[name](receptor_coords, tmp_path)
        if name in _INNERVATION_KWARGS
        else {}
    )
    if name in _DERIVES_OWN_CENTRES:
        # Passing centres only warns (tested in tests/unit/test_rf_*_builder.py).
        innervation = INNERVATION_REGISTRY.create(
            name, receptor_coords=receptor_coords, device="cpu", **kwargs
        )
    else:
        innervation = INNERVATION_REGISTRY.create(
            name,
            receptor_coords=receptor_coords,
            neuron_centers=neuron_centers,
            device="cpu",
            **kwargs,
        )
    check_component("innervation", cls, instance=innervation)


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

    check_component("stimulus", cls, instance=stim)


# ---------------------------------------------------------------------------
# Processing layers (Wave M3): forward(receptor_responses) -> receptor axis
# unchanged or grown; to_dict()/from_config() round trip the pipeline's
# {"method": ..., "params": {...}} config shape (F-058).
# ---------------------------------------------------------------------------

# A layer whose __init__ needs something check_component's cls() fallback
# can't supply (e.g. OnOffLayer's positional receptor_coords) needs its own
# builder here, the same way _STIMULUS_BUILDERS covers composite stimuli.
_PROCESSING_BUILDERS: Dict[str, Callable[[], Any]] = {
    "onoff": lambda: PROCESSING_REGISTRY.get_class("onoff")(
        receptor_coords=torch.rand(8, 2)
    ),
}


@pytest.mark.parametrize("name", sorted(PROCESSING_REGISTRY.list_registered()))
def test_processing_contract(name):
    # Deliberately does not consult _SKIP: those entries are keyed only by
    # name and were written for the filter/neuron placeholder aliases (a
    # registered "identity" filter is a `type(None)` placeholder; the
    # registered "identity" *processing* layer, IdentityLayer, is a real,
    # fully-behaved component and must be checked).
    cls = PROCESSING_REGISTRY.get_class(name)

    if name in _PROCESSING_BUILDERS:
        layer = _PROCESSING_BUILDERS[name]()
    else:
        layer = PROCESSING_REGISTRY.create(name)

    check_component("processing", cls, instance=layer)
