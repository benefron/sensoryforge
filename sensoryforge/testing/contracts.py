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

import inspect
from typing import Any, Callable, Dict, List, Optional

import torch

from sensoryforge.stimuli.base import ParamSpec

__all__ = ["check_component", "available_kinds"]


def _init_param_names(cls: type) -> List[str]:
    """Return ``cls.__init__``'s parameter names, excluding ``self`` and
    ``*args``/``**kwargs`` (F-045)."""
    sig = inspect.signature(cls.__init__)
    names = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        names.append(name)
    return names


def _assert_to_dict_roundtrip_complete(
    cls: type,
    instance: Any,
    extra_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Assert ``to_dict()``/``from_config()`` round-trip every constructor
    parameter (F-045).

    Checks:
    1. Every ``__init__`` parameter (except ``self``/``*args``/``**kwargs``,
       and any name listed in the class's ``_TO_DICT_EXCLUDE_PARAMS``, e.g.
       Izhikevich's ``preset`` -- a resolved-value convenience the class
       deliberately does not round-trip, see ``IzhikevichNeuronTorch``)
       appears as a key in ``instance.to_dict()``.
    2. ``cls.from_config(instance.to_dict()).to_dict() == instance.to_dict()``
       -- the round trip is a fixed point.

    Args:
        extra_config: Extra keys merged into ``instance.to_dict()`` before
            calling ``from_config()`` in check 2 only -- for a constructor
            parameter that is both required (no default) and listed in
            ``_TO_DICT_EXCLUDE_PARAMS`` (e.g. innervation's
            ``receptor_coords``/``neuron_centers`` tensors, which a bundle
            stores separately rather than through this dict), ``from_config``
            cannot proceed without it. These keys are never compared -- both
            ``d1`` and ``d2`` come from ``to_dict()``, which omits them.
    """
    excluded = set(getattr(cls, "_TO_DICT_EXCLUDE_PARAMS", ()))
    d1 = instance.to_dict()
    missing = [
        name
        for name in _init_param_names(cls)
        if name not in d1 and name not in excluded
    ]
    if missing:
        raise AssertionError(
            f"{cls.__name__}.to_dict() is missing constructor parameters: "
            f"{missing} -- every __init__ parameter must round-trip through "
            f"to_dict()/from_config() (F-045)"
        )
    reconstructed = cls.from_config({**d1, **(extra_config or {})})
    d2 = reconstructed.to_dict()
    if d1 != d2:
        raise AssertionError(
            f"{cls.__name__}.from_config(instance.to_dict()).to_dict() != "
            f"instance.to_dict() -- round trip is not a fixed point "
            f"(F-045). First: {d1!r} Second: {d2!r}"
        )


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
    """Canonical neuron shape: [batch, steps, features] -> v/spikes
    [batch, steps+1, features].

    ``spikes`` may be ``None`` (Phase 2, N2) for an analog (non-spiking)
    readout -- e.g. a DSL-compiled model with no threshold (N1); only
    ``v_trace``'s shape is then checked.
    """
    _assert_param_spec(cls)
    current = torch.randn(1, 5, 3)
    v_trace, spikes = instance(current)
    if tuple(v_trace.shape) != (1, 6, 3):
        raise AssertionError(
            f"{cls.__name__} forward(): expected v_trace shape (1, 6, 3), "
            f"got {tuple(v_trace.shape)}"
        )
    if spikes is not None and tuple(spikes.shape) != (1, 6, 3):
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
    _assert_to_dict_roundtrip_complete(cls, instance)


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
    _assert_to_dict_roundtrip_complete(cls, instance)


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
    _assert_to_dict_roundtrip_complete(cls, instance)


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
    _assert_to_dict_roundtrip_complete(cls, instance)


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
    _assert_to_dict_roundtrip_complete(
        cls,
        instance,
        extra_config={
            "receptor_coords": instance.receptor_coords,
            "neuron_centers": instance.neuron_centers,
        },
    )
    # Phase 2 (I3): every builder produces a ReceptiveFieldBank via build().
    from sensoryforge.core.rf_bank import ReceptiveFieldBank

    bank = instance.build()
    if not isinstance(bank, ReceptiveFieldBank):
        raise AssertionError(
            f"{cls.__name__}.build() must return a ReceptiveFieldBank, got "
            f"{type(bank)!r}"
        )
    if tuple(bank.weights.shape) != (bank.num_neurons, instance.num_receptors):
        raise AssertionError(
            f"{cls.__name__}.build(): bank weights must be "
            f"[N={bank.num_neurons}, M={instance.num_receptors}], got "
            f"{tuple(bank.weights.shape)}"
        )
    if tuple(bank.neuron_centers.shape) != (bank.num_neurons, 2):
        raise AssertionError(
            f"{cls.__name__}.build(): bank neuron_centers must be [N, 2], got "
            f"{tuple(bank.neuron_centers.shape)}"
        )
    if tuple(bank.receptor_coords.shape) != (instance.num_receptors, 2):
        raise AssertionError(
            f"{cls.__name__}.build(): bank receptor_coords must be [M, 2], got "
            f"{tuple(bank.receptor_coords.shape)}"
        )
    for key in ("builder", "builder_config", "sensoryforge_version"):
        if key not in bank.provenance:
            raise AssertionError(
                f"{cls.__name__}.build(): bank provenance is missing {key!r} "
                f"(has {sorted(bank.provenance)})"
            )
    if not isinstance(bank.provenance["builder_config"], dict):
        raise AssertionError(
            f"{cls.__name__}.build(): provenance['builder_config'] must be a "
            f"dict, got {type(bank.provenance['builder_config'])!r}"
        )


def _assert_stimulus_forward_shape(cls: type, out: Any, xx_shape: tuple) -> None:
    """Assert a stimulus ``forward()`` result is ``[H, W]`` or ``[T, H, W]``.

    Both are valid stimulus outputs (Phase 2, Wave K, Fact K-c): a
    single-frame stimulus returns the spatial shape alone, ``xx.shape``;
    a stimulus with its own internal motion or dynamics (``MovingStimulus``,
    the ported pressure-simulation stimuli in
    ``sensoryforge/stimuli/tactile.py``) returns a whole sequence with a
    leading time axis, ``[T, *xx.shape]``. Anything else is rejected.
    """
    out_shape = tuple(out.shape)
    if out_shape == xx_shape:
        return
    if len(out_shape) == len(xx_shape) + 1 and out_shape[1:] == xx_shape:
        return
    raise AssertionError(
        f"{cls.__name__} forward(): expected shape {xx_shape} or "
        f"[T, {', '.join(str(d) for d in xx_shape)}], got {out_shape}"
    )


def _check_stimulus(cls: type, instance: Any) -> None:
    """Canonical stimulus forward shape: ``[H, W] -> [H, W]`` or
    ``[H, W] -> [T, H, W]`` (Fact K-c; see
    :func:`_assert_stimulus_forward_shape`)."""
    _assert_param_spec(cls)
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 8), indexing="ij"
    )
    out = instance(xx, yy)
    _assert_stimulus_forward_shape(cls, out, tuple(xx.shape))
    reconstructed = cls.from_config(instance.to_dict())
    if not isinstance(reconstructed, cls):
        raise AssertionError(
            f"{cls.__name__}.from_config(instance.to_dict()) did not return "
            f"a {cls.__name__} instance (got {type(reconstructed)!r})"
        )
    out2 = reconstructed(xx, yy)
    _assert_stimulus_forward_shape(cls, out2, tuple(xx.shape))
    _assert_to_dict_roundtrip_complete(cls, instance)


def _infer_receptor_count(instance: Any) -> int:
    """Guess the receptor count a processing-layer instance was built for.

    Layers with no receptor-count dependence (``IdentityLayer``) don't care
    what ``M`` is, so any value works. Layers that precompute a receptor x
    receptor kernel (``OnOffLayer``'s ``dog_kernel``, and by the same shape
    convention any future centre-surround-style layer) fix ``M`` at
    construction time via a square buffer; find it generically instead of
    hard-coding ``OnOffLayer``'s attribute name, so a plugin layer with its
    own square kernel buffer is picked up the same way.
    """
    for _, buf in instance.named_buffers():
        if buf.ndim == 2 and buf.shape[0] == buf.shape[1] and buf.shape[0] > 0:
            return int(buf.shape[0])
    return 8


def _check_processing(cls: type, instance: Any) -> None:
    """Canonical processing-layer contract (Wave M3 shape, F-058).

    A processing layer's ``forward``/``to_dict``/``from_config`` shapes are
    deliberately not the same as the other five kinds (F-058): ``forward``
    takes ``(receptor_responses, metadata=None)`` rather than one positional
    tensor, ``to_dict()`` returns the ``{"method": ..., "params": {...}}``
    shape :class:`~sensoryforge.core.processing.ProcessingPipeline` reads
    rather than a flat kwargs dict, and a layer whose
    ``REQUIRES_RECEPTOR_COORDS`` is set needs ``receptor_coords`` passed to
    ``from_config`` explicitly (it cannot be recovered from ``to_dict()``,
    which never serialises the receptor axis). So this check is written
    against that actual contract instead of reusing
    :func:`_assert_to_dict_roundtrip_complete`, which assumes the flat shape.

    Checks:
    1. ``cls.get_param_spec()`` returns a list of ``ParamSpec``.
    2. ``forward(receptor_responses)`` on a ``[batch, time, M]`` input
       returns a tensor with the same leading (batch, time) shape and a
       receptor axis ``>= M`` (a layer may only grow it, e.g. ON/OFF
       splitting -- never silently shrink it).
    3. ``to_dict()`` includes a ``"method"`` or ``"type"`` key (either is
       ``ProcessingPipeline.from_config()``'s dispatch key into
       ``PROCESSING_REGISTRY``).
    4. ``cls.from_config(instance.to_dict(), receptor_coords=...)`` (the
       coordinates passed unconditionally; only a class with
       ``REQUIRES_RECEPTOR_COORDS`` uses them) round-trips to the same
       ``to_dict()``.

    Args:
        cls: The processing-layer class under test; must subclass
            ``BaseProcessingLayer``.
        instance: A constructed instance. Required (not optional) for any
            layer whose ``__init__`` needs ``receptor_coords`` positionally
            (e.g. ``OnOffLayer``) -- ``check_component`` falls back to
            ``cls()`` only when no instance is given, which fails for such
            layers, matching how the ``"innervation"`` kind already works.
    """
    _assert_param_spec(cls)
    m = _infer_receptor_count(instance)
    x = torch.randn(1, 5, m)
    out = instance(x)
    if tuple(out.shape[:-1]) != tuple(x.shape[:-1]):
        raise AssertionError(
            f"{cls.__name__} forward(): expected leading shape "
            f"{tuple(x.shape[:-1])}, got {tuple(out.shape[:-1])}"
        )
    if out.shape[-1] < m:
        raise AssertionError(
            f"{cls.__name__} forward(): receptor axis shrank from {m} to "
            f"{out.shape[-1]} -- a processing layer may only grow or "
            "preserve the receptor axis, per expand_receptor_coords()"
        )
    d1 = instance.to_dict()
    if "method" not in d1 and "type" not in d1:
        raise AssertionError(
            f"{cls.__name__}.to_dict() must include a 'method' or 'type' key "
            "-- ProcessingPipeline.from_config() dispatches on either "
            "(BaseProcessingLayer; 'type' is IdentityLayer's original key, "
            "'method' is PopulationInput.processing's, Wave M3)"
        )
    receptor_coords = torch.zeros(m, 2)
    if getattr(cls, "REQUIRES_RECEPTOR_COORDS", False):
        reconstructed = cls.from_config(d1, receptor_coords=receptor_coords)
    else:
        reconstructed = cls.from_config(d1)
    if not isinstance(reconstructed, cls):
        raise AssertionError(
            f"{cls.__name__}.from_config(instance.to_dict()) did not return "
            f"a {cls.__name__} instance (got {type(reconstructed)!r})"
        )
    d2 = reconstructed.to_dict()
    if d1 != d2:
        raise AssertionError(
            f"{cls.__name__}.from_config(instance.to_dict()).to_dict() != "
            f"instance.to_dict() -- round trip is not a fixed point "
            f"(F-058). First: {d1!r} Second: {d2!r}"
        )


_CHECKS: Dict[str, Callable[[type, Any], None]] = {
    "neuron": _check_neuron,
    "filter": _check_filter,
    "grid": _check_grid,
    "solver": _check_solver,
    "innervation": _check_innervation,
    "stimulus": _check_stimulus,
    "processing": _check_processing,
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
    4. For ``"innervation"``: ``instance.build()`` returns a
       :class:`~sensoryforge.core.rf_bank.ReceptiveFieldBank` with
       ``[N, M]`` weights, ``[N, 2]``/``[M, 2]`` coordinates and a provenance
       naming the builder (Phase 2, I3).

    Args:
        kind: One of ``"neuron"``, ``"filter"``, ``"grid"``, ``"solver"``,
            ``"innervation"``, ``"stimulus"``, ``"processing"``.
        cls: The component class under test. Must be (or subclass) the
            matching base class -- ``BaseNeuron``, ``BaseFilter``,
            ``BaseGrid``, ``BaseSolver``, ``BaseInnervation``,
            ``BaseStimulus``, or ``BaseProcessingLayer``.
        instance: An already-constructed instance to check against. Required
            for kinds without a parameter-free default constructor (e.g.
            ``innervation``, which always needs ``receptor_coords``/
            ``neuron_centers``; or a ``"processing"`` layer such as
            ``OnOffLayer`` that needs ``receptor_coords`` positionally) or
            when the default-constructed instance would not exercise the
            case under test (e.g. a composite stimulus needing a
            sub-stimulus). If omitted, ``cls()`` is used.

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
