"""Contract checks for the world registries (C4, F-049).

:func:`sensoryforge.testing.contracts.check_component` checks a component
*class*. The world registries in :mod:`sensoryforge.world` hold functions and
tables instead -- a shape is ``fn(x, y, params)`` plus ``ParamSpec`` rows, a
distribution is a sampler -- so they are checked here, by name, after the
plugin has registered them::

    from sensoryforge.testing.world_contracts import check_world_component

    check_world_component("shape", "ring")
    check_world_component("distribution", "my_dist", example={"scale": 2.0})

A check calls the registered function the way the renderer calls it (batch
first, numbers as tensors that broadcast against the coordinates, strings and
bools plain, ``amplitude`` taken out first) and raises ``AssertionError`` naming
the registry entry and the rule it breaks. Nothing here checks what happens when a
name is registered twice.

Checks, by kind (the same shapes the docs promise in
``docs/reference/world_contract.md``):

* every kind: its ``ParamSpec`` rows are well formed (unique names, a default of the
  declared type inside ``min_val``/``max_val`` and ``choices``, JSON-plain) and
  round-trip, both as rows and as the part a world file normalises to;
* ``shape``: ``amplitude`` is a spec; ``fn`` returns a finite tensor of the
  coordinates' shape and dtype, at the defaults and at the edges of every declared
  range; draw *i* alone equals draw *i* in a batch, bit for bit (guarantee 1); an
  indenter has no spec named ``depth`` and returns exactly 0 at depth 0; the
  defaults satisfy the shape's own ``check``;
* ``pattern``: ``x_mm`` and ``y_mm`` come together; ``fn`` returns equal-length
  lists of finite ``(x, y)`` positions and scales, the same on every call (the
  kernel caches by parameters); ``pattern_batch`` pads a group of them;
* ``modulation``: ``fn`` (``None`` means no modulation) returns values in
  ``[0, 1]`` of ``tc``'s shape, and draw *i* alone equals draw *i* in a batch;
* ``distribution``: a sample is one JSON-plain value per ``u`` in ``[0, 1)``, the
  same every call, each depending on its own ``u`` only; ``support`` holds every
  value drawn and ``bounds`` bracket every number drawn; the axis re-parses from its
  own ``to_dict``;
* ``class_kind``: the required methods are overridden, and on an example world the
  kind's fields resolve, its draws end and sit on a timeline, and its frames are
  finite, exactly 0 before 0 and from ``end_ms`` on, and the same for a draw alone.

Topic: testing, world
Tests: tests/unit/test_world_contracts.py,
    tests/contract/test_world_component_contracts.py
Entry points: check_world_component, available_world_kinds
Pinned by: docs/developer_guide/extensibility.md, docs/developer_guide/plugins.md,
    docs/reference/world_contract.md, .claude/rules/world-engine.md
Do not change without: a built-in shape, pattern, modulation, distribution or class
    kind that cannot pass is recorded as an open ledger item, never by weakening the
    check; no check here may depend on how a repeated registration behaves (F-5bd542f).
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sensoryforge.stimuli import layered
from sensoryforge.stimuli.base import ParamSpec
from sensoryforge.world import distributions, kernel, kinds
from sensoryforge.world.render import Canvas, movie_times, render
from sensoryforge.world.sampling import sample
from sensoryforge.world.schema import World

__all__ = ["check_world_component", "available_world_kinds"]

#: The ``dtype`` values a ``ParamSpec`` may declare. ``str`` is what the shipped
#: shapes use for a choice such as ``profile``.
_SPEC_DTYPES = ("float", "int", "bool", "str")
#: Tolerance on the ``[0, 1]`` range of a modulation (float32 rounding).
_RANGE_SLACK = 1.0e-6
#: The canvas of a shape check: ``[5, 7]`` points in mm, two draws, two times.
_AXES_MM = (
    torch.linspace(-1.5, 1.5, 5, dtype=torch.float64),
    torch.linspace(-2.0, 2.0, 7, dtype=torch.float64),
)
_OFFSETS_MM = torch.tensor([[0.1, -0.2], [0.0, 0.3]], dtype=torch.float64)
#: ``tc`` of a modulation check: ms since touch, two draws, four times.
_TC_MS = [[0.0, 5.0, 20.0, 80.0], [0.0, 1.0, 50.0, 200.0]]


def _fail(message: str) -> None:
    raise AssertionError(message)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _plain_json(value: Any, where: str) -> Any:
    """``value`` after a JSON round trip (tuples become lists); else a failure."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        _fail(f"{where}: {value!r} is not finite JSON-plain data ({exc})")


#: What a registered function can raise on input it was documented to accept.
_CALL_ERRORS = (
    ArithmeticError,
    AttributeError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _call(
    label: str, what: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """``fn(*args, **kwargs)``, or a contract failure naming ``what`` if it raises."""
    try:
        return fn(*args, **kwargs)
    except _CALL_ERRORS as exc:
        _fail(f"{label}: {what} raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- the specs


def _check_default(label: str, spec: ParamSpec) -> None:
    where = f"{label} spec {spec.name!r}"
    default, dtype = spec.default, spec.dtype
    if dtype == "bool":
        ok = isinstance(default, bool)
    elif dtype == "int":
        ok = isinstance(default, int) and not isinstance(default, bool)
    elif dtype == "str":
        ok = isinstance(default, str)
    else:
        # A float parameter may be a point or a list of points (pattern ``list``).
        ok = _number(default) or isinstance(default, (list, tuple))
    if not ok:
        _fail(f"{where}: default {default!r} is not a {dtype}")
    _plain_json(default, where)
    if _number(default):
        if spec.min_val is not None and default < spec.min_val:
            _fail(f"{where}: default {default!r} is below min_val {spec.min_val!r}")
        if spec.max_val is not None and default > spec.max_val:
            _fail(f"{where}: default {default!r} is above max_val {spec.max_val!r}")
    if spec.choices is not None and default not in spec.choices:
        _fail(f"{where}: default {default!r} is not one of choices {spec.choices!r}")


def _check_specs(label: str, specs: Any) -> List[ParamSpec]:
    """Check ``specs`` are well-formed ``ParamSpec`` rows; return them."""
    if not isinstance(specs, list) or not all(isinstance(s, ParamSpec) for s in specs):
        _fail(f"{label}: specs must be a list of ParamSpec, got {specs!r}")
    names = [s.name for s in specs]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        _fail(f"{label}: spec names {repeated} are repeated")
    for spec in specs:
        where = f"{label} spec {spec.name!r}"
        if not isinstance(spec.name, str) or not spec.name:
            _fail(f"{label}: a spec has no name")
        if spec.name == "kind":
            _fail(f"{where}: 'kind' is the key that names the part, not a field")
        if spec.dtype not in _SPEC_DTYPES:
            _fail(f"{where}: dtype {spec.dtype!r} is not one of {_SPEC_DTYPES}")
        if (
            _number(spec.min_val)
            and _number(spec.max_val)
            and spec.min_val > spec.max_val
        ):
            _fail(f"{where}: min_val {spec.min_val!r} exceeds max_val {spec.max_val!r}")
        _check_default(label, spec)
        rebuilt = ParamSpec(**spec.to_dict())
        if rebuilt.to_dict() != spec.to_dict():
            _fail(f"{where}: ParamSpec(**spec.to_dict()) does not rebuild the spec")
    return specs


def _normalised_part(part: str, source: Dict[str, Any], label: str) -> Dict[str, Any]:
    """The part a world file normalises ``source`` to (defaults filled in)."""
    try:
        layer = kinds.LayeredKind().normalise_layer({part: source}, "contract")
    except _CALL_ERRORS as exc:
        _fail(f"{label}: a world refuses {source!r}: {exc}")
    return layer[part]


def _check_part_round_trip(part: str, name: str, specs: List[ParamSpec]) -> None:
    """The specs round-trip through a world file.

    ``{kind: name}`` normalises to the kind plus every spec's default (but for the
    fields in ``layered.ADDED_FIELDS``, invisible until set); that part, written
    as JSON and read back, normalises to itself.
    """
    label = f"{part} {name!r}"
    added = layered.ADDED_FIELDS.get((part, name), ())
    first = _plain_json(_normalised_part(part, {"kind": name}, label), label)
    want = _plain_json(
        {"kind": name, **{s.name: s.default for s in specs if s.name not in added}},
        label,
    )
    if first != want:
        _fail(
            f"{label}: {{kind: {name}}} normalises to {first!r}, not the kind plus "
            f"its specs' defaults {want!r}"
        )
    second = _plain_json(_normalised_part(part, first, label), label)
    if second != first:
        _fail(
            f"{label}: normalising twice is not a fixed point: {first!r} then "
            f"{second!r}"
        )


# --------------------------------------------------------------- the values


def _draw_defaults(specs: Sequence[ParamSpec]) -> Dict[str, Any]:
    return {s.name: s.default for s in specs}


def _variant(specs: Sequence[ParamSpec], base: Dict[str, Any]) -> Dict[str, Any]:
    """A second draw: every number moved to another value inside its range.

    Up by half when the range allows it, else down by half, else to the middle of
    the range; an integer by half its size, at least one.
    """
    out = dict(base)
    for spec in specs:
        value = base[spec.name]
        if spec.dtype not in ("float", "int") or not _number(value):
            continue
        lo = -math.inf if spec.min_val is None else spec.min_val
        hi = math.inf if spec.max_val is None else spec.max_val
        if spec.dtype == "int":
            step = max(1, round(abs(value) * 0.5))
            candidates = [value + step, value - step]
        else:
            candidates = [value * 1.5 if value != 0 else 0.25, value * 0.5]
            if math.isfinite(lo + hi):
                candidates.append(0.5 * (lo + hi))
        for candidate in candidates:
            clipped = min(max(candidate, lo), hi)
            if clipped != value:
                out[spec.name] = int(clipped) if spec.dtype == "int" else clipped
                break
    return out


def _tensor_params(
    draws: Sequence[Dict[str, Any]], dtype: torch.dtype, view: Sequence[int]
) -> Dict[str, Any]:
    """Per-draw dicts -> one: numbers become tensors of ``view``, the rest pass.

    The renderer's grouping: the non-numeric values of a group are equal, so the
    first draw's are used.
    """
    out: Dict[str, Any] = {}
    for key in draws[0]:
        values = [d[key] for d in draws]
        if all(_number(v) for v in values):
            out[key] = torch.tensor([float(v) for v in values], dtype=dtype).view(*view)
        else:
            out[key] = values[0]
    return out


def _draw_edges(specs: Sequence[ParamSpec]) -> List[Tuple[ParamSpec, Any]]:
    """``(spec, value)`` for each declared end of each numeric range."""
    edges = []
    for spec in specs:
        if spec.dtype not in ("float", "int") or not _number(spec.default):
            continue
        for end in (spec.min_val, spec.max_val):
            if _number(end) and math.isfinite(end):
                edges.append((spec, int(end) if spec.dtype == "int" else float(end)))
    return edges


# ------------------------------------------------------------------- shapes


def _shape_accepts(shape: kernel.ShapeKind, draw: Dict[str, Any]) -> bool:
    """False when the shape's own load-time check refuses this draw."""
    if shape.check is None:
        return True
    fields = {k: distributions.FieldValues.exactly(v) for k, v in draw.items()}
    try:
        shape.check(fields, lambda field: f"shape.{field}")
    except ValueError:
        return False
    except _CALL_ERRORS as exc:
        _fail(
            f"shape check(values, path) raised {type(exc).__name__}: {exc} on "
            f"{draw!r}; it must raise ValueError for a draw it refuses"
        )
    return True


def _coordinates(
    g: int, k: int, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``x``, ``y`` of the renderer's shape ``[g, k, *S]``: mm from each centre."""
    xx, yy = torch.meshgrid(*_AXES_MM, indexing="ij")
    off_x = _OFFSETS_MM[:g, :k, None, None]
    off_y = _OFFSETS_MM.flip(0)[:g, :k, None, None]
    return (xx - off_x).to(dtype), (yy - off_y).to(dtype)


def _call_shape(
    name: str,
    shape: kernel.ShapeKind,
    draws: Sequence[Dict[str, Any]],
    dtype: torch.dtype,
    depth: Optional[float],
    first: int = 0,
) -> torch.Tensor:
    """Evaluate ``fn`` on a group of draws as the renderer does (amplitude out).

    The group is draws ``first`` onward of the two the check builds, each on its
    own coordinates, so a draw alone sees what it sees in the batch.
    """
    g, k = len(draws), 2
    x, y = (c[first : first + g] for c in _coordinates(2, k, dtype))
    params = _tensor_params(
        [{key: v for key, v in d.items() if key != "amplitude"} for d in draws],
        dtype,
        (g, 1, 1, 1),
    )
    if shape.indenter:
        params["depth"] = torch.full((g, k, 1, 1), float(depth), dtype=dtype)
    try:
        out = shape.fn(x, y, params)
    except _CALL_ERRORS as exc:
        _fail(
            f"shape {name!r}: fn(x, y, params) failed on {sorted(params)} with "
            f"tensors of shape {tuple(x.shape)} ({type(exc).__name__}: {exc})"
        )
    if not isinstance(out, torch.Tensor):
        _fail(f"shape {name!r}: fn must return a tensor, got {type(out)!r}")
    if tuple(out.shape) != tuple(x.shape):
        _fail(
            f"shape {name!r}: fn must return the coordinates' shape "
            f"{tuple(x.shape)}, got {tuple(out.shape)}"
        )
    if out.dtype != dtype:
        _fail(f"shape {name!r}: fn returned {out.dtype} for {dtype} coordinates")
    return out


def _check_finite(label: str, out: torch.Tensor, why: str) -> None:
    if not bool(torch.isfinite(out).all()):
        _fail(f"{label}: fn returned a value that is not finite {why}")


def _check_shape(name: str, example: Any) -> None:
    if name not in kernel.SHAPE_KINDS:
        raise ValueError(f"no shape {name!r}; registered: {sorted(kernel.SHAPE_KINDS)}")
    shape = kernel.SHAPE_KINDS[name]
    label = f"shape {name!r}"
    if not callable(shape.fn):
        _fail(f"{label}: fn is not callable")
    specs = _check_specs(label, shape.specs)
    if not any(s.name == "amplitude" for s in specs):
        _fail(f"{label}: its specs must include 'amplitude'")
    if shape.indenter and any(s.name == "depth" for s in specs):
        _fail(
            f"{label}: an indenter is passed params['depth'] (amplitude x envelope "
            "x modulation), so it cannot also have a spec named 'depth'"
        )
    _check_part_round_trip("shape", name, specs)

    base = _draw_defaults(specs)
    if not _shape_accepts(shape, base):
        _fail(f"{label}: its defaults {base!r} fail the shape's own check")
    second = _variant(specs, base)
    if not _shape_accepts(shape, second):
        second = base
    for dtype in (torch.float32, torch.float64):
        depth = 1.0 if shape.indenter else None
        batch = _call_shape(name, shape, [base, second], dtype, depth)
        _check_finite(label, batch, "at the defaults")
        for i, draw in enumerate((base, second)):
            alone = _call_shape(name, shape, [draw], dtype, depth, first=i)
            if not torch.equal(batch[i : i + 1], alone):
                _fail(
                    f"{label}: draw {i} alone differs from draw {i} in a batch of "
                    f"two ({dtype}) -- a loop or reduction over the batch must run "
                    "to the batch's maximum and mask each draw's surplus terms to "
                    "exact zeros (world contract guarantee 1)"
                )
        for spec, value in _draw_edges(specs):
            edge = {**base, spec.name: value}
            if spec.name == "amplitude" or not _shape_accepts(shape, edge):
                continue
            out = _call_shape(name, shape, [edge], dtype, depth)
            _check_finite(label, out, f"with {spec.name}={value!r} (a declared end)")
        if shape.indenter:
            flat = _call_shape(name, shape, [base, second], dtype, 0.0)
            if int(torch.count_nonzero(flat)) != 0:
                _fail(f"{label}: an indenter must return exactly 0 at depth 0")


# ----------------------------------------------------------------- patterns


def _check_pattern(name: str, example: Any) -> None:
    if name not in kernel.PATTERN_KINDS:
        raise ValueError(
            f"no pattern {name!r}; registered: {sorted(kernel.PATTERN_KINDS)}"
        )
    pattern = kernel.PATTERN_KINDS[name]
    label = f"pattern {name!r}"
    if not callable(pattern.fn):
        _fail(f"{label}: fn is not callable")
    specs = _check_specs(label, pattern.specs)
    names = {s.name for s in specs}
    if ("x_mm" in names) != ("y_mm" in names):
        _fail(f"{label}: x_mm and y_mm place a pattern together; it has only one")
    _check_part_round_trip("pattern", name, specs)

    full = _normalised_part("pattern", {"kind": name}, label)
    params = {k: v for k, v in full.items() if k not in ("kind", "x_mm", "y_mm")}
    results = []
    for _ in range(2):
        try:
            results.append(pattern.fn(dict(params)))
        except _CALL_ERRORS as exc:
            _fail(
                f"{label}: fn(params) failed on {sorted(params)} "
                f"({type(exc).__name__}: {exc})"
            )
    first, again = results
    if (
        not isinstance(first, tuple)
        or len(first) != 2
        or not all(isinstance(part, list) for part in first)
    ):
        _fail(
            f"{label}: fn must return (positions, scales) as two lists, got {first!r}"
        )
    positions, scales = first
    if len(positions) != len(scales):
        _fail(
            f"{label}: {len(positions)} positions but {len(scales)} scales -- one "
            "scale per element"
        )
    for point in positions:
        if len(point) != 2 or not all(_number(c) and math.isfinite(c) for c in point):
            _fail(f"{label}: a position must be a finite (x, y) in mm, got {point!r}")
    if not all(_number(s) and math.isfinite(s) for s in scales):
        _fail(f"{label}: every scale must be a finite number, got {scales!r}")
    if _plain_json(first, label) != _plain_json(again, label):
        _fail(
            f"{label}: fn(params) differs between two calls -- the kernel caches "
            "positions by parameters, so it must be a pure function of them"
        )
    batch = kernel.pattern_batch(name, [full, full], torch.float64, "cpu")
    rows = max(1, len(positions))
    if tuple(batch[0].shape) != (2, rows, 2) or tuple(batch[1].shape) != (2, rows):
        _fail(
            f"{label}: pattern_batch gave {tuple(batch[0].shape)} and "
            f"{tuple(batch[1].shape)}, expected (2, {rows}, 2) and (2, {rows})"
        )


# -------------------------------------------------------------- modulations


def _check_modulation(name: str, example: Any) -> None:
    if name not in kernel.MODULATION_KINDS:
        raise ValueError(
            f"no modulation {name!r}; registered: {sorted(kernel.MODULATION_KINDS)}"
        )
    modulation = kernel.MODULATION_KINDS[name]
    label = f"modulation {name!r}"
    specs = _check_specs(label, modulation.specs)
    _check_part_round_trip("modulation", name, specs)
    if modulation.fn is None:
        return
    if not callable(modulation.fn):
        _fail(f"{label}: fn must be callable or None (no modulation)")
    base = _draw_defaults(specs)
    second = _variant(specs, base)
    for dtype in (torch.float32, torch.float64):
        tc = torch.tensor(_TC_MS, dtype=dtype)

        def run(draws: List[Dict[str, Any]], rows: slice) -> torch.Tensor:
            params = _tensor_params(draws, dtype, (len(draws), 1))
            try:
                out = modulation.fn(tc[rows], params)
            except _CALL_ERRORS as exc:
                _fail(
                    f"{label}: fn(tc, params) failed on {sorted(params)} "
                    f"({type(exc).__name__}: {exc})"
                )
            if not isinstance(out, torch.Tensor) or out.shape != tc[rows].shape:
                _fail(
                    f"{label}: fn must return a tensor of tc's shape {tuple(tc.shape)}"
                )
            if out.dtype != dtype:
                _fail(f"{label}: fn returned {out.dtype} for {dtype} tc")
            return out

        batch = run([base, second], slice(0, 2))
        _check_finite(label, batch, "at the defaults")
        if float(batch.min()) < -_RANGE_SLACK or float(batch.max()) > 1 + _RANGE_SLACK:
            _fail(
                f"{label}: fn must return values in [0, 1], got "
                f"[{float(batch.min())!r}, {float(batch.max())!r}]"
            )
        for i, draw in enumerate((base, second)):
            if not torch.equal(batch[i : i + 1], run([draw], slice(i, i + 1))):
                _fail(
                    f"{label}: draw {i} alone differs from draw {i} in a batch of "
                    f"two ({dtype}) (world contract guarantee 1)"
                )


# ------------------------------------------------------------ distributions

_U = np.concatenate(
    [
        [0.0, 0.25, 0.5, 0.75, 1.0 - 2.0**-53],
        np.random.default_rng(0).random(27),
    ]
)


def _check_distribution(name: str, example: Any) -> None:
    if name not in distributions.DISTRIBUTIONS:
        raise ValueError(
            f"no distribution {name!r}; registered: "
            f"{sorted(distributions.DISTRIBUTIONS)}"
        )
    dist = distributions.DISTRIBUTIONS[name]
    label = f"distribution {name!r}"
    if example is not None and not isinstance(example, dict):
        raise ValueError("a distribution's example is a mapping of its options")
    options = dict(example or {})
    if not callable(dist.sample):
        _fail(f"{label}: sample is not callable")
    for hook in ("support", "check", "bounds"):
        value = getattr(dist, hook)
        if value is not None and not callable(value):
            _fail(f"{label}: {hook} must be callable or None")
    declaration = {"dist": name, **options}
    try:
        axis = distributions.AxisSpec.from_dict("contract", declaration)
    except ValueError as exc:
        _fail(
            f"{label}: a world refuses the axis {declaration!r}: {exc} -- pass "
            "example={...options...} that the distribution accepts"
        )
    except (KeyError, TypeError, AttributeError) as exc:
        _fail(
            f"{label}: check(axis) raised {type(exc).__name__}: {exc} on the axis "
            f"{declaration!r}; it must raise ValueError for options it refuses"
        )
    again = distributions.AxisSpec.from_dict("contract", axis.to_dict())
    if again != axis:
        _fail(
            f"{label}: the axis does not re-parse from its own to_dict(): "
            f"{axis.to_dict()!r}"
        )

    u = _U.copy()
    try:
        values = dist.sample(u.copy(), axis)
    except _CALL_ERRORS as exc:
        _fail(f"{label}: sample(u, axis) failed ({type(exc).__name__}: {exc})")
    if not isinstance(values, list) or len(values) != len(u):
        _fail(f"{label}: sample must return a list with one value per u")
    for value in values:
        if value is not None and not isinstance(value, (str, bool, int, float)):
            _fail(
                f"{label}: a drawn value must be a str, int, float or bool: {value!r}"
            )
        if _number(value) and not math.isfinite(value):
            _fail(f"{label}: a drawn number must be finite, got {value!r}")
    if values != dist.sample(u.copy(), axis):
        _fail(f"{label}: the same u gave different values on a second call")
    for i in (0, 7, len(u) - 1):
        if dist.sample(u[i : i + 1].copy(), axis) != [values[i]]:
            _fail(
                f"{label}: the value for u[{i}] depends on the other draws in the "
                "batch -- a draw must depend on its own u only (world contract "
                "guarantee 1)"
            )

    support = dist.support(axis) if dist.support is not None else None
    if support is not None:
        if not isinstance(support, list) or not support:
            _fail(f"{label}: support(axis) must be a non-empty list or None")
        missing = [v for v in values if v not in support]
        if missing:
            _fail(f"{label}: drawn values {missing[:3]!r} are outside support(axis)")
    if dist.quantile and not all(_number(v) for v in values):
        _fail(f"{label}: quantile=True means a continuous quantile function of u")
    if dist.bounds is not None:
        lo, hi = dist.bounds(axis)
        if not (_number(lo) and _number(hi) and math.isfinite(lo + hi) and lo <= hi):
            _fail(f"{label}: bounds(axis) must be a finite (lo, hi) with lo <= hi")
        outside = [v for v in values if _number(v) and not lo <= v <= hi]
        if outside:
            _fail(
                f"{label}: drawn numbers {outside[:3]!r} lie outside bounds(axis) "
                f"= ({lo!r}, {hi!r})"
            )


# -------------------------------------------------------------- class kinds

#: Methods :class:`~sensoryforge.world.kinds.ClassKind` leaves to the subclass.
_REQUIRED_METHODS = (
    "normalise_layer",
    "builtin_defaults",
    "resolve",
    "end_ms",
    "timeline",
    "render_group",
)
_CANVAS = Canvas.from_grid(rows=4, cols=5, spacing_mm=0.1)


def _check_class_fields(spec: Any, kind: kinds.ClassKind, label: str) -> None:
    layer = spec.layer
    again = _call(label, "normalise_layer(layer)", kind.normalise_layer, layer, label)
    if again != layer:
        _fail(f"{label}: normalise_layer of a normalised layer is not a fixed point")
    defaults = _call(label, "builtin_defaults(layer)", kind.builtin_defaults, layer)
    if not isinstance(defaults, dict):
        _fail(f"{label}: builtin_defaults must return a dict")
    for field in defaults:
        ref = _call(label, f"resolve({field!r})", kind.resolve, field, layer)
        if not (
            isinstance(ref, tuple)
            and len(ref) == 2
            and all(isinstance(part, str) for part in ref)
        ):
            _fail(f"{label}: resolve({field!r}) must return (part, field), got {ref!r}")
        domain = _call(label, f"domain({ref!r})", kind.domain, ref, layer)
        if not (isinstance(domain, (tuple, list)) and len(domain) == 2):
            _fail(f"{label}: domain({ref!r}) must return (lo, hi), got {domain!r}")
        info = _call(label, f"field_info({ref!r})", kind.field_info, ref, layer)
        if not isinstance(info, kinds.FieldInfo):
            _fail(f"{label}: field_info({ref!r}) must return a FieldInfo, got {info!r}")


def _check_class_draws(
    world: Any, spec: Any, kind: kinds.ClassKind, label: str
) -> None:
    draws = _call(
        label, "sample(world)", lambda: sample(world, 3, 0, classes=[spec.name])
    )
    ends = []
    for draw in draws:
        end = _call(label, "end_ms(values)", kind.end_ms, draw.values)
        if not (_number(end) and math.isfinite(end) and end >= 0.0):
            _fail(f"{label}: end_ms must be a finite number >= 0, got {end!r}")
        ends.append(float(end))
        cursor = 0.0
        for phase in _call(label, "timeline(values)", kind.timeline, draw.values):
            ok = (
                len(phase) == 3
                and isinstance(phase[0], str)
                and all(_number(t) for t in phase[1:])
            )
            if not ok or not cursor - 1e-9 <= phase[1] < phase[2] <= end + 1e-9:
                _fail(
                    f"{label}: timeline must be [[phase, start_ms, end_ms], ...] in "
                    f"order, none of zero length, inside [0, end_ms={end!r}]; got "
                    f"{phase!r}"
                )
            cursor = float(phase[2])
    span = max(ends) + 2.0
    times = movie_times(1.0, span)
    times = torch.cat([torch.tensor([-2.0, -1.0], dtype=times.dtype), times])
    frames = _call(
        label, "render(draws)", render, draws, _CANVAS, times, dtype=torch.float64
    )
    expected = (len(draws), len(times)) + tuple(_CANVAS.shape)
    if tuple(frames.shape) != expected:
        _fail(f"{label}: render gave {tuple(frames.shape)}, expected {expected}")
    _check_finite(label, frames, "in a rendered frame")
    for i, end in enumerate(ends):
        silent = (times < 0) | (times >= end)
        if bool((frames[i][silent] != 0).any()):
            _fail(
                f"{label}: draw {i} must render exactly 0 before 0 and from "
                f"end_ms={end!r} on"
            )
        alone = _call(
            label,
            f"render(draw {i})",
            render,
            [draws[i]],
            _CANVAS,
            times,
            dtype=torch.float64,
        )[0]
        if not torch.equal(frames[i], alone):
            _fail(f"{label}: draw {i} alone differs from draw {i} in a batch")


def _check_class_kind(name: str, example: Any) -> None:
    if name not in kinds.CLASS_KINDS:
        raise ValueError(
            f"no class kind {name!r}; registered: {sorted(kinds.CLASS_KINDS)}"
        )
    kind = kinds.CLASS_KINDS[name]
    label = f"class kind {name!r}"
    if not isinstance(kind, kinds.ClassKind):
        _fail(f"{label}: must be a ClassKind instance, got {type(kind)!r}")
    if kind.name != name:
        _fail(f"{label}: is registered as {name!r} but its name is {kind.name!r}")
    for method in _REQUIRED_METHODS:
        if getattr(type(kind), method) is getattr(kinds.ClassKind, method):
            _fail(f"{label}: does not override {method}()")
    if example is None:
        raise ValueError(
            "a class kind is checked on an example world: pass example=<the mapping "
            "World.from_dict takes> with at least one class of this kind"
        )
    try:
        world = World.from_dict(example)
    except _CALL_ERRORS as exc:
        _fail(f"{label}: the example world does not load: {exc}")
    classes = [c for c in world.classes.values() if c.kind == name]
    if not classes:
        raise ValueError(f"the example world has no class of kind {name!r}")
    for spec in classes:
        where = f"{label}, class {spec.name!r}"
        _check_class_fields(spec, kind, where)
        _check_class_draws(world, spec, kind, where)


_CHECKS: Dict[str, Callable[[str, Any], None]] = {
    "shape": _check_shape,
    "pattern": _check_pattern,
    "modulation": _check_modulation,
    "distribution": _check_distribution,
    "class_kind": _check_class_kind,
}


def available_world_kinds() -> List[str]:
    """Return the world registry kinds :func:`check_world_component` supports."""
    return sorted(_CHECKS)


def check_world_component(kind: str, name: str, example: Any = None) -> None:
    """Run the world contract checks on one registered entry.

    Register the entry first (``register_shape`` and friends, or the plugin's
    entry point), then check it by name. See the module docstring for the checks
    of each kind.

    Args:
        kind: One of :func:`available_world_kinds`: ``"shape"``, ``"pattern"``,
            ``"modulation"``, ``"distribution"``, ``"class_kind"``.
        name: The registered name (``shape.kind``, ``{dist: name}``, ...).
        example: Needed by two kinds. ``"distribution"``: a mapping of the
            options its axis takes (``{"travel_ratio": 2.0}``); omit it for a
            distribution without required options. ``"class_kind"``: a mapping
            as :meth:`sensoryforge.world.World.from_dict` takes, whose
            ``world.classes`` holds at least one class of this kind. Ignored by
            the other kinds.

    Raises:
        ValueError: If ``kind`` is unknown, ``name`` is not registered, or an
            ``example`` the kind needs is missing or malformed.
        AssertionError: If a contract check fails, naming the entry and the rule.

    Example:
        >>> from sensoryforge.testing.world_contracts import check_world_component
        >>> check_world_component("shape", "gaussian")
    """
    if kind not in _CHECKS:
        raise ValueError(
            f"Unknown world kind {kind!r}; choose one of {available_world_kinds()}"
        )
    _CHECKS[kind](name, example)
