"""World declarations: load, validate, bind, normalise and identify (spec §3)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sensoryforge.config.yaml_utils import load_yaml
from sensoryforge.world.distributions import (
    AxisSpec,
    is_number,
    plain,
    text_number_hint,
)
from sensoryforge.world.kinds import (
    CLASS_KINDS,
    AmbiguousField,
    ClassKind,
    FieldInfo,
    Ref,
    UnknownField,
    expect_mapping,
)

FORMAT = "sensoryforge-world/1"
_TOP_KEYS = {
    "name",
    "description",
    "modality",
    "units",
    "channels",
    "defaults",
    "groups",
    "classes",
    "held_out",
    "fixed_draws",
    "sessions",
}
_SESSION_KEYS = {"duration_ms", "contact_fraction", "gap_mean_ms", "types"}
_CLASS_KEYS = {"kind", "weight", "layer", "axes", "channel", "use"}
#: Names that become directory names (classes, held-out classes, fixed draws,
#: data-set splits): no separators, no leading dot.
NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


def check_name(name: str, where: str) -> str:
    """``name`` if it can name a directory, else a ``ValueError`` naming ``where``."""
    if not NAME_RE.match(name):
        raise ValueError(
            f"{where}: {name!r} names a directory, so it must start with a letter, "
            "digit or '_' and use only letters, digits, '_', '.' and '-'"
        )
    return name


@dataclass
class ClassSpec:
    """One class of a world, with every axis resolved (spec §3.1).

    Attributes:
        name: The class name.
        kind: The class kind's registered name.
        weight: Its weight in declared sampling (0 for a held-out class).
        channel: The world channel it draws on.
        layer: Its normalised layer (``None`` for ``quiet``).
        axes: Every bound name -> its axis, constants included.
        bindings: Every bound name -> ``(part, field)``.
        held_out: True for a held-out class.
    """

    name: str
    kind: str
    weight: float
    channel: str
    layer: Optional[Dict[str, Any]]
    axes: Dict[str, AxisSpec]
    bindings: Dict[str, Ref]
    held_out: bool = False

    @property
    def kind_obj(self) -> ClassKind:
        """The registered class kind."""
        return CLASS_KINDS[self.kind]

    @property
    def random_axes(self) -> List[AxisSpec]:
        """The axes that are sampled (not constants)."""
        return [a for a in self.axes.values() if a.is_random]

    def to_dict(self) -> Dict[str, Any]:
        """The normalised declaration."""
        out: Dict[str, Any] = {
            "kind": self.kind,
            "channel": self.channel,
            "layer": self.layer,
            "axes": {name: axis.to_dict() for name, axis in sorted(self.axes.items())},
        }
        if not self.held_out:
            out["weight"] = self.weight
        return out


@dataclass
class SessionModel:
    """The world's ``sessions:`` section (spec addendum, Task 11).

    Attributes:
        duration_ms: Axis of a session's length, ms (values > 0).
        contact_fraction: Axis of the share of a session spent in contact,
            in [0, 1].
        gap_mean_ms: Mean length of a quiet gap, ms (> 0).
        types: ``name -> {"weight": w, "classes": {class: w}}``, sorted by
            name; empty when the world declares no session types.
    """

    duration_ms: AxisSpec
    contact_fraction: AxisSpec
    gap_mean_ms: float
    types: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """The normalised section (``types`` only when there are some)."""
        out: Dict[str, Any] = {
            "duration_ms": self.duration_ms.to_dict(),
            "contact_fraction": self.contact_fraction.to_dict(),
            "gap_mean_ms": self.gap_mean_ms,
        }
        if self.types:
            out["types"] = {
                name: {"weight": t["weight"], "classes": dict(t["classes"])}
                for name, t in self.types.items()
            }
        return out


@dataclass
class World:
    """A declared stimulus world (spec §3). Build it with :func:`load_world`."""

    name: str
    description: str
    modality: str
    units: Dict[str, str]
    channels: List[str]
    classes: Dict[str, ClassSpec]
    held_out: Dict[str, ClassSpec]
    fixed: Dict[str, Dict[str, Any]]
    world_id: str = ""
    sessions: Optional[SessionModel] = None

    def class_spec(self, name: str) -> ClassSpec:
        """A class or held-out class by name."""
        if name in self.classes:
            return self.classes[name]
        if name in self.held_out:
            return self.held_out[name]
        raise ValueError(
            f"no class {name!r} in world {self.name!r}; classes: "
            f"{sorted(self.classes)}, held out: {sorted(self.held_out)}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """The normalised world; its canonical JSON is what :attr:`world_id` hashes."""
        out: Dict[str, Any] = {
            "format": FORMAT,
            "name": self.name,
            "modality": self.modality,
            "units": dict(self.units),
            "channels": list(self.channels),
            "classes": {n: c.to_dict() for n, c in self.classes.items()},
            "held_out": {n: c.to_dict() for n, c in self.held_out.items()},
            "fixed_draws": {
                n: {"class": f["class"], **f["values"]} for n, f in self.fixed.items()
            },
        }
        if self.sessions is not None:
            out["sessions"] = self.sessions.to_dict()
        return out

    def fixed_draw(self, name: str):
        """The named fixed draw (see :func:`sensoryforge.world.sampling.fixed_draw`)."""
        from sensoryforge.world.sampling import fixed_draw

        return fixed_draw(self, name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "World":
        """Parse a world mapping (optionally under a top-level ``world:`` key)."""
        return _parse_world(data)


def load_world(source: Union[str, Path, Dict[str, Any], World]) -> World:
    """Load a world from a YAML file or a mapping; a :class:`World` passes through."""
    if isinstance(source, World):
        return source
    if isinstance(source, dict):
        return World.from_dict(source)
    path = Path(source)
    with open(path, "r", encoding="utf-8") as stream:
        # Refuses a duplicated key (a class written twice) instead of keeping
        # the last one silently.
        data = load_yaml(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping with a 'world:' key")
    return World.from_dict(data)


def _named(
    mapping: Any, where: str, directories: bool = False
) -> List[Tuple[str, Any]]:
    """``(name, value)`` pairs of a mapping (``None``: none), sorted by name.

    Args:
        mapping: The parsed mapping.
        where: The path errors name.
        directories: The names become directory names (:func:`check_name`).

    Raises:
        ValueError: If it is not a mapping, two keys coincide as strings, or
            (``directories``) a name cannot name a directory.
    """
    out: Dict[str, Any] = {}
    for key, value in expect_mapping(mapping, where).items():
        name = str(key)
        if name in out:
            raise ValueError(f"{where}: {name!r} is given twice")
        if directories:
            check_name(name, where)
        out[name] = value
    return sorted(out.items(), key=lambda item: item[0])


def _parse_world(data: Any) -> World:
    raw = data.get("world", data) if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        raise ValueError(
            "a world is a mapping (optionally under a top-level 'world:' key)"
        )
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise ValueError(
            f"world: unknown keys {sorted(unknown)}; allowed: {sorted(_TOP_KEYS)}"
        )
    raw_channels = raw.get("channels")
    if raw_channels is not None and not isinstance(raw_channels, (list, tuple)):
        raise ValueError(
            f"world.channels: expected a list of names, got {raw_channels!r}"
        )
    channels = [str(c) for c in (raw_channels or ["value"])]
    if len(set(channels)) != len(channels) or not all(channels):
        raise ValueError(
            f"world.channels: give distinct non-empty names, got {channels}"
        )
    defaults = {
        name: AxisSpec.from_dict(name, spec, where=f"world.defaults.{name}")
        for name, spec in _named(raw.get("defaults"), "world.defaults")
    }
    groups: Dict[str, Dict[str, AxisSpec]] = {}
    for group_name, group_raw in _named(
        raw.get("groups"), "world.groups", directories=True
    ):
        group_where = f"world.groups.{group_name}"
        if not isinstance(group_raw, dict):
            raise ValueError(f"{group_where}: expected a mapping of axes")
        groups[group_name] = {
            axis_name: AxisSpec.from_dict(
                axis_name, spec, where=f"{group_where}.{axis_name}"
            )
            for axis_name, spec in _named(group_raw, group_where)
        }
    if not raw.get("classes"):
        raise ValueError("world.classes: declare at least one class")
    # Classes are kept sorted by name: nothing a world does may depend on the
    # order its YAML writes them in (yaml.safe_dump re-sorts keys).
    classes = {
        n: _parse_class(n, c, defaults, channels, held_out=False, groups=groups)
        for n, c in _named(raw["classes"], "world.classes", directories=True)
    }
    held = {
        n: _parse_class(n, c, defaults, channels, held_out=True, groups=groups)
        for n, c in _named(raw.get("held_out"), "world.held_out", directories=True)
    }
    clash = set(classes) & set(held)
    if clash:
        raise ValueError(f"world.held_out: {sorted(clash)} are also classes")
    if sum(c.weight for c in classes.values()) <= 0:
        raise ValueError("world.classes: the weights sum to 0")
    fixed = {
        n: _parse_fixed(n, f, classes, held)
        for n, f in _named(
            raw.get("fixed_draws"), "world.fixed_draws", directories=True
        )
    }
    sessions = (
        _parse_sessions(raw["sessions"], classes, held)
        if raw.get("sessions") is not None
        else None
    )
    units = expect_mapping(raw.get("units"), "world.units") or {
        "space": "mm",
        "time": "ms",
    }
    world = World(
        name=str(raw.get("name", "world")),
        description=str(raw.get("description", "")),
        modality=str(raw.get("modality", "")),
        units={str(k): str(v) for k, v in units.items()},
        channels=channels,
        classes=classes,
        held_out=held,
        fixed=fixed,
        sessions=sessions,
    )
    canonical = json.dumps(
        world.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    world.world_id = "w-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return world


def _axis_values(axis: AxisSpec) -> List[float]:
    """Every value (or both bounds) a session axis can take."""
    if axis.form == "constant":
        return [axis.value]
    if axis.form in ("numeric", "int"):
        return [axis.lo, axis.hi]
    return list(axis.values)


def _session_axis(
    raw: Dict[str, Any], key: str, lo: float, hi: Optional[float], open_lo: bool
) -> AxisSpec:
    """A session axis (constant, range or list) with its bounds checked."""
    where = f"world.sessions.{key}"
    if key not in raw:
        raise ValueError(f"world.sessions: needs {key}")
    axis = AxisSpec.from_dict(key, raw[key], where=where)
    if axis.form not in ("constant", "numeric", "int", "categorical"):
        raise ValueError(
            f"{where}: takes a constant, a range or a list of values, "
            f"not {axis.form!r} (same_as and distributions are not allowed)"
        )
    low = ">" if open_lo else ">="
    domain = f"{low} {lo:g}" if hi is None else f"in [{lo:g}, {hi:g}]"
    for value in _axis_values(axis):
        if not is_number(value) or not math.isfinite(value):
            raise ValueError(f"{where}: needs numbers, got {value!r}")
        bad = value <= lo if open_lo else value < lo
        if bad or (hi is not None and value > hi):
            raise ValueError(f"{where}: {value!r} must be {domain}")
    return axis


def _type_weight(raw: Any, where: str) -> float:
    if not is_number(raw) or not math.isfinite(raw) or raw < 0:
        raise ValueError(f"{where}: must be a finite number >= 0, got {raw!r}")
    return float(raw)


def _parse_sessions(
    raw: Any, classes: Dict[str, ClassSpec], held: Dict[str, ClassSpec]
) -> SessionModel:
    """The ``sessions:`` section, validated against the world's classes."""
    where = "world.sessions"
    raw = expect_mapping(raw, where)
    unknown = set(raw) - _SESSION_KEYS
    if unknown:
        raise ValueError(
            f"{where}: unknown keys {sorted(unknown)}; "
            f"allowed: {sorted(_SESSION_KEYS)}"
        )
    duration = _session_axis(raw, "duration_ms", 0.0, None, open_lo=True)
    fraction = _session_axis(raw, "contact_fraction", 0.0, 1.0, open_lo=False)
    if "gap_mean_ms" not in raw:
        raise ValueError(f"{where}: needs gap_mean_ms")
    gap = raw["gap_mean_ms"]
    if not is_number(gap):
        raise ValueError(
            f"{where}.gap_mean_ms: needs a number (ms) > 0, got {gap!r}"
            f"{text_number_hint(gap)}"
        )
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError(
            f"{where}.gap_mean_ms: must be a finite number > 0, got {gap!r}"
        )
    types: Dict[str, Dict[str, Any]] = {}
    if raw.get("types") is not None:
        named = _named(raw["types"], f"{where}.types", directories=True)
        if not named:
            raise ValueError(f"{where}.types: declare at least one type, or omit it")
        for type_name, type_raw in named:
            type_where = f"{where}.types.{type_name}"
            type_raw = expect_mapping(type_raw, type_where)
            extra = set(type_raw) - {"weight", "classes"}
            if extra:
                raise ValueError(
                    f"{type_where}: unknown keys {sorted(extra)}; "
                    "allowed: ['classes', 'weight']"
                )
            if "classes" not in type_raw:
                raise ValueError(f"{type_where}: needs classes: {{<class>: weight}}")
            weight = _type_weight(type_raw.get("weight", 1.0), f"{type_where}.weight")
            class_where = f"{type_where}.classes"
            members: Dict[str, float] = {}
            for class_name, class_weight in _named(type_raw["classes"], class_where):
                if class_name in held:
                    raise ValueError(
                        f"{class_where}: {class_name!r} is held out; a session "
                        "type draws from the world's own classes"
                    )
                if class_name not in classes:
                    raise ValueError(
                        f"{class_where}: unknown class {class_name!r}; "
                        f"the world's classes: {sorted(classes)}"
                    )
                members[class_name] = _type_weight(
                    class_weight, f"{class_where}.{class_name}"
                )
            if not members:
                raise ValueError(f"{class_where}: name at least one class")
            if sum(members.values()) <= 0:
                raise ValueError(f"{class_where}: the weights sum to 0")
            types[type_name] = {"weight": weight, "classes": members}
        if sum(t["weight"] for t in types.values()) <= 0:
            raise ValueError(f"{where}.types: the weights sum to 0")
    return SessionModel(
        duration_ms=duration,
        contact_fraction=fraction,
        gap_mean_ms=float(gap),
        types=types,
    )


def _parse_class(
    name: str,
    raw: Any,
    defaults: Dict[str, AxisSpec],
    channels: List[str],
    held_out: bool,
    groups: Optional[Dict[str, Dict[str, AxisSpec]]] = None,
) -> ClassSpec:
    where = f"world.{'held_out' if held_out else 'classes'}.{name}"
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping")
    unknown = set(raw) - _CLASS_KEYS
    if unknown:
        raise ValueError(
            f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(_CLASS_KEYS)}"
        )
    kind_name = str(raw.get("kind", "layered"))
    if kind_name not in CLASS_KINDS:
        raise ValueError(
            f"{where}: unknown class kind {kind_name!r}; known: {sorted(CLASS_KINDS)}"
        )
    kind = CLASS_KINDS[kind_name]
    raw_layer = raw.get("layer")
    layer = kind.normalise_layer(raw_layer, f"{where}.layer")
    if held_out:
        if "weight" in raw:
            raise ValueError(f"{where}: held-out classes take no weight")
        weight = 0.0
    else:
        raw_weight = raw.get("weight", 1.0)
        if not is_number(raw_weight) or not math.isfinite(raw_weight) or raw_weight < 0:
            raise ValueError(
                f"{where}.weight: must be a finite number >= 0, got {raw_weight!r}"
            )
        weight = float(raw_weight)
    channel = str(raw.get("channel", channels[0]))
    if channel not in channels:
        raise ValueError(
            f"{where}.channel: {channel!r} is not one of the world's "
            f"channels {channels}"
        )

    axes: Dict[str, AxisSpec] = {}
    bindings: Dict[str, Ref] = {}
    source: Dict[str, str] = {}
    paths: Dict[str, str] = {}

    def bind(axis_name: str, ref: Ref, axis: AxisSpec, origin: str, path: str) -> None:
        # A later source (built-in < world default < class axis) replaces an
        # earlier one; two names of one source binding one field would make
        # the result depend on the order they are written in, so they fail.
        for other, other_ref in list(bindings.items()):
            if other_ref == ref and other != axis_name:
                if source[other] == origin:
                    raise ValueError(
                        f"{where}: {other!r} and {axis_name!r} both set "
                        f"{'.'.join(ref)} (from {origin}); keep one"
                    )
                del bindings[other]
                del axes[other]
                del source[other]
                del paths[other]
        lo, hi = kind.domain(ref, layer)
        axes[axis_name] = axis.with_domain(lo, hi)
        bindings[axis_name] = ref
        source[axis_name] = origin
        paths[axis_name] = path

    for field_name, value in kind.builtin_defaults(layer).items():
        constant = AxisSpec(name=field_name, form="constant", value=value)
        ref = kind.resolve(field_name, layer)
        path = (
            f"{where}.layer.{ref[0]}.{ref[1]}"
            if layer is not None and ref[0] in layer
            else f"{where} (built-in {field_name})"
        )
        bind(field_name, ref, constant, "built-ins", path)
    for axis_name, axis in sorted(defaults.items(), key=lambda item: item[0]):
        try:
            ref = kind.resolve(axis_name, layer)
        except UnknownField:
            continue
        except AmbiguousField as exc:
            raise ValueError(f"{where}: world default {exc}") from None
        if kind.fixed_in_layer(ref, raw_layer):
            continue
        path = f"world.defaults.{axis_name} (in class {name!r})"
        bind(axis_name, ref, axis, "world.defaults", path)
    groups = groups or {}
    used = raw.get("use")
    if used is None:
        used = []
    if not isinstance(used, (list, tuple)):
        raise ValueError(f"{where}.use: expected a list of group names, got {used!r}")
    used = [str(g) for g in used]
    if len(set(used)) != len(used):
        raise ValueError(f"{where}.use: a group is named twice in {used}")
    set_by: Dict[Ref, str] = {}
    for group_name in used:
        if group_name not in groups:
            raise ValueError(
                f"{where}.use: unknown group {group_name!r}; "
                f"the world's groups: {sorted(groups)}"
            )
        for axis_name, axis in groups[group_name].items():
            path = f"world.groups.{group_name}.{axis_name} (in class {name!r})"
            try:
                ref = kind.resolve(axis_name, layer)
            except (UnknownField, AmbiguousField) as exc:
                raise ValueError(f"{path}: {exc}") from None
            if ref in set_by:
                raise ValueError(
                    f"{where}.use: groups {set_by[ref]!r} and {group_name!r} "
                    f"both set {'.'.join(ref)}; keep one"
                )
            set_by[ref] = group_name
            if kind.fixed_in_layer(ref, raw_layer):
                continue
            bind(axis_name, ref, axis, "world.groups", path)
    for axis_name, spec in _named(raw.get("axes"), f"{where}.axes"):
        try:
            ref = kind.resolve(axis_name, layer)
        except (UnknownField, AmbiguousField) as exc:
            raise ValueError(f"{where}.axes: {exc}") from None
        path = f"{where}.axes.{axis_name}"
        axis = AxisSpec.from_dict(axis_name, spec, where=path)
        bind(axis_name, ref, axis, f"{where}.axes", path)
    for axis_name, axis in axes.items():
        if axis.form == "link":
            continue
        info = kind.field_info(bindings[axis_name], layer)
        check_axis(paths[axis_name], axis, info)
    for axis_name, axis in axes.items():
        if axis.form != "link":
            continue
        target = axes.get(axis.link)
        if target is None:
            raise ValueError(
                f"{paths[axis_name]}: same_as names {axis.link!r}, which is not "
                f"an axis of class {name!r} (its axes: {sorted(axes)})"
            )
        if axis.link == axis_name:
            raise ValueError(
                f"{paths[axis_name]}: an axis cannot be the same as itself"
            )
        if target.form == "link":
            raise ValueError(
                f"{paths[axis_name]}: same_as names {axis.link!r}, which is itself "
                "a link; point at an axis that is drawn"
            )
        info = kind.field_info(bindings[axis_name], layer)
        check_axis(f"{paths[axis_name]} (copying {axis.link!r})", target, info)

    cls = ClassSpec(
        name=name,
        kind=kind_name,
        weight=weight,
        channel=channel,
        layer=layer,
        axes=dict(sorted(axes.items())),
        bindings=dict(sorted(bindings.items())),
        held_out=held_out,
    )
    kind.check(cls)
    return cls


def _parse_fixed(
    name: str, raw: Any, classes: Dict[str, ClassSpec], held: Dict[str, ClassSpec]
) -> Dict[str, Any]:
    where = f"world.fixed_draws.{name}"
    raw = expect_mapping(raw, where)
    if "class" not in raw:
        raise ValueError(f"{where}: needs a 'class'")
    class_name = str(raw["class"])
    spec = classes.get(class_name) or held.get(class_name)
    if spec is None:
        raise ValueError(f"{where}: unknown class {class_name!r}")
    values: Dict[str, Any] = {}
    for key, value in raw.items():
        if key == "class":
            continue
        try:
            ref = spec.kind_obj.resolve(str(key), spec.layer)
        except (UnknownField, AmbiguousField) as exc:
            raise ValueError(f"{where}: {exc}") from None
        bound = next((n for n, r in spec.bindings.items() if r == ref), None)
        if bound is None:
            raise ValueError(
                f"{where}: {key!r} is not an axis of class {class_name!r} "
                f"(its axes: {sorted(spec.axes)})"
            )
        if spec.axes[bound].form == "link":
            raise ValueError(
                f"{where}: {key!r} is a link (same_as {spec.axes[bound].link!r}); "
                "set the axis it copies instead"
            )
        if bound in values:
            raise ValueError(f"{where}: axis {bound!r} is set twice")
        value = plain(value)
        # Outside the axis's range is allowed (flagged out_of_range), outside
        # the field's domain is not.
        info = spec.kind_obj.field_info(ref, spec.layer)
        check_value(f"{where}.{key}", value, info)
        values[bound] = value
    return {"class": class_name, "values": dict(sorted(values.items()))}


def _whole(info: FieldInfo) -> str:
    at_least = "" if info.lo is None else f" >= {info.lo:g}"
    return (
        f"must be a whole number{at_least} "
        "(an int constant, an int: true range or a list of ints)"
    )


_TYPE_WORDS = {"str": "text", "bool": "true or false"}


def check_value(where: str, value: Any, info: FieldInfo) -> None:
    """Raise ``ValueError`` unless ``value`` suits a field described by ``info``.

    Args:
        where: The path the error names.
        value: A constant, range bound, categorical value or fixed-draw value.
        info: The bound field's type and domain.
    """
    numeric = info.dtype in ("number", "whole")
    if numeric or (info.dtype == "any" and is_number(value)):
        if info.dtype == "whole" and not isinstance(value, int):
            raise ValueError(f"{where}: {_whole(info)}, got {value!r}")
        if not is_number(value):
            hint = text_number_hint(value)
            raise ValueError(f"{where}: needs a number, got {value!r}{hint}")
        if not math.isfinite(value):
            raise ValueError(f"{where}: {value!r} is not finite")
        if (info.lo is not None and value < info.lo) or (
            info.hi is not None and value > info.hi
        ):
            lo = "-inf" if info.lo is None else f"{info.lo:g}"
            hi = "inf" if info.hi is None else f"{info.hi:g}"
            raise ValueError(
                f"{where}: {value!r} is outside the field's domain [{lo}, {hi}]"
            )
    elif info.dtype == "str":
        if not isinstance(value, str):
            raise ValueError(
                f"{where}: needs text, got {value!r}; quote it in YAML, "
                f"e.g. '{value}'"
            )
        if info.choices and value not in info.choices:
            raise ValueError(f"{where}: {value!r} is not one of {list(info.choices)}")
    elif info.dtype == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{where}: needs true or false, got {value!r}")


def check_axis(where: str, axis: AxisSpec, info: FieldInfo) -> None:
    """Raise ``ValueError`` unless every value ``axis`` can take suits its field.

    Constants, range bounds, categorical values and a registered
    distribution's finite support are each checked with :func:`check_value`.
    """
    if axis.form == "constant":
        check_value(where, axis.value, info)
    elif axis.form in ("numeric", "int"):
        if info.dtype in _TYPE_WORDS:
            raise ValueError(
                f"{where}: takes {_TYPE_WORDS[info.dtype]}, which a range cannot draw"
            )
        if info.dtype == "whole" and axis.form != "int":
            raise ValueError(f"{where}: {_whole(info)}, got a float range")
        for bound in (axis.lo, axis.hi):
            check_value(where, int(bound) if axis.form == "int" else bound, info)
    elif axis.form == "categorical":
        for value in axis.values:
            check_value(where, value, info)
    else:
        support = axis.support()
        if support is not None:
            for value in support:
                check_value(where, value, info)
        elif info.dtype == "whole":
            raise ValueError(f"{where}: {_whole(info)}, got distribution {axis.dist!r}")
