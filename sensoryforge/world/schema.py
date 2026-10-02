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
    "classes",
    "held_out",
    "fixed_draws",
}
_CLASS_KEYS = {"kind", "weight", "layer", "axes", "channel"}
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
        return {
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
    if not raw.get("classes"):
        raise ValueError("world.classes: declare at least one class")
    # Classes are kept sorted by name: nothing a world does may depend on the
    # order its YAML writes them in (yaml.safe_dump re-sorts keys).
    classes = {
        n: _parse_class(n, c, defaults, channels, held_out=False)
        for n, c in _named(raw["classes"], "world.classes", directories=True)
    }
    held = {
        n: _parse_class(n, c, defaults, channels, held_out=True)
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
    )
    canonical = json.dumps(
        world.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    world.world_id = "w-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return world


def _parse_class(
    name: str,
    raw: Any,
    defaults: Dict[str, AxisSpec],
    channels: List[str],
    held_out: bool,
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
    for axis_name, spec in _named(raw.get("axes"), f"{where}.axes"):
        try:
            ref = kind.resolve(axis_name, layer)
        except (UnknownField, AmbiguousField) as exc:
            raise ValueError(f"{where}.axes: {exc}") from None
        path = f"{where}.axes.{axis_name}"
        axis = AxisSpec.from_dict(axis_name, spec, where=path)
        bind(axis_name, ref, axis, f"{where}.axes", path)
    for axis_name, axis in axes.items():
        info = kind.field_info(bindings[axis_name], layer)
        check_axis(paths[axis_name], axis, info)

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
