"""World declarations: load, validate, bind, normalise and identify (spec §3)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from sensoryforge.world.distributions import AxisSpec, plain
from sensoryforge.world.kinds import (
    CLASS_KINDS,
    AmbiguousField,
    ClassKind,
    Ref,
    UnknownField,
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
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping with a 'world:' key")
    return World.from_dict(data)


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
    channels = [str(c) for c in (raw.get("channels") or ["value"])]
    if len(set(channels)) != len(channels) or not all(channels):
        raise ValueError(
            f"world.channels: give distinct non-empty names, got {channels}"
        )
    defaults = {
        str(name): AxisSpec.from_dict(str(name), spec)
        for name, spec in (raw.get("defaults") or {}).items()
    }
    if not raw.get("classes"):
        raise ValueError("world.classes: declare at least one class")
    classes = {
        str(n): _parse_class(str(n), c, defaults, channels, held_out=False)
        for n, c in raw["classes"].items()
    }
    held = {
        str(n): _parse_class(str(n), c, defaults, channels, held_out=True)
        for n, c in (raw.get("held_out") or {}).items()
    }
    clash = set(classes) & set(held)
    if clash:
        raise ValueError(f"world.held_out: {sorted(clash)} are also classes")
    if sum(c.weight for c in classes.values()) <= 0:
        raise ValueError("world.classes: the weights sum to 0")
    fixed = {
        str(n): _parse_fixed(str(n), f, classes, held)
        for n, f in (raw.get("fixed_draws") or {}).items()
    }
    units = raw.get("units") or {"space": "mm", "time": "ms"}
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
        weight = float(raw.get("weight", 1.0))
        if weight < 0:
            raise ValueError(f"{where}.weight: must be >= 0, got {weight}")
    channel = str(raw.get("channel", channels[0]))
    if channel not in channels:
        raise ValueError(
            f"{where}.channel: {channel!r} is not one of the world's "
            f"channels {channels}"
        )

    axes: Dict[str, AxisSpec] = {}
    bindings: Dict[str, Ref] = {}

    def bind(axis_name: str, ref: Ref, axis: AxisSpec) -> None:
        for other, other_ref in list(bindings.items()):
            if other_ref == ref and other != axis_name:
                del bindings[other]
                del axes[other]
        lo, hi = kind.domain(ref, layer)
        axes[axis_name] = axis.with_domain(lo, hi)
        bindings[axis_name] = ref

    for field_name, value in kind.builtin_defaults(layer).items():
        constant = AxisSpec(name=field_name, form="constant", value=value)
        bind(field_name, kind.resolve(field_name, layer), constant)
    for axis_name, axis in defaults.items():
        try:
            ref = kind.resolve(axis_name, layer)
        except UnknownField:
            continue
        except AmbiguousField as exc:
            raise ValueError(f"{where}: world default {exc}") from None
        if kind.fixed_in_layer(ref, raw_layer):
            continue
        bind(axis_name, ref, axis)
    for axis_name, spec in (raw.get("axes") or {}).items():
        axis_name = str(axis_name)
        try:
            ref = kind.resolve(axis_name, layer)
        except (UnknownField, AmbiguousField) as exc:
            raise ValueError(f"{where}.axes: {exc}") from None
        bind(axis_name, ref, AxisSpec.from_dict(axis_name, spec))

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
    if not isinstance(raw, dict) or "class" not in raw:
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
        values[bound] = plain(value)
    return {"class": class_name, "values": dict(sorted(values.items()))}
