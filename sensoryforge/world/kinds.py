"""Class kinds: how a world class binds axis names, times its draws and renders them.

Spec §3.1, §3.3, §3.4, §5.4. A class kind is registered by name; ``layered``
(a layered layer with random fields) and ``quiet`` (exactly zero) are built
in. A plugin adds a kind with :func:`register_class_kind`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from sensoryforge.stimuli.episode import contact_terms, span_progress
from sensoryforge.stimuli.layered import MOTIONS, defaults
from sensoryforge.world import kernel

Ref = Tuple[str, str]


class UnknownField(ValueError):
    """An axis name that is not a field of the class."""


class AmbiguousField(ValueError):
    """A bare axis name that matches fields in more than one part."""


#: Episode fields every layered class has (spec §3.3).
EPISODE_FIELDS: Tuple[str, ...] = (
    "delay_ms",
    "touch_ms",
    "hold_ms",
    "slide_ms",
    "release_ms",
    "contacts",
    "pause_ms",
    "speed_mm_per_ms",
    "direction_deg",
)
_EPISODE_DEFAULTS: Dict[str, Any] = {
    "delay_ms": 0.0,
    "touch_ms": 0.0,
    "hold_ms": 0.0,
    "slide_ms": 0.0,
    "release_ms": 0.0,
    "contacts": 1,
    "pause_ms": 0.0,
    "speed_mm_per_ms": 0.0,
    "direction_deg": 0.0,
}
_EPISODE_DOMAIN: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "contacts": (1.0, None),
    "direction_deg": (None, None),
}
_CONTACT_PHASES = (
    ("touch", "touch_ms"),
    ("hold", "hold_ms"),
    ("slide", "slide_ms"),
    ("release", "release_ms"),
)
_PARTS = ("shape", "pattern", "modulation")
_MOTION_KINDS = ("none", "linear", "circular", "path")


@dataclass(frozen=True)
class FieldInfo:
    """The values a field accepts; a world's axes are checked against it on load.

    Attributes:
        dtype: ``"number"`` (an int or a float, not a bool), ``"whole"`` (an
            int), ``"str"``, ``"bool"``, or ``"any"`` (numbers are still
            checked against ``lo``/``hi``).
        lo: The smallest valid value (``None``: unbounded).
        hi: The largest valid value (``None``: unbounded).
        choices: The valid values of a ``str`` field (``None``: any text).
    """

    dtype: str = "any"
    lo: Optional[float] = None
    hi: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None


class ClassKind:
    """Base class of class kinds; :class:`LayeredKind` is the worked example."""

    name = ""

    def normalise_layer(self, layer: Any, where: str) -> Optional[Dict[str, Any]]:
        """The class's ``layer`` with defaults filled (``None`` if it has none)."""
        raise NotImplementedError

    def builtin_defaults(self, layer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """``{axis name: constant}`` every class of this kind binds."""
        raise NotImplementedError

    def resolve(self, name: str, layer: Optional[Dict[str, Any]]) -> Ref:
        """The field an axis name binds to; raises UnknownField / AmbiguousField."""
        raise NotImplementedError

    def domain(
        self, ref: Ref, layer: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[float]]:
        """The valid range of a field (probes stay inside it)."""
        return (None, None)

    def field_info(self, ref: Ref, layer: Optional[Dict[str, Any]]) -> FieldInfo:
        """What values a field takes; by default any type within :meth:`domain`."""
        lo, hi = self.domain(ref, layer)
        return FieldInfo("any", lo, hi)

    def fixed_in_layer(self, ref: Ref, raw_layer: Any) -> bool:
        """True when the class's own layer sets this field (a world default yields)."""
        return False

    def check(self, spec: Any) -> None:
        """Kind-specific validation of a parsed class."""

    def end_ms(self, values: Dict[str, Any]) -> float:
        """When a draw with these values ends, ms."""
        raise NotImplementedError

    def timeline(self, values: Dict[str, Any]) -> List[List[Any]]:
        """``[[phase, start_ms, end_ms], ...]``, zero-length phases omitted."""
        raise NotImplementedError

    def to_layer(self, spec: Any, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The draw as an ordinary layered layer dict (``None`` if it has none)."""
        return None

    def render_group(
        self,
        spec: Any,
        draws: List[Any],
        X: torch.Tensor,
        Y: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Frames ``[g, K, *S]`` for a group of draws at times ``[g, K]`` (ms)."""
        raise NotImplementedError


def expect_mapping(value: Any, where: str) -> Dict[Any, Any]:
    """``value`` if it is a mapping, ``{}`` for ``None``; else a ``ValueError``.

    Args:
        value: A parsed YAML value.
        where: The path the error names (``world.classes.dots.axes``).
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"{where}: expected a mapping, got {type(value).__name__} {value!r}"
        )
    return value


def _fill(part: Any, default_kind: str, specs_fn, where: str) -> Dict[str, Any]:
    part = dict(expect_mapping(part, where) or {"kind": default_kind})
    kind = part.get("kind", default_kind)
    try:
        specs = specs_fn(kind)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None
    names = {s.name for s in specs}
    extra = set(part) - names - {"kind"}
    if extra:
        raise ValueError(
            f"{where}: {kind!r} has no fields {sorted(extra)}; it has {sorted(names)}"
        )
    return {
        "kind": kind,
        **defaults(specs),
        **{k: v for k, v in part.items() if k != "kind"},
    }


def _only_zero(axis: Any) -> bool:
    """True when an axis (already checked against its domain) can only give 0."""
    if axis.form == "constant":
        return isinstance(axis.value, (int, float)) and float(axis.value) == 0.0
    if axis.form in ("numeric", "int"):
        return axis.hi == 0.0
    support = axis.support()
    return support is not None and all(
        isinstance(v, (int, float)) and float(v) == 0.0 for v in support
    )


def _group_params(dicts, view, dtype, device) -> Dict[str, Any]:
    """Per-draw parameter dicts -> one dict: numbers become tensors of shape ``view``.

    Non-numeric values (strings, bools) are equal across a group (the
    renderer groups by them) and pass through as plain values.
    """
    out: Dict[str, Any] = {}
    for key in dicts[0]:
        if key == "kind":
            continue
        values = [d[key] for d in dicts]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            out[key] = torch.tensor(
                [float(v) for v in values], dtype=dtype, device=device
            ).view(view)
        else:
            out[key] = values[0]
    return out


class LayeredKind(ClassKind):
    """A class drawn as a layer: shape x pattern x modulation, timed by an episode."""

    name = "layered"

    def normalise_layer(self, layer, where):
        layer = dict(expect_mapping(layer, where))
        if "timing" in layer:
            raise ValueError(
                f"{where}: set timing with the episode axes (delay_ms, touch_ms, "
                "hold_ms, slide_ms, release_ms, contacts, pause_ms), not layer.timing"
            )
        unknown = set(layer) - {"shape", "pattern", "motion", "modulation"}
        if unknown:
            raise ValueError(
                f"{where}: unknown parts {sorted(unknown)}; "
                "a layer has shape, pattern, motion, modulation"
            )
        out: Dict[str, Any] = {
            "shape": _fill(
                layer.get("shape"), "gaussian", kernel.shape_specs, f"{where}.shape"
            ),
            "pattern": _fill(
                layer.get("pattern"), "single", kernel.pattern_specs, f"{where}.pattern"
            ),
            "modulation": _fill(
                layer.get("modulation"),
                "none",
                kernel.modulation_specs,
                f"{where}.modulation",
            ),
            "motion": None,
        }
        motion = layer.get("motion")
        if motion is not None:
            motion = dict(expect_mapping(motion, f"{where}.motion"))
            kind = motion.get("kind", "none")
            if kind not in _MOTION_KINDS:
                raise ValueError(
                    f"{where}.motion: unknown kind {kind!r}; "
                    f"known: {list(_MOTION_KINDS)}"
                )
            if "span" in motion:
                raise ValueError(
                    f"{where}.motion: a world class moves during its slides; "
                    "drop 'span'"
                )
            specs = [s for s in MOTIONS[kind] if s.name != "span"]
            extra = set(motion) - {"kind"} - {s.name for s in specs}
            if extra:
                raise ValueError(
                    f"{where}.motion: {kind!r} has no fields {sorted(extra)}"
                )
            out["motion"] = {"kind": kind, **defaults(specs), **motion}
        return out

    def _names(self, layer) -> Dict[str, set]:
        return {
            "shape": {s.name for s in kernel.shape_specs(layer["shape"]["kind"])},
            "pattern": {s.name for s in kernel.pattern_specs(layer["pattern"]["kind"])},
            "modulation": {
                s.name for s in kernel.modulation_specs(layer["modulation"]["kind"])
            },
        }

    def known(self, layer) -> List[str]:
        """Every name an axis of this class can bind."""
        names = self._names(layer)
        bare = set().union(*names.values())
        dotted = {f"{p}.{n}" for p in _PARTS for n in names[p]}
        return sorted(set(EPISODE_FIELDS) | bare | dotted)

    def builtin_defaults(self, layer):
        out = dict(_EPISODE_DEFAULTS)
        out["amplitude"] = layer["shape"]["amplitude"]
        if "x_mm" in self._names(layer)["pattern"]:
            out["x_mm"] = layer["pattern"]["x_mm"]
            out["y_mm"] = layer["pattern"]["y_mm"]
        return out

    def resolve(self, name, layer):
        if name in EPISODE_FIELDS:
            return ("episode", name)
        names = self._names(layer)
        if name == "amplitude":
            return ("shape", "amplitude")
        if name in ("x_mm", "y_mm"):
            if name in names["pattern"]:
                return ("pattern", name)
            raise UnknownField(
                f"{name!r}: pattern {layer['pattern']['kind']!r} has no placement"
            )
        if "." in name:
            part, field = name.split(".", 1)
            if part in names and field in names[part]:
                return (part, field)
            raise UnknownField(
                f"{name!r} is not a field of this class; known: {self.known(layer)}"
            )
        hits = [part for part in _PARTS if name in names[part]]
        if not hits:
            raise UnknownField(
                f"{name!r} is not a field of this class; known: {self.known(layer)}"
            )
        if len(hits) > 1:
            raise AmbiguousField(
                f"{name!r} is ambiguous; write one of {[h + '.' + name for h in hits]}"
            )
        return (hits[0], name)

    def _param_spec(self, ref, layer):
        part, field = ref
        specs_fn = {
            "shape": kernel.shape_specs,
            "pattern": kernel.pattern_specs,
            "modulation": kernel.modulation_specs,
        }[part]
        return next(s for s in specs_fn(layer[part]["kind"]) if s.name == field)

    def domain(self, ref, layer):
        part, field = ref
        if part == "episode":
            return _EPISODE_DOMAIN.get(field, (0.0, None))
        spec = self._param_spec(ref, layer)
        return (spec.min_val, spec.max_val)

    def field_info(self, ref, layer):
        """Numbers for episode fields (``contacts``: whole); else the ParamSpec's.

        Shape, pattern and modulation fields take their ``ParamSpec``'s type
        (``float``/``int``: a number, ``str``: text from ``choices`` if any,
        ``bool``: true or false) within its ``min_val``/``max_val``.
        """
        lo, hi = self.domain(ref, layer)
        part, field = ref
        if part == "episode":
            return FieldInfo("whole" if field == "contacts" else "number", lo, hi)
        spec = self._param_spec(ref, layer)
        dtype = {"float": "number", "int": "number", "str": "str", "bool": "bool"}
        choices = tuple(spec.choices) if spec.choices else None
        return FieldInfo(dtype.get(spec.dtype, "any"), lo, hi, choices)

    def fixed_in_layer(self, ref, raw_layer):
        part, field = ref
        return part in _PARTS and field in ((raw_layer or {}).get(part) or {})

    def check(self, spec):
        phases = [spec.axes[field] for _, field in _CONTACT_PHASES]
        if all(_only_zero(a) for a in phases):
            raise ValueError(
                f"class {spec.name!r}: touch_ms + hold_ms + slide_ms + release_ms is "
                "always 0, so it never touches; give it a hold_ms"
            )

    def end_ms(self, values):
        contacts = int(values["contacts"])
        contact = sum(float(values[field]) for _, field in _CONTACT_PHASES)
        return (
            float(values["delay_ms"])
            + contacts * contact
            + (contacts - 1) * float(values["pause_ms"])
        )

    def timeline(self, values):
        out: List[List[Any]] = []
        t = 0.0

        def add(phase: str, length: float) -> None:
            nonlocal t
            if length > 0:
                out.append([phase, t, t + length])
                t += length

        add("quiet", float(values["delay_ms"]))
        for k in range(int(values["contacts"])):
            if k:
                add("pause", float(values["pause_ms"]))
            for phase, field in _CONTACT_PHASES:
                add(phase, float(values[field]))
        return out

    def part_values(self, spec, values) -> Dict[str, Dict[str, Any]]:
        """The class's shape, pattern, modulation dicts with these axis values set."""
        parts = {part: dict(spec.layer[part]) for part in _PARTS}
        for name, value in values.items():
            part, field = spec.bindings[name]
            if part in parts:
                parts[part][field] = value
        return parts

    def to_layer(self, spec, values):
        parts = self.part_values(spec, values)
        v = values
        timing = {
            "onset_ms": float(v["delay_ms"]),
            "ramp_up_ms": float(v["touch_ms"]),
            "hold_ms": float(v["hold_ms"]),
            "slide_ms": float(v["slide_ms"]),
            "ramp_down_ms": float(v["release_ms"]),
            "contacts": int(v["contacts"]),
            "pause_ms": float(v["pause_ms"]),
        }
        declared = spec.layer.get("motion")
        if declared is not None:
            motion = {**declared, "span": "slide"}
        else:
            travel = (
                float(v["speed_mm_per_ms"]) * int(v["contacts"]) * float(v["slide_ms"])
            )
            theta = math.radians(float(v["direction_deg"]))
            motion = {
                "kind": "linear",
                "start": [0.0, 0.0],
                "end": [travel * math.cos(theta), travel * math.sin(theta)],
                "span": "slide",
            }
        return {
            "shape": parts["shape"],
            "pattern": parts["pattern"],
            "motion": motion,
            "timing": timing,
            "modulation": parts["modulation"],
        }

    def render_group(self, spec, draws, X, Y, times):
        """Frames ``[g, K, *S]``: amplitude x envelope x modulation x sum of shapes.

        Args:
            spec: The class.
            draws: The group's draws (same class, same non-numeric values).
            X, Y: Canvas coordinates ``[*S]`` in mm, in the output dtype/device.
            times: ``[g, K]`` ms since each draw's start (negative: before it).
        """
        dtype, device = X.dtype, X.device
        g, k_count = times.shape
        ones = (1,) * X.ndim
        lead = (g, k_count) + ones
        per_draw = (g, 1) + ones

        def column(name: str) -> torch.Tensor:
            values = [float(d.values[name]) for d in draws]
            return torch.tensor(values, dtype=dtype, device=device).view(g, 1)

        ep = {name: column(name) for name in EPISODE_FIELDS}
        env, tau, k, local = contact_terms(
            times,
            ep["delay_ms"],
            ep["touch_ms"],
            ep["hold_ms"],
            ep["slide_ms"],
            ep["release_ms"],
            ep["contacts"],
            ep["pause_ms"],
        )
        parts = [self.part_values(spec, d.values) for d in draws]
        modulation = kernel.MODULATION_KINDS[spec.layer["modulation"]["kind"]]
        if modulation.fn is not None:
            params = _group_params(
                [p["modulation"] for p in parts], (g, 1), dtype, device
            )
            env = env * modulation.fn(tau, params)
        progress = span_progress(
            tau,
            k,
            local,
            ep["contacts"],
            ep["touch_ms"] + ep["hold_ms"],
            ep["slide_ms"],
        )
        motion = spec.layer.get("motion")
        if motion is None:
            travel = ep["speed_mm_per_ms"] * ep["contacts"] * ep["slide_ms"]
            theta = torch.deg2rad(ep["direction_deg"])
            off_x = progress * (travel * torch.cos(theta))
            off_y = progress * (travel * torch.sin(theta))
        else:
            offsets = kernel.motion_offsets(motion, progress)
            off_x, off_y = offsets[..., 0], offsets[..., 1]
        off_x, off_y = off_x.reshape(lead), off_y.reshape(lead)

        shape = kernel.SHAPE_KINDS[spec.layer["shape"]["kind"]]
        params = _group_params([p["shape"] for p in parts], per_draw, dtype, device)
        amplitude = params.pop("amplitude")
        if shape.unbounded:
            total = shape.fn(X - off_x, Y - off_y, params)
        else:
            pos, scales = kernel.pattern_batch(
                spec.layer["pattern"]["kind"],
                [p["pattern"] for p in parts],
                dtype,
                device,
            )
            total = torch.zeros(
                (g, k_count) + tuple(X.shape), dtype=dtype, device=device
            )
            for slot in range(pos.shape[1]):
                px = pos[:, slot, 0].reshape(per_draw)
                py = pos[:, slot, 1].reshape(per_draw)
                weight = scales[:, slot].reshape(per_draw)
                total = total + weight * shape.fn(
                    X - px - off_x, Y - py - off_y, params
                )
        return amplitude * env.reshape(lead) * total


class QuietKind(ClassKind):
    """A class whose draws are exactly zero for ``quiet_ms``."""

    name = "quiet"

    def normalise_layer(self, layer, where):
        if layer:
            raise ValueError(f"{where}: a quiet class has no layer")
        return None

    def builtin_defaults(self, layer):
        return {"quiet_ms": 0.0}

    def resolve(self, name, layer):
        if name == "quiet_ms":
            return ("quiet", "quiet_ms")
        raise UnknownField(f"{name!r}: a quiet class has one axis, quiet_ms")

    def domain(self, ref, layer):
        return (0.0, None)

    def field_info(self, ref, layer):
        return FieldInfo("number", 0.0, None)

    def end_ms(self, values):
        return float(values["quiet_ms"])

    def timeline(self, values):
        q = float(values["quiet_ms"])
        return [["quiet", 0.0, q]] if q > 0 else []

    def render_group(self, spec, draws, X, Y, times):
        shape = (len(draws), times.shape[1]) + tuple(X.shape)
        return torch.zeros(shape, dtype=X.dtype, device=X.device)


CLASS_KINDS: Dict[str, ClassKind] = {}


def register_class_kind(kind: ClassKind, *, replace: bool = False) -> None:
    """Register a class kind under ``kind.name``.

    Raises:
        ValueError: If the name is empty, or taken and ``replace`` is false.
    """
    if not kind.name:
        raise ValueError("a class kind needs a name")
    if kind.name in CLASS_KINDS and not replace:
        raise ValueError(f"class kind {kind.name!r} is already registered")
    CLASS_KINDS[kind.name] = kind


register_class_kind(LayeredKind())
register_class_kind(QuietKind())
