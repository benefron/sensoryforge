"""The tactile declared adapter (roadmap A3, report contract section 8.3).

Reads the world a data set was built from (``dataset.json`` carries it) and, for each
manifest row, the values its declaration implies for the metric ids the audit also
measures. A value is written only where the declaration determines it; every other
case is left out with a reason (``modulated``, ``several shapes``, ``motion`` ...),
which the stage counts in ``declared.json``.

The contact intervals are the draw's ``timeline`` phases in the contact set (touch,
hold, slide, release), clipped to the entry's duration and merged where they touch.
For a session the same rule is applied to each item at its start time. Times are the
unrounded milliseconds of the world's timeline; the audit samples frames at ``k * dt``,
so a raw difference of up to about ``dt`` is expected and is not an error.

This module is outside the audit's import wall (it imports the world package). The
tactile *metrics* are in ``sensoryforge/evaluate/tactile/metrics.py``, inside it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from sensoryforge.evaluate.audit.units import QUANTITIES, unit_of
from sensoryforge.evaluate.declared.base import Declaration, DeclaredAdapter
from sensoryforge.evaluate.registry import DECLARED_ADAPTER_REGISTRY
from sensoryforge.world import kernel
from sensoryforge.world.sampling import Draw, Session
from sensoryforge.world.schema import FORMAT as WORLD_FORMAT
from sensoryforge.world.schema import ClassSpec, World

_CONTACT = ("touch", "hold", "slide", "release")
_SOURCE = "report_contract.md section 8.3; world_contract.md section 2"

#: Shapes whose kernel peaks at 1, so a declared peak is ``amplitude``. Any other shape
#: is left out ("shape not known"): the adapter does not guess a kernel's peak.
_UNIT_PEAK_SHAPES = ("gaussian", "disc", "bar", "grating")

#: Shapes that are also symmetric about their centre and finite, so a declared centroid
#: is the centre (a bar only when it has a length).
_CENTRED_SHAPES = ("gaussian", "disc", "bar")

#: ``(id, unit expression, basis)`` in catalogue order.
_CATALOGUE: Tuple[Tuple[str, str, str], ...] = (
    ("core.frames.duration", "time", "manifest"),
    ("core.frames.onset_time", "time", "timeline"),
    ("core.frames.offset_time", "time", "timeline"),
    ("core.frames.active_time", "time", "timeline"),
    ("core.frames.active_fraction", "1", "timeline"),
    ("core.frames.bouts", "count", "timeline"),
    ("core.frames.gap_mean", "time", "timeline"),
    ("core.frames.gap_max", "time", "timeline"),
    ("core.frames.peak", "frame", "shape peak"),
    ("core.frames.centroid_x", "space", "position"),
    ("core.frames.centroid_y", "space", "position"),
    ("core.frames.direction", "deg", "motion"),
    ("tactile.motion.speed", "space/time", "motion"),
    ("core.frames.dominant_frequency", "1/time", "modulation"),
    ("tactile.episode.rise_time", "time", "timeline"),
    ("tactile.episode.plateau_time", "time", "timeline"),
    ("tactile.episode.fall_time", "time", "timeline"),
)

#: The timeline metrics that a modulation reaching 0 inside a contact makes unknowable
#: (onset stays: the first tap starts the contact).
_TIMELINE_EXTENT = (
    "offset_time",
    "active_time",
    "active_fraction",
    "bouts",
    "gap_mean",
    "gap_max",
)

#: A declared number, or the reason it is left out.
Outcome = Union[float, int, str]


class _Omit(str):
    """A reason a value is left out (a ``str`` subclass, so it cannot be a value)."""


def _merge(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sort intervals and merge those that touch or overlap."""
    merged: List[List[float]] = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(a, b) for a, b in merged]


def contact_intervals(
    items: Sequence[Tuple[float, Draw]], duration_ms: float
) -> List[Tuple[float, float]]:
    """The merged contact intervals of a set of draws, clipped to the duration.

    Args:
        items: ``(start_ms, draw)`` pairs: one draw at 0, or a session's items.
        duration_ms: The entry's duration in ms.

    Returns:
        Sorted ``(start_ms, end_ms)`` pairs, each of positive length, in ms from the
        entry's start.
    """
    pieces: List[Tuple[float, float]] = []
    for start, draw in items:
        for phase, a, b in draw.timeline:
            if phase not in _CONTACT:
                continue
            lo, hi = min(start + a, duration_ms), min(start + b, duration_ms)
            if hi > lo:
                pieces.append((lo, hi))
    return _merge(pieces)


def _single_element(layer: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    """``(x, y, weight)`` of the pattern's only element in mm, else ``None``."""
    pattern = layer["pattern"]
    registered = kernel.PATTERN_KINDS[pattern["kind"]]
    params = {k: v for k, v in pattern.items() if k not in ("kind", "x_mm", "y_mm")}
    points, weights = registered.fn(params)
    if len(points) != 1:
        return None
    px, py = points[0]
    if registered.placement:
        px, py = px + float(pattern.get("x_mm", 0.0)), py + float(
            pattern.get("y_mm", 0.0)
        )
    return float(px), float(py), float(weights[0])


def _peak_problem(layer: Mapping[str, Any]) -> Optional[str]:
    """Why the shape does not fix a peak of ``amplitude``, else ``None``."""
    kind = layer["shape"]["kind"]
    if kind in _UNIT_PEAK_SHAPES:
        return None
    return "indenter" if kernel.SHAPE_KINDS[kind].indenter else "shape not known"


def _shape_problem(layer: Mapping[str, Any]) -> Optional[str]:
    """Why the shape does not fix a centroid at its centre, else ``None``."""
    shape = layer["shape"]
    registered = kernel.SHAPE_KINDS[shape["kind"]]
    if registered.unbounded or (
        shape["kind"] == "bar" and float(shape.get("length_mm", 0.0)) <= 0.0
    ):
        return "unbounded shape"
    if shape["kind"] not in _CENTRED_SHAPES:
        return "indenter" if registered.indenter else "shape not known"
    return None


def _reaches_zero(layer: Mapping[str, Any]) -> bool:
    """True when the draw's modulation takes the frames to exactly 0 inside a contact.

    A sine or a pulse train with ``depth`` 1 does (``1 - depth * ...`` reaches 0); any
    other modulation is not known to, so the timeline basis is left alone for it.
    """
    mod = layer["modulation"]
    return mod["kind"] in ("sine", "pulses") and float(mod.get("depth", 1.0)) >= 1.0


def _moves(layer: Mapping[str, Any], slide: float) -> bool:
    """True when the draw's motion carries the pattern during an in-duration slide."""
    motion = layer["motion"]
    if slide <= 0.0:
        return False
    if motion["kind"] == "linear":
        a, b = motion["start"], motion["end"]
        return math.hypot(b[0] - a[0], b[1] - a[1]) > 0.0
    return True


class TactileAdapter(DeclaredAdapter):
    """Declares what a tactile world implies for each entry's metrics.

    Contact timing comes from the draw's timeline; peak, centroid, direction, speed and
    frequency come from the layer the draw renders (report contract, section 8.3).
    """

    name = "tactile"
    version = "1"

    @classmethod
    def accepts(cls, dataset_meta: Mapping[str, Any]) -> bool:
        """True for a data set whose world has format ``sensoryforge-world/1``."""
        world = dataset_meta.get("world")
        return isinstance(world, dict) and world.get("format") == WORLD_FORMAT

    def load(self, dataset_meta: Mapping[str, Any]) -> World:
        """The world embedded in ``dataset.json``, checked against its ``world_id``."""
        raw = dataset_meta.get("world")
        if not isinstance(raw, dict):
            raise ValueError("dataset.json holds no world")
        # ``World.to_dict()`` adds a ``format`` key that the parser does not take.
        world = World.from_dict({k: v for k, v in raw.items() if k != "format"})
        if world.world_id != dataset_meta.get("world_id"):
            raise ValueError(
                f"dataset.json's world hashes to {world.world_id!r}, but it names "
                f"world_id {dataset_meta.get('world_id')!r}"
            )
        return world

    def describe(self, dataset_meta: Mapping[str, Any], world: World) -> Dict[str, str]:
        """The data set, world and world-format identifiers."""
        return {
            "dataset_id": str(dataset_meta.get("dataset_id", "")),
            "world_id": world.world_id,
            "world_format": WORLD_FORMAT,
        }

    def channels(self, world: World) -> int:
        """The world's channel count."""
        return len(world.channels)

    def units(self, world: World) -> Dict[str, Optional[Dict[str, str]]]:
        """Time and space as the world names them; every other quantity is ``None``."""
        out: Dict[str, Optional[Dict[str, str]]] = {q: None for q in QUANTITIES}
        for quantity in ("time", "space"):
            if world.units.get(quantity):
                out[quantity] = {
                    "symbol": world.units[quantity],
                    "origin": f"pack:{self.name}",
                    "source": f"world.units.{quantity}",
                }
        return out

    def metrics(
        self, units: Mapping[str, Optional[Mapping[str, str]]]
    ) -> List[Dict[str, Any]]:
        """The seventeen metrics section 8.3 declares, with the audit's units."""
        return [
            {
                "id": metric_id,
                "unit": unit_of(expression, units),
                "basis": basis,
                "source": _SOURCE,
            }
            for metric_id, expression, basis in _CATALOGUE
        ]

    # ------------------------------------------------------------------ entries

    def declare(self, row: Mapping[str, Any], world: World) -> Declaration:
        """The values the row's draw (or session) implies, per channel.

        Args:
            row: A manifest row (``draw``, ``duration_ms``, ...).
            world: The loaded world.

        Returns:
            The values in catalogue order and the omissions with their reasons.
        """
        record = row["draw"]
        duration = float(row["duration_ms"])
        is_session = record.get("sampling") == "session"
        if is_session:
            items = [(float(s), d) for s, d in Session.from_dict(record, world).items]
        else:
            items = [(0.0, Draw.from_dict(record, world))]
        multi = len(world.channels) > 1
        found: Dict[Tuple[str, int], Outcome] = {}
        for ci, channel in enumerate(world.channels):
            mine = [(s, d) for s, d in items if d.spec.channel == channel]
            for metric_id, outcome in self._channel(mine, duration, is_session).items():
                found[(metric_id, ci)] = outcome
        declaration = Declaration()
        for metric_id, _expr, _basis in _CATALOGUE:
            for ci in range(len(world.channels)):
                outcome = found[(metric_id, ci)]
                if isinstance(outcome, _Omit):
                    declaration.omitted.append((metric_id, str(outcome)))
                    continue
                value: Dict[str, Any] = {"id": metric_id, "value": outcome}
                if multi:
                    value["channel"] = ci
                declaration.values.append(value)
        return declaration

    def _channel(
        self,
        mine: Sequence[Tuple[float, Draw]],
        duration: float,
        is_session: bool,
    ) -> Dict[str, Outcome]:
        """Every catalogue metric for one channel, a number or an :class:`_Omit`."""
        out: Dict[str, Outcome] = {"core.frames.duration": duration}
        spans = contact_intervals(mine, duration)
        total = sum(b - a for a, b in spans)
        gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
        out["core.frames.active_time"] = total
        out["core.frames.active_fraction"] = total / duration
        out["core.frames.bouts"] = len(spans)
        if spans:
            out["core.frames.onset_time"] = spans[0][0]
            out["core.frames.offset_time"] = spans[-1][1]
        else:
            out["core.frames.onset_time"] = _Omit("no contact")
            out["core.frames.offset_time"] = _Omit("no contact")
        for key, stat in (("gap_mean", lambda g: sum(g) / len(g)), ("gap_max", max)):
            out[f"core.frames.{key}"] = (
                stat(gaps) if gaps else _Omit("fewer than two bouts")
            )
        # Frames that go to 0 inside a contact cut it into bouts the timeline cannot see.
        if any(
            layer is not None and _reaches_zero(layer)
            for layer in (d.to_layer() for _s, d in mine)
        ):
            for key in _TIMELINE_EXTENT:
                out[f"core.frames.{key}"] = _Omit("modulation reaches 0")
        rest = [i for i, _e, _b in _CATALOGUE if i not in out]
        if is_session:
            reason = "session"
        elif not mine or not spans:
            reason = "no contact"
        else:
            reason = ""
        if reason:
            out.update({i: _Omit(reason) for i in rest})
            return out
        draw = mine[0][1]
        layer = draw.to_layer()
        if layer is None:  # a quiet draw has spans == [], handled above
            out.update({i: _Omit("no contact") for i in rest})
            return out
        out.update(self._shape(draw, layer, duration, spans))
        return out

    def _shape(
        self,
        draw: Draw,
        layer: Mapping[str, Any],
        duration: float,
        spans: Sequence[Tuple[float, float]],
    ) -> Dict[str, Outcome]:
        """Peak, centroid, motion, frequency and episode phases of one draw."""
        timing = layer["timing"]
        modulation = layer["modulation"]
        modulated = modulation["kind"] != "none"
        slide_in_duration = any(
            phase == "slide" and a < duration for phase, a, _b in draw.timeline
        )
        out: Dict[str, Outcome] = {}

        # Where the shape sits, and what fixes its level.
        problem = _shape_problem(layer)
        peak_problem = _peak_problem(layer)
        # An unbounded kernel (a grating) is drawn once over the plane, so the pattern's
        # elements do not multiply it.
        unbounded = kernel.SHAPE_KINDS[layer["shape"]["kind"]].unbounded
        element = _single_element(layer)
        background = float(layer.get("background", 0.0))
        first_touch_end = next(
            (b for phase, _a, b in draw.timeline if phase == "touch"), 0.0
        )

        if modulated:
            out["core.frames.peak"] = _Omit("modulated")
        elif peak_problem is not None:
            out["core.frames.peak"] = _Omit(peak_problem)
        elif element is None and not unbounded:
            out["core.frames.peak"] = _Omit("several shapes")
        elif first_touch_end > duration:
            out["core.frames.peak"] = _Omit("peak not reached")
        else:
            weight = 1.0 if unbounded or element is None else element[2]
            out["core.frames.peak"] = (
                float(layer["shape"]["amplitude"]) * weight + background
            )

        moves = _moves(layer, float(timing["slide_ms"])) and slide_in_duration
        position_reason: Optional[str] = None
        if problem is not None:
            position_reason = problem
        elif moves:
            position_reason = "motion"
        elif element is None:
            position_reason = "several shapes"
        elif background != 0.0:
            position_reason = "background"
        for axis, k in (("x", 0), ("y", 1)):
            out[f"core.frames.centroid_{axis}"] = (
                _Omit(position_reason)
                if position_reason
                else element[k]  # type: ignore[index]
            )

        # Motion: direction and speed of the centroid during the slide.
        motion_reason: Optional[str] = None
        if problem is not None:
            motion_reason = problem
        elif not moves:
            motion_reason = "no motion"
        elif layer["motion"]["kind"] != "linear":
            motion_reason = "motion not linear"
        elif background != 0.0:
            motion_reason = "background"
        if motion_reason:
            out["core.frames.direction"] = _Omit(motion_reason)
            out["tactile.motion.speed"] = _Omit(motion_reason)
        else:
            direction, speed = self._velocity(draw, layer)
            out["core.frames.direction"] = direction
            out["tactile.motion.speed"] = speed

        # Modulation frequency, per ms.
        if not modulated:
            out["core.frames.dominant_frequency"] = _Omit("no modulation")
        elif modulation["kind"] == "sine":
            out["core.frames.dominant_frequency"] = (
                float(modulation["frequency_hz"]) / 1000.0
            )
        elif modulation["kind"] == "pulses":
            out["core.frames.dominant_frequency"] = (
                float(modulation["rate_hz"]) / 1000.0
            )
        else:
            out["core.frames.dominant_frequency"] = _Omit("modulation not known")

        # Episode phases, as the touch, hold + slide and release lengths.
        phase_reason: Optional[str] = None
        if modulated:
            phase_reason = "modulated"
        elif moves:  # the envelope of a moving shape is not flat across the canvas
            phase_reason = "motion"
        elif draw.end_ms > duration:
            phase_reason = "truncated"
        elif len(spans) != int(timing["contacts"]):
            phase_reason = "contacts merge"
        if phase_reason:
            for name in ("rise", "plateau", "fall"):
                out[f"tactile.episode.{name}_time"] = _Omit(phase_reason)
        else:
            out["tactile.episode.rise_time"] = float(timing["ramp_up_ms"])
            out["tactile.episode.plateau_time"] = float(timing["hold_ms"]) + float(
                timing["slide_ms"]
            )
            out["tactile.episode.fall_time"] = float(timing["ramp_down_ms"])
        return out

    @staticmethod
    def _velocity(draw: Draw, layer: Mapping[str, Any]) -> Tuple[float, float]:
        """``(direction_deg, speed)`` of a linear slide.

        A class without its own motion slides at the draw's ``speed_mm_per_ms`` towards
        its ``direction_deg``; a class that declares a linear motion slides along it,
        its travel spread over every contact's slide.
        """
        values = draw.values
        if (draw.spec.layer or {}).get("motion") is None:
            return float(values["direction_deg"]) % 360.0, float(
                values["speed_mm_per_ms"]
            )
        motion = layer["motion"]
        dx = motion["end"][0] - motion["start"][0]
        dy = motion["end"][1] - motion["start"][1]
        timing = layer["timing"]
        slide = float(timing["slide_ms"]) * int(timing["contacts"])
        return math.degrees(math.atan2(dy, dx)) % 360.0, math.hypot(dx, dy) / slide

    # ------------------------------------------------------------------- bounds

    def bounds(self, world: World) -> List[Dict[str, Any]]:
        """Ranges a class's axes allow for onset, peak and centroid (display only)."""
        out: List[Dict[str, Any]] = []
        for label in sorted({*world.classes, *world.held_out}):
            spec = world.class_spec(label)
            if spec.layer is None:
                continue
            delay = self._range(spec, None, "delay_ms")
            if delay is not None:
                out.append(
                    self._bound(label, "core.frames.onset_time", delay, "delay_ms")
                )
            out += self._level_bounds(label, spec)
        return out

    @staticmethod
    def _bound(
        label: str, metric_id: str, span: Tuple[float, float], source: str
    ) -> Dict[str, Any]:
        return {
            "grouping": "class",
            "label": label,
            "id": metric_id,
            "lo": float(span[0]),
            "hi": float(span[1]),
            "source": f"axis {source}",
        }

    @staticmethod
    def _range(
        spec: ClassSpec, part: Optional[str], field: str
    ) -> Optional[Tuple[float, float]]:
        """The ``(lo, hi)`` a layer field (or episode axis) can take, else ``None``."""
        if part is None:
            name = field if field in spec.axes else None
        else:
            name = next(
                (n for n, ref in spec.bindings.items() if ref == (part, field)), None
            )
        if name is not None:
            values = spec.axes[name].field_values()
            lo, hi = values.smallest(), values.largest()
            return None if lo is None or hi is None else (lo, hi)
        if part is None:
            return None
        const = (spec.layer or {}).get(part, {}).get(field)
        if isinstance(const, (int, float)) and not isinstance(const, bool):
            return float(const), float(const)
        return None

    def _level_bounds(self, label: str, spec: ClassSpec) -> List[Dict[str, Any]]:
        """Peak and centroid ranges of a class, when its layer fixes them."""
        layer = spec.layer or {}
        if layer.get("modulation", {}).get("kind", "none") != "none":
            return []
        element = _single_element(layer)
        unbounded = kernel.SHAPE_KINDS[layer["shape"]["kind"]].unbounded
        out: List[Dict[str, Any]] = []
        amp = self._range(spec, "shape", "amplitude")
        bg = self._range(spec, "layer", "background") or (0.0, 0.0)
        if _peak_problem(layer) is None and (element is not None or unbounded) and amp:
            weight = 1.0 if unbounded or element is None else element[2]
            out.append(
                self._bound(
                    label,
                    "core.frames.peak",
                    (weight * amp[0] + bg[0], weight * amp[1] + bg[1]),
                    "amplitude",
                )
            )
        slide = self._range(spec, None, "slide_ms")
        if (
            element is not None
            and _shape_problem(layer) is None
            and slide is not None
            and slide[1] == 0.0
            and bg == (0.0, 0.0)
        ):
            for axis, k in (("x", 0), ("y", 1)):
                span = self._range(spec, "pattern", f"{axis}_mm")
                if span is not None:
                    out.append(
                        self._bound(
                            label,
                            f"core.frames.centroid_{axis}",
                            (span[0] + element[k], span[1] + element[k]),
                            f"{axis}_mm",
                        )
                    )
        return out


DECLARED_ADAPTER_REGISTRY.register("tactile", TactileAdapter)
