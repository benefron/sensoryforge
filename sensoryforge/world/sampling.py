"""Sampling a world: draws, fixed draws and sessions (spec §4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sensoryforge.world import rng
from sensoryforge.world.distributions import fill_links
from sensoryforge.world.schema import ClassSpec, World

_CONTACT_PHASES = {"touch", "hold", "slide", "release"}


@dataclass(frozen=True)
class Draw:
    """One draw from a world: a class and a value for every axis it binds.

    Attributes:
        world: The world it belongs to (not part of its record).
        class_name: The class drawn.
        values: Every bound axis name -> its value, constants included.
        seed: The sampling seed (``None`` for a fixed draw).
        index: The draw's index under that seed.
        draw_seed: ``H(seed, index)``; every random choice derives from it.
        sampling: ``declared``, ``stratified``, ``probe`` or ``fixed``.
        out_of_range: Axes set outside their declared range.
    """

    world: World = field(repr=False, compare=False)
    class_name: str
    values: Dict[str, Any]
    seed: Optional[int] = None
    index: Optional[int] = None
    draw_seed: Optional[int] = None
    sampling: str = "declared"
    out_of_range: Tuple[str, ...] = ()

    @property
    def spec(self) -> ClassSpec:
        """The draw's class."""
        return self.world.class_spec(self.class_name)

    @property
    def end_ms(self) -> float:
        """When the draw ends (the last release; ``quiet_ms`` for quiet), ms."""
        return float(self.spec.kind_obj.end_ms(self.values))

    @property
    def timeline(self) -> List[List[Any]]:
        """``[[phase, start_ms, end_ms], ...]`` from time 0, the entry's start."""
        return self.spec.kind_obj.timeline(self.values)

    def to_layer(self) -> Optional[Dict[str, Any]]:
        """The draw as an ordinary layered layer dict (``None`` for quiet)."""
        return self.spec.kind_obj.to_layer(self.spec, self.values)

    def to_dict(self) -> Dict[str, Any]:
        """The JSON-ready record (spec §4.4)."""
        return {
            "world_id": self.world.world_id,
            "seed": self.seed,
            "index": self.index,
            "draw_seed": self.draw_seed,
            "class": self.class_name,
            "sampling": self.sampling,
            "values": dict(sorted(self.values.items())),
            "timeline": self.timeline,
            "end_ms": self.end_ms,
            "out_of_range": list(self.out_of_range),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], world: World) -> "Draw":
        """Rebuild a draw from its record and the world it came from."""
        if data.get("world_id") != world.world_id:
            raise ValueError(
                f"this draw belongs to world {data.get('world_id')!r}, "
                f"not {world.world_id!r}"
            )
        spec = world.class_spec(data["class"])
        missing = set(spec.axes) - set(data["values"])
        if missing:
            raise ValueError(f"draw record lacks values for {sorted(missing)}")
        return cls(
            world=world,
            class_name=data["class"],
            values=dict(data["values"]),
            seed=data.get("seed"),
            index=data.get("index"),
            draw_seed=data.get("draw_seed"),
            sampling=data.get("sampling", "declared"),
            out_of_range=tuple(data.get("out_of_range", ())),
        )


def class_pool(
    world: World, classes: Optional[Sequence[str]] = None
) -> List[ClassSpec]:
    """The classes declared sampling weighs, sorted by name.

    Args:
        world: The world.
        classes: Class or held-out class names, each once; ``None`` means
            every class (held-out ones excluded).

    Raises:
        ValueError: For an empty list, a name given twice, or an unknown name.
    """
    if classes is None:
        return [world.classes[name] for name in sorted(world.classes)]
    names = [str(c) for c in classes]
    if not names:
        raise ValueError("no classes to sample from")
    seen = set()
    for name in names:
        if name in seen:
            raise ValueError(f"classes: {name!r} is named more than once")
        seen.add(name)
    return [world.class_spec(name) for name in sorted(names)]


def sample(
    world: World,
    n: Optional[int] = None,
    seed: int = 0,
    *,
    indices: Optional[Iterable[int]] = None,
    classes: Optional[Sequence[str]] = None,
) -> List[Draw]:
    """Draws from the world's declared distribution (spec §4.2).

    Draw ``i`` depends only on ``(world, seed, i)`` and on the *set* of
    ``classes``: ask for ``indices=[i]`` to regenerate it alone. Classes are
    weighed in order of their names, so neither the order a world writes
    its classes in nor the order ``classes`` lists them changes a draw.

    Args:
        world: The world.
        n: How many draws (indices ``0 .. n-1``); or give ``indices``.
        seed: The sampling seed.
        indices: Which draw indices to produce.
        classes: Restrict to these classes (held-out ones allowed), each
            named once; their weights are renormalised (equal if they sum
            to 0). ``None`` means every (not held-out) class.

    Returns:
        One :class:`Draw` per index, in order.

    Raises:
        ValueError: For both or neither of ``n`` and ``indices``, an empty
            or repeating ``classes``, or an unknown class.
    """
    if (n is None) == (indices is None):
        raise ValueError("give exactly one of n or indices")
    idx = (
        np.arange(int(n), dtype=np.int64)
        if indices is None
        else np.asarray(list(indices), dtype=np.int64)
    )
    pool = class_pool(world, classes)
    seeds = rng.draw_seeds(seed, idx)
    weights = np.array([c.weight for c in pool], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(pool))
    cum = np.cumsum(weights) / weights.sum()
    chosen = np.minimum(
        np.searchsorted(cum, rng.uniforms(seeds, "class"), side="right"), len(pool) - 1
    )
    draws: List[Optional[Draw]] = [None] * idx.size
    for ci, cls in enumerate(pool):
        rows = np.nonzero(chosen == ci)[0]
        if rows.size == 0:
            continue
        sub = seeds[rows]
        columns = {
            name: (
                axis.sample(rng.uniforms(sub, name))
                if axis.is_random
                else [axis.value] * rows.size
            )
            for name, axis in cls.axes.items()
            if axis.form != "link"
        }
        for j, row in enumerate(rows.tolist()):
            values = {name: columns[name][j] for name in columns}
            fill_links(cls.axes, values)
            draws[row] = Draw(
                world=world,
                class_name=cls.name,
                values={name: values[name] for name in cls.axes},
                seed=int(seed),
                index=int(idx[row]),
                draw_seed=int(seeds[row]),
                sampling="declared",
            )
    return draws  # type: ignore[return-value]


def fixed_draw(world: World, name: str) -> Draw:
    """A named fixed draw: given values, every other axis at its midpoint.

    See spec §3.5.
    """
    if name not in world.fixed:
        raise ValueError(f"no fixed draw {name!r}; the world has {sorted(world.fixed)}")
    entry = world.fixed[name]
    spec = world.class_spec(entry["class"])
    values = {
        axis_name: axis.midpoint()
        for axis_name, axis in spec.axes.items()
        if axis.form != "link"
    }
    outside = []
    for axis_name, value in entry["values"].items():
        values[axis_name] = value
        if not spec.axes[axis_name].contains(value):
            outside.append(axis_name)
    fill_links(spec.axes, values)
    return Draw(
        world=world,
        class_name=spec.name,
        values=values,
        sampling="fixed",
        out_of_range=tuple(sorted(outside)),
    )


@dataclass(frozen=True)
class Session:
    """Draws laid end to end over ``duration_ms`` (spec §4.5).

    Attributes:
        world: The world (not part of the record).
        seed: The sampling seed.
        index: The session's index under that seed.
        session_seed: ``H(seed, index)``; draw ``k`` is sampled with
            ``sample(world, indices=[k], seed=session_seed)``.
        duration_ms: The session's length.
        items: ``((start_ms, draw), ...)``; the last draw may run past the end.
    """

    world: World = field(repr=False, compare=False)
    seed: int
    index: int
    session_seed: int
    duration_ms: float
    items: Tuple[Tuple[float, Draw], ...]

    @property
    def end_ms(self) -> float:
        """The session's length, ms."""
        return self.duration_ms

    @property
    def truncated(self) -> bool:
        """True when the last draw is cut at ``duration_ms``."""
        start, last = self.items[-1]
        return start + last.end_ms > self.duration_ms

    @property
    def contact_ms(self) -> float:
        """Time in contact (touch, hold, slide, release) within the session, ms."""
        total = 0.0
        for start, draw in self.items:
            for phase, a, b in draw.timeline:
                if phase in _CONTACT_PHASES:
                    lo = min(start + a, self.duration_ms)
                    hi = min(start + b, self.duration_ms)
                    total += max(hi - lo, 0.0)
        return total

    @property
    def quiet_fraction(self) -> float:
        """The share of the session with nothing touching."""
        return 1.0 - self.contact_ms / self.duration_ms

    def to_dict(self) -> Dict[str, Any]:
        """The JSON-ready record."""
        return {
            "world_id": self.world.world_id,
            "sampling": "session",
            "seed": self.seed,
            "index": self.index,
            "session_seed": self.session_seed,
            "duration_ms": self.duration_ms,
            "items": [[start, draw.to_dict()] for start, draw in self.items],
            "truncated": self.truncated,
            "quiet_fraction": self.quiet_fraction,
            "end_ms": self.end_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], world: World) -> "Session":
        """Rebuild a session from its record and its world."""
        if data.get("world_id") != world.world_id:
            raise ValueError(
                f"this session belongs to world {data.get('world_id')!r}, "
                f"not {world.world_id!r}"
            )
        items = tuple(
            (float(start), Draw.from_dict(d, world)) for start, d in data["items"]
        )
        return cls(
            world=world,
            seed=int(data["seed"]),
            index=int(data["index"]),
            session_seed=int(data["session_seed"]),
            duration_ms=float(data["duration_ms"]),
            items=items,
        )


def session(world: World, duration_ms: float, seed: int, index: int = 0) -> Session:
    """Draws from the world laid end to end until ``duration_ms`` is filled.

    Raises:
        ValueError: If a draw has zero length (the session would never fill).
    """
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be > 0, got {duration_ms}")
    session_seed = rng.seed53(seed, index)
    items: List[Tuple[float, Draw]] = []
    t = 0.0
    k = 0
    while t < duration_ms:
        draw = sample(world, indices=[k], seed=session_seed)[0]
        if draw.end_ms <= 0:
            raise ValueError(
                f"session draw {k} (class {draw.class_name!r}) has zero length; "
                "give its class a positive duration"
            )
        items.append((t, draw))
        t += draw.end_ms
        k += 1
    return Session(
        world=world,
        seed=int(seed),
        index=int(index),
        session_seed=int(session_seed),
        duration_ms=float(duration_ms),
        items=tuple(items),
    )
