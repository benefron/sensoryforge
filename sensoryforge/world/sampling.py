"""Sampling a world: draws, fixed draws and sessions (spec §4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    weights: Optional[Mapping[str, float]] = None,
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
        weights: ``class -> weight`` overriding the classes' own weights: the
            classes are the mapping's keys (held-out ones allowed), weighed
            in order of their names, renormalised (equal if they sum to 0).
            ``None`` runs the declared weights. Not combined with ``classes``.

    Returns:
        One :class:`Draw` per index, in order.

    Raises:
        ValueError: For both or neither of ``n`` and ``indices``, an empty
            or repeating ``classes``, an unknown class, or both ``classes``
            and ``weights``.
    """
    if (n is None) == (indices is None):
        raise ValueError("give exactly one of n or indices")
    if weights is not None and classes is not None:
        raise ValueError("give classes or weights, not both")
    idx = (
        np.arange(int(n), dtype=np.int64)
        if indices is None
        else np.asarray(list(indices), dtype=np.int64)
    )
    if weights is None:
        pool = class_pool(world, classes)
        class_weights = np.array([c.weight for c in pool], dtype=np.float64)
    else:
        pool = class_pool(world, list(weights))
        class_weights = np.array(
            [float(weights[c.name]) for c in pool], dtype=np.float64
        )
        if (class_weights < 0).any() or not np.isfinite(class_weights).all():
            raise ValueError("weights must be finite numbers >= 0")
    seeds = rng.draw_seeds(seed, idx)
    if class_weights.sum() <= 0:
        class_weights = np.ones(len(pool))
    cum = np.cumsum(class_weights) / class_weights.sum()
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
            Quiet gaps are the holes between items (they render exactly 0).
        session_type: The type drawn (a world with ``sessions: types:`` only).
        contact_fraction: The declared target share in contact (a world with
            ``sessions:`` only).
    """

    world: World = field(repr=False, compare=False)
    seed: int
    index: int
    session_seed: int
    duration_ms: float
    items: Tuple[Tuple[float, Draw], ...]
    session_type: Optional[str] = None
    contact_fraction: Optional[float] = None

    @property
    def end_ms(self) -> float:
        """The session's length, ms."""
        return self.duration_ms

    @property
    def truncated(self) -> bool:
        """True when the last draw is cut at ``duration_ms``."""
        if not self.items:
            return False
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
        out: Dict[str, Any] = {
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
        if self.session_type is not None:
            out["session_type"] = self.session_type
        if self.contact_fraction is not None:
            out["contact_fraction"] = self.contact_fraction
        return out

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
            session_type=data.get("session_type"),
            contact_fraction=(
                None
                if data.get("contact_fraction") is None
                else float(data["contact_fraction"])
            ),
        )


def session(
    world: World,
    duration_ms: Optional[float] = None,
    seed: int = 0,
    index: int = 0,
) -> Session:
    """A session: draws laid end to end over ``duration_ms``.

    A world without a ``sessions:`` section lays draws end to end until
    ``duration_ms`` is filled (spec §4.5). With one, the session is budgeted
    (see :func:`_model_session`): its length and target contact fraction come
    from the section's axes, episodes come from its type's class mix until
    their contact time reaches ``contact_fraction * duration_ms``, and the
    rest of the length is spent as quiet gaps.

    Args:
        world: The world.
        duration_ms: The session's length, ms. Required for a world without
            ``sessions:``; with one, ``None`` draws it from the section's
            ``duration_ms`` axis and a number overrides that.
        seed: The sampling seed.
        index: The session's index under that seed.

    Raises:
        ValueError: For a length <= 0 or missing, or a draw of zero length
            (the session would never fill).
    """
    if duration_ms is not None and duration_ms <= 0:
        raise ValueError(f"duration_ms must be > 0, got {duration_ms}")
    if world.sessions is not None:
        return _model_session(world, duration_ms, seed, index)
    if duration_ms is None:
        raise ValueError(
            "duration_ms is required: this world declares no sessions: section "
            "(world.sessions.duration_ms)"
        )
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


def _contact_ms(draw: Draw) -> float:
    return sum(b - a for phase, a, b in draw.timeline if phase in _CONTACT_PHASES)


def _model_session(
    world: World, duration_ms: Optional[float], seed: int, index: int
) -> Session:
    """A budgeted session of a world that declares ``sessions:``.

    With ``s = seed53(seed, index)``: the type comes from its weights
    (``uniforms([s], "session_type")``), the length ``D`` and the target
    contact fraction ``f`` from their axes (slots ``duration_ms`` and
    ``contact_fraction``). Episodes ``k = 0, 1, ...`` are
    ``sample(indices=[k], seed=s, weights=<the type's classes>)`` until their
    contact time reaches ``f * D`` or their total length reaches ``D``. The
    quiet budget ``Q = max(0, D - E)`` is spent as ``G = min(n + 1,
    max(1, round(Q / gap_mean_ms)))`` gaps at distinct episode boundaries,
    their lengths the spacings of ``Q`` cut at sorted uniforms.
    """
    model = world.sessions
    assert model is not None
    session_seed = rng.seed53(seed, index)
    one = np.array([session_seed], dtype=np.uint64)
    session_type: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    if model.types:
        names = sorted(model.types)
        type_weights = np.array([model.types[n]["weight"] for n in names], dtype=float)
        cum = np.cumsum(type_weights) / type_weights.sum()
        pick = int(
            min(
                np.searchsorted(
                    cum, rng.uniforms(one, "session_type")[0], side="right"
                ),
                len(names) - 1,
            )
        )
        session_type = names[pick]
        weights = dict(model.types[session_type]["classes"])
    total = (
        float(model.duration_ms.sample(rng.uniforms(one, "duration_ms"))[0])
        if duration_ms is None
        else float(duration_ms)
    )
    target = float(
        model.contact_fraction.sample(rng.uniforms(one, "contact_fraction"))[0]
    )
    episodes: List[Draw] = []
    contact = 0.0
    length = 0.0
    k = 0
    while length < total and contact < target * total:
        draw = sample(world, indices=[k], seed=session_seed, weights=weights)[0]
        if draw.end_ms <= 0:
            raise ValueError(
                f"session draw {k} (class {draw.class_name!r}) has zero length; "
                "give its class a positive duration"
            )
        episodes.append(draw)
        contact += _contact_ms(draw)
        length += draw.end_ms
        k += 1
    n = len(episodes)
    quiet = max(0.0, total - length)
    gap_at: Dict[int, float] = {}
    if quiet > 0:
        count = min(n + 1, max(1, int(round(quiet / model.gap_mean_ms))))
        boundaries = sorted(
            int(b)
            for b in rng.permutation(n + 1, session_seed, "gap_boundaries")[:count]
        )
        cuts = np.sort(
            rng.uniforms(rng.draw_seeds(session_seed, np.arange(count - 1)), "gap_cut")
        )
        edges = quiet * np.concatenate([[0.0], cuts, [1.0]])
        for boundary, gap in zip(boundaries, np.diff(edges).tolist()):
            gap_at[boundary] = gap
    items: List[Tuple[float, Draw]] = []
    t = 0.0
    for j in range(n):
        t += gap_at.get(j, 0.0)
        items.append((t, episodes[j]))
        t += episodes[j].end_ms
    return Session(
        world=world,
        seed=int(seed),
        index=int(index),
        session_seed=int(session_seed),
        duration_ms=total,
        items=tuple(items),
        session_type=session_type,
        contact_fraction=target,
    )
