"""The ``sessions:`` model: declared length, contact fraction, gaps and types."""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from sensoryforge.world import (
    Canvas,
    Session,
    build_dataset,
    load_dataset,
    load_world,
    movie_times,
    render_movie,
    sample,
    session,
)
from sensoryforge.world.schema import World

WORLDS = Path(__file__).resolve().parents[1] / "fixtures" / "worlds"
WORLD = load_world(WORLDS / "elements_v1_2.yml")
MODEL = WORLD.sessions
CONTACT = {"touch", "hold", "slide", "release"}
TYPE_CLASSES = {name: set(spec["classes"]) for name, spec in MODEL.types.items()}

BASE = {
    "name": "sess",
    "defaults": {
        "amplitude": {"range": [0.5, 1.0]},
        "touch_ms": {"range": [8, 14]},
        "hold_ms": {"range": [20, 40]},
        "release_ms": {"range": [8, 14]},
    },
    "classes": {
        "a": {"weight": 1, "layer": {"shape": {"kind": "gaussian", "sigma_mm": 0.2}}},
        "b": {"weight": 1, "layer": {"shape": {"kind": "gaussian", "sigma_mm": 0.3}}},
    },
    "held_out": {
        "h": {"layer": {"shape": {"kind": "gaussian", "sigma_mm": 0.4}}},
    },
    "sessions": {
        "duration_ms": {"range": [300, 500]},
        "contact_fraction": {"range": [0.2, 0.5]},
        "gap_mean_ms": 30,
    },
}


def _world(mutate=None):
    raw = copy.deepcopy(BASE)
    if mutate is not None:
        mutate(raw)
    return World.from_dict({"world": raw})


def _gaps(s):
    """The quiet holes between a session's items, ms (zero-length ones left out)."""
    out = []
    t = 0.0
    for start, draw in s.items:
        out.append(start - t)
        t = start + draw.end_ms
    out.append(s.duration_ms - t)
    return [g for g in out if g > 1e-9]


def _digest(obj):
    def rounded(v):
        if isinstance(v, float):
            return float(f"{v:.10g}")
        if isinstance(v, dict):
            return {k: rounded(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [rounded(x) for x in v]
        return v

    text = json.dumps(rounded(obj), sort_keys=True).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def test_a_world_without_sessions_keeps_v1_1_sessions():
    want = json.loads((WORLDS / "v1_1_0_reference" / "reference.json").read_text())[
        "session"
    ]
    old = load_world(WORLDS / f"{want['world']}.yml")
    assert old.sessions is None and "sessions" not in old.to_dict()
    record = session(old, want["duration_ms"], want["seed"], want["index"]).to_dict()
    assert _digest(record) == want["digest"]
    assert "session_type" not in record and "contact_fraction" not in record


def test_session_length_and_target_fraction_come_from_their_axes():
    world = _world(
        lambda raw: raw["sessions"].update(
            duration_ms={"value": 400}, contact_fraction={"value": 0.3}
        )
    )
    s = session(world, seed=3, index=1)
    assert s.duration_ms == 400.0 and s.contact_fraction == 0.3
    assert session(world, 250.0, seed=3, index=1).duration_ms == 250.0
    ranged = [session(WORLD, seed=3, index=i) for i in range(20)]
    lo, hi = MODEL.duration_ms.lo, MODEL.duration_ms.hi
    assert all(lo <= s.duration_ms <= hi for s in ranged)
    assert len({s.duration_ms for s in ranged}) > 10
    flo, fhi = MODEL.contact_fraction.lo, MODEL.contact_fraction.hi
    assert all(flo <= s.contact_fraction <= fhi for s in ranged)
    with pytest.raises(ValueError, match="duration_ms"):
        session(_world(lambda raw: raw.pop("sessions")), seed=1)


def test_the_realised_contact_fraction_reaches_the_target():
    for i in range(50):
        s = session(WORLD, seed=11, index=i)
        longest = max(d.end_ms for _, d in s.items)
        realised = s.contact_ms / s.duration_ms
        target = s.contact_fraction
        assert target - 1e-9 <= realised <= target + longest / s.duration_ms + 1e-9, i
        assert realised == pytest.approx(1.0 - s.quiet_fraction)


def test_gaps_render_exactly_zero():
    canvas = Canvas.from_grid(8, 8, 0.15)
    found_gap = False
    for i in range(4):
        s = session(WORLD, seed=2, index=i)
        times = movie_times(1.0, s.duration_ms)
        frames = render_movie(s, canvas, 1.0, s.duration_ms, dtype=torch.float64)
        contact = torch.zeros(len(times), dtype=torch.bool)
        for start, draw in s.items:
            for phase, a, b in draw.timeline:
                if phase in CONTACT:
                    contact |= (times >= start + a) & (times < start + b)
        assert torch.count_nonzero(frames[~contact]) == 0
        assert torch.count_nonzero(frames[contact]) > 0
        found_gap |= bool(_gaps(s))
    assert found_gap


def test_gap_count_and_mean_follow_gap_mean_ms():
    counts, lengths = [], []
    for i in range(60):
        s = session(WORLD, seed=9, index=i)
        quiet = s.duration_ms - sum(d.end_ms for _, d in s.items)
        if quiet <= 0:
            continue
        gaps = _gaps(s)
        # the budget is spent as round(Q / gap_mean) gaps (at least one), at
        # most one per episode boundary
        want = min(len(s.items) + 1, max(1, round(quiet / MODEL.gap_mean_ms)))
        assert len(gaps) == want, i
        assert sum(gaps) == pytest.approx(quiet)
        counts.append(len(gaps))
        if round(quiet / MODEL.gap_mean_ms) <= len(s.items) + 1:
            # not capped by the number of boundaries: the mean is gap_mean_ms
            lengths.extend(gaps)
        else:
            assert len(gaps) == len(s.items) + 1
    assert len(counts) > 30 and len(lengths) > 30
    assert np.mean(lengths) == pytest.approx(MODEL.gap_mean_ms, rel=0.25)


def test_type_frequencies_follow_their_weights():
    n = 2000
    types = [session(WORLD, 300.0, seed=4, index=i).session_type for i in range(n)]
    weights = {name: spec["weight"] for name, spec in MODEL.types.items()}
    total = sum(weights.values())
    for name, w in weights.items():
        p = w / total
        got = types.count(name)
        sigma = math.sqrt(n * p * (1 - p))
        assert abs(got - n * p) < 4 * sigma, (name, got, n * p)


def test_episodes_come_only_from_their_types_classes():
    seen = set()
    for i in range(80):
        s = session(WORLD, seed=6, index=i)
        seen.add(s.session_type)
        assert {d.class_name for _, d in s.items} <= TYPE_CLASSES[s.session_type]
    assert seen == set(TYPE_CLASSES)


def test_sessions_are_deterministic_and_differ_by_index():
    a = session(WORLD, seed=1, index=3).to_dict()
    assert session(WORLD, seed=1, index=3).to_dict() == a
    assert session(WORLD, seed=1, index=4).to_dict() != a
    assert session(WORLD, seed=2, index=3).to_dict() != a


def test_a_session_record_round_trips():
    s = session(WORLD, seed=8, index=2)
    record = json.loads(json.dumps(s.to_dict()))
    assert record["session_type"] == s.session_type
    assert record["contact_fraction"] == s.contact_fraction
    back = Session.from_dict(record, WORLD)
    assert back.to_dict() == s.to_dict()
    assert back.session_type == s.session_type
    assert back.contact_fraction == s.contact_fraction


def test_a_sessions_split_takes_each_sessions_own_length():
    spec = load_dataset(WORLDS / "dataset_v1_2.yml")
    own = [e for e in build_dataset(spec) if e.split == "sessions"]
    assert len(own) == 3
    for e in own:
        assert e.duration_ms == e.item.duration_ms
    assert len({e.duration_ms for e in own}) == 3
    explicit = [e for e in build_dataset(spec) if e.split == "long_sessions"]
    assert [e.duration_ms for e in explicit] == [250.0, 250.0]
    assert spec.splits[0].to_dict().get("duration_ms") is None
    # without a model a sessions split still needs its own length
    plain = {
        "name": "d",
        "world": {"world": {k: v for k, v in BASE.items() if k != "sessions"}},
        "seed": 1,
        "duration_ms": 100,
        "splits": {"sessions": {"n": 1}},
    }
    with pytest.raises(ValueError, match="splits.sessions: duration_ms must be > 0"):
        load_dataset(plain)


def test_sample_weights_none_is_unchanged_and_weights_reweigh():
    world = _world()
    base = [d.to_dict() for d in sample(world, n=30, seed=5)]
    assert [d.to_dict() for d in sample(world, n=30, seed=5, weights=None)] == base
    only_b = sample(world, n=20, seed=5, weights={"b": 1.0})
    assert {d.class_name for d in only_b} == {"b"}
    mostly_a = sample(world, n=400, seed=5, weights={"a": 9.0, "b": 1.0})
    share = sum(d.class_name == "a" for d in mostly_a) / 400
    assert 0.85 < share < 0.95
    swapped = sample(world, n=30, seed=5, weights={"b": 1.0, "a": 9.0})
    assert [d.to_dict() for d in swapped] == [
        d.to_dict() for d in sample(world, n=30, seed=5, weights={"a": 9.0, "b": 1.0})
    ]
    held = sample(world, n=3, seed=5, weights={"h": 1.0})
    assert {d.class_name for d in held} == {"h"}
    with pytest.raises(ValueError, match="classes or weights"):
        sample(world, n=2, seed=0, classes=["a"], weights={"a": 1.0})


def _bad(mutate):
    def go(raw):
        mutate(raw["sessions"])

    return go


BAD = [
    (_bad(lambda s: s.pop("duration_ms")), r"world\.sessions: needs duration_ms"),
    (_bad(lambda s: s.pop("gap_mean_ms")), r"world\.sessions: needs gap_mean_ms"),
    (_bad(lambda s: s.update(extra=1)), r"world\.sessions: unknown keys \['extra'\]"),
    (
        _bad(lambda s: s.update(duration_ms={"range": [0, 100]})),
        r"world\.sessions\.duration_ms: .*> 0",
    ),
    (
        _bad(lambda s: s.update(contact_fraction={"range": [0.2, 1.5]})),
        r"world\.sessions\.contact_fraction: .*\[0, 1\]",
    ),
    (
        _bad(lambda s: s.update(contact_fraction={"same_as": "duration_ms"})),
        r"world\.sessions\.contact_fraction: .*same_as",
    ),
    (_bad(lambda s: s.update(gap_mean_ms=0)), r"world\.sessions\.gap_mean_ms: .*> 0"),
    (
        _bad(lambda s: s.update(gap_mean_ms="40")),
        r"world\.sessions\.gap_mean_ms: .*number",
    ),
    (
        _bad(lambda s: s.update(types={"t": {"classes": {"zzz": 1}}})),
        r"world\.sessions\.types\.t\.classes: unknown class 'zzz'",
    ),
    (
        _bad(lambda s: s.update(types={"t": {"classes": {"h": 1}}})),
        r"world\.sessions\.types\.t\.classes: 'h' is held out",
    ),
    (
        _bad(lambda s: s.update(types={"t": {"classes": {"a": 0}}})),
        r"world\.sessions\.types\.t\.classes: the weights sum to 0",
    ),
    (
        _bad(lambda s: s.update(types={"t": {"classes": {"a": -1, "b": 2}}})),
        r"world\.sessions\.types\.t\.classes\.a: .*>= 0",
    ),
    (
        _bad(lambda s: s.update(types={"t": {"weight": 0, "classes": {"a": 1}}})),
        r"world\.sessions\.types: the weights sum to 0",
    ),
    (
        _bad(lambda s: s.update(types={"t": {"weight": 1}})),
        r"world\.sessions\.types\.t: needs classes",
    ),
    (_bad(lambda s: s.update(types={})), r"world\.sessions\.types: .*at least one"),
]


@pytest.mark.parametrize("mutate,message", BAD)
def test_bad_session_sections_fail_naming_the_path(mutate, message):
    with pytest.raises(ValueError, match=message):
        _world(mutate)
