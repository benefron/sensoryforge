"""Sampling: independence from n, weights, ranges, records, fixed draws, sessions."""

import json
from pathlib import Path

import pytest
import yaml

from sensoryforge.world.sampling import Draw, Session, sample, session
from sensoryforge.world.schema import World, load_world

WORLD_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "worlds" / "tactile_small.yml"
)
WORLD = load_world(WORLD_PATH)


def _reversed(value):
    """Every mapping in ``value`` with its keys in reverse order (lists kept)."""
    if isinstance(value, dict):
        return {k: _reversed(value[k]) for k in reversed(list(value))}
    if isinstance(value, list):
        return [_reversed(v) for v in value]
    return value


def test_draw_i_does_not_depend_on_n():
    many = sample(WORLD, n=200, seed=7)
    some = sample(WORLD, indices=[5, 150], seed=7)
    assert many[5].to_dict() == some[0].to_dict()
    assert many[150].to_dict() == some[1].to_dict()
    assert many[5].index == 5 and many[5].seed == 7


def test_class_order_changes_neither_the_id_nor_the_draws():
    raw = yaml.safe_load(WORLD_PATH.read_text())
    flipped = World.from_dict(_reversed(raw))
    assert list(raw["world"]["classes"])[0] == "dots"  # written in another order
    assert flipped.world_id == WORLD.world_id
    want = [d.to_dict() for d in sample(WORLD, n=300, seed=7)]
    assert [d.to_dict() for d in sample(flipped, n=300, seed=7)] == want
    held = sorted(WORLD.held_out) + ["dots", "edges"]
    assert [d.to_dict() for d in sample(flipped, n=40, seed=2, classes=held)] == [
        d.to_dict() for d in sample(WORLD, n=40, seed=2, classes=held[::-1])
    ]
    assert flipped.fixed_draw("braille_H").to_dict() == (
        WORLD.fixed_draw("braille_H").to_dict()
    )


def test_class_lists_must_name_each_class_once():
    with pytest.raises(ValueError, match="no classes to sample from"):
        sample(WORLD, n=3, seed=0, classes=[])
    with pytest.raises(ValueError, match="'dots'"):
        sample(WORLD, n=3, seed=0, classes=["dots", "edges", "dots"])


def test_exactly_one_of_n_and_indices():
    with pytest.raises(ValueError, match="exactly one of n or indices"):
        sample(WORLD, n=3, indices=[1], seed=0)


def test_class_frequencies_follow_the_weights():
    draws = sample(WORLD, n=20000, seed=1)
    total = sum(c.weight for c in WORLD.classes.values())
    for name, cls in WORLD.classes.items():
        share = sum(d.class_name == name for d in draws) / len(draws)
        assert abs(share - cls.weight / total) < 0.015, name


def test_values_lie_in_their_ranges_and_constants_are_kept():
    for d in sample(WORLD, n=500, seed=2):
        assert set(d.values) == set(d.spec.axes)
        for name, axis in d.spec.axes.items():
            assert axis.contains(d.values[name]), (d.class_name, name, d.values[name])


def test_restricting_classes_reaches_held_out_ones():
    draws = sample(WORLD, n=50, seed=3, classes=["gratings"])
    assert {d.class_name for d in draws} == {"gratings"}


def test_records_round_trip_through_json():
    for d in sample(WORLD, n=30, seed=4):
        record = json.loads(json.dumps(d.to_dict()))
        assert Draw.from_dict(record, WORLD).to_dict() == d.to_dict()
        assert record["end_ms"] == pytest.approx(d.end_ms)
        assert record["timeline"][-1][2] == pytest.approx(d.end_ms)


def test_a_record_from_another_world_is_refused():
    record = sample(WORLD, n=1, seed=0)[0].to_dict()
    record["world_id"] = "w-000000000000"
    with pytest.raises(ValueError, match="belongs to world"):
        Draw.from_dict(record, WORLD)


def test_fixed_draws_take_midpoints_and_flag_out_of_range():
    h = WORLD.fixed_draw("braille_H")
    assert h.values["dots"] == "125" and h.values["hold_ms"] == 40
    assert h.values["touch_ms"] == pytest.approx(10.0)  # midpoint of [5, 15]
    assert h.out_of_range == () and h.sampling == "fixed" and h.seed is None
    wide = WORLD.fixed_draw("wide_dot")
    assert wide.out_of_range == ("sigma_mm",)


def test_a_session_lays_draws_end_to_end():
    s = session(WORLD, duration_ms=1000.0, seed=11, index=0)
    assert s.items[0][0] == 0.0
    for (a, da), (b, _) in zip(s.items, s.items[1:]):
        assert b == pytest.approx(a + da.end_ms)
    last_start, last = s.items[-1]
    assert last_start < 1000.0 <= last_start + last.end_ms
    assert 0.0 < s.quiet_fraction < 1.0
    assert s.end_ms == 1000.0


def test_sessions_are_reproducible_and_differ_by_index():
    a = session(WORLD, 500.0, seed=11, index=0)
    b = session(WORLD, 500.0, seed=11, index=0)
    c = session(WORLD, 500.0, seed=11, index=1)
    assert a.to_dict() == b.to_dict() and a.to_dict() != c.to_dict()
    record = json.loads(json.dumps(a.to_dict()))
    assert Session.from_dict(record, WORLD).to_dict() == a.to_dict()


def test_a_zero_length_draw_stops_a_session():
    world = World.from_dict(
        {
            "world": {
                "classes": {"q": {"kind": "quiet", "axes": {"quiet_ms": {"value": 0}}}}
            }
        }
    )
    with pytest.raises(ValueError, match="zero length"):
        session(world, 100.0, seed=0)
