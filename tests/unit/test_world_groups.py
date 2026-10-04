"""World ``groups:`` and ``use:``: sugar that resolves into the class's axes."""

import copy

import pytest

from sensoryforge.world.sampling import sample
from sensoryforge.world.schema import World

BASE = {
    "name": "groups",
    "defaults": {"amplitude": {"range": [0.5, 1.0]}},
    "classes": {
        "a": {
            "weight": 1,
            "layer": {"shape": {"kind": "gaussian", "sigma_mm": 0.2}},
            "axes": {"x_mm": {"range": [-0.1, 0.1]}},
        },
        "b": {
            "weight": 1,
            "layer": {"shape": {"kind": "gaussian", "sigma_mm": 0.3}},
            "axes": {"hold_ms": {"range": [10, 20]}},
        },
    },
}
GROUPS = {
    "tap": {
        "touch_ms": {"dist": "uniform", "range": [5.0, 15.0]},
        "hold_ms": {"range": [20, 60]},
        "release_ms": {"same_as": "touch_ms"},
    },
    "where": {"y_mm": {"range": [-0.2, 0.2]}},
    "size": {"sigma_mm": {"range": [0.1, 0.3]}},
}


def _raw(mutate=None):
    raw = copy.deepcopy(BASE)
    raw["groups"] = copy.deepcopy(GROUPS)
    raw["classes"]["a"]["use"] = ["tap"]
    if mutate is not None:
        mutate(raw)
    return {"world": raw}


def test_a_used_group_sets_the_class_axes():
    world = World.from_dict(_raw())
    a = world.classes["a"]
    assert a.axes["touch_ms"].to_dict() == {"dist": "uniform", "range": [5.0, 15.0]}
    assert a.axes["release_ms"].link == "touch_ms"
    assert a.bindings["hold_ms"] == ("episode", "hold_ms")
    assert world.classes["b"].axes["hold_ms"].to_dict() == {
        "dist": "uniform",
        "range": [10.0, 20.0],
    }
    assert world.classes["a"].axes["y_mm"].to_dict() == {"value": 0.0}  # unused
    assert "groups" not in world.to_dict()
    assert "use" not in world.to_dict()["classes"]["a"]


def test_class_axes_override_a_group():
    def over(raw):
        raw["classes"]["a"]["axes"]["hold_ms"] = {"range": [1, 2]}

    world = World.from_dict(_raw(over))
    assert world.classes["a"].axes["hold_ms"].to_dict() == {
        "dist": "uniform",
        "range": [1.0, 2.0],
    }
    assert world.classes["a"].axes["touch_ms"].to_dict() == {
        "dist": "uniform",
        "range": [5.0, 15.0],
    }


def test_a_group_beats_the_world_defaults():
    def over(raw):
        raw["defaults"]["hold_ms"] = {"range": [100, 200]}

    world = World.from_dict(_raw(over))
    assert world.classes["a"].axes["hold_ms"].to_dict() == {
        "dist": "uniform",
        "range": [20.0, 60.0],
    }


def test_a_field_the_layer_fixes_wins_over_a_group():
    def fix(raw):
        raw["classes"]["a"]["use"] = ["tap", "size"]

    world = World.from_dict(_raw(fix))
    assert "sigma_mm" not in world.classes["a"].axes


def test_two_groups_on_one_field_fail():
    def clash(raw):
        raw["groups"]["where"]["hold_ms"] = {"range": [1, 2]}
        raw["classes"]["a"]["use"] = ["tap", "where"]

    with pytest.raises(ValueError, match=r"world\.classes\.a.*tap.*where.*hold_ms"):
        World.from_dict(_raw(clash))


def test_two_group_names_on_one_field_fail():
    def clash(raw):
        raw["groups"]["where"] = {"hold": {"range": [1, 2]}}
        raw["classes"]["a"]["use"] = ["tap", "where"]

    with pytest.raises(ValueError, match="hold_ms"):
        World.from_dict(_raw(clash))


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda r: r["classes"]["a"].update(use=["nope"]),
            r"world\.classes\.a\.use.*'nope'.*tap",
        ),
        (
            lambda r: r["classes"]["a"].update(use="tap"),
            r"world\.classes\.a\.use.*list",
        ),
        (
            lambda r: r["groups"]["tap"].update(sigma_mm_typo={"range": [1, 2]}),
            r"world\.groups\.tap.*sigma_mm_typo",
        ),
        (
            lambda r: r["groups"].update(tap=[1]),
            r"world\.groups\.tap",
        ),
        (
            lambda r: r["groups"]["tap"].update(touch_ms={"range": [5]}),
            r"world\.groups\.tap\.touch_ms",
        ),
    ],
)
def test_an_unknown_group_or_field_fails_naming_the_path(mutate, match):
    with pytest.raises(ValueError, match=match):
        World.from_dict(_raw(mutate))


def test_a_world_written_with_groups_equals_the_same_world_written_out():
    grouped = World.from_dict(_raw())
    plain_raw = _raw()["world"]
    del plain_raw["groups"]
    del plain_raw["classes"]["a"]["use"]
    plain_raw["classes"]["a"]["axes"].update(copy.deepcopy(GROUPS["tap"]))
    plain = World.from_dict({"world": plain_raw})
    assert grouped.world_id == plain.world_id
    assert grouped.to_dict() == plain.to_dict()
    da = [d.to_dict() for d in sample(grouped, 6, seed=3)]
    db = [d.to_dict() for d in sample(plain, 6, seed=3)]
    assert da == db


def test_held_out_classes_use_groups():
    def held(raw):
        raw["held_out"] = {
            "h": {
                "layer": {"shape": {"kind": "gaussian", "sigma_mm": 0.2}},
                "use": ["tap", "where"],
            }
        }

    world = World.from_dict(_raw(held))
    assert "y_mm" in world.held_out["h"].axes
