"""Axis features: ``stratify: false`` and ``same_as`` (P3)."""

import copy

import pytest

from sensoryforge.world import World, sample
from sensoryforge.world.dataset import build_dataset, load_dataset, stratify_class
from sensoryforge.world.schema import load_world

BASE = {
    "defaults": {
        "delay_ms": {"range": [0, 20]},
        "touch_ms": {"range": [5, 15]},
        "hold_ms": {"range": [20, 60]},
        "release_ms": {"range": [5, 15]},
        "amplitude": {"range": [0.5, 1.0]},
    },
    "classes": {
        "press": {
            "layer": {"shape": {"kind": "gaussian", "sigma_mm": 0.2}},
            "axes": {
                "release_ms": {"same_as": "touch_ms"},
                "contacts": {"range": [2, 4], "int": True},
            },
        }
    },
}


def _world(**class_axes):
    data = copy.deepcopy(BASE)
    data["classes"]["press"]["axes"].update(class_axes)
    return World.from_dict({"world": data})


def _dataset(world_dict, splits):
    return load_dataset(
        {
            "dataset": {
                "world": {"world": world_dict},
                "seed": 4,
                "duration_ms": 200,
                "splits": splits,
            }
        }
    )


def test_same_as_copies_in_declared_strata_probes_and_fixed_draws():
    data = copy.deepcopy(BASE)
    data["fixed_draws"] = {"fx": {"class": "press", "touch_ms": 9.0}}
    world = World.from_dict({"world": data})
    for d in sample(world, n=20, seed=1):
        assert d.values["release_ms"] == d.values["touch_ms"]
    fx = world.fixed_draw("fx")
    assert fx.values["release_ms"] == fx.values["touch_ms"] == 9.0
    spec = _dataset(
        data,
        {
            "train": {"n": 6},
            "test": {"stratified": {"bins": 3, "per_bin": 2}},
            "probes": {"per_bin": 3, "bins": 4},
        },
    )
    entries = build_dataset(spec)
    kinds = {e.split for e in entries}
    assert kinds == {"train", "test", "probes"}
    for e in entries:
        v = e.item.values
        assert v["release_ms"] == v["touch_ms"]
        if e.split == "test":
            assert "release_ms" not in e.bins
    probed = [e for e in entries if e.probe and e.probe["axis"] == "touch_ms"]
    assert probed


def test_a_link_to_a_missing_axis_or_to_a_link_fails_at_load():
    with pytest.raises(
        ValueError, match=r"world\.classes\.press\.axes\.release_ms.*nope"
    ):
        _world(release_ms={"same_as": "nope"})
    with pytest.raises(ValueError, match="itself"):
        _world(release_ms={"same_as": "release_ms"})
    with pytest.raises(ValueError, match="link"):
        _world(touch_ms={"same_as": "hold_ms"})
    with pytest.raises(ValueError, match="same_as"):
        _world(release_ms={"same_as": "touch_ms", "range": [1, 2]})


def test_a_link_whose_target_leaves_its_field_domain_fails():
    # contacts is a whole number >= 1; touch_ms is a float range
    with pytest.raises(ValueError, match="world.classes.press.axes.contacts"):
        _world(contacts={"same_as": "touch_ms"})
    # a whole-number target that can be 0, linked to contacts (>= 1)
    with pytest.raises(ValueError, match="world.classes.press.axes.contacts"):
        _world(hold_ms={"range": [0, 3], "int": True}, contacts={"same_as": "hold_ms"})


def test_a_fixed_draw_may_not_set_a_link():
    data = copy.deepcopy(BASE)
    data["fixed_draws"] = {"fx": {"class": "press", "release_ms": 5.0}}
    with pytest.raises(ValueError, match="link"):
        World.from_dict({"world": data})


def test_stratify_false_samples_iid_with_no_label():
    world = _world(hold_ms={"range": [20, 60], "stratify": False})
    cls = world.classes["press"]
    draws, labels = stratify_class(world, cls, 4, 3, 77)
    assert all("hold_ms" not in label for label in labels)
    assert all("touch_ms" in label for label in labels)
    vals = [d.values["hold_ms"] for d in draws]
    assert len(set(vals)) == len(vals)
    assert all(20 <= v <= 60 for v in vals)
    again, _ = stratify_class(world, cls, 4, 3, 77)
    assert [d.values["hold_ms"] for d in again] == vals


def test_a_large_int_seed_axis_with_stratify_false_builds_a_test_split():
    with pytest.raises(ValueError, match="at most"):
        w = _world(contacts={"range": [1, 1000], "int": True})
        stratify_class(w, w.classes["press"], 4, 2, 5)
    world = _world(contacts={"range": [1, 1000], "int": True, "stratify": False})
    draws, labels = stratify_class(world, world.classes["press"], 4, 2, 5)
    assert len(draws) == 8
    assert all("contacts" not in label for label in labels)
    assert all(isinstance(d.values["contacts"], int) for d in draws)


def test_the_new_keys_change_the_world_id_only_when_used():
    plain_data = copy.deepcopy(BASE)
    plain_data["classes"]["press"]["axes"].pop("release_ms")
    a = World.from_dict({"world": plain_data})
    again = World.from_dict({"world": copy.deepcopy(plain_data)})
    assert a.world_id == again.world_id
    axes = a.classes["press"].axes
    assert all("stratify" not in x.to_dict() for x in axes.values())
    on = _world(hold_ms={"range": [20, 60], "stratify": True})
    off = _world(hold_ms={"range": [20, 60], "stratify": False})
    base = _world()
    assert on.world_id == base.world_id
    assert off.world_id != base.world_id
    assert off.classes["press"].axes["hold_ms"].to_dict()["stratify"] is False
    assert base.classes["press"].axes["release_ms"].to_dict() == {"same_as": "touch_ms"}
    assert load_world(base).world_id == base.world_id
