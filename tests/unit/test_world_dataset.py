"""Data sets: ids, counts, seeds, strata, probes, truncation, world pin, manifest."""

import copy
import json
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from sensoryforge.world.dataset import (
    build_dataset,
    load_dataset,
    load_manifest,
    skipped_probes,
    write_dataset,
)

WORLDS = Path(__file__).resolve().parents[1] / "fixtures" / "worlds"
SPEC = load_dataset(WORLDS / "dataset_small.yml")
ENTRIES = build_dataset(SPEC)
RAW = yaml.safe_load((WORLDS / "dataset_small.yml").read_text())


def _split(name):
    return [e for e in ENTRIES if e.split == name]


def test_ids_and_counts():
    assert re.fullmatch(r"d-[0-9a-f]{12}", SPEC.dataset_id)
    assert len({e.entry for e in ENTRIES}) == len(ENTRIES)
    assert len(_split("train")) == 12 and {e.repeat for e in _split("train")} == {0, 1}
    assert len(_split("validation")) == 8
    assert {e.noise_repeat for e in _split("validation")} == {0, 1}
    assert len(_split("test")) == len(SPEC.world.classes) * 4
    assert len(_split("held_out")) == 2
    assert len(_split("sessions")) == 1 and len(_split("fixed")) == 1
    assert _split("train")[0].entry == "train/r0/00000"
    assert _split("validation")[1].entry == "validation/r0/00000.n1"
    assert _split("test")[0].entry == "test/braille/0000"  # classes by name
    assert _split("sessions")[0].entry == "sessions/000"
    assert _split("fixed")[0].entry == "fixed/braille_H"


def test_building_is_deterministic():
    again = build_dataset(load_dataset(WORLDS / "dataset_small.yml"))
    assert [e.to_dict() for e in again] == [e.to_dict() for e in ENTRIES]


def test_no_draw_seed_is_in_two_splits_and_noise_seeds_are_unique():
    noise = [e.seeds["noise"] for e in ENTRIES]
    assert len(set(noise)) == len(noise)
    owners = {}
    for e in ENTRIES:
        if e.seeds["draw"] is not None:
            owners.setdefault(e.seeds["draw"], set()).add(e.split)
    assert all(len(splits) == 1 for splits in owners.values())
    first, second = _split("validation")[:2]
    assert first.seeds["draw"] == second.seeds["draw"]
    assert first.seeds["noise"] != second.seeds["noise"]


def test_train_repeats_draw_fresh_stimuli():
    r0 = {e.seeds["draw"] for e in _split("train") if e.repeat == 0}
    r1 = {e.seeds["draw"] for e in _split("train") if e.repeat == 1}
    assert not r0 & r1


def test_every_test_bin_holds_per_bin_draws_per_class():
    for cls in SPEC.world.classes.values():
        rows = [e for e in _split("test") if e.class_name == cls.name]
        assert len(rows) == 4
        for axis in cls.random_axes:
            counts = Counter(e.bins[axis.name] for e in rows)
            if axis.support() is None:
                assert sorted(counts.values()) == [2, 2], (cls.name, axis.name)
                for e in rows:
                    lo, hi = (float(x) for x in e.bins[axis.name][1:-1].split(", "))
                    value = e.item.values[axis.name]
                    assert lo - 1e-9 <= value <= hi + 1e-9
            else:
                assert max(counts.values()) - min(counts.values()) <= 1


def test_probes_are_labelled_and_outside_their_range():
    probes = _split("probes")
    assert probes
    for e in probes:
        name = e.probe["axis"]
        axis = e.item.spec.axes[name]
        value = e.item.values[name]
        assert e.bins == {name: e.probe["side"]}
        assert value < axis.lo if e.probe["side"] == "below" else value > axis.hi
        assert e.item.out_of_range == (name,)
    skipped = skipped_probes(SPEC)
    assert "dots/delay_ms-below" in skipped  # delay_ms starts at 0
    assert not any(e.entry.startswith("probes/dots/delay_ms-below") for e in probes)
    assert not any(e.probe["axis"] == "direction_deg" for e in probes)  # circular


def test_long_draws_are_marked_truncated():
    # 'twice' (two contacts) reaches 215 ms in 120 ms entries.
    twice = [e for e in ENTRIES if e.class_name == "twice"]
    assert any(e.truncated for e in twice)
    for e in twice:
        assert e.truncated == (e.item.end_ms > 120.0)
    assert all(e.duration_ms == 120.0 for e in ENTRIES if e.split != "sessions")
    assert _split("sessions")[0].duration_ms == 300.0


def _reversed(value):
    """Every mapping in ``value`` with its keys in reverse order (lists kept)."""
    if isinstance(value, dict):
        return {k: _reversed(value[k]) for k in reversed(list(value))}
    if isinstance(value, list):
        return [_reversed(v) for v in value]
    return value


def test_class_order_does_not_change_the_entries():
    world = yaml.safe_load((WORLDS / "tactile_small.yml").read_text())
    raw = copy.deepcopy(RAW)
    raw["dataset"]["world"] = _reversed(world)
    raw["dataset"]["world_id"] = SPEC.world.world_id
    flipped = load_dataset(raw, base_dir=WORLDS)
    assert flipped.dataset_id == SPEC.dataset_id
    assert [e.to_dict() for e in build_dataset(flipped)] == [
        e.to_dict() for e in ENTRIES
    ]
    assert skipped_probes(flipped) == skipped_probes(SPEC)


def test_world_pin_mismatch_fails():
    raw = copy.deepcopy(RAW)
    raw["dataset"]["world_id"] = "w-000000000000"
    with pytest.raises(ValueError, match="w-000000000000.*" + SPEC.world.world_id):
        load_dataset(raw, base_dir=WORLDS)
    raw["dataset"]["world_id"] = SPEC.world.world_id
    assert load_dataset(raw, base_dir=WORLDS).dataset_id != ""


def test_manifest_and_dataset_json(tmp_path):
    out = write_dataset(SPEC, ENTRIES, tmp_path / "ds")
    info = json.loads((out / "dataset.json").read_text())
    assert (
        info["dataset_id"] == SPEC.dataset_id
        and info["world_id"] == SPEC.world.world_id
    )
    assert info["n_entries"] == len(ENTRIES) and info["counts"]["test"]["dots"] == 4
    assert info["sensoryforge"]["source"] in ("git", "direct_url", "unknown")
    assert "dots/delay_ms-below" in info["skipped_probes"]
    assert load_manifest(out) == [json.loads(json.dumps(e.to_dict())) for e in ENTRIES]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["splits"].update(colours={"n": 3}), "unknown kind"),
        (lambda d: d["splits"]["train"].update(n=0), "n >= 1"),
        (lambda d: d["splits"]["fixed"].update(draws=["nope"]), "no fixed draw"),
        (
            lambda d: d["splits"]["test"].update(stratified={"bins": 0, "per_bin": 2}),
            "bins",
        ),
        (lambda d: d.pop("seed"), "seed"),
        (lambda d: d["splits"]["train"].update(repeats=0), "repeats"),
    ],
)
def test_invalid_specs_are_named(mutate, message):
    raw = copy.deepcopy(RAW)
    mutate(raw["dataset"])
    with pytest.raises(ValueError, match=message):
        load_dataset(raw, base_dir=WORLDS)


# ------------------------------------------- loading: YAML, mappings, keys, names


def test_a_duplicated_split_key_in_yaml_is_refused(tmp_path):
    path = tmp_path / "d.yml"
    path.write_text(
        "dataset:\n"
        f"  world: {WORLDS / 'tactile_small.yml'}\n"
        "  seed: 1\n"
        "  duration_ms: 50\n"
        "  splits:\n"
        "    train: {n: 2}\n"
        "    train: {n: 3}\n"
    )
    with pytest.raises(ValueError, match="Duplicate key 'train'"):
        load_dataset(path)


def _put(*path, value):
    def mutate(d):
        target = d
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    "mutate, message",
    [
        # Mappings.
        (
            _put("splits", value=["train", "test"]),
            r"dataset\.splits: expected a mapping",
        ),
        (
            _put("splits", "train", value=5),
            r"dataset\.splits\.train: expected a mapping",
        ),
        (
            _put("splits", "test", "stratified", value=3),
            r"dataset\.splits\.test\.stratified: expected a mapping",
        ),
        (
            _put("splits", "held_out", "stratified", value=[2, 1]),
            r"dataset\.splits\.held_out\.stratified: expected a mapping",
        ),
        (
            _put("splits", "fixed", "draws", value="braille_H"),
            r"dataset\.splits\.fixed\.draws: expected a list",
        ),
        # Keys that do not belong to the split's kind.
        (
            _put("splits", "train", value={"n": 3, "stratified": {"bins": 2}}),
            r"dataset\.splits\.train: \['stratified'\].*declared",
        ),
        (
            _put("splits", "test", value={"duration_ms": 100}),
            r"dataset\.splits\.test: \['duration_ms'\].*stratified",
        ),
        (
            _put("splits", "probes", "draws", value=["braille_H"]),
            r"dataset\.splits\.probes: \['draws'\].*probes",
        ),
        (
            _put("splits", "sessions", "per_bin", value=2),
            r"dataset\.splits\.sessions: \['per_bin'\].*sessions",
        ),
        (
            _put("splits", "fixed", "n", value=2),
            r"dataset\.splits\.fixed: \['n'\].*fixed",
        ),
        (
            _put("splits", "held_out", "n", value=2),
            r"dataset\.splits\.held_out: give stratified or n, not both",
        ),
        # Names that become directories.
        (
            _put("splits", "my train", value={"kind": "declared", "n": 2}),
            "'my train'",
        ),
        (
            _put("splits", "a/b", value={"kind": "declared", "n": 2}),
            "'a/b'",
        ),
    ],
)
def test_declarations_of_the_wrong_shape_are_named(mutate, message):
    raw = copy.deepcopy(RAW)
    mutate(raw["dataset"])
    with pytest.raises(ValueError, match=message):
        load_dataset(raw, base_dir=WORLDS)


def test_a_top_level_that_is_not_a_mapping_is_named():
    with pytest.raises(ValueError, match="a data set is a mapping"):
        load_dataset({"dataset": ["train"]})
