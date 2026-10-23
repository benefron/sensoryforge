"""Unit-free field names and a declared space unit (roadmap A1b follow-up).

A world may write ``sigma`` for ``sigma_mm``; it must then equal the canonical world
bit for bit, and a world that writes no alias must be untouched.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from sensoryforge.io.bundle import load_bundle
from sensoryforge.world.names import alias_table, neutralise
from sensoryforge.world.runner import declared_units
from sensoryforge.world.schema import load_world
from tests.fixtures import toy_vision as toy

WORLDS = Path(__file__).resolve().parents[1] / "fixtures" / "worlds"

CANONICAL = {
    "world": {
        "name": "w",
        "defaults": {
            "hold_ms": {"value": 30},
            "speed_mm_per_ms": {"range": [0.01, 0.02]},
            "x_mm": {"range": [-1, 1]},
            "y_mm": {"range": [-1, 1]},
        },
        "classes": {
            "blobs": {
                "layer": {"shape": {"kind": "gaussian", "sigma_mm": 1.0}},
                "axes": {"amplitude": {"range": [0.5, 1.0]}},
            },
            "wide": {
                "layer": {"shape": {"kind": "gaussian"}},
                "axes": {"sigma_mm": {"range": [1.0, 2.0]}},
            },
            "dotted": {
                "layer": {"shape": {"kind": "gaussian"}},
                "axes": {"shape.sigma_mm": {"range": [1.0, 2.0]}},
            },
        },
        "fixed_draws": {"one": {"class": "wide", "sigma_mm": 1.5}},
    }
}


def _aliased():
    raw = copy.deepcopy(CANONICAL)
    w = raw["world"]
    w["defaults"] = {
        "hold_ms": {"value": 30},
        "speed": {"range": [0.01, 0.02]},
        "x": {"range": [-1, 1]},
        "y": {"range": [-1, 1]},
    }
    w["classes"]["blobs"]["layer"]["shape"] = {"kind": "gaussian", "sigma": 1.0}
    w["classes"]["wide"]["axes"] = {"sigma": {"range": [1.0, 2.0]}}
    w["classes"]["dotted"]["axes"] = {"shape.sigma": {"range": [1.0, 2.0]}}
    w["fixed_draws"]["one"] = {"class": "wide", "sigma": 1.5}
    return raw


def test_alias_world_equals_canonical_world():
    a, c = load_world(_aliased()), load_world(CANONICAL)
    assert a.world_id == c.world_id
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(
        c.to_dict(), sort_keys=True
    )


def test_a_world_without_aliases_is_untouched():
    raw = yaml.safe_load((WORLDS / "tactile_small.yml").read_text())["world"]
    assert neutralise(raw) == raw


def test_both_spellings_of_one_field_are_refused():
    raw = copy.deepcopy(CANONICAL)
    raw["world"]["defaults"]["x"] = {"range": [0, 1]}
    with pytest.raises(ValueError, match="same field"):
        load_world(raw)


def test_same_as_names_the_canonical_axis():
    twin = neutralise(
        {"classes": {"c": {"axes": {"x": {"same_as": "sigma"}, "sigma": {"value": 1}}}}}
    )
    axes = twin["classes"]["c"]["axes"]
    assert axes["x_mm"] == {"same_as": "sigma_mm"}
    assert "sigma_mm" in axes


def test_an_existing_field_name_is_never_an_alias():
    table = alias_table()
    assert table["sigma"] == "sigma_mm"
    assert table["speed"] == "speed_mm_per_ms"
    # `amplitude`, `hold_ms`, ... are fields themselves and are not in the table
    assert not {"amplitude", "hold_ms", "contacts"} & set(table)


def test_tactile_worlds_declare_no_units_block_in_a_bundle():
    for name in ("tactile_small.yml", "elements_v1_2.yml"):
        assert declared_units(load_world(WORLDS / name)) == {}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return toy.build_toy_dataset(tmp_path_factory.mktemp("toy_units"))


def test_the_toy_world_declares_pixels_and_the_bundle_says_so(built):
    assert declared_units(built.spec.world) == {"space": "px"}
    for e in built.entries[:3]:
        config = load_bundle(built.data / e.entry).meta["config_json"]
        assert config["units"] == {"space": "px"}
        assert config["grid"]["spacing_mm"] == toy.SPACING


def test_records_keep_the_canonical_names(built):
    values = built.entries[0].to_dict()["draw"]["values"]
    assert "sigma_mm" in values and "sigma" not in values
