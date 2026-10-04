"""biased_direction: lateral-biased scan directions and quantile strata."""

import copy
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from sensoryforge.world.dataset import build_dataset, load_dataset
from sensoryforge.world.distributions import AxisSpec, _travel_stretch
from sensoryforge.world.schema import load_world

WORLDS = Path(__file__).resolve().parents[1] / "fixtures" / "worlds"


def _axis(**options):
    return AxisSpec.from_dict("direction_deg", {"dist": "biased_direction", **options})


def _theta(axis, n=1000, offset=0.5):
    return np.asarray(axis.sample((np.arange(n) + offset) / n))


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.5, 4.0])
def test_travel_ratio_is_honoured(ratio):
    axis = _axis(travel_ratio=ratio)
    theta = np.radians(_theta(axis, n=2_000_000))
    got = np.abs(np.cos(theta)).mean() / np.abs(np.sin(theta)).mean()
    assert got == pytest.approx(ratio, abs=1e-6, rel=0)


def test_known_stretch_for_ratio_2_5():
    assert _travel_stretch(2.5) == pytest.approx(3.87625, abs=1e-5)


def test_ratio_one_is_uniform():
    axis = _axis(travel_ratio=1.0)
    u = np.linspace(0.0, 0.999, 37)
    assert np.allclose(axis.sample(u), 360.0 * u)


def test_axis_rotates_the_bias():
    base = _theta(_axis(travel_ratio=3.0))
    rotated = _theta(_axis(travel_ratio=3.0, axis_deg=90.0))
    assert np.allclose((base + 90.0) % 360.0, rotated)
    t = np.radians(rotated)
    assert np.abs(np.sin(t)).mean() > np.abs(np.cos(t)).mean()


def test_values_lie_in_0_360():
    for ratio in (0.3, 1.0, 5.0):
        for axis_deg in (0.0, 123.0, -50.0, 400.0):
            v = _theta(_axis(travel_ratio=ratio, axis_deg=axis_deg), n=997)
            assert (v >= 0.0).all() and (v < 360.0).all()


def test_sampling_is_deterministic_and_draw_i_is_independent_of_n():
    axis = _axis(travel_ratio=2.5, axis_deg=30.0)
    u = np.random.default_rng(3).random(50)
    assert axis.sample(u) == axis.sample(u)
    assert axis.sample(u[:7]) == axis.sample(u)[:7]
    assert axis.sample(u[7:8]) == axis.sample(u)[7:8]


@pytest.fixture(scope="module")
def strata_entries():
    raw = yaml.safe_load((WORLDS / "dataset_small.yml").read_text())
    raw = copy.deepcopy(raw)
    world = yaml.safe_load((WORLDS / "elements_v1_2.yml").read_text())
    raw["dataset"]["world"] = world
    return raw


def test_quantile_strata_hold_per_bin_each(tmp_path, strata_entries):
    raw = strata_entries
    wpath = tmp_path / "w.yml"
    wpath.write_text(yaml.safe_dump(raw["dataset"]["world"]))
    raw["dataset"]["world"] = str(wpath)
    raw["dataset"]["splits"] = {
        "test": {"stratified": {"bins": 4, "per_bin": 3}},
    }
    dpath = tmp_path / "d.yml"
    dpath.write_text(yaml.safe_dump(raw))
    spec = load_dataset(dpath)
    axis = spec.world.classes["lateral_slide"].axes["direction_deg"]
    rows = [e for e in build_dataset(spec) if e.class_name == "lateral_slide"]
    assert len(rows) == 12
    counts = Counter(e.bins["direction_deg"] for e in rows)
    assert len(counts) == 4 and set(counts.values()) == {3}
    labels = [axis.quantile_label(b, 4) for b in range(4)]
    assert set(counts) == set(labels)
    assert all(re.fullmatch(r"\[[-0-9.e+]+, [-0-9.e+]+\)", lab) for lab in labels)
    for e in rows:
        b = labels.index(e.bins["direction_deg"])
        lo, hi = b / 4, (b + 1) / 4
        value = e.item.values["direction_deg"]
        # the stratum is u in [lo, hi): the value is the quantile of some u there
        us = np.linspace(lo, hi, 20001, endpoint=False)
        vals = np.asarray(axis.sample(us))
        step = np.abs(np.diff(vals)).max() + 1e-9
        assert np.abs(vals - value).min() <= max(step, 1e-3)


@pytest.mark.parametrize(
    "options, message",
    [
        ({}, r"needs travel_ratio"),
        ({"travel_ratio": 0}, r"must be > 0"),
        ({"travel_ratio": -1.0}, r"must be > 0"),
        ({"travel_ratio": 2.0, "bogus": 1}, r"unknown options \['bogus'\]"),
        ({"travel_ratio": 2.0, "axis_deg": "x"}, r"axis_deg"),
    ],
)
def test_bad_options_fail_at_load_naming_the_path(options, message):
    world = yaml.safe_load((WORLDS / "elements_v1_2.yml").read_text())
    world["world"]["classes"]["lateral_slide"]["axes"]["direction_deg"] = {
        "dist": "biased_direction",
        **options,
    }
    with pytest.raises(ValueError, match=r"lateral_slide.*direction_deg.*" + message):
        load_world(world)


def test_braille_cells_is_still_stratified_by_support():
    axis = AxisSpec.from_dict("cells", {"dist": "braille_cells"})
    assert axis.support() is not None and len(axis.support()) == 63


# ------------------------- continuous distributions against their field's domain


def _one_class(layer, axes):
    return {
        "world": {
            "classes": {
                "c": {"layer": layer, "axes": {"hold_ms": {"value": 10.0}, **axes}}
            }
        }
    }


def test_biased_direction_declares_its_bounds():
    from sensoryforge.world.distributions import DISTRIBUTIONS

    axis = _axis(travel_ratio=2.5, axis_deg=30.0)
    assert DISTRIBUTIONS["biased_direction"].bounds(axis) == (0.0, 360.0)


def test_a_continuous_distribution_outside_its_fields_domain_fails_at_load():
    # biased_direction draws [0, 360); grating duty takes [0.01, 0.99]. v1.2.0
    # loaded this world and rendered nonsense duties.
    world = _one_class(
        {"shape": {"kind": "grating", "profile": "square"}},
        {"duty": {"dist": "biased_direction", "travel_ratio": 2.0}},
    )
    with pytest.raises(
        ValueError,
        match=r"world\.classes\.c\.axes\.duty: .*biased_direction.*\[0, 360\]"
        r".*domain \[0\.01, 0\.99\]",
    ):
        load_world(world)


def test_a_continuous_distribution_inside_its_fields_domain_loads():
    world = _one_class(
        {"shape": {"kind": "grating"}},
        {"orientation_deg": {"dist": "biased_direction", "travel_ratio": 2.0}},
    )
    assert load_world(world).classes["c"].axes["orientation_deg"].dist == (
        "biased_direction"
    )


def test_a_numeric_distribution_cannot_bind_a_text_field():
    world = _one_class(
        {"shape": {"kind": "gaussian"}, "pattern": {"kind": "braille"}},
        {"text": {"dist": "biased_direction", "travel_ratio": 2.0}},
    )
    with pytest.raises(ValueError, match=r"axes\.text: .*biased_direction.*numbers"):
        load_world(world)


def test_a_continuous_distribution_without_bounds_cannot_bind_a_number_field():
    # multi-letter text has no finite support and declares no bounds
    world = _one_class(
        {"shape": {"kind": "gaussian"}},
        {"sigma_mm": {"dist": "letter_text", "cells": 2, "stratify": False}},
    )
    with pytest.raises(
        ValueError, match=r"axes\.sigma_mm: .*letter_text.*declares no bounds"
    ):
        load_world(world)


def test_a_registered_distribution_with_bounds_is_checked_against_the_domain():
    from sensoryforge.world.distributions import DISTRIBUTIONS, register_distribution

    register_distribution(
        "_t_wide",
        lambda u, axis: (0.5 + 99.5 * np.asarray(u)).tolist(),
        quantile=True,
        bounds=lambda axis: (0.5, 100.0),
    )
    try:
        # gaussian sigma_mm takes [0.001, 50]
        with pytest.raises(
            ValueError, match=r"axes\.sigma_mm: .*_t_wide.*\[0\.5, 100\]"
        ):
            load_world(
                _one_class(
                    {"shape": {"kind": "gaussian"}}, {"sigma_mm": {"dist": "_t_wide"}}
                )
            )
        # bar width_mm takes [0.001, 100]
        load_world(
            _one_class({"shape": {"kind": "bar"}}, {"width_mm": {"dist": "_t_wide"}})
        )
    finally:
        DISTRIBUTIONS.pop("_t_wide")
