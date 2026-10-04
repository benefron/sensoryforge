"""Braille pages and letters: line_spacing_mm, '/' line breaks, letter_text."""

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import stats

from sensoryforge.stimuli.layered import (
    ADDED_FIELDS,
    PATTERNS,
    _BRAILLE,
    pattern_positions,
)
from sensoryforge.world import rng
from sensoryforge.world.distributions import AxisSpec
from sensoryforge.world.schema import load_world

WORLDS = Path(__file__).resolve().parents[1] / "fixtures" / "worlds"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _dot(cell_x, y0, pitch, dot):
    n = int(dot) - 1
    col, row = divmod(n, 3)
    return (cell_x + (col - 0.5) * pitch, y0 + (1 - row) * pitch)


def test_two_lines_put_dots_at_known_positions():
    pattern = {
        "kind": "braille",
        "text": "ab/c",
        "dot_spacing_mm": 1.0,
        "cell_spacing_mm": 5.0,
        "line_spacing_mm": 8.0,
        "x_mm": 2.0,
        "y_mm": 3.0,
    }
    positions, scales = pattern_positions(pattern)
    want = []
    for dot in _BRAILLE["a"]:
        want.append(_dot(2.0, 3.0, 1.0, dot))
    for dot in _BRAILLE["b"]:
        want.append(_dot(2.0 + 5.0, 3.0, 1.0, dot))
    for dot in _BRAILLE["c"]:  # line 1: the cell index restarts, y drops by 8
        want.append(_dot(2.0, 3.0 - 8.0, 1.0, dot))
    assert positions == pytest.approx(want, abs=1e-12)
    assert scales == [1.0] * len(want)


def test_dots_split_on_slash_too():
    pattern = {
        "kind": "braille",
        "dots": "1 2/12",
        "dot_spacing_mm": 1.0,
        "cell_spacing_mm": 5.0,
        "line_spacing_mm": 4.0,
    }
    positions, _ = pattern_positions(pattern)
    want = [_dot(0.0, 0.0, 1.0, "1"), _dot(5.0, 0.0, 1.0, "2")]
    want += [_dot(0.0, -4.0, 1.0, d) for d in "12"]
    assert positions == pytest.approx(want, abs=1e-12)


def test_blank_cells_still_count_on_a_line():
    pattern = {
        "kind": "braille",
        "text": "a b/ a",
        "dot_spacing_mm": 1.0,
        "cell_spacing_mm": 5.0,
        "line_spacing_mm": 6.0,
    }
    positions, _ = pattern_positions(pattern)
    want = [_dot(0.0, 0.0, 1.0, "1")]
    want += [_dot(10.0, 0.0, 1.0, d) for d in "12"]
    want += [_dot(5.0, -6.0, 1.0, "1")]
    assert positions == pytest.approx(want, abs=1e-12)


def test_single_line_braille_renders_as_before():
    plain = {"kind": "braille", "text": "hi", "dot_spacing_mm": 2.5}
    assert pattern_positions(plain) == pattern_positions(
        {**plain, "line_spacing_mm": 3.0}
    )
    reference = json.loads((WORLDS / "v1_1_0_reference" / "reference.json").read_text())
    world = load_world(WORLDS / "tactile_small.yml")
    assert world.world_id == reference["worlds"]["tactile_small"]["world_id"]


def test_an_old_braille_layer_normalises_without_line_spacing():
    world = load_world(WORLDS / "tactile_small.yml")
    layer = world.classes["braille"].layer
    assert "line_spacing_mm" not in layer["pattern"]
    assert ("pattern", "braille") in ADDED_FIELDS
    assert ADDED_FIELDS[("pattern", "braille")] == ("line_spacing_mm",)
    assert any(s.name == "line_spacing_mm" for s in PATTERNS["braille"])


def test_line_spacing_is_kept_when_the_file_sets_it():
    raw = yaml.safe_load((WORLDS / "tactile_small.yml").read_text())
    raw = copy.deepcopy(raw)
    raw["world"]["classes"]["braille"]["layer"]["pattern"]["line_spacing_mm"] = 1.5
    world = load_world(raw)
    assert world.classes["braille"].layer["pattern"]["line_spacing_mm"] == 1.5


def _axis(**options):
    return AxisSpec.from_dict("text", {"dist": "letter_text", **options})


def _letters(axis, n):
    u = (np.arange(n) + 0.5) / n
    return axis.sample(u)


def test_letter_text_is_deterministic_and_follows_its_weights():
    weights = [float(i + 1) for i in range(26)]
    axis = _axis(weights=weights)

    n = 20000
    seeds = rng.draw_seeds(7, np.arange(n))
    u = rng.uniforms(seeds, "text")
    one = axis.sample(u)
    assert one == axis.sample(u)
    counts = Counter(one)
    observed = np.array([counts[ch] for ch in LETTERS], dtype=float)
    expected = np.array(weights) / sum(weights) * n
    p = stats.chisquare(observed, expected).pvalue
    assert p > 0.001


def test_one_cell_letter_text_is_stratified_by_letter():
    axis = _axis(letters="abc")
    assert axis.support() == ["a", "b", "c"]
    assert _axis().support() == list(LETTERS)


def test_multi_cell_letter_text_has_no_support_and_joins_lines():
    axis = AxisSpec.from_dict(
        "text", {"dist": "letter_text", "cells": 3, "lines": 2, "stratify": False}
    )
    assert axis.support() is None
    for text in _letters(axis, 50):
        lines = text.split("/")
        assert len(lines) == 2 and all(len(line) == 3 for line in lines)
        assert set(text) <= set(LETTERS + "/")


def test_multi_cell_letter_text_without_stratify_false_fails_at_load_with_a_hint():
    with pytest.raises(ValueError, match="stratify: false"):
        AxisSpec.from_dict("text", {"dist": "letter_text", "cells": 2})
    with pytest.raises(ValueError, match="stratify: false"):
        AxisSpec.from_dict("text", {"dist": "letter_text", "lines": 2})


def test_unknown_letters_fail_at_load():
    with pytest.raises(ValueError, match="letters"):
        _axis(letters="ab1")
    with pytest.raises(ValueError, match="weights"):
        _axis(letters="abc", weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="cells"):
        AxisSpec.from_dict(
            "text", {"dist": "letter_text", "cells": 0, "stratify": False}
        )
    with pytest.raises(ValueError, match="unknown options"):
        _axis(bogus=1)


def test_a_space_is_a_blank_letter():
    axis = _axis(letters="a ", weights=[1.0, 1.0])
    assert set(_letters(axis, 40)) == {"a", " "}
