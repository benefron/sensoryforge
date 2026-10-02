"""Axes: forms, sampling, strata, probes, errors."""

import numpy as np
import pytest

from sensoryforge.world import rng
from sensoryforge.world.distributions import (
    BRAILLE_CELLS,
    DISTRIBUTIONS,
    AxisSpec,
    register_distribution,
)

U = rng.uniforms(rng.draw_seeds(3, np.arange(20000)), "u")


def test_constant_axis():
    axis = AxisSpec.from_dict("contacts", {"value": 2})
    assert not axis.is_random and axis.sample(U[:3]) == [2, 2, 2]
    assert axis.to_dict() == {"value": 2}


def test_uniform_and_log_uniform_stay_in_range():
    lin = AxisSpec.from_dict("a", {"range": [0.15, 0.45]})
    log = AxisSpec.from_dict("b", {"range": [0.01, 0.1], "dist": "log_uniform"})
    v = np.array(lin.sample(U))
    w = np.array(log.sample(U))
    assert v.min() >= 0.15 and v.max() < 0.45
    assert w.min() >= 0.01 and w.max() <= 0.1
    assert abs(np.median(np.log(w)) - np.log(np.sqrt(0.001))) < 0.05


def test_int_axis_reaches_both_ends():
    axis = AxisSpec.from_dict("contacts", {"range": [1, 3], "int": True})
    values = axis.sample(U)
    assert set(values) == {1, 2, 3} and all(isinstance(v, int) for v in values)
    assert axis.support() == [1, 2, 3]


def test_categorical_follows_its_weights():
    axis = AxisSpec.from_dict("letter", {"values": ["a", "b"], "weights": [3, 1]})
    values = axis.sample(U)
    assert abs(values.count("a") / len(values) - 0.75) < 0.02


def test_braille_cells_are_the_63_non_empty_cells():
    assert len(BRAILLE_CELLS) == 63 == len(set(BRAILLE_CELLS))
    axis = AxisSpec.from_dict("dots", {"dist": "braille_cells"})
    values = axis.sample(U)
    assert set(values) == set(BRAILLE_CELLS)
    assert axis.support() == BRAILLE_CELLS


def test_strata_cover_equal_bins_on_the_sampling_scale():
    log = AxisSpec.from_dict("s", {"range": [0.01, 1.0], "dist": "log_uniform"})
    edges = log.bin_edges(2)
    assert edges == pytest.approx([0.01, 0.1, 1.0])
    values = log.stratum_values(np.array([0, 1]), 2, np.array([0.5, 0.5]))
    assert values[0] < 0.1 < values[1]
    assert log.bin_label(0, 2) == "[0.01, 0.1)" and log.bin_label(1, 2) == "[0.1, 1]"


def test_probes_lie_strictly_outside_and_inside_the_domain():
    axis = AxisSpec.from_dict("sigma_mm", {"range": [0.15, 0.45]}).with_domain(
        0.001, 50.0
    )
    below = np.array(axis.probe_values("below", 5, U))
    above = np.array(axis.probe_values("above", 5, U))
    assert below.max() < 0.15 and below.min() >= 0.15 - 0.06 - 1e-12
    assert above.min() > 0.45 and above.max() <= 0.45 + 0.06 + 1e-12
    worst = axis.probe_values("below", 5, np.array([1.0 - 2.0**-53]))
    assert worst[0] < 0.15


def test_probes_skip_a_side_with_no_room():
    axis = AxisSpec.from_dict("hold_ms", {"range": [0, 100]}).with_domain(0.0, None)
    assert axis.probe_values("below", 5, U[:3]) is None
    assert axis.probe_values("above", 5, U[:3]) is not None
    circular = AxisSpec.from_dict("d", {"range": [0, 360], "circular": True})
    assert circular.probe_values("above", 5, U[:3]) is None


def test_midpoint_and_contains():
    log = AxisSpec.from_dict("s", {"range": [0.01, 1.0], "dist": "log_uniform"})
    assert log.midpoint() == pytest.approx(0.1)
    assert log.contains(0.5) and not log.contains(2.0)
    cat = AxisSpec.from_dict("c", {"values": ["x", "y"]})
    assert cat.midpoint() == "x" and not cat.contains("z")


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"range": [2, 1]}, "lo > hi"),
        ({"range": [0, 1], "dist": "log_uniform"}, "needs lo > 0"),
        ({"range": [0, 1], "dist": "normal"}, "must be one of"),
        ({"values": []}, "is empty"),
        ({"dist": "nope"}, "unknown distribution"),
        ({"rang": [0, 1]}, "unknown keys"),
        ({"value": 1, "range": [0, 1]}, "takes only 'value'"),
        ({"range": [0.5, 2], "int": True}, "integer bounds"),
        ({"range": ["1e-1", 1]}, "range bounds must be numbers.*3.0e-1"),
        ({"range": [0, float("inf")]}, "range bounds must be finite"),
        ({"range": [True, 2]}, "range bounds must be numbers"),
        ({"value": float("nan")}, "constants must be finite"),
        ({"values": [1, float("-inf")]}, "values must be finite"),
        ({"values": [1, 2], "weights": ["1", 1]}, "weights must be numbers"),
    ],
)
def test_invalid_axes_are_named(spec, message):
    with pytest.raises(ValueError, match=message):
        AxisSpec.from_dict("x", spec)


def test_a_taken_distribution_name_needs_replace():
    original = DISTRIBUTIONS["braille_cells"]
    with pytest.raises(ValueError, match="'braille_cells' is already registered"):
        register_distribution("braille_cells", lambda u, axis: list(u))
    assert DISTRIBUTIONS["braille_cells"] is original
    try:
        register_distribution("braille_cells", lambda u, axis: list(u), replace=True)
        assert DISTRIBUTIONS["braille_cells"] is not original
    finally:
        DISTRIBUTIONS["braille_cells"] = original
    with pytest.raises(ValueError, match="built-in numeric"):
        register_distribution("uniform", lambda u, axis: list(u), replace=True)
