"""Layered stimuli: shapes, patterns, motion, timing, stacking and presets."""

import math

import pytest
import torch

from sensoryforge.config.schema import GridConfig, SensoryForgeConfig, StimulusConfig
from sensoryforge.stimuli.layered import (
    PATTERNS,
    SHAPES,
    default_layer,
    layer_envelope,
    pattern_positions,
    render_layers,
)
from sensoryforge.stimuli.presets import PRESETS, preset
from sensoryforge.stimuli.render import render_for_config

H = 81
XS = torch.linspace(-8.0, 8.0, H)  # 0.2 mm pitch, 0 at index 40
XX, YY = torch.meshgrid(XS, XS, indexing="ij")


def _render(layers, total_ms=100.0, combine="sum"):
    return render_layers(layers, XX, YY, dt_ms=1.0, total_ms=total_ms, combine=combine)


def _layer(shape, pattern=None, motion=None, timing=None):
    layer = default_layer(shape["kind"])
    layer["shape"].update(shape)
    if pattern:
        layer["pattern"] = pattern
    if motion:
        layer["motion"] = motion
    layer["timing"] = timing or {
        "onset_ms": 0,
        "ramp_up_ms": 0,
        "hold_ms": None,
        "ramp_down_ms": 0,
    }
    return layer


def _at(frame, x, y):
    return float(frame[int(round((x + 8) / 0.2)), int(round((y + 8) / 0.2))])


def _centroid(frame):
    total = frame.sum()
    return float((frame * XX).sum() / total), float((frame * YY).sum() / total)


# ------------------------------------------------------------------ shapes


@pytest.mark.parametrize("kind", sorted(SHAPES))
def test_every_shape_is_non_negative_and_peaks_at_its_amplitude(kind):
    frames = _render([_layer({"kind": kind, "amplitude": 2.5})])
    assert float(frames.min()) >= 0.0
    assert float(frames.max()) == pytest.approx(2.5, rel=1e-3)


def test_disc_is_flat_inside_and_zero_outside():
    frame = _render([_layer({"kind": "disc", "diameter_mm": 2.0, "edge_mm": 0.2})])[0]
    assert _at(frame, 0.0, 0.0) == 1.0 and _at(frame, 0.8, 0.0) == 1.0
    assert _at(frame, 1.2, 0.0) == 0.0


def test_grating_period_and_orientation():
    vertical = _render([_layer({"kind": "grating", "wavelength_mm": 2.0})])[0]
    assert _at(vertical, 0.0, 0.0) == pytest.approx(1.0)
    assert _at(vertical, 1.0, 0.0) == pytest.approx(0.0, abs=1e-6)  # half period
    assert _at(vertical, 0.0, 3.0) == pytest.approx(1.0)  # constant along y
    turned = _render(
        [_layer({"kind": "grating", "wavelength_mm": 2.0, "orientation_deg": 90.0})]
    )[0]
    assert _at(turned, 0.0, 1.0) == pytest.approx(0.0, abs=1e-6)


def test_bar_has_a_finite_length_when_set():
    frame = _render(
        [_layer({"kind": "bar", "width_mm": 0.3, "length_mm": 4.0, "profile": "flat"})]
    )[0]
    # orientation 0: across = y, along = x
    assert _at(frame, 1.0, 0.0) == 1.0 and _at(frame, 3.0, 0.0) == 0.0


# ---------------------------------------------------------------- patterns


def test_grid_mask_places_only_the_marked_cells():
    positions, _ = pattern_positions(
        {"kind": "grid", "rows": 3, "cols": 2, "spacing_mm": 2.0, "mask": "10 01 11"}
    )
    assert sorted(positions) == sorted(
        [(-1.0, 2.0), (1.0, 0.0), (-1.0, -2.0), (1.0, -2.0)]
    )


def test_braille_letters_follow_the_standard_dot_numbers():
    # h = dots 1, 2, 5: top-left, middle-left, middle-right
    positions, _ = pattern_positions(
        {"kind": "braille", "text": "h", "dot_spacing_mm": 2.0}
    )
    assert sorted(positions) == sorted([(-1.0, 2.0), (-1.0, 0.0), (1.0, 0.0)])
    with pytest.raises(ValueError, match="no cell"):
        pattern_positions({"kind": "braille", "text": "h!"})


def test_random_pattern_is_reproducible_and_keeps_its_distance():
    spec = {
        "kind": "random",
        "count": 25,
        "width_mm": 10,
        "height_mm": 10,
        "min_distance_mm": 1.0,
        "seed": 7,
    }
    first, _ = pattern_positions(spec)
    assert first == pattern_positions(spec)[0]
    assert first != pattern_positions({**spec, "seed": 8})[0]
    for i, (x1, y1) in enumerate(first):
        for x2, y2 in first[i + 1 :]:
            assert math.hypot(x1 - x2, y1 - y2) >= 1.0


# ------------------------------------------------------- timing and motion


def test_timing_onset_ramps_and_hold():
    t = torch.arange(200, dtype=torch.float32)
    env = layer_envelope(
        {"onset_ms": 20, "ramp_up_ms": 40, "hold_ms": 60, "ramp_down_ms": 40}, t, 200
    )
    assert float(env[10]) == 0.0
    assert float(env[40]) == pytest.approx(0.5)
    assert float(env[100]) == 1.0
    assert float(env[140]) == pytest.approx(0.5)
    assert float(env[170]) == 0.0


def test_an_unset_hold_lasts_until_the_ramp_down_ends_the_run():
    t = torch.arange(100, dtype=torch.float32)
    env = layer_envelope(
        {"onset_ms": 0, "ramp_up_ms": 10, "hold_ms": None, "ramp_down_ms": 10}, t, 100
    )
    assert float(env[50]) == 1.0 and float(env[95]) == pytest.approx(0.5)


def test_linear_motion_carries_the_pattern_from_start_to_end_during_the_hold():
    layer = _layer(
        {"kind": "gaussian", "sigma_mm": 0.5},
        motion={"kind": "linear", "start": [-4, 0], "end": [4, 0], "span": "hold"},
        timing={"onset_ms": 0, "ramp_up_ms": 20, "hold_ms": 60, "ramp_down_ms": 20},
    )
    frames = _render([layer], total_ms=100)
    assert _centroid(frames[10])[0] == pytest.approx(-4.0, abs=0.05)  # still at start
    assert _centroid(frames[50])[0] == pytest.approx(0.0, abs=0.2)
    assert _centroid(frames[90])[0] == pytest.approx(4.0, abs=0.05)


# ---------------------------------------------------------------- stacking


def test_layers_stack_with_their_own_onsets_and_combine_by_sum_or_max():
    a = _layer(
        {"kind": "disc", "diameter_mm": 2.0},
        timing={"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": 50, "ramp_down_ms": 0},
    )
    b = _layer(
        {"kind": "disc", "diameter_mm": 2.0},
        timing={"onset_ms": 25, "ramp_up_ms": 0, "hold_ms": 50, "ramp_down_ms": 0},
    )
    summed = _render([a, b])
    assert _at(summed[10], 0, 0) == 1.0
    assert _at(summed[40], 0, 0) == 2.0  # both on
    assert _at(summed[60], 0, 0) == 1.0  # only b
    assert _at(_render([a, b], combine="max")[40], 0, 0) == 1.0


def test_a_layered_config_round_trips_through_yaml_and_renders_the_same():
    config = SensoryForgeConfig(grids=[GridConfig(name="g", rows=30, cols=30)])
    stimulus = StimulusConfig(type="layered")
    chosen = preset("braille_word", 300)
    stimulus.layers = chosen["layers"]
    stimulus.combine = "max"
    config.stimulus = stimulus
    again = SensoryForgeConfig.from_yaml(config.to_yaml())
    assert again.stimulus.layers == chosen["layers"] and again.stimulus.combine == "max"
    first = render_for_config(config, duration_ms=300, dt_ms=1.0)[0]
    second = render_for_config(again, duration_ms=300, dt_ms=1.0)[0]
    assert torch.equal(first, second)


def test_unknown_kinds_are_named_in_the_error():
    with pytest.raises(ValueError, match="shape kind 'blob'"):
        _render([_layer({"kind": "gaussian"}) | {"shape": {"kind": "blob"}}])
    with pytest.raises(ValueError, match="pattern kind"):
        pattern_positions({"kind": "spiral"})
    assert set(PATTERNS) == {"single", "grid", "list", "random", "braille"}


# ----------------------------------------------------------------- presets


def _named_and_preset(name, duration_ms):
    config = SensoryForgeConfig(
        grids=[GridConfig(name="g", rows=60, cols=60, spacing=0.2)]
    )
    config.stimulus = StimulusConfig(type=name)
    named = render_for_config(config, duration_ms=duration_ms, dt_ms=1.0)[0]
    chosen = preset(name, duration_ms)
    layered = StimulusConfig(type="layered")
    layered.layers = chosen["layers"]
    layered.combine = chosen["combine"]
    config.stimulus = layered
    return named, render_for_config(config, duration_ms=duration_ms, dt_ms=1.0)[0]


@pytest.mark.parametrize(
    "name", ["moving_edge", "braille", "drifting_grating", "ramp_gaussian"]
)
@pytest.mark.parametrize("duration_ms", [330.0, 900.0])
def test_pressure_simulation_presets_reproduce_their_named_type(name, duration_ms):
    """Within 2% of peak: the named types sample their ramps one step differently."""
    named, layered = _named_and_preset(name, duration_ms)
    assert named.shape == layered.shape
    peak = float(named.abs().max())
    assert float((named - layered).abs().max()) <= 0.021 * peak


@pytest.mark.parametrize("name", ["gaussian", "moving", "repeated_pattern"])
def test_legacy_presets_reproduce_their_named_type(name):
    named, layered = _named_and_preset(name, 400.0)
    assert float((named - layered).abs().max()) <= 0.01 * float(named.abs().max())


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_renders_something_that_changes_over_time(name):
    config = SensoryForgeConfig(grids=[GridConfig(name="g", rows=40, cols=40)])
    chosen = preset(name, 400)
    stimulus = StimulusConfig(type="layered")
    stimulus.layers = chosen["layers"]
    config.stimulus = stimulus
    frames = render_for_config(config, duration_ms=400, dt_ms=1.0)[0][0]
    assert float(frames.max()) > 0
    assert not torch.equal(frames[5], frames[200])
