"""The background contact and the floor: ``background`` and ``clamp_min`` (P6)."""

import math

import pytest
import torch

from sensoryforge.stimuli.episode import contact_terms
from sensoryforge.stimuli.layered import render_layers
from sensoryforge.world import Canvas, World, render, sample

CANVAS = Canvas.from_grid(rows=16, cols=16, spacing_mm=0.05)
TIMES = torch.arange(0, 120, 1.0, dtype=torch.float64)

EPISODE = {
    "delay_ms": {"value": 10},
    "touch_ms": {"value": 20},
    "hold_ms": {"value": 30},
    "release_ms": {"value": 20},
    "contacts": {"value": 2},
    "pause_ms": {"value": 5},
}


def _world(layer, axes, defaults=None):
    return World.from_dict(
        {
            "world": {
                "defaults": defaults or {},
                "classes": {"c": {"layer": layer, "axes": {**EPISODE, **axes}}},
            }
        }
    )


def _env(draw, times=TIMES):
    v = draw.values
    t = times.view(1, -1)
    col = lambda name: torch.tensor([[float(v[name])]], dtype=torch.float64)  # noqa
    env, _, _, _ = contact_terms(
        t,
        col("delay_ms"),
        col("touch_ms"),
        col("hold_ms"),
        col("slide_ms"),
        col("release_ms"),
        col("contacts"),
        col("pause_ms"),
    )
    return env[0]


def _frames(world):
    draw = sample(world, n=1, seed=3)[0]
    return draw, render([draw], CANVAS, TIMES, dtype=torch.float64)[0]


def test_background_alone_is_level_times_envelope():
    world = _world(
        {"shape": {"kind": "gaussian"}},
        {"amplitude": {"value": 0.0}, "background": {"value": 0.3}},
    )
    draw, frames = _frames(world)
    expected = 0.3 * _env(draw).view(-1, 1, 1)
    assert torch.equal(frames, expected.expand_as(frames))
    assert float(frames.max()) > 0


def test_background_adds_under_a_feature():
    world = _world(
        {"shape": {"kind": "gaussian", "sigma_mm": 0.05}},
        {
            "amplitude": {"value": 0.8},
            "background": {"value": 0.25},
            "x_mm": {"value": 0.0},
            "y_mm": {"value": 0.0},
        },
    )
    draw, frames = _frames(world)
    env = _env(draw)
    gauss = torch.exp(-(CANVAS.xx**2 + CANVAS.yy**2) / (2 * 0.05**2))
    expected = env.view(-1, 1, 1) * (0.8 * gauss + 0.25)
    torch.testing.assert_close(frames, expected, atol=1e-12, rtol=0)
    far = (CANVAS.xx**2 + CANVAS.yy**2).argmax()
    far_ix = divmod(int(far), CANVAS.shape[1])
    torch.testing.assert_close(
        frames[:, far_ix[0], far_ix[1]],
        env * (0.8 * gauss[far_ix] + 0.25),
        atol=1e-12,
        rtol=0,
    )


def _relief_world():
    return _world(
        {
            "shape": {"kind": "grating", "signed": True, "wavelength_mm": 0.3},
            "clamp_min": 0.0,
        },
        {
            "amplitude": {"value": 1.0},
            "background": {"value": 0.3},
            "orientation_deg": {"value": 0.0},
        },
    )


def test_the_floor_keeps_relief_nonnegative():
    draw, frames = _frames(_relief_world())
    env = _env(draw).view(-1, 1, 1)
    wave = torch.cos(2 * math.pi * CANVAS.xx / 0.3)
    expected = torch.where(env > 0, (env * (wave + 0.3)).clamp(min=0.0), 0.0)
    torch.testing.assert_close(frames, expected, atol=1e-12, rtol=0)
    assert float(frames.min()) >= 0.0
    # the unfloored total would dip below zero, so the floor does act
    assert float((env * (wave + 0.3)).min()) < 0.0


def test_quiet_stretches_stay_exactly_zero():
    draw, frames = _frames(_relief_world())
    env = _env(draw)
    quiet = env == 0
    assert quiet.any()
    assert torch.all(frames[quiet] == 0.0)
    assert torch.all(frames[TIMES >= draw.end_ms] == 0.0)
    assert torch.all(frames[TIMES < 10.0] == 0.0)
    pause = (TIMES >= 10 + 70) & (TIMES < 10 + 70 + 5)
    assert torch.all(frames[pause] == 0.0)


def test_a_world_default_background_binds_every_layered_class_and_skips_quiet():
    world = World.from_dict(
        {
            "world": {
                "defaults": {"background": {"range": [0.1, 0.2]}},
                "classes": {
                    "a": {
                        "layer": {"shape": {"kind": "gaussian"}},
                        "axes": {"hold_ms": {"value": 20}},
                    },
                    "b": {
                        "layer": {"shape": {"kind": "disc"}},
                        "axes": {"hold_ms": {"value": 20}},
                    },
                    "q": {"kind": "quiet", "axes": {"quiet_ms": {"value": 10}}},
                },
            }
        }
    )
    assert world.classes["a"].bindings["background"] == ("layer", "background")
    assert world.classes["b"].bindings["background"] == ("layer", "background")
    assert "background" not in world.classes["q"].bindings
    lo, hi = (
        world.classes["a"].axes["background"].lo,
        world.classes["a"].axes["background"].hi,
    )
    assert (lo, hi) == (0.1, 0.2)


def test_background_is_recorded_and_carried_into_to_layer():
    world = _world(
        {"shape": {"kind": "gaussian"}, "clamp_min": 0.0},
        {"background": {"range": [0.1, 0.4]}},
    )
    draw = sample(world, n=1, seed=5)[0]
    assert "background" in draw.to_dict()["values"]
    layer = draw.to_layer()
    assert layer["background"] == draw.values["background"]
    assert layer["clamp_min"] == 0.0
    # a layer that sets neither emits neither
    plain = sample(_world({"shape": {"kind": "gaussian"}}, {}), n=1, seed=5)[0]
    assert "background" not in plain.to_layer()
    assert "clamp_min" not in plain.to_layer()
    assert "clamp_min" not in plain.spec.layer


def test_layers_without_the_keys_render_as_before():
    world = _world({"shape": {"kind": "gaussian", "sigma_mm": 0.1}}, {})
    draw = sample(world, n=1, seed=1)[0]
    xx, yy = CANVAS.xx.float(), CANVAS.yy.float()
    frames = render_layers([draw.to_layer()], xx, yy, dt_ms=1.0, total_ms=120.0)
    assert torch.isfinite(frames).all()


def test_a_layer_background_must_be_nonnegative_and_clamp_min_a_number():
    with pytest.raises(ValueError, match="background"):
        _world({"shape": {"kind": "gaussian"}, "background": -1.0}, {})
    with pytest.raises(ValueError, match="clamp_min"):
        _world({"shape": {"kind": "gaussian"}, "clamp_min": "zero"}, {})
