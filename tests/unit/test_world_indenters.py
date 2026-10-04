"""Indenter shapes and ``curved_contact`` (P5): depth-driven, exact geometry."""

import math

import pytest
import torch

from sensoryforge.stimuli import layered
from sensoryforge.stimuli.episode import contact_terms
from sensoryforge.world import Canvas, World, kernel, render, sample
from sensoryforge.world.surfaces import curved_contact, step_edge

UNIT = 0.05
RADIUS = 0.8
CANVAS = Canvas.from_grid(rows=41, cols=41, spacing_mm=0.02)
TIMES = torch.arange(0, 140, 1.0, dtype=torch.float64)

EPISODE = {
    "delay_ms": {"value": 10},
    "touch_ms": {"value": 40},
    "hold_ms": {"value": 30},
    "release_ms": {"value": 20},
    "contacts": {"value": 2},
    "pause_ms": {"value": 6},
}


def _world(shape=None, layer_extra=None, axes=None):
    layer = {
        "shape": {
            "kind": "curved_contact",
            "form": "sphere",
            "radius_mm": RADIUS,
            "unit_mm": UNIT,
            **(shape or {}),
        },
        **(layer_extra or {}),
    }
    return World.from_dict(
        {
            "world": {
                "classes": {
                    "c": {
                        "layer": layer,
                        "axes": {
                            **EPISODE,
                            "amplitude": {"value": 0.8},
                            **(axes or {}),
                        },
                    }
                }
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


def _p(depth, form="sphere", radius=RADIUS, unit=UNIT, theta=0.0):
    t = lambda v: torch.tensor(float(v), dtype=torch.float64)  # noqa
    return {
        "depth": t(depth),
        "radius_mm": t(radius),
        "unit_mm": t(unit),
        "orientation_deg": t(theta),
        "form": form,
    }


def _line(n=2001, half=1.0):
    x = torch.linspace(-half, half, n, dtype=torch.float64)
    return x, torch.zeros_like(x)


def _centre_index():
    return CANVAS.xx.shape[0] // 2, CANVAS.xx.shape[1] // 2


def _draw():
    return sample(_world(), n=1, seed=3)[0]


def test_registered_as_an_indenter_and_others_are_not():
    assert kernel.SHAPE_KINDS["curved_contact"].indenter
    assert not kernel.SHAPE_KINDS["curved_contact"].unbounded
    for name in ("gaussian", "disc", "bar", "grating", "gabor", "self_affine"):
        assert not kernel.SHAPE_KINDS[name].indenter


def test_centre_depth_equals_amplitude_times_envelope():
    draw = _draw()
    frames = render([draw], CANVAS, TIMES, dtype=torch.float64)[0]
    i, j = _centre_index()
    assert abs(float(CANVAS.xx[i, j])) < 1e-12 and abs(float(CANVAS.yy[i, j])) < 1e-12
    want = 0.8 * _env(draw)
    torch.testing.assert_close(frames[:, i, j], want, atol=1e-12, rtol=0)
    assert float(want.max()) == pytest.approx(0.8)


@pytest.mark.parametrize("form", ["sphere", "cylinder"])
@pytest.mark.parametrize("depth", [0.4, 1.0, 2.0])
def test_contact_radius_follows_the_geometry(form, depth):
    # delta = d * unit_mm; the value reaches 0 at a = sqrt(2 R delta - delta^2)
    delta = depth * UNIT
    a = math.sqrt(2.0 * RADIUS * delta - delta**2)
    x, y = _line()
    # a cylinder along y (orientation 90): across = x sin(90) + y cos(90) = x
    theta = 90.0 if form == "cylinder" else 0.0
    value = curved_contact(x, y, _p(depth, form, theta=theta))
    inside = x.abs() < a - 1e-6
    outside = x.abs() > a + 1e-6
    assert bool((value[inside] > 0).all())
    assert bool((value[outside] == 0).all())
    # the value itself: d - sag/unit, with sag = R - sqrt(R^2 - r^2)
    probe = torch.tensor([0.0, 0.5 * a, 0.9 * a], dtype=torch.float64)
    got = curved_contact(probe, torch.zeros_like(probe), _p(depth, form, theta=theta))
    sag = RADIUS - torch.sqrt(RADIUS**2 - probe**2)
    torch.testing.assert_close(got, depth - sag / UNIT, atol=1e-9, rtol=0)
    # exactly zero at and beyond the contact radius
    edge = torch.tensor([a * (1 + 1e-9), 1.5 * a, RADIUS * 1.2], dtype=torch.float64)
    zero = curved_contact(edge, torch.zeros_like(edge), _p(depth, form, theta=theta))
    assert bool((zero == 0).all())


def test_the_footprint_grows_during_the_rise():
    draw = _draw()
    frames = render([draw], CANVAS, TIMES, dtype=torch.float64)[0]
    env = _env(draw)
    # first rise: delay 10, touch 40; half the rise is t = 30 (env 0.5)
    k_half = int(torch.argmin((TIMES - 30.0).abs()))
    k_full = int(torch.argmax(env))
    assert float(env[k_half]) == pytest.approx(0.5)
    want = curved_contact(CANVAS.xx, CANVAS.yy, _p(0.8 * 0.5))
    torch.testing.assert_close(frames[k_half], want, atol=1e-12, rtol=0)
    area_half = int((frames[k_half] > 0).sum())
    area_full = int((frames[k_full] > 0).sum())
    assert 0 < area_half < area_full
    # radius of the contact at half depth, from the geometry
    delta = 0.8 * 0.5 * UNIT
    a_half = math.sqrt(2.0 * RADIUS * delta - delta**2)
    x = CANVAS.xx[:, 0]
    reach = float(x[frames[k_half][:, _centre_index()[1]] > 0].abs().max())
    assert a_half - 0.02 <= reach <= a_half + 1e-9


def test_a_cylinder_is_constant_along_its_axis():
    world = _world({"form": "cylinder", "orientation_deg": 0.0})
    draw = sample(world, n=1, seed=1)[0]
    frames = render([draw], CANVAS, TIMES, dtype=torch.float64)[0]
    k = int(torch.argmax(_env(draw)))
    frame = frames[k]
    # orientation 0: across = y, the axis runs along x (down the rows of the grid)
    # xx runs down the rows and yy along the columns: across = y varies by column
    across_profile = frame[0]
    for row in (5, 20, 35):
        assert torch.equal(frame[row], across_profile)
    assert float(frame.max()) > 0 and not torch.equal(
        frame[:, 0], frame[:, _centre_index()[1]]
    )


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"layer_extra": {"clamp_min": 0.0}},
        {"axes": {"background": {"value": 0.3}}},
        {"layer_extra": {"clamp_min": 0.0}, "axes": {"background": {"value": 0.3}}},
    ],
)
def test_depth_zero_is_exactly_zero(extra):
    draw = sample(_world(**extra), n=1, seed=5)[0]
    frames = render([draw], CANVAS, TIMES, dtype=torch.float64)[0]
    env = _env(draw)
    assert bool((env == 0).any())
    quiet = env == 0
    assert bool((frames[quiet] == 0).all())
    # also: lead-in (t < 10), the pause between contacts, and after end_ms
    assert bool((frames[TIMES < 10] == 0).all())
    assert bool((frames[TIMES >= draw.end_ms] == 0).all())
    assert float(frames.abs().max()) > 0


def test_background_adds_its_own_draw_and_the_floor_holds():
    draw = sample(_world(axes={"background": {"value": 0.3}}), n=1, seed=5)[0]
    frames = render([draw], CANVAS, TIMES, dtype=torch.float64)[0]
    env = _env(draw)
    i, j = _centre_index()
    torch.testing.assert_close(frames[:, i, j], env * (0.8 + 0.3), atol=1e-12, rtol=0)
    # outside the contact only the background remains
    torch.testing.assert_close(frames[:, 0, 0], env * 0.3, atol=1e-12, rtol=0)


def test_a_default_background_reaches_an_indenter():
    """A ``background`` set in ``defaults`` binds every layered class, the
    indenters included (contract section 9)."""
    shape = {"kind": "curved_contact", "radius_mm": RADIUS, "unit_mm": UNIT}
    world = World.from_dict(
        {
            "world": {
                "defaults": {"background": {"value": 0.3}},
                "classes": {
                    "c": {
                        "layer": {"shape": shape},
                        "axes": {**EPISODE, "amplitude": {"value": 0.8}},
                    }
                },
            }
        }
    )
    draw = sample(world, n=1, seed=5)[0]
    assert draw.values["background"] == 0.3
    frames = render([draw], CANVAS, TIMES, dtype=torch.float64)[0]
    torch.testing.assert_close(frames[:, 0, 0], _env(draw) * 0.3, atol=1e-12, rtol=0)


def _layered_curved(draw, canvas, total):
    return layered.render_layers(
        [draw.to_layer()],
        canvas.xx.float(),
        canvas.yy.float(),
        dt_ms=1.0,
        total_ms=total,
    ).double()


@pytest.mark.parametrize("form", ["sphere", "cylinder"])
def test_layered_equals_the_world_renderer(form):
    world = _world(
        {"form": form, "orientation_deg": 30.0},
        layer_extra={
            "clamp_min": 0.0,
            "pattern": {"kind": "grid", "rows": 2, "cols": 2, "spacing_mm": 0.4},
        },
        axes={"background": {"value": 0.2}},
    )
    draw = sample(world, n=1, seed=2)[0]
    total = math.ceil(draw.end_ms) + 5.0
    times = torch.arange(0, total, 1.0, dtype=torch.float64)
    frames = render([draw], CANVAS, times, dtype=torch.float64)[0]
    torch.testing.assert_close(
        frames, _layered_curved(draw, CANVAS, total)[: len(times)], atol=1e-5, rtol=0
    )


def test_a_registered_indenter_works_in_a_layered_stimulus():
    def cone(x, y, p):
        r = torch.sqrt(x**2 + y**2)
        return (p["depth"] - r).clamp(min=0.0)

    specs = [layered._f("amplitude", 1.0, 0.0, 10.0)]
    kernel.register_shape("test_cone", cone, specs, indenter=True)
    try:
        assert kernel.SHAPE_KINDS["test_cone"].indenter
        layer = layered.default_layer()
        layer["shape"] = {"kind": "test_cone", "amplitude": 0.5}
        layer["timing"] = {
            "onset_ms": 0,
            "ramp_up_ms": 10,
            "hold_ms": 10,
            "ramp_down_ms": 10,
        }
        c = torch.linspace(-1.0, 1.0, 9)
        xx, yy = c.view(-1, 1).expand(9, 9), c.view(1, -1).expand(9, 9)
        frames = layered.render_layers([layer], xx, yy, dt_ms=1.0, total_ms=30.0)
        # at t = 5 the ramp is at 0.5: the depth is 0.5 * 0.5
        want = cone(xx, yy, {"depth": torch.tensor(0.25)})
        torch.testing.assert_close(frames[5], want, atol=1e-6, rtol=0)
        assert float(frames[5].max()) == pytest.approx(0.25)
        assert bool((frames[29] <= 0.5 / 10 + 1e-6).all())
    finally:
        del kernel.SHAPE_KINDS["test_cone"]


def test_non_indenter_shapes_are_unchanged():
    # the v1.1.0 guard (test_world_v1_1_compat.py) holds the bytes; here the
    # registered kinds carry the flag off and layered's built-ins are not indenters
    for name, kind in kernel.SHAPE_KINDS.items():
        assert kind.indenter is (name in ("curved_contact", "step_edge"))


def test_a_cylinder_and_a_sphere_differ():
    x, y = torch.tensor([0.0, 0.1]), torch.tensor([0.0, 0.1])
    a = curved_contact(x.double(), y.double(), _p(1.0, "sphere"))
    b = curved_contact(x.double(), y.double(), _p(1.0, "cylinder"))
    assert float(a[0]) == float(b[0]) == 1.0
    assert float(a[1]) != float(b[1])


def test_unknown_form_raises():
    x, y = _line(5)
    with pytest.raises(ValueError, match="form"):
        curved_contact(x, y, _p(1.0, form="cone"))


# ------------------------------------------------------------------ step_edge

RHO = 0.5


def _sp(depth, rho=RHO, unit=UNIT, theta=0.0):
    t = lambda v: torch.tensor(float(v), dtype=torch.float64)  # noqa
    return {
        "depth": t(depth),
        "shoulder_radius_mm": t(rho),
        "unit_mm": t(unit),
        "orientation_deg": t(theta),
    }


def test_step_edge_is_a_registered_indenter():
    kind = kernel.SHAPE_KINDS["step_edge"]
    assert kind.indenter and not kind.unbounded


def test_plate_side_is_the_depth():
    x = torch.tensor([-3.0, -1.0, -1e-9, 0.0], dtype=torch.float64)
    got = step_edge(x, torch.zeros_like(x), _sp(1.7, theta=90.0))
    # theta = 90: across = x, so the plate lies where x <= 0
    torch.testing.assert_close(got, torch.full_like(x, 1.7), atol=0, rtol=0)


def test_shoulder_profile_is_circular():
    depth = 4.0  # delta = 0.2 < rho
    p = torch.tensor([0.05, 0.1, 0.2, 0.3], dtype=torch.float64)
    got = step_edge(p, torch.zeros_like(p), _sp(depth, theta=90.0))
    sag = RHO - torch.sqrt(RHO**2 - p**2)
    torch.testing.assert_close(got, depth - sag / UNIT, atol=1e-12, rtol=0)


@pytest.mark.parametrize("depth", [1.0, 4.0, 8.0])
def test_contact_ends_where_the_shoulder_rises_above_the_depth(depth):
    delta = depth * UNIT
    x = torch.linspace(1e-4, 1.2 * RHO, 20001, dtype=torch.float64)
    got = step_edge(x, torch.zeros_like(x), _sp(depth, theta=90.0))
    if delta < RHO:
        a = math.sqrt(2.0 * RHO * delta - delta**2)
        assert bool((got[x < a - 1e-4] > 0).all())
        assert bool((got[x > a + 1e-4] == 0).all())
    else:
        # the contact runs to the end of the shoulder, then drops to 0
        below = x < RHO - 1e-9
        assert bool((got[below] > 0).all())
        last = step_edge(
            torch.tensor([RHO * (1 - 1e-12)], dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            _sp(depth, theta=90.0),
        )
        assert float(last) == pytest.approx(depth - RHO / UNIT, abs=1e-6)
        assert bool((got[x >= RHO] == 0).all())


def test_a_sharp_step_is_a_hard_edge():
    x = torch.tensor([-0.2, -1e-9, 1e-9, 0.2], dtype=torch.float64)
    got = step_edge(x, torch.zeros_like(x), _sp(2.0, rho=0.0, theta=90.0))
    assert got.tolist() == [2.0, 2.0, 0.0, 0.0]


def test_orientation_flips_the_plate_side():
    x = torch.tensor([-0.3, 0.3], dtype=torch.float64)
    z = torch.zeros_like(x)
    a = step_edge(x, z, _sp(2.0, rho=0.0, theta=90.0))
    b = step_edge(x, z, _sp(2.0, rho=0.0, theta=270.0))
    assert a.tolist() == [2.0, 0.0]
    assert b.tolist() == [0.0, 2.0]


def test_a_step_is_constant_along_its_edge():
    t = torch.linspace(-0.5, 0.5, 11, dtype=torch.float64)
    x = torch.full_like(t, 0.1)
    got = step_edge(x, t, _sp(4.0, theta=90.0))
    assert bool((got == got[0]).all())


def test_step_depth_zero_is_exactly_zero():
    x, y = _line(101)
    got = step_edge(x, y, _sp(0.0, theta=90.0))
    assert bool((got == 0).all())
    draw = sample(_step_world(), n=1, seed=3)[0]
    frames = render([draw], CANVAS, TIMES, dtype=torch.float64)[0]
    assert bool((frames[TIMES < 10] == 0).all())
    assert bool((frames[TIMES >= draw.end_ms] == 0).all())
    assert float(frames.abs().max()) > 0


def _step_world():
    return World.from_dict(
        {
            "world": {
                "classes": {
                    "c": {
                        "layer": {
                            "shape": {
                                "kind": "step_edge",
                                "shoulder_radius_mm": RHO,
                                "unit_mm": UNIT,
                            },
                            "clamp_min": 0.0,
                        },
                        "axes": {
                            **EPISODE,
                            "amplitude": {"value": 0.8},
                            "orientation_deg": {"value": 90.0},
                        },
                    }
                }
            }
        }
    )


def test_step_layered_equals_the_world_renderer():
    world = World.from_dict(
        {
            "world": {
                "classes": {
                    "c": {
                        "layer": {
                            "shape": {
                                "kind": "step_edge",
                                "shoulder_radius_mm": 0.3,
                                "unit_mm": UNIT,
                            },
                            "clamp_min": 0.0,
                        },
                        "axes": {
                            **EPISODE,
                            "amplitude": {"value": 0.8},
                            "orientation_deg": {"value": 40.0},
                            "background": {"value": 0.2},
                        },
                    }
                }
            }
        }
    )
    draw = sample(world, n=1, seed=2)[0]
    total = math.ceil(draw.end_ms) + 5.0
    times = torch.arange(0, total, 1.0, dtype=torch.float64)
    frames = render([draw], CANVAS, times, dtype=torch.float64)[0]
    torch.testing.assert_close(
        frames, _layered_curved(draw, CANVAS, total)[: len(times)], atol=1e-5, rtol=0
    )
