"""Rendering: equal to layered; batch invariance; windows; grids; quiet; sessions."""

import math
from pathlib import Path

import pytest
import torch

from sensoryforge.config.schema import GridConfig
from sensoryforge.stimuli import layered
from sensoryforge.stimuli.canvas import stimulus_canvas
from sensoryforge.stimuli.layered import render_layers
from sensoryforge.world import (
    Canvas,
    World,
    load_world,
    movie_times,
    render,
    render_movie,
    sample,
    session,
)

WORLD = load_world(
    Path(__file__).resolve().parents[1] / "fixtures" / "worlds" / "tactile_small.yml"
)
CANVAS = Canvas.from_grid(rows=24, cols=24, spacing_mm=0.05)
DT = 1.0


def _layered(draw, total_ms):
    xx, yy = CANVAS.xx.float(), CANVAS.yy.float()
    return render_layers(
        [draw.to_layer()], xx, yy, dt_ms=DT, total_ms=total_ms
    ).double()


@pytest.mark.parametrize(
    "class_name", sorted(n for n in WORLD.classes if n != "quiet") + ["gratings"]
)
def test_world_render_equals_layered(class_name):
    draws = sample(WORLD, n=3, seed=21, classes=[class_name])
    total = math.ceil(max(d.end_ms for d in draws)) + 5.0
    frames = render(draws, CANVAS, movie_times(DT, total), dtype=torch.float64)
    for i, draw in enumerate(draws):
        torch.testing.assert_close(frames[i], _layered(draw, total), atol=1e-5, rtol=0)


#: What the fixture world lacks: grid, list and random patterns, declared
#: linear and path motion, gabor, a signed square grating, a hard disc and a
#: flat bar (the last three hard-edged).
EXTRA = World.from_dict(
    {
        "world": {
            "defaults": {
                "delay_ms": {"range": [0, 20]},
                "touch_ms": {"range": [5, 15]},
                "hold_ms": {"range": [20, 60]},
                "release_ms": {"range": [5, 15]},
                "speed_mm_per_ms": {"range": [0.005, 0.02], "dist": "log_uniform"},
                "direction_deg": {"range": [0, 360], "circular": True},
                "amplitude": {"range": [0.5, 1.0]},
                "x_mm": {"range": [-0.2, 0.2]},
                "y_mm": {"range": [-0.2, 0.2]},
            },
            "classes": {
                "grid_dots": {
                    "layer": {
                        "shape": {"kind": "gaussian", "sigma_mm": 0.08},
                        "pattern": {"kind": "grid", "rows": 2, "cols": 3},
                    },
                    "axes": {
                        "spacing_mm": {"range": [0.2, 0.3]},
                        "mask": {"values": ["", "101 011", "111 010"]},
                    },
                },
                "list_dots": {
                    "layer": {
                        "shape": {"kind": "gaussian"},
                        "pattern": {
                            "kind": "list",
                            "positions": [[-0.3, 0.1], [0.25, -0.2], [0.0, 0.3]],
                            "amplitudes": [1.0, 0.6, 0.8],
                        },
                    },
                    "axes": {"sigma_mm": {"range": [0.05, 0.15]}},
                },
                "random_dots": {
                    "layer": {
                        "shape": {"kind": "gaussian", "sigma_mm": 0.07},
                        "pattern": {
                            "kind": "random",
                            "count": 6,
                            "width_mm": 0.8,
                            "height_mm": 0.8,
                            "min_distance_mm": 0.1,
                            "amplitude_jitter": 0.3,
                        },
                    },
                    "axes": {"seed": {"range": [0, 1000], "int": True}},
                },
                "linear_slide": {
                    "layer": {
                        "shape": {"kind": "gaussian", "sigma_mm": 0.15},
                        "motion": {
                            "kind": "linear",
                            "start": [-0.3, 0],
                            "end": [0.3, 0.2],
                        },
                    },
                    "axes": {
                        "slide_ms": {"range": [20, 40]},
                        "hold_ms": {"range": [5, 20]},
                    },
                },
                "path_slide": {
                    "layer": {
                        "shape": {"kind": "gaussian", "sigma_mm": 0.15},
                        "motion": {
                            "kind": "path",
                            "waypoints": [[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3]],
                        },
                    },
                    "axes": {
                        "slide_ms": {"range": [20, 40]},
                        "hold_ms": {"range": [5, 20]},
                        "contacts": {"value": 2},
                        "pause_ms": {"range": [5, 10]},
                    },
                },
                "gabor": {
                    "layer": {"shape": {"kind": "gabor"}},
                    "axes": {
                        "sigma_mm": {"range": [0.2, 0.4]},
                        "wavelength_mm": {"range": [0.2, 0.5]},
                        "orientation_deg": {"range": [0, 180]},
                        "phase_deg": {"range": [0, 360]},
                        "signed": {"values": [True, False]},
                    },
                },
                "square_signed": {
                    "layer": {
                        "shape": {
                            "kind": "grating",
                            "profile": "square",
                            "signed": True,
                        }
                    },
                    "axes": {
                        "wavelength_mm": {"range": [0.3, 0.6]},
                        "duty": {"range": [0.2, 0.7]},
                        "orientation_deg": {"range": [0, 180]},
                        "phase_deg": {"range": [0, 360]},
                        "slide_ms": {"range": [10, 20]},
                    },
                },
                "hard_disc": {
                    "layer": {
                        "shape": {"kind": "disc", "edge_mm": 0},
                        "pattern": {"kind": "grid", "rows": 1, "cols": 2},
                    },
                    "axes": {
                        "diameter_mm": {"range": [0.2, 0.4]},
                        "spacing_mm": {"range": [0.4, 0.5]},
                        "slide_ms": {"range": [10, 20]},
                    },
                },
                "flat_bar": {
                    "layer": {"shape": {"kind": "bar", "profile": "flat"}},
                    "axes": {
                        "width_mm": {"range": [0.1, 0.3]},
                        "length_mm": {"range": [0.3, 0.8]},
                        "orientation_deg": {"range": [0, 180]},
                        "slide_ms": {"range": [10, 20]},
                    },
                },
            },
        }
    }
)
HARD = {"square_signed", "hard_disc", "flat_bar"}
EDGE_MM = 1e-4


def _edge_band(draw, total_ms):
    """``[T, *S]``: canvas points within 1e-4 mm of a hard edge at each frame.

    Computed from the draw's geometry in float64: the element positions,
    the motion offset at each frame, and the shape's edges (a disc's rim, a
    flat bar's sides and ends, a square grating's stripe borders).
    """
    layer = draw.to_layer()
    shape = layer["shape"]
    times = movie_times(DT, total_ms)
    offsets = layered.motion_offsets(layer["motion"], layer["timing"], times, total_ms)
    if offsets is None:
        offsets = torch.zeros(len(times), 2, dtype=torch.float64)
    X, Y = CANVAS.xx.unsqueeze(0), CANVAS.yy.unsqueeze(0)
    ox, oy = offsets[:, 0].view(-1, 1, 1), offsets[:, 1].view(-1, 1, 1)
    if shape["kind"] == "grating":
        theta = math.radians(shape["orientation_deg"])
        across = (X - ox) * math.cos(theta) + (Y - oy) * math.sin(theta)
        wavelength = shape["wavelength_mm"]
        phase = torch.remainder(
            2.0 * math.pi * across / wavelength + math.radians(shape["phase_deg"]),
            2.0 * math.pi,
        )
        centred = torch.minimum(phase, 2.0 * math.pi - phase)
        gap = (centred - math.pi * shape["duty"]).abs() * wavelength / (2.0 * math.pi)
        return gap < EDGE_MM
    band = torch.zeros((len(times),) + CANVAS.shape, dtype=torch.bool)
    positions, _ = layered.pattern_positions(layer["pattern"])
    for px, py in positions:
        x, y = X - px - ox, Y - py - oy
        if shape["kind"] == "disc":
            rim = torch.sqrt(x**2 + y**2) - shape["diameter_mm"] / 2.0
            band |= rim.abs() < EDGE_MM
        else:
            theta = math.radians(shape["orientation_deg"])
            across = x * math.sin(theta) + y * math.cos(theta)
            along = x * math.cos(theta) - y * math.sin(theta)
            band |= (across.abs() - shape["width_mm"] / 2.0).abs() < EDGE_MM
            band |= (along.abs() - shape["length_mm"] / 2.0).abs() < EDGE_MM
    return band


@pytest.mark.parametrize("class_name", sorted(EXTRA.classes))
def test_every_shape_pattern_and_motion_equals_layered(class_name):
    draws = sample(EXTRA, n=4, seed=22, classes=[class_name])
    total = math.ceil(max(d.end_ms for d in draws)) + 5.0
    frames = render(draws, CANVAS, movie_times(DT, total), dtype=torch.float64)
    assert float(frames.abs().amax()) > 0.1
    for i, draw in enumerate(draws):
        want = _layered(draw, total)
        keep = torch.ones_like(want, dtype=torch.bool)
        if class_name in HARD:
            keep = ~_edge_band(draw, total)
            assert float(keep.double().mean()) > 0.95
        torch.testing.assert_close(frames[i][keep], want[keep], atol=1e-5, rtol=0)


def test_draw_i_alone_equals_draw_i_in_a_batch_bit_for_bit():
    draws = sample(WORLD, n=40, seed=8)
    times = movie_times(DT, 120.0)
    batch = render(draws, CANVAS, times, dtype=torch.float64)
    one_per_chunk = render(draws, CANVAS, times, dtype=torch.float64, max_elements=1)
    assert torch.equal(batch, one_per_chunk)
    for i in (0, 7, 39):
        assert torch.equal(
            render([draws[i]], CANVAS, times, dtype=torch.float64)[0], batch[i]
        )


def test_windows_equal_movie_frames_bit_for_bit():
    draws = sample(WORLD, n=12, seed=9)
    movie = render(draws, CANVAS, movie_times(DT, 120.0), dtype=torch.float64)
    steps = [(k - 8, k, k + 8) for k in range(10, 22)]
    windows = render(
        draws, CANVAS, torch.tensor(steps, dtype=torch.float64), dtype=torch.float64
    )
    for i, triple in enumerate(steps):
        for j, step in enumerate(triple):
            assert torch.equal(windows[i, j], movie[i, step])


def test_a_draw_agrees_on_40x40_and_80x80_where_they_overlap():
    small = Canvas.from_grid(40, 40, 0.15)
    large = Canvas.from_grid(80, 80, 0.15)
    torch.testing.assert_close(small.xx, large.xx[20:60, 20:60], atol=1e-12, rtol=0)
    draws = sample(WORLD, n=6, seed=10)
    times = torch.tensor([30.0, 60.0], dtype=torch.float64)
    a = render(draws, small, times, dtype=torch.float64)
    b = render(draws, large, times, dtype=torch.float64)[:, :, 20:60, 20:60]
    torch.testing.assert_close(a, b, atol=1e-12, rtol=0)


def test_quiet_is_exactly_zero():
    draws = sample(WORLD, n=200, seed=12)
    frames = render(draws, CANVAS, movie_times(DT, 220.0), dtype=torch.float64)
    for i, d in enumerate(draws):
        if d.class_name == "quiet":
            assert torch.count_nonzero(frames[i]) == 0
            continue
        lead = int(d.values["delay_ms"])  # frames k < delay_ms
        assert torch.count_nonzero(frames[i, :lead]) == 0
        end = int(math.ceil(d.end_ms + 1e-9))
        assert torch.count_nonzero(frames[i, end:]) == 0
        if d.class_name == "twice":
            pause_from = d.values["delay_ms"] + sum(
                d.values[f] for f in ("touch_ms", "hold_ms", "slide_ms", "release_ms")
            )
            first, last = math.ceil(pause_from + 1e-9), math.floor(
                pause_from + d.values["pause_ms"] - 1e-9
            )
            assert torch.count_nonzero(frames[i, first : last + 1]) == 0


STEP_RELEASE = World.from_dict(
    {
        "world": {
            "defaults": {
                "delay_ms": {"range": [0, 20]},
                "touch_ms": {"range": [0, 15]},
                "hold_ms": {"range": [5, 60]},
                "release_ms": {"value": 0},
                "x_mm": {"range": [-0.2, 0.2]},
                "y_mm": {"range": [-0.2, 0.2]},
            },
            "classes": {
                "dots": {"layer": {"shape": {"kind": "gaussian", "sigma_mm": 0.3}}},
                "twice": {
                    "layer": {"shape": {"kind": "gaussian", "sigma_mm": 0.3}},
                    "axes": {"contacts": {"value": 2}, "pause_ms": {"range": [3, 9]}},
                },
                "slides": {
                    "layer": {"shape": {"kind": "gaussian", "sigma_mm": 0.3}},
                    "axes": {
                        "slide_ms": {"range": [5, 20]},
                        "speed_mm_per_ms": {"value": 0.01},
                    },
                },
            },
        }
    }
)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_a_step_release_is_exactly_zero_from_end_ms(dtype):
    draws = sample(STEP_RELEASE, n=300, seed=4)
    ends = torch.tensor([d.end_ms for d in draws], dtype=torch.float64)
    after = torch.nextafter(ends, torch.tensor(math.inf, dtype=torch.float64))
    at_end = render(draws, CANVAS, torch.stack([ends, after], 1), dtype=dtype)
    assert torch.count_nonzero(at_end) == 0
    # Not vacuous: half a millisecond earlier every draw is still touching.
    before = render(draws, CANVAS, (ends - 0.5).unsqueeze(1), dtype=dtype)
    assert bool((before.flatten(1).abs().amax(1) > 0.1).all())


def test_a_session_renders_each_draw_from_its_start():
    s = session(WORLD, duration_ms=400.0, seed=13)
    times = movie_times(DT, 400.0)
    frames = render([s], CANVAS, times, dtype=torch.float64)[0]
    assert torch.equal(frames, render_movie(s, CANVAS, DT, 400.0, dtype=torch.float64))
    for j, (start, draw) in enumerate(s.items):
        stop = s.items[j + 1][0] if j + 1 < len(s.items) else 400.0
        ks = [k for k in range(400) if start <= k < stop]
        if not ks:
            continue
        local = torch.tensor([k - start for k in ks], dtype=torch.float64)
        alone = render([draw], CANVAS, local, dtype=torch.float64)[0]
        assert torch.equal(frames[ks], alone)


def test_records_render_with_their_world():
    draws = sample(WORLD, n=5, seed=14)
    times = torch.tensor([20.0], dtype=torch.float64)
    direct = render(draws, CANVAS, times, dtype=torch.float64)
    from_records = render(
        [d.to_dict() for d in draws], CANVAS, times, dtype=torch.float64, world=WORLD
    )
    assert torch.equal(direct, from_records)
    with pytest.raises(ValueError, match="needs world="):
        render([draws[0].to_dict()], CANVAS, times)


def test_random_pattern_seed_axis_matches_layered():
    world = World.from_dict(
        {
            "world": {
                "classes": {
                    "bumps": {
                        "layer": {
                            "shape": {"kind": "gaussian", "sigma_mm": 0.1},
                            "pattern": {
                                "kind": "random",
                                "count": 5,
                                "width_mm": 1.0,
                                "height_mm": 1.0,
                            },
                        },
                        "axes": {
                            "seed": {"range": [0, 1000], "int": True},
                            "hold_ms": {"value": 20},
                        },
                    }
                }
            }
        }
    )
    draws = sample(world, n=4, seed=1)
    assert len({d.values["seed"] for d in draws}) > 1
    frames = render(draws, CANVAS, movie_times(DT, 25.0), dtype=torch.float64)
    for i, draw in enumerate(draws):
        torch.testing.assert_close(frames[i], _layered(draw, 25.0), atol=1e-5, rtol=0)


def test_a_string_axis_splits_groups_without_changing_results():
    world = World.from_dict(
        {
            "world": {
                "classes": {
                    "bars": {
                        "layer": {"shape": {"kind": "bar", "length_mm": 0.0}},
                        "axes": {
                            "profile": {"values": ["gaussian", "flat"]},
                            "width_mm": {"range": [0.1, 0.3]},
                            "hold_ms": {"value": 20},
                        },
                    }
                }
            }
        }
    )
    draws = sample(world, n=10, seed=2)
    assert {d.values["profile"] for d in draws} == {"gaussian", "flat"}
    times = movie_times(DT, 25.0)
    batch = render(draws, CANVAS, times, dtype=torch.float64)
    for i, draw in enumerate(draws):
        assert torch.equal(
            batch[i], render([draw], CANVAS, times, dtype=torch.float64)[0]
        )


def test_a_multi_channel_world_draws_each_class_on_its_plane():
    world = World.from_dict(
        {
            "world": {
                "channels": ["pressure", "vibration"],
                "classes": {
                    "press": {
                        "layer": {"shape": {"kind": "gaussian"}},
                        "axes": {"hold_ms": {"value": 20}},
                    },
                    "buzz": {
                        "channel": "vibration",
                        "layer": {
                            "shape": {"kind": "gaussian"},
                            "modulation": {"kind": "sine", "frequency_hz": 100.0},
                        },
                        "axes": {"hold_ms": {"value": 20}},
                    },
                },
            }
        }
    )
    draws = sample(world, n=20, seed=3)
    frames = render(draws, CANVAS, [2.0], dtype=torch.float64)
    assert frames.shape == (20, 1, 2, 24, 24)
    for i, draw in enumerate(draws):
        on = 0 if draw.class_name == "press" else 1
        assert float(frames[i, 0, on].abs().max()) > 0.0
        assert float(frames[i, 0, 1 - on].abs().max()) == 0.0


def test_float64_on_mps_is_refused():
    with pytest.raises(ValueError, match="MPS has no float64"):
        render(
            sample(WORLD, n=1, seed=0), CANVAS, [0.0], dtype=torch.float64, device="mps"
        )


def test_canvas_from_a_grid_config_spans_the_stimulus_canvas():
    grid = GridConfig(name="g", rows=8, cols=8, spacing=0.15)
    canvas = Canvas.from_grid_config(grid)
    torch.testing.assert_close(
        canvas.xx, stimulus_canvas(grid).xx.double(), atol=1e-6, rtol=0
    )
    assert canvas.shape == (8, 8) and canvas.xx.dtype == torch.float64
    assert Canvas.from_points(torch.tensor([[0.0, 0.0], [0.1, 0.2]])).shape == (2,)


def _record_render_group(monkeypatch):
    from sensoryforge.world.kinds import LayeredKind, QuietKind

    calls = []
    for cls in (LayeredKind, QuietKind):
        original = cls.render_group

        def wrapped(self, spec, draws, X, Y, times, _orig=original):
            calls.append((len(draws), times.shape[1], X.numel()))
            return _orig(self, spec, draws, X, Y, times)

        monkeypatch.setattr(cls, "render_group", wrapped)
    return calls


def test_a_session_evaluates_each_frame_once(monkeypatch):
    s = session(WORLD, duration_ms=600.0, seed=13)
    calls = _record_render_group(monkeypatch)
    frames = render_movie(s, CANVAS, 1.0, 600.0, dtype=torch.float64)
    assert sum(g * k for g, k, _ in calls) <= 600
    monkeypatch.undo()
    for j, (start, draw) in enumerate(s.items):
        stop = s.items[j + 1][0] if j + 1 < len(s.items) else 600.0
        ks = [k for k in range(600) if start <= k < stop]
        if not ks:
            continue
        local = torch.tensor([k - start for k in ks], dtype=torch.float64)
        alone = render([draw], CANVAS, local, dtype=torch.float64)[0]
        assert torch.equal(frames[ks], alone)


def test_no_render_call_exceeds_the_element_budget(monkeypatch):
    draws = sample(WORLD, n=3, seed=5)
    times = movie_times(1.0, 300.0)
    budget = 24 * 24 * 7
    calls = _record_render_group(monkeypatch)
    small = render(draws, CANVAS, times, dtype=torch.float64, max_elements=budget)
    assert calls and all(g * k * n <= budget for g, k, n in calls)
    monkeypatch.undo()
    default = render(draws, CANVAS, times, dtype=torch.float64)
    assert torch.equal(small, default)
