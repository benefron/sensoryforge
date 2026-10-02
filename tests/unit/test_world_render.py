"""Rendering: equal to layered; batch invariance; windows; grids; quiet; sessions."""

import math
from pathlib import Path

import pytest
import torch

from sensoryforge.config.schema import GridConfig
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
