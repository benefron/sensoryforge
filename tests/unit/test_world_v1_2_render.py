"""Every v1.2 element class: equals layered, batch invariant, grid invariant."""

import math
from pathlib import Path

import pytest
import torch

from sensoryforge.stimuli.layered import render_layers
from sensoryforge.world import Canvas, load_world, movie_times, render, sample

WORLD = load_world(
    Path(__file__).resolve().parents[1] / "fixtures" / "worlds" / "elements_v1_2.yml"
)
CLASSES = sorted(n for n in WORLD.classes if WORLD.classes[n].kind != "quiet")
CANVAS = Canvas.from_grid(rows=24, cols=24, spacing_mm=0.05)
DT = 1.0


def _layered(draw, total_ms, canvas=CANVAS):
    return render_layers(
        [draw.to_layer()],
        canvas.xx.float(),
        canvas.yy.float(),
        dt_ms=DT,
        total_ms=total_ms,
    ).double()


@pytest.mark.parametrize("class_name", CLASSES)
def test_v1_2_class_equals_layered(class_name):
    draws = sample(WORLD, n=3, seed=21, classes=[class_name])
    total = math.ceil(max(d.end_ms for d in draws)) + 5.0
    frames = render(draws, CANVAS, movie_times(DT, total), dtype=torch.float64)
    for i, draw in enumerate(draws):
        torch.testing.assert_close(frames[i], _layered(draw, total), atol=1e-5, rtol=0)


@pytest.mark.parametrize("class_name", CLASSES)
@pytest.mark.parametrize("max_elements", [None, 1])
def test_v1_2_draw_alone_equals_draw_in_a_batch(class_name, max_elements):
    draws = sample(WORLD, n=4, seed=8, classes=[class_name])
    total = math.ceil(max(d.end_ms for d in draws)) + 5.0
    times = movie_times(DT, total)
    kwargs = {} if max_elements is None else {"max_elements": max_elements}
    batch = render(draws, CANVAS, times, dtype=torch.float64, **kwargs)
    for i, draw in enumerate(draws):
        alone = render([draw], CANVAS, times, dtype=torch.float64, **kwargs)
        assert torch.equal(alone[0], batch[i])


@pytest.mark.parametrize("class_name", CLASSES)
def test_v1_2_draw_agrees_on_40x40_and_80x80(class_name):
    draw = sample(WORLD, n=1, seed=4, classes=[class_name])[0]
    total = math.ceil(draw.end_ms) + 2.0
    times = movie_times(DT, total)
    # 41x41 at 0.15 mm and 81x81 at 0.075 mm span the same 6 mm and nest:
    # every coarse point is a fine one (an even grid would sit half a pitch off)
    coarse = Canvas.from_grid(rows=41, cols=41, spacing_mm=0.15)
    fine = Canvas.from_grid(rows=81, cols=81, spacing_mm=0.075)
    a = render([draw], coarse, times, dtype=torch.float64)[0]
    b = render([draw], fine, times, dtype=torch.float64)[0]
    # xx runs down the rows and yy along the columns (Canvas.from_grid)
    ax, ay = coarse.xx[:, 0], coarse.yy[0]
    bx, by = fine.xx[:, 0], fine.yy[0]
    ir = torch.stack([(bx - x).abs().argmin() for x in ax])
    ic = torch.stack([(by - y).abs().argmin() for y in ay])
    assert torch.allclose(bx[ir], ax, atol=1e-9)
    assert torch.allclose(by[ic], ay, atol=1e-9)
    torch.testing.assert_close(a, b[:, ir][:, :, ic], atol=1e-12, rtol=0)


@pytest.mark.parametrize("class_name", CLASSES)
def test_v1_2_determinism_by_seed(class_name):
    one = sample(WORLD, n=3, seed=2, classes=[class_name])
    two = sample(WORLD, n=3, seed=2, classes=[class_name])
    assert [d.to_dict() for d in one] == [d.to_dict() for d in two]
    total = math.ceil(max(d.end_ms for d in one)) + 2.0
    times = movie_times(DT, total)
    assert torch.equal(
        render(one, CANVAS, times, dtype=torch.float64),
        render(two, CANVAS, times, dtype=torch.float64),
    )
