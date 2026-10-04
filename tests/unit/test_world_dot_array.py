"""``dot_array``: a patch-filling lattice of dots (P9)."""

import math

import pytest
import torch

from sensoryforge.stimuli import layered
from sensoryforge.world import World, kernel
from sensoryforge.world.dataset import load_dataset
from sensoryforge.world.kernel import _gaussian
from sensoryforge.world.surfaces import dot_array


def _p(dtype=torch.float64, **kw):
    base = {
        "sigma_mm": 0.2,
        "spacing_mm": 1.0,
        "row_spacing_mm": 0.0,
        "orientation_deg": 0.0,
        "arrangement": "square",
    }
    base.update(kw)
    return {
        k: v if isinstance(v, str) else torch.tensor(float(v), dtype=dtype)
        for k, v in base.items()
    }


def _canvas(half=3.0, n=41, dtype=torch.float64):
    c = torch.linspace(-half, half, n, dtype=dtype)
    return c.view(-1, 1).expand(n, n).contiguous(), c.view(1, -1).expand(n, n)


def _brute(x, y, sigma, a, b, theta_deg, arrangement, dx=0.0, dy=0.0):
    """Explicit sum over every lattice site within 10 sigma of the canvas."""
    th = math.radians(theta_deg)
    reach = 10 * sigma + max(abs(x).max().item(), abs(y).max().item()) * 2 + 4 * a
    n = int(reach / min(a, b)) + 2
    total = torch.zeros_like(x)
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            off = 0.5 * a * (j % 2) if arrangement == "hexagonal" else 0.0
            lu, lv = i * a + off, j * b
            sx = dx + lu * math.cos(th) - lv * math.sin(th)
            sy = dy + lu * math.sin(th) + lv * math.cos(th)
            total = total + torch.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * sigma**2))
    return total


def test_registered_and_patch_filling():
    assert "dot_array" in kernel.SHAPE_KINDS
    assert not kernel.SHAPE_KINDS["dot_array"].unbounded


@pytest.mark.parametrize("arrangement", ["square", "hexagonal"])
def test_equals_a_brute_force_sum_over_lattice_sites(arrangement):
    x, y = _canvas(half=2.0, n=21)
    a, sigma, theta = 0.7, 0.2, 23.0
    b = a if arrangement == "square" else a * math.sqrt(3) / 2
    dx, dy = 0.13, -0.21
    got = dot_array(
        x - dx,
        y - dy,
        _p(
            sigma_mm=sigma,
            spacing_mm=a,
            orientation_deg=theta,
            arrangement=arrangement,
        ),
    )
    want = _brute(x, y, sigma, a, b, theta, arrangement, dx, dy)
    torch.testing.assert_close(got, want, atol=1e-12, rtol=0)


def test_explicit_row_spacing_is_used():
    x, y = _canvas(half=2.0, n=15)
    got = dot_array(x, y, _p(spacing_mm=0.8, row_spacing_mm=1.1))
    want = _brute(x, y, 0.2, 0.8, 1.1, 0.0, "square")
    torch.testing.assert_close(got, want, atol=1e-12, rtol=0)


def test_peak_is_amplitude_when_dots_are_far_apart():
    x, y = _canvas()
    h = dot_array(x, y, _p(sigma_mm=0.1, spacing_mm=1.0))
    assert h.max().item() == pytest.approx(1.0, abs=1e-12)
    assert h.min().item() >= 0.0


def test_equals_a_layered_grid_of_gaussians_on_the_canvas():
    x, y = _canvas(half=3.0, n=31)
    got = dot_array(x, y, _p(sigma_mm=0.25, spacing_mm=0.5))
    positions, _ = layered.pattern_positions(
        {"kind": "grid", "rows": 41, "cols": 41, "spacing_mm": 0.5}
    )
    want = torch.zeros_like(x)
    for px, py in positions:
        want = want + _gaussian(
            x - px, y - py, {"sigma_mm": torch.tensor(0.25, dtype=x.dtype)}
        )
    torch.testing.assert_close(got, want, atol=1e-12, rtol=0)


def test_a_batch_with_mixed_sigma_and_spacing_equals_each_alone_bit_for_bit():
    g = 4
    x, y = _canvas(half=2.0, n=13)
    x = x.expand(g, 2, 13, 13).contiguous()
    y = y.expand(g, 2, 13, 13).contiguous()

    def col(vals):
        return torch.tensor(vals, dtype=torch.float64).view(g, 1, 1, 1)

    p = {
        "sigma_mm": col([0.1, 0.3, 0.2, 0.5]),
        "spacing_mm": col([1.0, 0.5, 0.8, 0.4]),
        "row_spacing_mm": col([0.0, 0.0, 0.9, 0.0]),
        "orientation_deg": col([0.0, 30.0, 75.0, 10.0]),
        "arrangement": "hexagonal",
    }
    batch = dot_array(x, y, p)
    for i in range(g):
        alone = dot_array(
            x[i : i + 1],
            y[i : i + 1],
            {k: v if isinstance(v, str) else v[i : i + 1] for k, v in p.items()},
        )
        assert torch.equal(alone[0], batch[i])


def test_float32_and_float64_agree_to_1e_5():
    x, y = _canvas()
    a = dot_array(x, y, _p(arrangement="hexagonal", orientation_deg=17.0))
    b = dot_array(
        x.float(),
        y.float(),
        _p(torch.float32, arrangement="hexagonal", orientation_deg=17.0),
    )
    torch.testing.assert_close(a, b.double(), atol=1e-5, rtol=0)


def test_x_mm_shifts_the_lattice():
    layer = lambda x_mm: {  # noqa: E731
        "shape": {
            "kind": "dot_array",
            "sigma_mm": 0.2,
            "spacing_mm": 1.0,
        },
        "pattern": {"kind": "single", "x_mm": x_mm, "y_mm": 0.0},
        "timing": {
            "delay_ms": 0.0,
            "touch_ms": 0.0,
            "hold_ms": 10.0,
            "release_ms": 0.0,
        },
    }
    xs = torch.arange(20, dtype=torch.float32) * 0.1
    xx, yy = xs.view(-1, 1).expand(20, 20), xs.view(1, -1).expand(20, 20)
    base = layered.render_layers([layer(0.0)], xx, yy, dt_ms=1.0, total_ms=5.0)
    shifted = layered.render_layers([layer(0.5)], xx, yy, dt_ms=1.0, total_ms=5.0)
    assert not torch.allclose(base, shifted)
    torch.testing.assert_close(shifted[:, 5:], base[:, :-5], atol=1e-4, rtol=0)


def test_too_wide_a_dot_for_its_spacing_is_refused():
    x, y = _canvas(n=5)
    with pytest.raises(ValueError, match="sigma_mm.*spacing_mm"):
        dot_array(x, y, _p(sigma_mm=10.0, spacing_mm=0.1))


# ------------------------------------------- the site limit, checked at load


def _dots(shape, axes, fixed=None):
    """A world mapping with one dot-array class ``dots``."""
    return {
        "world": {
            "classes": {
                "dots": {
                    "layer": {"shape": {"kind": "dot_array", **shape}},
                    "axes": {"hold_ms": {"value": 10.0}, **axes},
                }
            },
            "fixed_draws": fixed or {},
        }
    }


def test_a_world_that_can_exceed_the_site_limit_fails_at_load_naming_class_and_axis():
    # m = ceil(7.5 * 10 / 1.3) = 58 sites each way at the range's worst case;
    # v1.2.0 loaded this world and failed only when such a draw was rendered
    with pytest.raises(
        ValueError,
        match=r"world\.classes\.dots\.axes\.sigma_mm.*58 lattice sites.*limit 32",
    ):
        World.from_dict(
            _dots({"spacing_mm": 1.3}, {"sigma_mm": {"range": [0.1, 10.0]}})
        )


def test_the_worst_spacing_names_the_spacing_axis():
    # m = ceil(7.5 * 0.5 / 0.05) = 75
    with pytest.raises(
        ValueError, match=r"world\.classes\.dots\.axes\.spacing_mm.*75 lattice sites"
    ):
        World.from_dict(
            _dots({"sigma_mm": 0.5}, {"spacing_mm": {"range": [0.05, 2.0]}})
        )


def test_the_load_check_takes_the_hexagonal_row_spacing():
    # rows of a hexagonal lattice are spacing * sqrt(3)/2 apart: sigma 1.0 over
    # spacing 0.25 needs ceil(7.5 / 0.2165) = 35 sites (square: 30, allowed)
    shape = {"sigma_mm": 1.0, "spacing_mm": 0.25}
    World.from_dict(_dots({**shape, "arrangement": "square"}, {}))
    with pytest.raises(ValueError, match=r"35 lattice sites"):
        World.from_dict(_dots({**shape, "arrangement": "hexagonal"}, {}))
    with pytest.raises(ValueError, match=r"35 lattice sites"):
        World.from_dict(
            _dots(shape, {"arrangement": {"values": ["square", "hexagonal"]}})
        )


def test_a_row_spacing_range_reaching_zero_fails_at_load():
    with pytest.raises(ValueError, match=r"axes\.row_spacing_mm.*close to 0"):
        World.from_dict(_dots({}, {"row_spacing_mm": {"range": [0.0, 2.0]}}))


def test_a_world_within_the_site_limit_loads():
    # m = ceil(7.5 * 1.0 / 0.3) = 25
    world = World.from_dict(
        _dots({"spacing_mm": 0.3}, {"sigma_mm": {"range": [0.1, 1.0]}})
    )
    assert "dots" in world.classes


def test_a_fixed_draw_beyond_the_site_limit_fails_at_load():
    # outside its axis range but inside the field's domain: allowed until v1.2.0
    with pytest.raises(
        ValueError, match=r"world\.fixed_draws\.wide.*sigma_mm.*38 lattice sites"
    ):
        World.from_dict(
            _dots(
                {"spacing_mm": 1.0},
                {"sigma_mm": {"range": [0.1, 0.2]}},
                fixed={"wide": {"class": "dots", "sigma_mm": 5.0}},
            )
        )


def test_probes_that_can_exceed_the_site_limit_fail_at_dataset_load():
    axes = {"spacing_mm": {"range": [0.3, 2.0]}}

    def spec():
        return {
            "dataset": {
                "world": _dots({"sigma_mm": 1.0}, axes),
                "seed": 1,
                "duration_ms": 20,
                "splits": {"probes": {"per_bin": 1, "bins": 5}},
            }
        }

    # the declared worst case needs ceil(7.5 / 0.3) = 25 sites, but the
    # 'below' probes of spacing_mm reach the field's lowest value, 0.01 mm
    with pytest.raises(
        ValueError,
        match=r"dataset\.splits\.probes.*dots.*spacing_mm.*below.*750 lattice sites",
    ):
        load_dataset(spec())
    axes["spacing_mm"]["probes"] = False
    load_dataset(spec())
