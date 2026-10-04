"""``self_affine``: a patch-filling self-affine relief (P7)."""

import math

import numpy as np
import pytest
import torch

from sensoryforge.stimuli.layered import render_layers
from sensoryforge.world import kernel
from sensoryforge.world.surfaces import self_affine, self_affine_table


def _surface(
    seed, x, y, hurst=0.8, rolloff=2.0, cutoff=0.3, components=256, dtype=None
):
    dtype = dtype or x.dtype
    p = {
        "seed": torch.tensor(float(seed), dtype=dtype),
        "hurst": torch.tensor(hurst, dtype=dtype),
        "rolloff_mm": torch.tensor(rolloff, dtype=dtype),
        "cutoff_mm": torch.tensor(cutoff, dtype=dtype),
        "components": torch.tensor(float(components), dtype=dtype),
    }
    return self_affine(x, y, p)


def _grid(n, pitch):
    c = (torch.arange(n, dtype=torch.float64) - n / 2) * pitch
    return (
        c.view(-1, 1).expand(n, n).contiguous(),
        c.view(1, -1).expand(n, n).contiguous(),
    )


def test_registered_and_patch_filling():
    assert "self_affine" in kernel.SHAPE_KINDS
    assert not kernel.SHAPE_KINDS["self_affine"].unbounded


def test_one_component_equals_its_cosine():
    x, y = _grid(12, 0.1)
    table = self_affine_table(5, 0.8, 2.0, 0.3, 16)
    expected = torch.zeros_like(x)
    for qx, qy, ph in table:
        expected = expected + 16**-0.5 * torch.cos(qx * x + qy * y + ph)
    torch.testing.assert_close(
        _surface(5, x, y, components=16), expected, atol=1e-12, rtol=0
    )


def test_mean_and_rms():
    x, y = _grid(128, 0.03125)  # 4 mm wide: 4 roll-off lengths at rolloff_mm = 1
    means, rmss = [], []
    for seed in range(64):
        h = _surface(seed, x, y, rolloff=1.0, cutoff=0.3)
        means.append(h.mean().item())
        rmss.append(h.pow(2).mean().sqrt().item())
    assert abs(np.mean(means)) < 0.05
    assert abs(np.mean(rmss) - 1 / math.sqrt(2)) < 0.1 / math.sqrt(2)


@pytest.mark.parametrize("hurst", [0.6, 1.0])
def test_spectral_slope_matches_hurst(hurst):
    rolloff, cutoff = 4.0, 0.2
    q0, q1 = 2 * math.pi / rolloff, 2 * math.pi / cutoff
    n, pitch = 256, 0.05
    x, y = _grid(n, pitch)
    freq = np.fft.fftfreq(n, d=pitch) * 2 * math.pi
    qr = np.sqrt(freq[:, None] ** 2 + freq[None, :] ** 2)
    edges = np.geomspace(2 * q0, q1 / 2, 12)
    # a Hann window: a rectangular window's leakage flattens a steep spectrum
    window = np.outer(np.hanning(n), np.hanning(n))
    power = np.zeros(len(edges) - 1)
    seeds = 24
    for seed in range(seeds):
        h = _surface(seed, x, y, hurst, rolloff, cutoff, components=2048).numpy()
        spec = np.abs(np.fft.fft2(h * window)) ** 2
        for b in range(len(edges) - 1):
            sel = (qr >= edges[b]) & (qr < edges[b + 1])
            power[b] += spec[sel].mean() / seeds
    centres = np.sqrt(edges[:-1] * edges[1:])
    slope = np.polyfit(np.log(centres), np.log(power), 1)[0]
    assert abs(slope - (-2 * (hurst + 1))) < 0.2


def test_no_power_above_the_cutoff():
    for hurst in (0.0, 0.5, 1.0):
        table = self_affine_table(3, hurst, 2.0, 0.3, 512)
        q = np.hypot(table[:, 0], table[:, 1])
        assert q.max() <= 2 * math.pi / 0.3 + 1e-9


def test_same_seed_same_surface_other_seed_differs():
    x, y = _grid(10, 0.1)
    assert torch.equal(_surface(7, x, y), _surface(7, x, y))
    assert not torch.allclose(_surface(7, x, y), _surface(8, x, y))


def test_a_batch_with_mixed_components_equals_each_alone_bit_for_bit():
    g = 3
    x, y = _grid(9, 0.1)
    x = x.expand(g, 2, 9, 9).contiguous()
    y = y.expand(g, 2, 9, 9).contiguous()
    seeds, comps = [1, 2, 3], [16, 40, 97]
    p = {
        "seed": torch.tensor(seeds, dtype=torch.float64).view(g, 1, 1, 1),
        "hurst": torch.tensor([0.5, 0.8, 1.0], dtype=torch.float64).view(g, 1, 1, 1),
        "rolloff_mm": torch.full((g, 1, 1, 1), 2.0, dtype=torch.float64),
        "cutoff_mm": torch.full((g, 1, 1, 1), 0.3, dtype=torch.float64),
        "components": torch.tensor(comps, dtype=torch.float64).view(g, 1, 1, 1),
    }
    batch = self_affine(x, y, p)
    for i in range(g):
        alone = self_affine(
            x[i : i + 1], y[i : i + 1], {k: v[i : i + 1] for k, v in p.items()}
        )
        assert torch.equal(alone[0], batch[i])


def test_float32_and_float64_agree_to_1e_5():
    x, y = _grid(16, 0.1)
    for seed in (0, 16777215):
        a = _surface(seed, x, y, components=128)
        b = _surface(seed, x.float(), y.float(), components=128)
        torch.testing.assert_close(a, b.double(), atol=1e-5, rtol=0)


def test_x_mm_translates_the_surface():
    layer = lambda x_mm: {  # noqa: E731
        "shape": {"kind": "self_affine", "seed": 4, "components": 32},
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
    base = render_layers([layer(0.0)], xx, yy, dt_ms=1.0, total_ms=5.0)
    shifted = render_layers([layer(0.5)], xx, yy, dt_ms=1.0, total_ms=5.0)
    assert not torch.allclose(base, shifted)
    # shifting the pattern by 5 samples moves the surface by 5 rows
    torch.testing.assert_close(shifted[:, 5:], base[:, :-5], atol=1e-4, rtol=0)
