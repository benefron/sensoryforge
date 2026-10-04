"""``self_affine``: a patch-filling self-affine relief (P7).

Since v1.2.1 the components sit on strata of equal width on the scale
``xi = (q**2 / q0**2 - 1) / 2`` below the roll-off ``q0`` and ``ln(q / q0)``
above it (one component per stratum), each with the stratum's share of the
declared power, so every octave between roll-off and cut-off carries
components in every surface. The helpers below restate the declared spectrum
and that layout independently of the code: the tests compare the code with
them.
"""

import math

import numpy as np
import pytest
import torch

from sensoryforge.stimuli.layered import render_layers
from sensoryforge.world import Canvas, World, kernel, render, sample
from sensoryforge.world.dataset import load_dataset
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


def _wavenumbers(table):
    return np.hypot(table[:, 0], table[:, 1])


def _power(table):
    """Each component's power ``a_j**2 / 2`` (per unit amplitude)."""
    return table[:, 3] ** 2 / 2.0


# ---------------------------------------------- the declared spectrum, restated


def _xi(q, q0):
    """The strata's scale: ``(q**2/q0**2 - 1)/2`` below ``q0``, ``ln(q/q0)`` above."""
    return (q * q / (q0 * q0) - 1.0) / 2.0 if q <= q0 else math.log(q / q0)


def _mass(xi, hurst, xi1):
    """Declared radial power ``int q C(q) dq`` from 0 to ``xi`` (units ``C0 q0**2``).

    The power per unit ``xi`` is 1 below the roll-off (``xi <= 0``) and
    ``exp(-2 H xi)`` above it, up to the cut-off ``xi1``.
    """
    xi = min(max(xi, -0.5), xi1)
    if xi <= 0:
        return xi + 0.5
    if hurst == 0:
        return 0.5 + xi
    return 0.5 + (1.0 - math.exp(-2.0 * hurst * xi)) / (2.0 * hurst)


def _band(hurst, rolloff, cutoff, n, fine_mm, coarse_mm):
    """``(expected band power, the most one surface can differ from it)``.

    The band holds every wavelength in ``[fine_mm, coarse_mm]``. Its expected
    power is half its share of the declared power (a surface's variance is
    1/2). Each stratum, of width ``W / n`` in ``xi``, holds one component with
    the stratum's share of the power, so a surface's band power can differ from
    the expectation only through the strata cut by the band's two edges: by at
    most the power within ``W / n`` of each edge.
    """
    q0, q1 = 2 * math.pi / rolloff, 2 * math.pi / cutoff
    xi1 = math.log(q1 / q0)
    step = (0.5 + xi1) / n
    total = _mass(xi1, hurst, xi1)
    edges = [_xi(2 * math.pi / coarse_mm, q0), _xi(min(2 * math.pi / fine_mm, q1), q0)]
    expected = 0.5 * (_mass(edges[1], hurst, xi1) - _mass(edges[0], hurst, xi1)) / total
    spread = sum(
        0.5 * (_mass(e + step, hurst, xi1) - _mass(e - step, hurst, xi1)) / total
        for e in edges
    )
    return expected, spread


# A placeholder configuration with v1.2.0's defect: a long roll-off over a fine
# band of wavelengths (0.1-0.3 mm) next to the cut-off.
ROLLOFF, CUTOFF, N = 4.0, 0.1, 256


# ------------------------------------------------------------------- the shape


def test_registered_and_patch_filling():
    assert "self_affine" in kernel.SHAPE_KINDS
    assert not kernel.SHAPE_KINDS["self_affine"].unbounded


def test_one_component_equals_its_cosine():
    x, y = _grid(12, 0.1)
    table = self_affine_table(5, 0.8, 2.0, 0.3, 16)
    expected = torch.zeros_like(x)
    for qx, qy, ph, amp in table:
        expected = expected + amp * torch.cos(qx * x + qy * y + ph)
    torch.testing.assert_close(
        _surface(5, x, y, components=16), expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("hurst", [0.0, 0.6, 1.0])
@pytest.mark.parametrize("rolloff, cutoff, n", [(2.0, 0.3, 256), (4.0, 0.1, 16)])
def test_every_surface_has_variance_one_half(hurst, rolloff, cutoff, n):
    """The powers sum to 1/2 in every surface: RMS amplitude / sqrt(2) on the plane."""
    for seed in (0, 1, 16777215):
        table = self_affine_table(seed, hurst, rolloff, cutoff, n)
        assert table.shape == (n, 4)
        assert _power(table).sum() == pytest.approx(0.5, abs=1e-12)


def test_mean_and_rms():
    x, y = _grid(128, 0.03125)  # 4 mm wide: 4 roll-off lengths at rolloff_mm = 1
    means, rmss = [], []
    for seed in range(64):
        h = _surface(seed, x, y, rolloff=1.0, cutoff=0.3)
        means.append(h.mean().item())
        rmss.append(h.pow(2).mean().sqrt().item())
    assert abs(np.mean(means)) < 0.05
    assert abs(np.mean(rmss) - 1 / math.sqrt(2)) < 0.1 / math.sqrt(2)


@pytest.mark.parametrize("hurst", [0.6, 0.8, 1.0])
def test_every_octave_carries_components_in_every_surface(hurst):
    """Each stratum of width ``W / N`` holds one component, so an octave
    (``ln 2`` in ``xi``) holds at least ``floor(N ln 2 / W) - 1`` of them in
    every surface. v1.2.0 put 0 or 1 component in the top 1.6 octaves here."""
    q0, q1 = 2 * math.pi / ROLLOFF, 2 * math.pi / CUTOFF
    width = 0.5 + math.log(q1 / q0)
    least = math.floor(N * math.log(2.0) / width) - 1
    assert least >= 1
    octaves = []
    lo = q0
    while 2 * lo <= q1 * (1 + 1e-12):
        octaves.append(lo)
        lo *= 2
    for seed in range(200):
        q = _wavenumbers(self_affine_table(seed, hurst, ROLLOFF, CUTOFF, N))
        for lo in octaves:
            count = int(((q >= lo) & (q < 2 * lo)).sum())
            assert count >= least, (seed, lo, count)


@pytest.mark.parametrize("hurst", [0.6, 0.8, 1.0])
def test_fine_band_power_varies_little_between_surfaces(hurst):
    """The 0.1-0.3 mm band's power in single surfaces stays within the
    derived bound of its expectation.

    Measured over seeds 0-199 with v1.2.0's synthesis (equal-power
    components), the coefficient of variation of this band's power was
    0.084, 0.169 and 0.808 at H = 0.6, 0.8 and 1.0 (its components per surface:
    5.3, 1.9 and 0.6). Every surface now lies within ``spread`` of the
    expectation, so the CV is at most ``spread / (expected - spread)``.
    """
    expected, spread = _band(hurst, ROLLOFF, CUTOFF, N, CUTOFF, 3 * CUTOFF)
    qa, qb = 2 * math.pi / (3 * CUTOFF), 2 * math.pi / CUTOFF
    powers = []
    for seed in range(200):
        table = self_affine_table(seed, hurst, ROLLOFF, CUTOFF, N)
        q = _wavenumbers(table)
        powers.append(_power(table)[(q >= qa) & (q <= qb)].sum())
    powers = np.array(powers)
    assert np.abs(powers - expected).max() <= spread + 1e-15
    cv = powers.std() / powers.mean()
    assert cv <= spread / (expected - spread)


@pytest.mark.parametrize("hurst", [0.0, 0.6, 1.0])
def test_expected_band_power_is_the_declared_spectrum(hurst):
    """Averaged over 400 surfaces, each band's power is its share of the
    declared Persson power (within 4 standard errors, the per-surface spread
    bounding the standard deviation)."""
    rolloff, cutoff, n, seeds = 2.0, 0.1, 64, 400
    bands = [(1.0, 2.0), (2.0, 4.0), (0.5, 1.0), (0.25, 0.5), (0.1, 0.25)]
    tables = [self_affine_table(s, hurst, rolloff, cutoff, n) for s in range(seeds)]
    for fine, coarse in bands:
        expected, spread = _band(hurst, rolloff, cutoff, n, fine, coarse)
        qa, qb = 2 * math.pi / coarse, 2 * math.pi / fine
        mean = np.mean(
            [
                _power(t)[(_wavenumbers(t) >= qa) & (_wavenumbers(t) < qb)].sum()
                for t in tables
            ]
        )
        assert abs(mean - expected) <= 4 * spread / math.sqrt(seeds), (fine, coarse)


@pytest.mark.parametrize("hurst", [0.6, 0.8, 1.0])
def test_spectral_slope_matches_hurst(hurst):
    """The ensemble periodogram of rendered surfaces falls as q**(-2(H+1))."""
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
    seeds = 16
    for seed in range(seeds):
        h = _surface(seed, x, y, hurst, rolloff, cutoff, components=512).numpy()
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
        assert _wavenumbers(table).max() <= 2 * math.pi / 0.3 + 1e-9


def test_a_cutoff_at_or_above_the_rolloff_is_refused():
    x, y = _grid(6, 0.1)
    for rolloff, cutoff in ((0.5, 2.0), (0.5, 0.5)):
        with pytest.raises(ValueError, match="cutoff_mm.*rolloff_mm"):
            _surface(0, x, y, rolloff=rolloff, cutoff=cutoff)


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


# --------------------------------------------- worlds, and the checks at load


def _rough(shape, axes, fixed=None):
    """A world mapping with one self-affine class ``rough``."""
    return {
        "world": {
            "classes": {
                "rough": {
                    "layer": {"shape": {"kind": "self_affine", **shape}},
                    "axes": {
                        "hold_ms": {"value": 10.0},
                        "seed": {
                            "range": [0, 16777215],
                            "int": True,
                            "stratify": False,
                        },
                        **axes,
                    },
                }
            },
            "fixed_draws": fixed or {},
        }
    }


def test_a_rolloff_longer_than_the_patch_shows_as_an_offset():
    """Wavelengths longer than the patch render as an offset across it: the
    patch mean's share of the RMS grows with the roll-off (contract section 9)."""
    canvas = Canvas.from_grid(20, 20, 0.1)  # a 2 mm patch
    share = {}
    for rolloff in (0.2, 6.0):
        world = World.from_dict(_rough({"rolloff_mm": rolloff, "cutoff_mm": 0.05}, {}))
        draws = sample(world, n=32, seed=2)
        times = torch.tensor([5.0], dtype=torch.float64)
        frames = render(draws, canvas, times, dtype=torch.float64)[:, 0]
        offset = frames.mean(dim=(1, 2)).abs() * math.sqrt(2)
        share[rolloff] = offset.mean().item()
    assert share[6.0] > 3 * share[0.2]


@pytest.mark.parametrize(
    "shape, axes, path",
    [
        # a fixed cut-off above a fixed roll-off (v1.2.0 rendered a flat spectrum)
        ({"rolloff_mm": 0.5, "cutoff_mm": 2.0}, {}, r"layer\.shape\.cutoff_mm"),
        # a roll-off range reaching down to the cut-off
        (
            {"cutoff_mm": 0.3},
            {"rolloff_mm": {"range": [0.2, 2.0]}},
            r"axes\.rolloff_mm",
        ),
        # a cut-off range reaching up to the roll-off
        ({"rolloff_mm": 1.0}, {"cutoff_mm": {"range": [0.1, 1.0]}}, r"axes\.cutoff_mm"),
    ],
)
def test_a_world_whose_cutoff_can_reach_the_rolloff_fails_at_load(shape, axes, path):
    with pytest.raises(
        ValueError, match=r"world\.classes\.rough\..*" + path + r".*below rolloff_mm"
    ):
        World.from_dict(_rough(shape, axes))


def test_a_cutoff_below_every_rolloff_loads():
    axes = {"rolloff_mm": {"range": [0.5, 2.0]}, "cutoff_mm": {"range": [0.1, 0.4]}}
    assert "rough" in World.from_dict(_rough({}, axes)).classes


def test_a_fixed_draw_whose_cutoff_reaches_the_rolloff_fails_at_load():
    with pytest.raises(ValueError, match=r"world\.fixed_draws\.flat.*cutoff_mm"):
        World.from_dict(
            _rough(
                {"cutoff_mm": 0.3},
                {"rolloff_mm": {"range": [0.5, 2.0]}},
                fixed={"flat": {"class": "rough", "rolloff_mm": 0.25}},
            )
        )


def test_probes_whose_rolloff_can_reach_the_cutoff_fail_at_dataset_load():
    axes = {"rolloff_mm": {"range": [0.4, 2.0], "dist": "log_uniform"}}

    def spec():
        return {
            "dataset": {
                "world": _rough({"cutoff_mm": 0.3}, axes),
                "seed": 1,
                "duration_ms": 20,
                "splits": {"probes": {"per_bin": 1, "bins": 5}},
            }
        }

    # the 'below' probes of rolloff_mm reach 0.4 * 5**-0.2 = 0.290 mm < 0.3 mm
    with pytest.raises(
        ValueError, match=r"dataset\.splits\.probes.*rough.*rolloff_mm.*below"
    ):
        load_dataset(spec())
    axes["rolloff_mm"]["probes"] = False
    load_dataset(spec())
