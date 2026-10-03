"""Tests for the event encoders (2026-10-01): level-crossing (RA), sigma-delta (SA).

Each test pins one property the two decisions rest on:

* level-crossing: ON events at rate slope/theta on a rising ramp, none on a
  hold, OFF events at the same rate on a falling ramp, the signed event sum
  times theta reconstructs the drive to within theta, white noise well below
  theta is almost silent, and a refractory period is respected;
* sigma-delta: the rate is linear in a constant drive with no rheobase (exact
  to 1e-3), a low-pass of the spikes recovers a slow sinusoid, and the
  quantisation error spectrum is high-pass (noise shaping).

The measured numbers are printed (``pytest -s``) so a human can read them.
"""

import math

import numpy as np
import pytest
import torch

from sensoryforge.neurons.event_encoders import LevelCrossingNeuron, SigmaDeltaNeuron
from sensoryforge.registry import NEURON_REGISTRY
from sensoryforge.testing.contracts import check_component

DT = 0.1  # ms


def _time(steps: int) -> torch.Tensor:
    return torch.arange(steps, dtype=torch.float64) * DT


def _events(model, x: torch.Tensor) -> torch.Tensor:
    """Run ``model`` on a 1-D drive and return its per-step events [T] (float64)."""
    _, ev = model(x.view(1, -1, 1))
    return ev[0, 1:, 0].double()


# --------------------------------------------------------------------------- #
# Registry and contract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name, cls",
    [("level_crossing", LevelCrossingNeuron), ("sigma_delta", SigmaDeltaNeuron)],
)
def test_registered_and_contract(name, cls):
    assert NEURON_REGISTRY.get_class(name) is cls
    check_component("neuron", cls)


def test_signed_flag():
    assert LevelCrossingNeuron.SIGNED_EVENTS is True
    assert SigmaDeltaNeuron.SIGNED_EVENTS is False


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (LevelCrossingNeuron, {"theta": 0.0}),
        (LevelCrossingNeuron, {"refractory_ms": -1.0}),
        (LevelCrossingNeuron, {"initial_reference": "middle"}),
        (SigmaDeltaNeuron, {"theta": -1.0}),
        (SigmaDeltaNeuron, {"leak_tau_ms": 0.0}),
    ],
)
def test_invalid_parameters_raise(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_output_dtypes_and_initial_sample():
    x = torch.randn(2, 30, 4)
    for model in (LevelCrossingNeuron(dt=DT, theta=0.2), SigmaDeltaNeuron(dt=DT)):
        state, ev = model(x)
        assert state.shape == (2, 31, 4) and ev.shape == (2, 31, 4)
        assert ev.dtype == torch.int16
        assert torch.all(ev[:, 0] == 0)


# --------------------------------------------------------------------------- #
# Level-crossing (RA)
# --------------------------------------------------------------------------- #

THETA = 0.5  # mA
SLOPE = 0.01  # mA/ms -> 0.02 events/ms expected


def test_level_crossing_rising_ramp_on_rate_and_silent_hold():
    n = 10_000  # 1000 ms ramp then 1000 ms hold
    t = _time(n)
    ramp = SLOPE * t
    x = torch.cat([ramp, torch.full((n,), float(ramp[-1]))])
    ev = _events(LevelCrossingNeuron(dt=DT, theta=THETA), x)
    on_rate = float((ev[:n] > 0).sum()) / (n * DT)
    print(f"\nlevel-crossing ramp: ON rate {on_rate:.4f}/ms (s/theta = {SLOPE/THETA})")
    # Within one event over the ramp: floor(final level / theta) events.
    assert abs(on_rate - SLOPE / THETA) <= 1.0 / (n * DT) + 1e-12
    assert float((ev[:n] < 0).sum()) == 0
    assert float(ev[n:].abs().sum()) == 0, "a hold must produce no events"


def test_level_crossing_falling_ramp_off_rate():
    n = 10_000
    t = _time(n)
    x = SLOPE * (t[-1] - t)  # falls from 10 mA to 0
    ev = _events(LevelCrossingNeuron(dt=DT, theta=THETA, initial_reference="first"), x)
    off_rate = float((ev < 0).sum()) / (n * DT)
    print(f"\nlevel-crossing falling ramp: OFF rate {off_rate:.4f}/ms")
    assert abs(off_rate - SLOPE / THETA) <= 1.0 / (n * DT) + 1e-12
    assert float((ev > 0).sum()) == 0


def test_level_crossing_signed_sum_reconstructs_within_theta():
    torch.manual_seed(1)
    n = 20_000
    t = _time(n)
    # A signed, multi-scale drive: slow sinusoid + faster one + a step.
    x = (
        3.0 * torch.sin(2 * math.pi * 1e-3 * t)
        + 0.8 * torch.sin(2 * math.pi * 2e-2 * t)
        + 2.0 * (t > 700).double()
    )
    for ref0 in ("zero", "first"):
        model = LevelCrossingNeuron(dt=DT, theta=THETA, initial_reference=ref0)
        ev = _events(model, x)
        start = 0.0 if ref0 == "zero" else float(x[0])
        recon = start + torch.cumsum(ev, 0) * THETA
        err = float((x - recon).abs().max()) / THETA
        print(
            f"\nlevel-crossing reconstruction ({ref0}): max |x - sum*theta| = "
            f"{err:.4f} theta"
        )
        assert err < 1.0
    # The step emits several events in one step, as one signed count.
    assert float(ev.abs().max()) >= 4


def test_level_crossing_multiple_events_per_step_as_signed_count():
    x = torch.tensor([0.0, 2.6, 2.6, -0.4], dtype=torch.float64)
    ev = _events(LevelCrossingNeuron(dt=DT, theta=THETA), x)
    assert ev.tolist() == [0.0, 5.0, 0.0, -5.0]


@pytest.mark.parametrize("ratio, max_rate", [(0.1, 0.0), (0.2, 1e-5), (0.3, 5e-3)])
def test_level_crossing_small_noise_is_nearly_silent(ratio, max_rate):
    """White noise of std sigma << theta: events per sample vs sigma/theta.

    Measured (10M samples: 20k steps x 500 neurons): sigma/theta 0.1 -> 0, 0.2 -> 1e-6,
    0.3 -> 1.7e-3, 0.4 -> 2.5e-2, 0.5 -> 8.7e-2 events per sample.
    """
    torch.manual_seed(0)
    noise = torch.randn(1, 20_000, 500, dtype=torch.float64) * ratio * THETA
    _, ev = LevelCrossingNeuron(dt=DT, theta=THETA)(noise)
    rate = float(ev.abs().double().mean())
    print(f"\nlevel-crossing noise sigma/theta={ratio}: {rate:.2e} events/sample")
    assert rate <= max_rate


def test_level_crossing_refractory_respected():
    n = 20_000
    x = 1.0 * _time(n)  # 1 mA/ms -> 2 events/ms without a refractory period
    refractory = 2.0
    ev = _events(LevelCrossingNeuron(dt=DT, theta=THETA, refractory_ms=refractory), x)
    idx = torch.nonzero(ev).flatten()
    assert float(ev.abs().max()) == 1.0, "at most one event per step"
    isi = idx.diff().double() * DT
    assert float(isi.min()) >= refractory - 1e-9
    rate = len(idx) / (n * DT)
    print(
        f"\nlevel-crossing refractory {refractory} ms: rate {rate:.3f}/ms, "
        f"min ISI {float(isi.min()):.2f} ms"
    )
    assert rate == pytest.approx(1.0 / refractory, rel=1e-2)


# --------------------------------------------------------------------------- #
# Sigma-delta (SA)
# --------------------------------------------------------------------------- #

SD_THETA = 10.0  # mA*ms


def test_sigma_delta_rate_linear_in_drive_no_rheobase():
    """Constant drive -> rate drive/theta exactly (1e-3), even far below 1 mA.

    One run, one neuron per drive level, 50 s at 1 ms steps (the integration
    is exact at any step without a leak); the first 1 s is warm-up.
    """
    drives = torch.tensor([0.2, 0.5, 1.0, 5.0, 20.0, 0.01], dtype=torch.float64)
    n, warm, dt = 50_000, 1_000, 1.0
    x = drives.view(1, 1, -1).expand(1, n, -1).contiguous()
    _, sp = SigmaDeltaNeuron(dt=dt, theta=SD_THETA)(x)
    rates = sp[0, 1 + warm :, :].double().sum(0) / ((n - warm) * dt)
    for d, r in zip(drives.tolist(), rates.tolist()):
        print(f"\nsigma-delta drive {d} mA: rate {r:.6f}/ms (expected {d / SD_THETA})")
    expected = drives / SD_THETA
    # >= 980 spikes for every drive but the last: count error <= 1 -> 1e-3.
    assert torch.allclose(rates[:-1], expected[:-1], rtol=1e-3, atol=0)
    # No rheobase: 1/1000 of the smallest drive above still fires, linearly.
    assert float(rates[-1]) == pytest.approx(float(expected[-1]), rel=3e-2)


def test_sigma_delta_zero_drive_is_silent():
    sp = _events(SigmaDeltaNeuron(dt=DT, theta=SD_THETA), torch.zeros(5_000))
    assert float(sp.sum()) == 0


def _sinusoid_run():
    n = 100_000  # 10 s
    t = _time(n)
    x = 5.0 + 4.0 * torch.sin(2 * math.pi * 2e-3 * t)  # 2 Hz, 1..9 mA
    sp = _events(SigmaDeltaNeuron(dt=DT, theta=SD_THETA), x).numpy()
    return x.numpy(), sp


def test_sigma_delta_lowpass_reconstructs_slow_sinusoid():
    """Boxcar low-pass of the spikes vs the drive, by window.

    Measured (2 Hz, 1..9 mA, theta 10 mA*ms, 100..900 Hz): RMS error against
    the equally smoothed drive (pure quantisation) 0.85 / 0.39 / 0.20 / 0.083 /
    0.041 mA at 5 / 10 / 20 / 50 / 100 ms, i.e. ~ theta / window; against the
    raw drive it is smallest near 50 ms (0.095 mA, 2.4% of the amplitude),
    where the boxcar's own attenuation of the 2 Hz sinusoid starts to dominate.
    """
    x, sp = _sinusoid_run()
    errs = {}
    for window_ms in (5, 10, 20, 50, 100):
        k = int(window_ms / DT)
        kernel = np.ones(k) / k
        est = np.convolve(sp, kernel, "same") * SD_THETA / DT
        smooth = np.convolve(x, kernel, "same")
        sl = slice(k, -k)
        e_raw = float(np.sqrt(np.mean((est[sl] - x[sl]) ** 2)))
        e_q = float(np.sqrt(np.mean((est[sl] - smooth[sl]) ** 2)))
        errs[window_ms] = (e_raw, e_q)
        print(
            f"\nsigma-delta window {window_ms} ms: rms vs drive {e_raw:.3f} mA, "
            f"vs smoothed drive {e_q:.3f} mA"
        )
    # Quantisation error falls roughly as 1/window...
    assert errs[100][1] < errs[10][1] / 5
    # ...and a 50 ms window recovers the 2 Hz, 4 mA sinusoid to < 5%.
    assert errs[50][0] < 0.05 * 4.0


def test_sigma_delta_quantisation_error_is_high_pass():
    x, sp = _sinusoid_run()
    err = sp - x * DT / SD_THETA  # spikes minus expected count per step
    power = np.abs(np.fft.rfft(err)) ** 2
    freq_khz = np.fft.rfftfreq(len(err), DT)
    bands = [(0.001, 0.01), (0.01, 0.1), (0.1, 1.0), (1.0, 5.0)]
    means = []
    for lo, hi in bands:
        m = (freq_khz >= lo) & (freq_khz < hi)
        means.append(float(power[m].mean()))
        print(
            f"\nsigma-delta error power {lo*1000:.0f}-{hi*1000:.0f} Hz: {means[-1]:.3g}"
        )
    # Measured: 4.3e-3, 12.6, 4.4e3, 4.8e3 -- rising by >= 100x per decade
    # below 100 Hz (first-order noise shaping), flat once past the rate.
    assert means[0] < means[1] < means[2]
    assert means[0] < 1e-4 * means[2]


def test_sigma_delta_refractory_caps_rate_without_windup():
    n = 20_000
    x = torch.full((n,), 50.0, dtype=torch.float64)  # 5/ms unrestricted
    sp = _events(SigmaDeltaNeuron(dt=DT, theta=SD_THETA, refractory_ms=1.0), x)
    idx = torch.nonzero(sp).flatten()
    assert float(sp.max()) == 1.0
    assert float((idx.diff().double() * DT).min()) >= 1.0 - 1e-9
    assert len(idx) / (n * DT) == pytest.approx(1.0, rel=1e-2)
    # Anti-windup: after the drive stops, at most one more spike.
    x2 = torch.cat([x, torch.zeros(n, dtype=torch.float64)])
    sp2 = _events(SigmaDeltaNeuron(dt=DT, theta=SD_THETA, refractory_ms=1.0), x2)
    assert float(sp2[n:].sum()) <= 1


def test_sigma_delta_leak_adds_rheobase():
    # Leak tau 10 ms: rheobase theta / tau = 1 mA; below it, silence.
    model = SigmaDeltaNeuron(dt=DT, theta=SD_THETA, leak_tau_ms=10.0)
    assert float(_events(model, torch.full((20_000,), 0.9)).sum()) == 0
    assert float(_events(model, torch.full((20_000,), 1.5)).sum()) > 0
