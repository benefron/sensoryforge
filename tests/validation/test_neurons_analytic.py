"""Analytic/published validation of the Izhikevich RS and FS presets (Wave S, S1).

**What this test can and cannot claim.** The Wave S spec asks for RS/FS
rate-vs-current curves "against the published figures of Izhikevich (2003)".
This environment has no network access and no digitised copy of that
paper's figures, so we cannot compare against literal digitised figure
points -- doing so without the real numbers would be exactly the fabricated
reference the Wave S brief forbids. What we validate instead, honestly
scoped:

1. A **closed-form rheobase** derived directly from the model's own defining
   equations (Izhikevich 2003, Eq. 1 -- the same equations
   ``IzhikevichNeuronTorch`` implements, see its docstring), independent of
   any figure or recorded output.
2. The **qualitative published claim** in Izhikevich (2003) Fig. 2's own
   caption/discussion, reproduced in the class's docstring and standard
   secondary treatments (e.g. Izhikevich, *Dynamical Systems in
   Neuroscience*, 2007, sec. 8.1.1): RS shows spike-frequency adaptation to
   a comparatively low steady rate, while FS sustains a much higher rate
   (Izhikevich states FS trains reach into the "several hundred Hz" range)
   for the same suprathreshold drive -- both presets share ``b=0.2``, and
   only ``a`` (the recovery time constant) and ``d`` (the reset jump)
   differ, so this ordering is a direct, checkable consequence of the
   published parameter table, not an independent claim.

Closed-form rheobase derivation (recorded here so the reference is
reproducible without external sources): treat the recovery variable as
adiabatically slaved to v at the fixed point, ``u* = b*v*`` (this is the
standard small-a approximation and is exact in the limit ``a -> 0``; the
model's ``dv/dt=0`` condition ``0.04 v^2 + 5v + 140 - u + I = 0`` becomes,
after substituting ``u = b*v``,

    0.04 v^2 + (5 - b) v + (140 + I) = 0

whose two real roots (a stable node and a saddle -- the subthreshold rest
state and the firing threshold) exist only while the discriminant is
non-negative:

    (5 - b)^2 - 4 * 0.04 * (140 + I) >= 0

The rheobase (saddle-node bifurcation, where the two roots merge and
disappear -- the current above which no stable rest state exists and the
neuron must fire repetitively) is at equality:

    I_rheobase = (5 - b)^2 / 0.16 - 140

For ``b = 0.2`` (both RS and FS): ``I_rheobase = 4.8^2/0.16 - 140 = 4.0``.

Tolerance: the closed form assumes u tracks v instantly (u = b*v at the fixed
point), which the real dynamics do not do exactly, so the simulated
threshold is not exactly 4.0. Measured by bisection: 3.774 for RS (a=0.02)
and 3.859 for FS (a=0.10), gaps of 0.23 and 0.14 mA. The assertion allows
0.5 mA, about twice the larger gap. (An earlier version of this docstring
said the gap shrinks as ``a`` gets smaller; the two measurements show the
opposite, so that claim is removed rather than kept.)

What this test catches, and how -- established by experiment on 2026-09-17,
correcting an earlier record of a result that did not happen:

* **An error in the integrated equations.** Changing ``140`` to ``135`` in
  the membrane update, in both integration paths, moves the simulated
  thresholds to 8.774 (RS) and 8.859 (FS) while the closed-form target, which
  is written from the published equation rather than read from the
  implementation, stays at 4.0. The test fails by about 4.8 mA. This is the
  physics comparison doing its job: the reference is independent of the
  code under test.
* **A wrong preset value.** Changing RS's ``b`` from 0.20 to 0.10 does *not*
  make the physics comparison fail. The simulated threshold moves to 10.012
  and the closed form recomputed from that ``b`` is 10.063; they agree,
  because the model is still integrated correctly. That change is caught
  only by the explicit ``b == 0.20`` and ``target == 4.0`` assertions, which
  pin the published parameter value. An earlier version of this docstring
  claimed the simulated threshold "stays near 3.8" under this perturbation.
  It does not; the failure that record describes came from those hard-coded
  assertions, not from the physics.

Both halves are needed: the closed-form comparison validates the
integration, and the pinned published value validates the preset.
"""

from __future__ import annotations

import torch

from sensoryforge.neurons.izhikevich import IZHIKEVICH_PRESETS, IzhikevichNeuronTorch


def _closed_form_rheobase(b: float) -> float:
    return (5.0 - b) ** 2 / 0.16 - 140.0


def _mean_rate_hz(
    preset: str,
    current: float,
    dt: float = 0.05,
    t_ms: float = 1200.0,
    settle_ms: float = 400.0,
) -> float:
    neuron = IzhikevichNeuronTorch(preset=preset, dt=dt, v_floor=-120.0)
    steps = int(t_ms / dt)
    I_in = torch.full((1, steps, 1), current, dtype=torch.float32)
    _, spikes = neuron(I_in)
    settle_steps = int(settle_ms / dt)
    n_spikes = float(spikes[0, settle_steps:, 0].sum())
    duration_s = (t_ms - settle_ms) / 1000.0
    return n_spikes / duration_s


def _empirical_rheobase(
    preset: str, lo: float = 0.0, hi: float = 10.0, iters: int = 14
) -> float:
    """Bisection search for the current at which sustained firing begins."""
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if _mean_rate_hz(preset, mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def test_rheobase_matches_closed_form_bifurcation():
    """RS and FS empirical spiking thresholds match the closed-form saddle-node
    rheobase.

    See module docstring for the derivation, reference value (4.0 mA for
    b=0.2), and the 0.5 mA tolerance's justification.
    """
    target = _closed_form_rheobase(IZHIKEVICH_PRESETS["RS"]["b"])
    assert target == 4.0  # sanity: both presets' published b=0.2 gives this

    for preset in ("RS", "FS"):
        assert IZHIKEVICH_PRESETS[preset]["b"] == 0.20
        empirical = _empirical_rheobase(preset)
        error = abs(empirical - target)
        assert (
            error < 0.5
        ), f"{preset}: empirical rheobase {empirical} vs closed-form {target}"


def test_below_rheobase_neither_preset_fires():
    """Well below the closed-form rheobase (I=2.0 << 4.0), neither preset spikes.

    Reference: same closed-form bifurcation as above -- below I_rheobase a
    stable subthreshold fixed point exists and the trajectory settles there
    (no limit cycle), so a correct implementation must show zero spikes.
    This is a coarse, unambiguous sanity bound independent of the adiabatic
    approximation's residual error (I=2.0 is 2 mA, four times the ~0.5 mA
    tolerance band, below the closed-form value).
    """
    for preset in ("RS", "FS"):
        rate = _mean_rate_hz(preset, current=2.0)
        assert rate == 0.0, f"{preset} fired below rheobase: rate={rate}"


def test_fs_sustains_much_higher_rate_than_rs_at_same_suprathreshold_current():
    """FS fires at a much higher steady rate than RS for the same drive
    (Izhikevich 2003).

    Reference: Izhikevich (2003) describes FS neurons as capable of
    sustained firing into the "several hundred Hz" range, distinctly above
    RS, whose spike-frequency adaptation caps it at a much lower rate for
    the same input -- both classes share the published ``b=0.2``, so the
    gap is driven by ``a`` (RS: 0.02, FS: 0.10) and ``d`` (RS: 8, FS: 2),
    also from the published table (``IZHIKEVICH_PRESETS``).

    Tolerance: we require FS's rate to exceed 2x RS's rate at I=20 mA.
    Measured values (recorded 2026-09-16) are RS=43.3 Hz, FS=302.7 Hz (a
    ~7x gap) -- the 2x bound leaves more than 3x of headroom below the
    actual measurement, so ordinary run-to-run float nondeterminism cannot
    flip it, while a preset table with FS's ``a``/``d`` swapped for RS's
    values (making the two presets numerically identical) would collapse
    the ratio to 1x and fail outright.
    """
    rs_rate = _mean_rate_hz("RS", current=20.0)
    fs_rate = _mean_rate_hz("FS", current=20.0)
    assert rs_rate > 0.0
    assert fs_rate > 2.0 * rs_rate, f"RS={rs_rate} Hz, FS={fs_rate} Hz"
