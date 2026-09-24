"""S2: a sanity comparison of SA/RA responses against published afferent-class
behaviour.

**Read ``tests/fixtures/reference/README.md`` first.** Genuine TouchSim
output (Saal et al. 2017, PNAS) was not obtainable in this environment --
`touchsim` is not installed and may not become a dependency (Fact P4-a),
and no digitised figure data was available either. This test therefore
compares against the qualitative, citable published bounds recorded in
``tests/fixtures/reference/sa_ra_qualitative_reference.json``, not against
TouchSim's numbers.

**This is a sanity check, not an equivalence claim.** TouchSim's SA1/RA1
afferent models and SensoryForge's SA/RA filters are different models of
related but distinct afferent populations (see the fixture README for the
full statement). Passing this test shows SensoryForge's filters are in the
correct qualitative adaptation regime; it does not show quantitative
agreement with TouchSim and would not catch a miscalibration that preserved
the SA/RA qualitative shape (e.g. a wrong gain).

Perturbation proof (recorded 2026-09-16, applied in this test file only --
the fixture and the implementation are both untouched -- tested, then
reverted): swapping which filter class plays the "SA" role (constructing
``sa_filter`` from ``RAFilterTorch`` instead of ``SAFilterTorch``) makes
``test_sa_ra_filters_show_published_adaptation_classes`` fail: the measured
SA hold/peak ratio collapses from 0.71 to ~4.6e-16 (the RA filter decays to
exactly zero during a static hold, as it must), far below the SA bound.
Measured ratios on the unperturbed code (recorded 2026-09-16): SA=0.71,
RA=4.6e-16. Since 2026-09-24 SA's filter has a stronger ramp term, fitted to
TouchSim's SA1 (k2 3.0 -> 8.0, D-f4d0967): SA's ratio is now 0.34, and the
fixture's SA bound was lowered from 0.5 to 0.25 because TouchSim's own SA1
holds at 0.19-0.26 of its ramp rate. RA's bound (<= 0.15) is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sensoryforge.filters.sa_ra import RAFilterTorch, SAFilterTorch

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "reference"
    / "sa_ra_qualitative_reference.json"
)


def _load_reference() -> dict:
    with open(_FIXTURE) as f:
        return json.load(f)


def _ramp_and_hold(
    ramp_ms: float, hold_ms: float, peak: float, dt: float
) -> torch.Tensor:
    """[1, T, 1] ramp-and-hold-and-return stimulus, matching the fixture protocol."""
    n_ramp = int(round(ramp_ms / dt))
    n_hold = int(round(hold_ms / dt))
    ramp_up = torch.linspace(0.0, peak, n_ramp + 1)[1:]
    hold = torch.full((n_hold,), peak)
    ramp_down = torch.linspace(peak, 0.0, n_ramp + 1)[1:]
    trace = torch.cat([ramp_up, hold, ramp_down])
    return trace.view(1, -1, 1), n_ramp, n_hold


def test_sa_ra_filters_show_published_adaptation_classes():
    """SA response stays high through hold; RA response decays to near-zero during hold.

    See module and fixture README docstrings for what this does and does
    not validate, and exactly why a genuine TouchSim comparison was not
    possible here.
    """
    ref = _load_reference()
    proto = ref["stimulus_protocol"]
    bounds = ref["bounds"]

    stim, n_ramp, n_hold = _ramp_and_hold(
        proto["ramp_ms"], proto["hold_ms"], proto["peak_amplitude_ma"], proto["dt_ms"]
    )

    sa_filter = SAFilterTorch(dt=proto["dt_ms"])
    ra_filter = RAFilterTorch(dt=proto["dt_ms"])

    sa_out = sa_filter(stim, reset_states=True).squeeze()
    ra_out = ra_filter(stim, reset_states=True).squeeze()

    hold_start = n_ramp
    hold_end = n_ramp + n_hold
    late_hold = slice(hold_end - int(0.1 * n_hold), hold_end)  # last 10% of hold

    sa_peak = sa_out[:hold_end].max()
    sa_late = sa_out[late_hold].mean()
    sa_ratio = float(sa_late / sa_peak.clamp(min=1e-9))

    ra_peak = ra_out[: hold_start + 1].max()  # onset transient
    ra_late = ra_out[late_hold].mean()
    ra_ratio = float(ra_late / ra_peak.clamp(min=1e-9))

    assert sa_ratio >= bounds["sa_hold_to_peak_ratio_min"], (
        f"SA hold/peak ratio {sa_ratio} below published lower bound "
        f"{bounds['sa_hold_to_peak_ratio_min']} -- SA response is decaying "
        "(RA-like) instead of sustaining through the hold."
    )
    assert ra_ratio <= bounds["ra_hold_to_peak_ratio_max"], (
        f"RA hold/peak ratio {ra_ratio} above published upper bound "
        f"{bounds['ra_hold_to_peak_ratio_max']} -- RA response is staying "
        "elevated (SA-like) instead of decaying through the hold."
    )
