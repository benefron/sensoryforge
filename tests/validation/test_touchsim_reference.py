"""Integrity checks on the genuine TouchSim ramp-and-hold reference fixture.

Loads ``tests/fixtures/reference/touchsim_ramp_hold.json`` -- real TouchSim
(Saal, Delhaye, Rayhaun & Bensmaia, 2017, PNAS) output, generated once in a
throwaway environment per ledger D-4aafcdc and committed as static data (see
``tests/fixtures/reference/README.md`` and
``scripts/validation/generate_touchsim_reference.py`` for provenance and
regeneration). TouchSim itself is not installed here and never becomes a
SensoryForge dependency; this test only ever needs ``json``, ``numpy`` (and,
in principle, ``torch``) to run.

This test checks that the *fixture itself* -- TouchSim's own output --
reproduces the textbook-defining properties of the SA1 and RA1
(``touchsim``'s ``'RA'``) afferent classes:

* SA1 sustains firing throughout a static hold.
* RA1 fires mainly at the onset and offset transients and is silent (or
  near-silent) during the static hold.
* Firing rates increase with indentation depth for afferents near the probe.

**This does not compare TouchSim against SensoryForge's own SA/RA filters.**
That quantitative comparison is deliberately deferred (see the Wave brief
this fixture was built for) until after a gain recalibration being done
elsewhere; ``tests/validation/test_touchsim_sanity.py`` remains the only
SensoryForge-side check today, against a qualitative literature bound, not
against this fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "reference"
    / "touchsim_ramp_hold.json"
)

_REQUIRED_PROVENANCE_KEYS = {
    "touchsim_repo_url",
    "touchsim_commit_sha",
    "touchsim_version",
    "touchsim_license",
    "reference_paper",
    "python_version",
    "numpy_version",
    "scipy_version",
    "date_generated",
    "generating_command",
    "random_seed",
}


def _load_fixture() -> dict:
    with open(_FIXTURE) as f:
        return json.load(f)


def _afferents_by_class_near_probe(ref: dict) -> dict:
    """Maps affclass -> afferent id string, for the afferents at distance 0."""
    near = [a for a in ref["afferents"] if a["distance_mm"] == 0.0]
    return {a["affclass"]: str(a["id"]) for a in near}


def test_fixture_has_full_provenance():
    """The fixture records where it came from -- repo, commit, licence, env, command."""
    ref = _load_fixture()
    provenance = ref["provenance"]
    missing = _REQUIRED_PROVENANCE_KEYS - provenance.keys()
    assert not missing, f"Fixture provenance is missing keys: {sorted(missing)}"

    assert provenance["touchsim_repo_url"] == "https://github.com/hsaal/touchsim"
    assert len(provenance["touchsim_commit_sha"]) == 40, "Expected a full git SHA"
    assert provenance["random_seed"] is not None, "Seed must be recorded and fixed"


def test_stimulus_protocol_is_ramp_and_hold():
    """Stimulus is a punctate-probe trapezoidal ramp/hold/ramp, as specified."""
    ref = _load_fixture()
    proto = ref["stimulus_protocol"]

    assert proto["ramp_ms"] == 50.0
    assert 400.0 <= proto["hold_ms"] <= 500.0
    assert proto["pin_radius_mm"] > 0.0
    assert len(proto["depths_mm"]) >= 5, "Expect several indentation depth levels"
    depths = proto["depths_mm"]
    assert depths == sorted(depths), "Depths should be recorded low to high"
    assert depths[0] < 0.1, "Lowest depth should be near threshold"
    assert (
        1.0 <= depths[-1] <= 1.6
    ), "Highest depth should be in touchsim's typical range"


def test_sa1_sustains_firing_through_the_hold():
    """SA1 keeps firing through the hold at a suprathreshold depth, near the probe."""
    ref = _load_fixture()
    by_class = _afferents_by_class_near_probe(ref)
    assert "SA1" in by_class
    sa1_id = by_class["SA1"]

    # Largest depth is comfortably suprathreshold for an afferent at the probe centre.
    entry = ref["by_depth"][-1]
    rates = entry["rates_hz"][sa1_id]

    assert rates["onset"] > 0.0, "SA1 should respond to the ramp-up"
    assert rates["sustained"] > 0.0, (
        "SA1 should keep firing during the hold (excluding the first 100 ms) -- "
        f"got {rates['sustained']} Hz at depth {entry['depth_mm']} mm"
    )

    # Sustained firing should not be a vanishing fraction of the onset transient.
    assert rates["sustained"] >= 0.1 * rates["onset"], (
        "SA1 sustained rate collapsed relative to onset -- looks RA-like, not SA-like: "
        f"onset={rates['onset']} Hz, sustained={rates['sustained']} Hz"
    )


def test_ra_fires_mainly_at_onset_and_offset_and_is_silent_during_hold():
    """RA1 (touchsim's 'RA') is silent during the hold at every depth, near the probe."""
    ref = _load_fixture()
    by_class = _afferents_by_class_near_probe(ref)
    assert "RA" in by_class
    ra_id = by_class["RA"]

    sustained_rates = [
        entry["rates_hz"][ra_id]["sustained"] for entry in ref["by_depth"]
    ]
    assert all(r == 0.0 for r in sustained_rates), (
        "RA1 should not sustain firing during a static hold at any depth -- "
        f"got sustained rates {sustained_rates} Hz across depths "
        f"{[e['depth_mm'] for e in ref['by_depth']]}"
    )

    # At the largest (clearly suprathreshold) depth, RA1 should respond to both transients.
    entry = ref["by_depth"][-1]
    rates = entry["rates_hz"][ra_id]
    assert rates["onset"] > 0.0, "RA1 should respond to the ramp-up transient"
    assert rates["offset"] > 0.0, "RA1 should respond to the ramp-down transient"


def test_rates_increase_with_indentation_depth_near_probe():
    """Firing rates rise monotonically with depth for afferents at the probe centre."""
    ref = _load_fixture()
    by_class = _afferents_by_class_near_probe(ref)
    depths = [entry["depth_mm"] for entry in ref["by_depth"]]
    assert depths == sorted(depths)

    checks = [("SA1", "onset"), ("SA1", "sustained"), ("RA", "onset")]
    for affclass, window in checks:
        aff_id = by_class[affclass]
        rates = np.array(
            [entry["rates_hz"][aff_id][window] for entry in ref["by_depth"]]
        )

        diffs = np.diff(rates)
        assert np.all(diffs >= -1e-9), (
            f"{affclass} {window} rate is not monotonically non-decreasing with "
            f"depth: depths={depths}, rates={rates.tolist()}"
        )
        assert rates[-1] > rates[0], (
            f"{affclass} {window} rate did not increase from the lowest to the "
            f"highest depth: rates={rates.tolist()}"
        )


def test_afferents_have_known_positions_and_distances():
    """Every afferent's (x, y) position and distance from the probe are recorded."""
    ref = _load_fixture()
    for a in ref["afferents"]:
        assert {"id", "affclass", "x_mm", "y_mm", "distance_mm"} <= a.keys()
        expected_distance = float(np.hypot(a["x_mm"], a["y_mm"]))
        assert abs(a["distance_mm"] - expected_distance) < 1e-9

    distances = sorted({a["distance_mm"] for a in ref["afferents"]})
    assert distances[0] == 0.0, "At least one afferent should sit at the probe centre"
    assert distances[-1] >= 3.0, "Placement should reach out to at least a few mm"
