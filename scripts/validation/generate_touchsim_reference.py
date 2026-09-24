"""Generate a genuine TouchSim ramp-and-hold reference fixture.

Writes ``tests/fixtures/reference/touchsim_ramp_hold.json`` (resolved relative
to this file). This is the ledger-D-4aafcdc fixture: TouchSim output
generated **once**, in a throwaway environment, and committed as static
data. TouchSim (Saal, Delhaye, Rayhaun & Bensmaia, *Simulating tactile
signals from the whole hand with millisecond precision*, PNAS 2017) never
becomes a SensoryForge dependency -- this script is not imported by the
package or by the test suite, and no TouchSim source is copied into this
repository. See ``tests/fixtures/reference/README.md`` for the full
provenance and ``tests/validation/test_touchsim_reference.py`` for what the
fixture is used to check.

Building the throwaway environment (never the ``sensoryforge`` conda env --
F-053 also forbids editable installs of *anything* from a worktree, and this
script has nothing to do with the SensoryForge package anyway)::

    /opt/miniconda3/bin/conda create -y -p /path/to/throwaway/env \\
        python=3.10 numpy scipy numba matplotlib scikit-image pytest \\
        -c conda-forge

    git clone https://github.com/hsaal/touchsim.git /path/to/throwaway/touchsim_src
    cd /path/to/throwaway/touchsim_src
    /path/to/throwaway/env/bin/pip install --no-deps .

    # touchsim's setup.py data_files puts surfaces/hand.png at <env>/surfaces/hand.png,
    # but touchsim/surface.py's hand_surface (built at import time, even though this
    # script never uses it) looks for it at <site-packages>/surfaces/hand.png. Copy it:
    mkdir -p /path/to/throwaway/env/lib/python3.10/site-packages/surfaces
    cp /path/to/throwaway/env/surfaces/hand.png \\
        /path/to/throwaway/env/lib/python3.10/site-packages/surfaces/hand.png

Running (from the throwaway env, with this file's directory on no particular
path -- it only imports ``touchsim``, ``numpy`` and the standard library)::

    /path/to/throwaway/env/bin/python scripts/validation/generate_touchsim_reference.py

The exact commit used to generate the committed fixture:
touchsim @ 4ec9f5c382e7d48410566de743d7a4f05de75cec (2025-04-05,
"Remove holoviews from requirements."), https://github.com/hsaal/touchsim.
"""

from __future__ import annotations

import json
import platform
import random
from datetime import date
from pathlib import Path

import numpy as np
import scipy

import touchsim as ts
from touchsim.classes import Afferent

OUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "reference"
    / "touchsim_ramp_hold.json"
)

# --- Fixed protocol -----------------------------------------------------

SEED = 42

# The exact commit of https://github.com/hsaal/touchsim this fixture was
# generated from (recorded manually -- the installed package is a built
# wheel, with no git metadata of its own to introspect).
TOUCHSIM_COMMIT_SHA = "4ec9f5c382e7d48410566de743d7a4f05de75cec"

PIN_RADIUS_MM = 0.5  # punctate probe radius, touchsim's own default
FS_HZ = 5000.0
RAMP_MS = 50.0
HOLD_MS = 450.0
TOTAL_MS = 2 * RAMP_MS + HOLD_MS  # 550 ms

# 7 indentation depths, near-threshold (25 um) to touchsim's typical upper range.
DEPTHS_MM = [0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 1.25]

# Afferent classes touchsim ships (its 'RA' is what the literature calls RA1).
AFFCLASSES = ["SA1", "RA", "PC"]

# Distances (mm) from the probe centre, along a single ray -- touchsim's
# receptive fields are radially symmetric around the probe, so a ray fully
# characterises the response as a function of distance.
DISTANCES_MM = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

# Every afferent uses the first (idx=0) of touchsim's built-in single-neuron
# parameter sets for its class, for full reproducibility independent of the
# afferent-model-selection RNG (touchsim draws a random idx via the stdlib
# `random` module when idx is left None).
AFF_IDX = 0

# Analysis windows, ms from stimulus onset (t=0).
WINDOWS_MS = {
    "onset": (0.0, RAMP_MS),  # ramp-up transient
    "sustained": (RAMP_MS + 100.0, RAMP_MS + HOLD_MS),  # hold, excluding first 100 ms
    "offset": (RAMP_MS + HOLD_MS, TOTAL_MS),  # ramp-down transient
}


def build_population():
    """Returns (afferents_meta, AfferentPopulation) in a fixed, stable order."""
    afferents_meta = []
    pop = None
    for dist in DISTANCES_MM:
        for affclass in AFFCLASSES:
            loc = np.array([[dist, 0.0]])
            aff = Afferent(affclass, location=loc, idx=AFF_IDX, surface=ts.null_surface)
            pop = aff if pop is None else pop + aff
            afferents_meta.append(
                {
                    "id": len(afferents_meta),
                    "affclass": affclass,
                    "touchsim_idx": AFF_IDX,
                    "x_mm": float(dist),
                    "y_mm": 0.0,
                    "distance_mm": float(dist),
                }
            )
    return afferents_meta, pop


def build_stimulus(depth_mm: float):
    return ts.stim_ramp(
        amp=depth_mm,
        len=TOTAL_MS / 1000.0,
        ramp_len=RAMP_MS / 1000.0,
        ramp_type="lin",
        fs=FS_HZ,
        pin_radius=PIN_RADIUS_MM,
    )


def _rate_in_window(spike_times_s: np.ndarray, window_ms: tuple[float, float]) -> float:
    lo, hi = window_ms[0] / 1000.0, window_ms[1] / 1000.0
    n = int(np.sum((spike_times_s >= lo) & (spike_times_s < hi)))
    duration_s = hi - lo
    return n / duration_s if duration_s > 0 else float("nan")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    afferents_meta, pop = build_population()

    by_depth = []
    for depth_mm in DEPTHS_MM:
        stim = build_stimulus(depth_mm)
        response = pop.response(stim)
        spikes = response.spikes  # list of arrays, one per afferent, seconds

        spike_times = {}
        rates_hz = {}
        for meta, sp in zip(afferents_meta, spikes):
            aff_id = str(meta["id"])
            sp = np.asarray(sp, dtype=float)
            spike_times[aff_id] = [round(float(t), 6) for t in sp]
            rates_hz[aff_id] = {
                window: round(_rate_in_window(sp, bounds), 4)
                for window, bounds in WINDOWS_MS.items()
            }

        by_depth.append(
            {
                "depth_mm": depth_mm,
                "spike_times_s": spike_times,
                "rates_hz": rates_hz,
            }
        )

    fixture = {
        "provenance": {
            "touchsim_repo_url": "https://github.com/hsaal/touchsim",
            "touchsim_commit_sha": TOUCHSIM_COMMIT_SHA,
            "touchsim_version": "0.1.1",
            "touchsim_license": (
                "No LICENSE file in the repository and GitHub's license "
                "detector reports none (checked 2026-09-24); treated as "
                "all rights reserved by the author (Hannes Saal / "
                "BensmaiaLab) absent an explicit open-source grant. Only "
                "numeric outputs (this fixture) are committed here; no "
                "touchsim source is copied into SensoryForge, and touchsim "
                "never becomes a dependency (ledger D-4aafcdc)."
            ),
            "reference_paper": (
                "Saal, H.P., Delhaye, B.P., Rayhaun, B.C. & Bensmaia, S.J. "
                "(2017). Simulating tactile signals from the whole hand "
                "with millisecond precision. PNAS 114(28), E5693-E5702."
            ),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "date_generated": date.today().isoformat(),
            "generating_command": (
                "<throwaway-env>/bin/python "
                "scripts/validation/generate_touchsim_reference.py"
            ),
            "random_seed": SEED,
            "notes": (
                "Every afferent uses touchsim's idx=0 single-neuron model "
                "for its class (deterministic); the seed fixes touchsim's "
                "membrane-noise RNG (numpy) and its afferent-model-selection "
                "RNG (stdlib random), the latter unused here since idx is "
                "always given explicitly."
            ),
        },
        "stimulus_protocol": {
            "type": "ramp_and_hold_punctate_probe",
            "pin_radius_mm": PIN_RADIUS_MM,
            "ramp_ms": RAMP_MS,
            "hold_ms": HOLD_MS,
            "total_duration_ms": TOTAL_MS,
            "fs_hz": FS_HZ,
            "ramp_type": "lin",
            "depths_mm": DEPTHS_MM,
        },
        "windows_ms": {k: list(v) for k, v in WINDOWS_MS.items()},
        "afferents": afferents_meta,
        "by_depth": by_depth,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=False)

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
