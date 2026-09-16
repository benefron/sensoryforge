"""Export a golden fixture of the four Wave K stimuli from pressure-simulation.

Writes ``tests/fixtures/stimulus_golden.npz`` (resolved relative to this file,
so it lands in the SensoryForge repo regardless of the current working
directory). ``tests/integration/test_stimulus_parity.py`` compares
``sensoryforge.stimuli.render.render_stimulus`` against this fixture at zero
tolerance.

Imports pressure-simulation's own functions directly -- it does not use
SensoryForge's port, so a bug shared between the two would not be masked.

Run with the sensoryforge conda Python, from anywhere; set
``PRESSURE_SIM_ROOT`` to override the default
``~/Documents/pressure simulation``::

    /opt/miniconda3/envs/sensoryforge/bin/python \\
        scripts/regenerate_stimulus_golden.py

The script skips (prints a message, exits 0) when pressure-simulation is not
found at ``PRESSURE_SIM_ROOT``, so it never blocks CI, which does not have
that repository checked out.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

OUT_PATH = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "stimulus_golden.npz"
)

# Every 10th time sample, to keep the fixture small (K3).
TIME_STRIDE = 10

GRID_SIZE = 80
SPACING = 0.15
CENTER = (0.0, 0.0)


def _pressure_sim_root() -> Path:
    return Path(
        os.environ.get("PRESSURE_SIM_ROOT", "~/Documents/pressure simulation")
    ).expanduser()


def _load_movie_module(repo: Path):
    """Import ``scripts/ebkf/_ebkf_pres_movies.py`` for ``gen_braille_H`` and
    ``gen_drifting_grating`` without executing its module-level side effects
    that need a stimulus bundle on disk (F-052/K3: only these two pure
    functions are used)."""
    path = repo / "scripts" / "ebkf" / "_ebkf_pres_movies.py"
    spec = importlib.util.spec_from_file_location("_ebkf_pres_movies_golden", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    repo = _pressure_sim_root()
    if not repo.exists():
        print(
            f"pressure-simulation not found at {repo}; skipping golden "
            "regeneration (set PRESSURE_SIM_ROOT to override)."
        )
        return 0

    sys.path.insert(0, str(repo))

    from encoding.grid_torch import GridManager
    from encoding.encode_runner import generate_stimulus_from_json

    grid = GridManager(grid_size=GRID_SIZE, spacing=SPACING, center=CENTER)
    xx, yy = grid.get_coordinates()

    mv = _load_movie_module(repo)

    # ramp_gaussian: experiments/ncn2026/_harness.py::gen_ramp_gaussian.
    # That module isn't a clean import (it loads the movie script with a
    # gitignored bundle path), so the six-line function is transcribed here
    # verbatim from pressure-simulation's source rather than imported --
    # this script's job is to prove SensoryForge's port matches
    # pressure-simulation's own output, not to avoid all duplication.
    def gen_ramp_gaussian(xx, yy, total_ms=1100, ramp_ms=50, sigma_mm=1.0):
        blob = torch.exp(-((xx**2 + yy**2) / (2.0 * sigma_mm**2)))
        amp = torch.ones(total_ms)
        if ramp_ms > 0:
            amp[:ramp_ms] = torch.linspace(0.0, 1.0, ramp_ms)
        return blob.unsqueeze(0) * amp.view(total_ms, 1, 1)

    ramp_gaussian = gen_ramp_gaussian(xx, yy, 1100, 50, 1.0)

    # moving_edge: encode_runner.generate_stimulus_from_json, type="edge",
    # motion="moving", with the K2 defaults.
    payload = {
        "type": "edge",
        "motion": "moving",
        "start": [-7.11, 0.0],
        "end": [7.0, 0.0],
        "spread": 1.0,
        "orientation_deg": 50.0,
        "amplitude": 1.0,
        "ramp_up_ms": 20.0,
        "plateau_ms": 300.0,
        "ramp_down_ms": 10.0,
        "total_ms": 330.0,
        "dt_ms": 1.0,
    }
    moving_edge, _ = generate_stimulus_from_json(payload, grid)

    braille_h = mv.gen_braille_H(xx, yy, 900, 75, 20.0)
    drifting_grating = mv.gen_drifting_grating(xx, yy, 1000, 100, 0.25, 15.0)

    stimuli = {
        "ramp_gaussian": ramp_gaussian,
        "moving_edge": moving_edge,
        "braille": braille_h,
        "drifting_grating": drifting_grating,
    }

    arrays = {}
    for name, frames in stimuli.items():
        arrays[name] = frames[::TIME_STRIDE].to(torch.float32).numpy()

    meta = {
        "source": {
            "ramp_gaussian": "experiments/ncn2026/_harness.py::gen_ramp_gaussian "
            "(transcribed here; see comment above)",
            "moving_edge": "encoding/encode_runner.py::generate_stimulus_from_json",
            "braille": "scripts/ebkf/_ebkf_pres_movies.py::gen_braille_H",
            "drifting_grating": "scripts/ebkf/_ebkf_pres_movies.py::gen_drifting_grating",
        },
        "params": {
            "ramp_gaussian": {"total_ms": 1100, "ramp_ms": 50, "sigma_mm": 1.0},
            "moving_edge": payload,
            "braille": {
                "total_ms": 900,
                "ramp_ms": 75,
                "v_mms": 20.0,
                "sigma_dot": 0.40,
            },
            "drifting_grating": {
                "total_ms": 1000,
                "ramp_ms": 100,
                "spatial_freq": 0.25,
                "v_mms": 15.0,
            },
        },
        "grid": {"grid_size": GRID_SIZE, "spacing": SPACING, "center": list(CENTER)},
        "time_stride": TIME_STRIDE,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        meta=json.dumps(meta),
        **arrays,
    )
    print(f"Wrote {OUT_PATH}")
    for name, arr in arrays.items():
        print(f"  {name}: {arr.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
