"""Golden parity test against pressure-simulation for the four Wave K stimuli
(task K3).

Compares ``render_stimulus`` (K1) rendering ``sensoryforge.stimuli.tactile``'s
four ported stimuli (K2) against a fixture exported directly from
pressure-simulation's own functions (see
``scripts/regenerate_stimulus_golden.py`` and
``tests/fixtures/stimulus_golden.npz``), the way
``tests/integration/test_pressure_sim_parity.py`` (task E5) compares filtered
responses. The comparison is exact (zero tolerance), matching that test's
convention, since both sides do the identical floating-point arithmetic in
the identical order -- the port is a transcription, not a reimplementation.

The fixture is subsampled to every 10th time sample (``TIME_STRIDE``) to keep
it small; this test applies the same stride to SensoryForge's output before
comparing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.register_components import register_all
from sensoryforge.stimuli.render import render_stimulus

register_all()

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "stimulus_golden.npz"


@pytest.fixture(scope="module")
def golden():
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"golden fixture not found at {FIXTURE_PATH}; run "
            "scripts/regenerate_stimulus_golden.py with pressure-simulation "
            "available first"
        )
    data = np.load(FIXTURE_PATH, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return data, meta


@pytest.fixture(scope="module")
def grid(golden):
    _, meta = golden
    g = meta["grid"]
    return ReceptorGrid(
        grid_size=(g["grid_size"], g["grid_size"]),
        spacing=g["spacing"],
        center=tuple(g["center"]),
    )


_STIMULUS_PARAMS = {
    "ramp_gaussian": {"total_ms": 1100.0, "ramp_ms": 50.0, "sigma_mm": 1.0},
    "moving_edge": {
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
    },
    "braille": {
        "total_ms": 900.0,
        "ramp_ms": 75.0,
        "v_mms": 20.0,
        "sigma_dot": 0.40,
    },
    "drifting_grating": {
        "total_ms": 1000.0,
        "ramp_ms": 100.0,
        "spatial_freq": 0.25,
        "v_mms": 15.0,
    },
}


def _render(grid, name: str, stride: int) -> torch.Tensor:
    xx, yy = grid.get_coordinates()
    frames, _ = render_stimulus(name, _STIMULUS_PARAMS[name], xx, yy, dt_ms=1.0)
    return frames[::stride]


@pytest.mark.parametrize("name", sorted(_STIMULUS_PARAMS))
def test_stimulus_matches_pressure_simulation_exactly(golden, grid, name):
    data, meta = golden
    stride = int(meta["time_stride"])
    expected = torch.from_numpy(data[name])
    actual = _render(grid, name, stride)
    assert (
        actual.shape == expected.shape
    ), f"{name}: shape {tuple(actual.shape)} != golden {tuple(expected.shape)}"
    assert torch.equal(actual, expected), f"{name}: values differ from golden"


# ---------------------------------------------------------------------------
# K3 requires proving the test above actually bites: four independent
# mutations to the port, each of which must break the comparison. This is
# done directly against the golden fixture rather than as permanent test
# cases (a "test" that is expected to fail is not a test to keep in the
# suite) -- see the Wave K report for the four mutations and their output.
# ---------------------------------------------------------------------------
