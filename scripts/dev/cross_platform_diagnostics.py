"""Measure how far each golden fixture comparison is from exact on this machine.

Run from the repository root. For every test that compares against a fixture
generated on macOS arm64, report whether the *structure* matches exactly
(which receptors connect to which neuron, i.e. the non-zero pattern) and how
large the *value* differences are. Structure differing is a real
cross-platform bug; small value differences with identical structure are
floating-point rounding. This decides how each test should be fixed.
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare(actual, expected) -> dict:
    a = torch.as_tensor(np.asarray(actual)).double()
    e = torch.as_tensor(np.asarray(expected)).double()
    out = {"shape_equal": list(a.shape) == list(e.shape)}
    if not out["shape_equal"]:
        out["shapes"] = [list(a.shape), list(e.shape)]
        return out
    out["exact"] = bool(torch.equal(a, e))
    out["pattern_equal"] = bool(torch.equal(a != 0, e != 0))
    diff = (a - e).abs()
    out["max_abs"] = float(diff.max())
    denom = e.abs().clamp_min(1e-30)
    nz = e != 0
    out["max_rel_nonzero"] = float((diff[nz] / denom[nz]).max()) if nz.any() else 0.0
    out["n_differing"] = int((diff > 0).sum())
    out["n_total"] = int(diff.numel())
    return out


report = {
    "platform": f"{platform.system()} {platform.machine()}",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "results": {},
}
R = report["results"]

# 1. Receptive-field builders vs rf_builder_golden_weights.pt
m = load("tests/unit/test_rf_builders_build.py", "rfb")
golden = torch.load(m.GOLDEN, weights_only=False)
for method in m.METHODS:
    bank = m._builder(method, golden).build()
    R[f"rf_builder/{method}"] = compare(bank.weights, golden[method]["weights"])

# 2. Engine grid path vs rf_engine_golden_weights.pt
m2 = load("tests/unit/test_engine_rf_banks.py", "erb")
g2 = torch.load(m2.GOLDEN, weights_only=False)
eng = m2.SimulationEngine(m2._config("gaussian"))
bank = eng.populations[0]["bank"]
rec = g2["engine"]
R["engine/weights"] = compare(
    bank.weights, rec["weights"].reshape(9, m2.ROWS * m2.COLS)
)
R["engine/neuron_centers"] = compare(bank.neuron_centers, rec["neuron_centers"])
R["engine/receptor_coords"] = compare(bank.receptor_coords, rec["receptor_coords"])

# 3. Stimulus parity
m3 = load("tests/integration/test_stimulus_parity.py", "sp")
data = np.load(m3.FIXTURE_PATH, allow_pickle=False)
meta = json.loads(str(data["meta"]))
stride = int(meta["time_stride"])
grid = (
    m3.GridManager(grid_size=80, spacing=0.15, center=(0.0, 0.0))
    if hasattr(m3, "GridManager")
    else None
)
if grid is None:
    from sensoryforge.core.grid import GridManager

    grid = GridManager(grid_size=80, spacing=0.15, center=(0.0, 0.0))
for name in ("ramp_gaussian", "moving_edge", "braille", "drifting_grating"):
    actual = m3._render(grid, name, stride)
    R[f"stimulus/{name}"] = compare(actual, torch.from_numpy(data[name]))

# 4. pressure-simulation parity
m4 = load("tests/integration/test_pressure_sim_parity.py", "psp")
gold = np.load(
    m4.GOLDEN_PATH
    if hasattr(m4, "GOLDEN_PATH")
    else REPO / "tests/fixtures/pressure_sim_golden/case_small.npz"
)
res = (
    m4.engine_result.__wrapped__(gold)
    if hasattr(m4.engine_result, "__wrapped__")
    else None
)
if res is not None:
    for pop, key in (("SA Pop", "sa"), ("RA Pop", "ra")):
        R[f"psim/{key}_drive"] = compare(res[pop]["drive"], gold[f"{key}_drive"])
        R[f"psim/{key}_filtered"] = compare(
            res[pop]["filtered"], gold[f"{key}_filtered"]
        )

print(json.dumps(report, indent=1))
