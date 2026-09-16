"""``examples/vision_rgb_onoff.py --quick`` writes one reloadable bundle
(Phase 2, Wave M, M4) -- the CI-executed generality demo.

Run as a subprocess with an explicit cwd/PYTHONPATH (F-053: a script run
directly puts its own directory on sys.path[0], not the repo root -- see
tests/integration/test_pressure_simulation_recipe.py for the same pattern).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sensoryforge.io.bundle import load_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples" / "vision_rgb_onoff.py"
OUTPUT_DIR = REPO_ROOT / "examples" / "output" / "vision_rgb_onoff"


@pytest.fixture
def clean_output_dir():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    yield
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def test_quick_run_writes_reloadable_bundle(clean_output_dir):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env["PYTHONPATH"]]
        if env.get("PYTHONPATH")
        else [str(REPO_ROOT)]
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--quick"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert result.returncode == 0, (
        f"vision_rgb_onoff.py --quick exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    bundle_dir = OUTPUT_DIR / "run"
    assert bundle_dir.exists()
    bundle = load_bundle(bundle_dir)

    assert set(bundle.populations.keys()) == {
        "RG OnOff Population",
        "RGB Concat Population",
    }

    rg = bundle.populations["RG OnOff Population"]
    rgb = bundle.populations["RGB Concat Population"]

    # "RG OnOff Population" (combine: sum, two inputs, neurons_per_row=8):
    # its neuron count is the single-input lattice size, not doubled --
    # sum shares one set of neurons across inputs.
    assert rg["spikes"].shape[-1] == 8 * 8

    # "RGB Concat Population" (combine: concat, three inputs,
    # neurons_per_row=6): 3 * 6x6 blocks -- concat's neuron count really is
    # N * len(inputs), the plausible-looking failure mode this guards.
    assert rgb["spikes"].shape[-1] == 3 * 6 * 6

    # Some real signal in both populations, not a silently-zero pipeline.
    assert int(rg["spikes"].sum().item()) > 0
    assert int(rgb["spikes"].sum().item()) > 0

    assert bundle.stimulus is not None
    assert bundle.stimulus.shape[-3:] == (3, 32, 32)  # [C, H, W], 3 channels
