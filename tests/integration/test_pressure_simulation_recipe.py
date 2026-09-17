"""``examples/pressure_simulation_recipe.py --quick`` writes four bundles
that reload (Phase 2, Wave K, K5).

Run as a subprocess with an explicit cwd/PYTHONPATH (F-053: a script run
directly puts its own directory on sys.path[0], not the repo root, so
``import sensoryforge`` can silently resolve to a different checkout's
editable install -- see tests/docs/test_docs_examples.py for the same
pattern).
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
SCRIPT_PATH = REPO_ROOT / "examples" / "pressure_simulation_recipe.py"
OUTPUT_DIR = REPO_ROOT / "examples" / "output" / "pressure_simulation_recipe"

_EXPECTED_STIMULI = {"ramp_gaussian", "moving_edge", "braille", "drifting_grating"}


@pytest.fixture
def clean_output_dir():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    yield
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def test_quick_run_writes_four_reloadable_bundles(clean_output_dir):
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
        f"pressure_simulation_recipe.py --quick exited with code "
        f"{result.returncode}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    assert OUTPUT_DIR.exists()
    bundle_names = {d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()}
    assert bundle_names == _EXPECTED_STIMULI

    for name in _EXPECTED_STIMULI:
        bundle = load_bundle(OUTPUT_DIR / name)
        assert set(bundle.populations.keys()) == {"SA Population", "RA Population"}
        for pop_name, pop_data in bundle.populations.items():
            assert "spikes" in pop_data
            assert pop_data["spikes"].shape[-1] == 900  # d=0.40 on 80x80/0.15mm
        assert bundle.stimulus is not None
        assert bundle.stimulus.shape[-2:] == (80, 80)
