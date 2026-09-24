"""The committed TouchSim comparison still describes the current models (F-070).

``scripts/validation/compare_with_touchsim.py`` compares the tactile recipes'
SA/RA responses with TouchSim's SA1/RA afferents on a ramp-and-hold probe and
commits the result to ``benchmarks/results/touchsim_comparison/``. This test
re-runs the comparison at the committed amplitude-per-mm fit and requires the
same verdict for every feature: a change to a filter, a neuron preset or a
recipe gain that makes a feature start or stop agreeing with TouchSim fails
here until the script is re-run and its report committed. It also checks the
fit still holds: SA's hold rate at the fitted depth stays within 25% of
TouchSim's.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "validation" / "compare_with_touchsim.py"
COMMITTED = REPO / "benchmarks" / "results" / "touchsim_comparison" / "comparison.json"


@pytest.fixture(scope="module")
def compare():
    spec = importlib.util.spec_from_file_location("compare_with_touchsim", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def committed():
    return json.loads(COMMITTED.read_text())


@pytest.mark.parametrize("model", ["Izhikevich", "AdEx"])
def test_every_feature_verdict_matches_the_committed_comparison(
    compare, committed, model
):
    touchsim = compare.load_touchsim()
    recorded = committed[model]
    rows = compare.compare(recorded["preset"], touchsim, recorded["amplitude_per_mm"])
    assert compare.summarise(rows) == recorded["summary"], (
        "a feature's agreement with TouchSim changed; re-run "
        "scripts/validation/compare_with_touchsim.py and commit its report"
    )
    fitted = next(
        r
        for r in rows
        if r["feature"] == "SA hold rate" and r["depth_mm"] == compare.FIT_DEPTH_MM
    )
    assert fitted["sensoryforge"] == pytest.approx(fitted["touchsim"], rel=0.25)


def test_the_comparison_records_its_touchsim_provenance(committed):
    assert committed["fit_depth_mm"] == 1.25
    assert committed["target_sa_hold_hz"] > 0
    for model in ("Izhikevich", "AdEx"):
        assert committed[model]["amplitude_per_mm"] > 0
        assert committed[model]["summary"]["RA hold rate"] is True
