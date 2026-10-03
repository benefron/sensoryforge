"""Separating sensor from membrane noise changes no existing config (C-130).

``tests/fixtures/noise_split_golden.pt`` was recorded by
``scripts/dev/export_noise_split_golden.py`` with the code from **before**
``PopulationConfig.sensor_noise_std`` / ``membrane_noise_std`` existed: the
spikes (and filtered current) of every shipped preset, and of two configs using
the now-deprecated ``noise_std`` alias (seeded per population, and seeded by the
run). The current code must reproduce them: bit-identically on the platform the
fixture was recorded on, and within a few threshold-edge spikes elsewhere
(F-071 -- other platforms round float32 differently).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "noise_split_golden.pt"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "dev"))

import export_noise_split_golden as recorder  # noqa: E402


@pytest.fixture(scope="module")
def golden():
    return torch.load(FIXTURE, weights_only=False)


def _case_names():
    return [name for name, _ in recorder.cases()]


def test_fixture_covers_every_shipped_preset(golden):
    from sensoryforge.presets import list_presets

    recorded = set(golden["cases"])
    for name in list_presets():
        assert f"preset:{name}" in recorded


@pytest.mark.parametrize("case_name", _case_names())
def test_case_reproduces_pre_split_golden(golden, case_name):
    config_dict = dict(recorder.cases())[case_name]
    actual = recorder.run_case(config_dict)
    expected = golden["cases"][case_name]
    assert set(actual) == set(expected)

    same_platform = golden["platform"] == recorder.platform_signature()
    for key, want in expected.items():
        got = actual[key]
        if key.endswith("/spikes"):
            got_dense, want_dense = got.to_dense(), want.to_dense()
            if same_platform:
                assert torch.equal(
                    got_dense, want_dense
                ), f"{case_name} {key}: not bit-identical"
            else:
                mismatched = int((got_dense != want_dense).sum())
                allowed = max(5, int(0.005 * max(1, int(want_dense.sum()))))
                assert (
                    mismatched <= allowed
                ), f"{case_name} {key}: {mismatched} bins differ (allowed {allowed})"
        else:
            assert got["shape"] == want["shape"]
            if same_platform:
                assert (
                    got["sha256"] == want["sha256"]
                ), f"{case_name} {key}: filtered current not bit-identical"
            else:
                assert got["sum"] == pytest.approx(want["sum"], rel=1e-4, abs=1e-3)
                assert got["sq_sum"] == pytest.approx(want["sq_sum"], rel=1e-4)
