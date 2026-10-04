"""The v1.1.0 compatibility guard: old worlds, data sets and bundles stay bit-equal.

The v1.2.0 world additions are *invisible until used* (IU): a world, data set
or session file that uses no v1.2 key must load to the same normalised dict,
``world_id``, draws, records, manifest rows, session records and frames, bit
for bit, as v1.1.0, and a v1.1.0 bundle's own record must still rebuild and
re-render to its own frames.

The reference is ``tests/fixtures/worlds/v1_1_0_reference/``, recorded on a
clean copy of tag v1.1.0 by ``tests/fixtures/make_world_v1_1_reference.py``
(the same script on the working tree gave a byte-identical ``reference.json``).

**A failure here means an old world changed.** Do not update the reference and
do not loosen a tolerance: fix the change so that a file without the new key
renders and hashes as before. Frame hashes are exact only on macOS arm64,
where the reference was made; elsewhere frames are compared to a tolerance
(F-071).
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sensoryforge.config.schema import GridConfig
from sensoryforge.testing.golden import assert_matches_golden
from sensoryforge.world import (
    Canvas,
    Draw,
    Session,
    build_dataset,
    load_dataset,
    load_world,
    render,
    render_movie,
    sample,
    session,
)

ROOT = Path(__file__).resolve().parents[2]
WORLDS = ROOT / "tests" / "fixtures" / "worlds"
REFERENCE_DIR = WORLDS / "v1_1_0_reference"
REFERENCE = json.loads((REFERENCE_DIR / "reference.json").read_text())
WORLD_FILES = {
    "tactile_small": WORLDS / "tactile_small.yml",
    "every_element_v1_1": WORLDS / "every_element_v1_1.yml",
}
DATASET_FILE = WORLDS / "dataset_small.yml"
ON_REFERENCE_PLATFORM = (
    sys.platform == REFERENCE["platform"]["sys_platform"]
    and platform.machine() == REFERENCE["platform"]["machine"]
)
BUNDLES = ("bundle_test", "bundle_session")


def _rounded(value):
    if isinstance(value, float):
        return float(f"{value:.10g}")
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    return value


def _digest(obj) -> str:
    text = json.dumps(_rounded(obj), sort_keys=True).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def _worlds():
    return {name: load_world(path) for name, path in WORLD_FILES.items()}


def _frames(world) -> torch.Tensor:
    draws = sample(world, n=12, seed=7)
    return render(
        draws,
        Canvas.from_grid(12, 12, 0.1),
        torch.arange(0.0, 120.0, 5.0, dtype=torch.float64),
        dtype=torch.float64,
    )


def _same_frames(actual: torch.Tensor, want: torch.Tensor, what: str, **tol) -> None:
    """Bit-equal on the reference platform, to the golden tolerance elsewhere."""
    if ON_REFERENCE_PLATFORM:
        assert torch.equal(actual, want), f"{what}: frames changed (bit-exact)"
    else:
        assert_matches_golden(actual, want, what=what, **tol)


def test_old_world_ids_and_normal_forms_are_unchanged():
    for name, world in _worlds().items():
        want = REFERENCE["worlds"][name]
        assert world.world_id == want["world_id"], name
        normal = json.dumps(world.to_dict(), sort_keys=True).encode("utf-8")
        assert hashlib.sha256(normal).hexdigest() == want["normal_form_sha256"], name


def test_old_draw_records_are_unchanged():
    for name, world in _worlds().items():
        draws = sample(world, n=50, seed=7)
        assert _digest([d.to_dict() for d in draws]) == (
            REFERENCE["worlds"][name]["draws_digest"]
        ), name


def test_old_session_record_is_unchanged():
    want = REFERENCE["session"]
    world = load_world(WORLD_FILES[want["world"]])
    record = session(world, want["duration_ms"], want["seed"], want["index"])
    assert _digest(record.to_dict()) == want["digest"]


def test_old_dataset_id_and_manifest_are_unchanged():
    spec = load_dataset(DATASET_FILE)
    assert spec.dataset_id == REFERENCE["dataset"]["dataset_id"]
    rows = [e.to_dict() for e in build_dataset(spec)]
    assert _digest(rows) == REFERENCE["dataset"]["manifest_digest"]


def test_old_frames_are_unchanged():
    recorded = torch.load(REFERENCE_DIR / "frames.pt", weights_only=True)
    for name, world in _worlds().items():
        frames = _frames(world)
        assert tuple(frames.shape) == tuple(recorded[name].shape), name
        if ON_REFERENCE_PLATFORM:
            sha = hashlib.sha256(frames.numpy().tobytes()).hexdigest()
            assert sha == REFERENCE["worlds"][name]["frames_sha256"], name
        _same_frames(frames, recorded[name], f"{name} frames", atol=1e-12)


@pytest.mark.parametrize("bundle", BUNDLES)
def test_a_v1_1_bundle_record_rebuilds_and_rerenders_bit_equal(bundle):
    record = json.loads((REFERENCE_DIR / bundle / "stimulus.json").read_text())
    entry = record["entry"]
    world = load_world(WORLD_FILES["tactile_small"])
    data = entry["draw"]
    # from_dict raises if the world id no longer matches the record's.
    if data.get("sampling") == "session":
        item = Session.from_dict(data, world)
    else:
        item = Draw.from_dict(data, world)
    grid = GridConfig.from_dict({"name": "bundle", **record["grid"]})
    movie = render_movie(
        item,
        Canvas.from_grid_config(grid),
        record["dt_ms"],
        entry["duration_ms"],
        dtype=torch.float64,
    ).float()
    want = torch.from_numpy(np.load(REFERENCE_DIR / bundle / "frames.npy"))
    assert tuple(movie.shape) == tuple(want.shape), bundle
    _same_frames(movie, want, f"{bundle} frames")
