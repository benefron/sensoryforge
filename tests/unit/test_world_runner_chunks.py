"""Long entries render in time chunks, equal to rendering them whole."""

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from sensoryforge.cli import cmd_batch, create_parser
from sensoryforge.config.schema import GridConfig
from sensoryforge.io.bundle import load_bundle
from sensoryforge.testing.golden import assert_matches_golden
from sensoryforge.world import (
    Canvas,
    Draw,
    Session,
    load_world,
    render_movie,
    sample,
    session,
)
from sensoryforge.world import runner
from sensoryforge.world.runner import render_frames

TESTS = Path(__file__).resolve().parents[1]
WORLDS = TESTS / "fixtures" / "worlds"
FIXTURE = TESTS / "fixtures" / "design_8x8"
MODEL_WORLD = load_world(WORLDS / "elements_v1_2.yml")
CANVAS = Canvas.from_grid(8, 8, 0.15)
DT = 1.0
# One frame is 64 elements: 640 elements make a chunk of 10 frames.
TINY = 640


def _whole(item, duration):
    return render_movie(item, CANVAS, DT, duration, dtype=torch.float64).to(
        torch.float32
    )


def test_chunked_frames_equal_whole_frames_bit_for_bit():
    model_session = session(MODEL_WORLD, seed=3, index=1)
    plain = sample(MODEL_WORLD, n=1, seed=4)[0]
    for item, duration in ((model_session, model_session.duration_ms), (plain, 120.0)):
        assert isinstance(item, (Session, Draw))
        assert duration / DT >= 5 * (TINY // 64)
        want = _whole(item, duration)
        got = render_frames(item, CANVAS, DT, duration, None, 1, chunk_elements=TINY)
        assert got.dtype == torch.float32
        assert torch.equal(got, want)
    # the default budget is one chunk and gives the same frames
    assert torch.equal(
        render_frames(plain, CANVAS, DT, 120.0, None, 1), _whole(plain, 120.0)
    )


def test_the_bundle_still_equals_the_in_process_render(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "CHUNK_ELEMENTS", TINY)
    spec_path = tmp_path / "ds.yml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "chunky",
                    "world": str(WORLDS / "tactile_small.yml"),
                    "seed": 7,
                    "duration_ms": 120,
                    "splits": {
                        "test": {"stratified": {"bins": 1, "per_bin": 1}},
                        "sessions": {"n": 1, "duration_ms": 250},
                    },
                }
            }
        )
    )
    out = tmp_path / "run"
    argv = ["batch", "--design", FIXTURE, "--dataset", spec_path, "--output", out]
    parser = create_parser()
    assert cmd_batch(parser.parse_args([str(a) for a in argv])) == 0
    from sensoryforge.io.design import load_design
    from sensoryforge.world import build_dataset, load_dataset

    spec = load_dataset(spec_path)
    config = load_design(FIXTURE)
    canvas = Canvas.from_grid_config(config.grids[0])
    entries = build_dataset(spec)
    assert any(e.duration_ms / DT > 10 for e in entries)
    for e in entries:
        want = render_movie(
            e.item,
            canvas,
            config.simulation.dt_ms,
            e.duration_ms,
            dtype=torch.float64,
        ).to(torch.float32)
        assert torch.equal(load_bundle(out / e.entry).stimulus, want), e.entry


def test_a_chunked_render_reproduces_the_frozen_v1_1_0_session_bundle():
    """Task 1's guard, through the chunked path: the v1.1.0 session frames."""
    ref = WORLDS / "v1_1_0_reference"
    record = json.loads((ref / "bundle_session" / "stimulus.json").read_text())
    entry = record["entry"]
    world = load_world(WORLDS / "tactile_small.yml")
    item = Session.from_dict(entry["draw"], world)
    grid = GridConfig.from_dict({"name": "bundle", **record["grid"]})
    want = torch.from_numpy(np.load(ref / "bundle_session" / "frames.npy"))
    cells = grid.rows * grid.cols
    got = render_frames(
        item,
        Canvas.from_grid_config(grid),
        record["dt_ms"],
        entry["duration_ms"],
        None,
        1,
        chunk_elements=cells * 20,  # 15 chunks of 20 frames
    )
    assert tuple(got.shape) == tuple(want.shape)
    platform_ref = json.loads((ref / "reference.json").read_text())["platform"]
    on_reference = (
        sys.platform == platform_ref["sys_platform"]
        and platform.machine() == platform_ref["machine"]
        and torch.__version__ == platform_ref["torch"]
    )
    if on_reference:
        assert torch.equal(got, want)
    else:
        assert_matches_golden(got, want, what="chunked v1.1.0 session frames")
