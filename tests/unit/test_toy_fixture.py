"""The toy non-tactile data set (A1b): it builds and holds its declared statistics."""

from __future__ import annotations

import math
import time

import h5py
import numpy as np
import pytest

from sensoryforge.io.bundle import load_bundle
from sensoryforge.world.render import Canvas
from sensoryforge.world.runner import read_batch_index
from tests.fixtures import toy_vision as toy

BLOBS = {"red_blob": "red", "green_blob": "green", "blue_blob": "blue"}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    started = time.perf_counter()
    ds = toy.build_toy_dataset(tmp_path_factory.mktemp("toy"))
    ds.seconds = time.perf_counter() - started
    return ds


def _frames(built, entry):
    with h5py.File(built.data / entry.entry / "data.h5", "r") as f:
        return np.asarray(f["stimulus/frames"])


def test_builds_in_seconds_with_the_batch_layout(built):
    assert built.seconds < 30
    assert built.summary == {"ok": len(built.entries), "failed": 0, "skipped": 0}
    assert len(built.entries) == 8 + 4
    rows = read_batch_index(built.data)
    assert [r["entry"] for r in rows] == [e.entry for e in built.entries]
    assert {r["status"] for r in rows} == {"ok"}
    manifest = (built.data / "manifest.jsonl").read_text().splitlines()
    assert len(manifest) == len(built.entries)
    assert (built.data / "dataset.json").exists()


def test_no_data_files_are_committed():
    from pathlib import Path

    stray = [
        p for p in Path(toy.__file__).parent.glob("toy_vision*") if p.suffix != ".py"
    ]
    assert stray == []


def test_three_channels_two_populations(built):
    bundle = load_bundle(built.data / built.entries[0].entry)
    config = bundle.meta["config_json"]["config"]
    assert config["grids"][0]["channels"] == toy.CHANNELS
    assert [p["name"] for p in config["populations"]] == ["luma", "chroma"]
    assert bundle.stimulus.shape == (toy.DURATION_MS, 3, toy.ROWS, toy.COLS)
    with h5py.File(built.data / built.entries[0].entry / "data.h5", "r") as f:
        assert f["populations/luma/spikes"].shape[-1] == 16
        assert f["populations/chroma/spikes"].shape[-1] == 32


def test_each_blob_drives_only_its_own_channel(built):
    seen = set()
    for e in built.entries:
        frames = _frames(built, e)
        if e.class_name == "dark":
            assert not frames.any()
            continue
        own = toy.CHANNELS.index(BLOBS[e.class_name])
        others = [c for c in range(3) if c != own]
        assert frames[:, own].max() > 0
        assert not frames[:, others].any()
        seen.add(e.class_name)
    assert seen == set(BLOBS)


def test_populations_respond_to_blobs_and_stay_quiet_on_dark(built):
    totals = {"luma": 0, "chroma": 0}
    for e in built.entries:
        with h5py.File(built.data / e.entry / "data.h5", "r") as f:
            spikes = {
                k: int(np.asarray(f[f"populations/{k}/spikes"]).sum()) for k in totals
            }
        if e.class_name == "dark":
            assert spikes == {"luma": 0, "chroma": 0}
        for k in totals:
            totals[k] += spikes[k]
    assert totals["luma"] > 0 and totals["chroma"] > 0


def test_frames_hold_the_declared_statistics(built):
    """Measured from the frames alone against the draw's record: the fixture's point."""
    canvas = Canvas.from_grid_config(toy.toy_config().grids[0])
    xx, yy = canvas.xx.numpy(), canvas.yy.numpy()
    checked = 0
    for e in built.entries:
        if e.class_name not in BLOBS:
            continue
        values = e.to_dict()["draw"]["values"]
        frames = _frames(built, e)[:, toy.CHANNELS.index(BLOBS[e.class_name])]
        end = int(e.to_dict()["draw"]["end_ms"])
        # active exactly while the declared timeline runs, exactly zero after
        active = frames.reshape(len(frames), -1).any(axis=1)
        assert active[:end].all() and not active[end:].any()
        assert frames.max() <= values["amplitude"] * (1 + 1e-6)

        def centroid(k):
            w = frames[k]
            return (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()

        # keep the blob well inside the array so its edge does not bias the centroid
        sigma, speed = values["sigma_mm"], values["speed_mm_per_ms"]
        theta = math.radians(values["direction_deg"])
        x0, y0 = values["x_mm"], values["y_mm"]
        reach = 60 * speed
        half = toy.ROWS * toy.SPACING / 2
        ends = (x0 + reach * math.cos(theta), y0 + reach * math.sin(theta))
        if max(abs(x0), abs(y0), abs(ends[0]), abs(ends[1])) + 2.5 * sigma > half:
            continue
        cx, cy = centroid(0)
        assert (cx, cy) == pytest.approx((x0, y0), abs=0.15)
        (ax, ay), (bx, by) = centroid(30), centroid(89)
        measured_speed = math.hypot(bx - ax, by - ay) / 59
        assert measured_speed == pytest.approx(speed, rel=0.1)
        measured_dir = math.degrees(math.atan2(by - ay, bx - ax)) % 360
        diff = (measured_dir - values["direction_deg"] + 180) % 360 - 180
        assert abs(diff) < 5
        checked += 1
    assert checked >= 3
