"""Pressure-simulation's contract with the world engine (its brief, §6), run here.

PS's Phase 2b runs these against the tagged sha; running them in SensoryForge
first means the tag is known to pass them. Each test carries the brief's
number. docs/reference/world_contract.md cites them.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml

from sensoryforge.cli import cmd_batch, create_parser
from sensoryforge.io.bundle import load_bundle
from sensoryforge.io.design import load_design, read_manifest
from sensoryforge.provenance import source_info
from sensoryforge.world import (
    Canvas,
    Draw,
    Session,
    build_dataset,
    load_dataset,
    load_world,
    movie_times,
    read_batch_index,
    render,
    render_movie,
    sample,
    session,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "design_8x8"
WORLDS = ROOT / "tests" / "fixtures" / "worlds"
WORLD_PATH = WORLDS / "tactile_small.yml"
WORLD = load_world(WORLD_PATH)

_DIGEST_SCRIPT = """
import hashlib, json, sys, torch
from sensoryforge.world import Canvas, load_world, render, sample
world = load_world(sys.argv[1])
draws = sample(world, n=50, seed=7)
frames = render(draws, Canvas.from_grid(16, 16, 0.1), torch.arange(0.0, 120.0, 5.0),
                dtype=torch.float64)
records = json.dumps([d.to_dict() for d in draws], sort_keys=True).encode()
print(json.dumps({"draws": hashlib.sha256(records).hexdigest(),
                  "frames": hashlib.sha256(frames.numpy().tobytes()).hexdigest()}))
"""


def _digests(draws, frames):
    records = json.dumps([d.to_dict() for d in draws], sort_keys=True).encode()
    return {
        "draws": hashlib.sha256(records).hexdigest(),
        "frames": hashlib.sha256(frames.numpy().tobytes()).hexdigest(),
    }


def _digests_in_a_new_process():
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    out = subprocess.run(
        [sys.executable, "-c", _DIGEST_SCRIPT, str(WORLD_PATH)],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_1_determinism_across_processes_and_batch_sizes():
    first, second = _digests_in_a_new_process(), _digests_in_a_new_process()
    assert first == second
    draws = sample(WORLD, n=50, seed=7)
    canvas = Canvas.from_grid(16, 16, 0.1)
    times = torch.arange(0.0, 120.0, 5.0)
    frames = render(draws, canvas, times, dtype=torch.float64)
    assert _digests(draws, frames) == first
    alone = sample(WORLD, indices=[31], seed=7)[0]
    assert alone.to_dict() == draws[31].to_dict()
    assert torch.equal(
        render([alone], canvas, times, dtype=torch.float64)[0], frames[31]
    )


@pytest.fixture(scope="module")
def batch_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("contract")
    spec_path = root / "dataset.yml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "contract",
                    "world": str(WORLD_PATH),
                    "seed": 2026,
                    "duration_ms": 120,
                    "splits": {
                        "test": {"stratified": {"bins": 1, "per_bin": 1}},
                        "validation": {"n": 2, "noise_repeats": 2},
                        "sessions": {"n": 1, "duration_ms": 250},
                        "fixed": {"draws": ["braille_H"]},
                    },
                }
            }
        )
    )
    sensor = root / "sensor.yml"
    sensor.write_text(yaml.safe_dump({"simulation": {"receptor_noise_std": 0.05}}))
    out = root / "run"
    argv = [
        "batch",
        str(sensor),
        "--design",
        str(FIXTURE),
        "--dataset",
        str(spec_path),
        "--output",
        str(out),
    ]
    assert cmd_batch(create_parser().parse_args(argv)) == 0
    spec = load_dataset(spec_path)
    return {
        "out": out,
        "spec": spec,
        "entries": build_dataset(spec),
        "spec_path": spec_path,
        "sensor": sensor,
    }


def _item_from_bundle(bundle_dir):
    """What PS does: rebuild the draw (or session) from the bundle's own record."""
    record = json.loads((bundle_dir / "stimuli" / "stimulus.json").read_text())[
        "entry"
    ]["draw"]
    if record.get("sampling") == "session":
        return Session.from_dict(record, WORLD)
    return Draw.from_dict(record, WORLD)


def test_2_the_bundle_records_exactly_the_in_process_render(batch_run):
    config = load_design(FIXTURE)
    canvas = Canvas.from_grid_config(config.grids[0])
    for e in batch_run["entries"]:
        bundle_dir = batch_run["out"] / e.entry
        item = _item_from_bundle(bundle_dir)
        want = render_movie(
            item, canvas, config.simulation.dt_ms, e.duration_ms, dtype=torch.float64
        ).to(torch.float32)
        assert torch.equal(load_bundle(bundle_dir).stimulus, want), e.entry


def test_3_windows_agree_with_movies():
    draws = sample(WORLD, n=8, seed=3)
    canvas = Canvas.from_grid(8, 8, 0.15)
    movie = render(draws, canvas, movie_times(1.0, 120.0), dtype=torch.float64)
    triples = torch.tensor([[42.0, 50.0, 58.0]] * 8, dtype=torch.float64)
    windows = render(draws, canvas, triples, dtype=torch.float64)
    for j, step in enumerate((42, 50, 58)):
        assert torch.equal(windows[:, j], movie[:, step])


def test_4_noise_seeds(batch_run, tmp_path):
    validation = [e for e in batch_run["entries"] if e.split == "validation"]
    a, b = validation[0], validation[1]  # one draw, two noise repeats
    assert a.seeds["draw"] == b.seeds["draw"] and a.seeds["noise"] != b.seeds["noise"]
    bundle_a = load_bundle(batch_run["out"] / a.entry)
    bundle_b = load_bundle(batch_run["out"] / b.entry)
    assert torch.equal(bundle_a.stimulus, bundle_b.stimulus)
    # The 53-bit noise seed itself (not its 32-bit truncation) seeds the receptors,
    # and the bundle records it, so the realisation can be rebuilt from the record.
    for e, bundle in ((a, bundle_a), (b, bundle_b)):
        recorded = bundle.meta["config_json"]["config"]["simulation"]
        assert recorded["receptor_noise_seed"] == e.seeds["noise"]
    assert any(
        not torch.equal(
            bundle_a.populations[p]["filtered"], bundle_b.populations[p]["filtered"]
        )
        for p in bundle_a.populations
    )
    rerun = tmp_path / "rerun"
    argv = [
        "batch",
        str(batch_run["sensor"]),
        "--design",
        str(FIXTURE),
        "--dataset",
        str(batch_run["spec_path"]),
        "--output",
        str(rerun),
        "--splits",
        "validation",
        "--entries",
        "0:1",
    ]
    assert cmd_batch(create_parser().parse_args(argv)) == 0
    again = load_bundle(rerun / a.entry)
    for name, data in bundle_a.populations.items():
        for key, tensor in data.items():
            assert torch.equal(tensor, again.populations[name][key]), (name, key)


def test_5_splits_strata_and_probes():
    spec = load_dataset(WORLDS / "dataset_small.yml")
    entries = build_dataset(spec)
    owners = {}
    for e in entries:
        if e.seeds["draw"] is not None:
            owners.setdefault(e.seeds["draw"], set()).add(e.split)
    assert all(len(splits) == 1 for splits in owners.values())
    noise = [e.seeds["noise"] for e in entries]
    assert len(noise) == len(set(noise))
    for cls in spec.world.classes.values():
        rows = [e for e in entries if e.split == "test" and e.class_name == cls.name]
        for axis in cls.random_axes:
            counts = Counter(e.bins[axis.name] for e in rows)
            if axis.support() is None:
                assert set(counts.values()) == {2}, (cls.name, axis.name)
            else:
                assert max(counts.values()) - min(counts.values()) <= 1
    probes = [e for e in entries if e.split == "probes"]
    assert probes
    for e in probes:
        axis = e.item.spec.axes[e.probe["axis"]]
        value = e.item.values[e.probe["axis"]]
        assert e.bins[e.probe["axis"]] == e.probe["side"]
        assert value < axis.lo if e.probe["side"] == "below" else value > axis.hi


def test_6_one_draw_on_40x40_and_80x80():
    draws = sample(WORLD, n=6, seed=10)
    times = torch.tensor([30.0, 60.0], dtype=torch.float64)
    small = render(draws, Canvas.from_grid(40, 40, 0.15), times, dtype=torch.float64)
    large = render(draws, Canvas.from_grid(80, 80, 0.15), times, dtype=torch.float64)
    torch.testing.assert_close(small, large[:, :, 20:60, 20:60], atol=1e-12, rtol=0)


def test_7_session_quiet_stretches_are_exactly_zero():
    s = session(WORLD, duration_ms=600.0, seed=17)
    times = movie_times(1.0, 600.0)
    frames = render_movie(
        s, Canvas.from_grid(8, 8, 0.15), 1.0, 600.0, dtype=torch.float64
    )
    contact = torch.zeros(len(times), dtype=torch.bool)
    for start, draw in s.items:
        for phase, a, b in draw.timeline:
            if phase in ("touch", "hold", "slide", "release"):
                contact |= (times >= start + a) & (times < start + b)
    assert torch.count_nonzero(frames[~contact]) == 0
    assert torch.count_nonzero(frames[contact]) > 0


def test_8_every_bundle_carries_its_provenance(batch_run):
    manifest = read_manifest(FIXTURE)
    sha = source_info()["sha"]
    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    for e in batch_run["entries"]:
        bundle_dir = batch_run["out"] / e.entry
        bundle = load_bundle(bundle_dir)
        cfg = bundle.meta["config_json"]
        assert cfg["design"] == manifest
        assert cfg["world"] == {
            "world_id": WORLD.world_id,
            "dataset_id": batch_run["spec"].dataset_id,
            "entry": e.entry,
        }
        assert cfg["sensoryforge_sha"] == sha and bundle.meta["sensoryforge_sha"] == sha
        payload = json.loads((bundle_dir / "stimuli" / "stimulus.json").read_text())
        assert payload["entry"] == json.loads(json.dumps(e.to_dict()))
    rows = read_batch_index(batch_run["out"])
    assert {r["sensoryforge_sha"] for r in rows} == {sha}
    assert {r["design_id"] for r in rows} == {manifest["design_id"]}
