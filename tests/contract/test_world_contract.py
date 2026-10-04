"""Pressure-simulation's contract with the world engine (its brief, §6), run here.

PS's Phase 2b runs these against the tagged sha; running them in SensoryForge
first means the tag is known to pass them. Each test carries the brief's
number. docs/reference/world_contract.md cites them.
"""

import hashlib
import importlib.util
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
    # Not a comparison of zeros: every one of these draws touches at these times.
    assert bool((windows.flatten(1).abs().sum(1) > 0).all())
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
    # Not a comparison of zeros: every one of these draws touches at these times.
    assert bool((small.flatten(1).abs().sum(1) > 0).all())
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


def test_7_model_session_quiet_stretches_are_exactly_zero():
    model_world = load_world(
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "worlds"
        / "elements_v1_2.yml"
    )
    s = session(model_world, seed=3, index=1)
    assert s.session_type is not None
    times = movie_times(1.0, s.duration_ms)
    frames = render_movie(
        s, Canvas.from_grid(8, 8, 0.15), 1.0, s.duration_ms, dtype=torch.float64
    )
    contact = torch.zeros(len(times), dtype=torch.bool)
    for start, draw in s.items:
        for phase, a, b in draw.timeline:
            if phase in ("touch", "hold", "slide", "release"):
                contact |= (times >= start + a) & (times < start + b)
    assert (~contact).any() and contact.any()
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


def test_9_old_worlds_and_bundles_are_unchanged():
    """v1.1.0's worlds, data sets and bundles keep their ids, records and frames.

    The checks live in ``tests/unit/test_world_v1_1_compat.py`` (the invisible-
    until-used rule); this runs them as part of the contract.
    """
    path = ROOT / "tests" / "unit" / "test_world_v1_1_compat.py"
    spec = importlib.util.spec_from_file_location("world_v1_1_compat_checks", path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    checks.test_old_world_ids_and_normal_forms_are_unchanged()
    checks.test_old_draw_records_are_unchanged()
    checks.test_old_session_record_is_unchanged()
    checks.test_old_dataset_id_and_manifest_are_unchanged()
    checks.test_old_frames_are_unchanged()
    for bundle in checks.BUNDLES:
        checks.test_a_v1_1_bundle_record_rebuilds_and_rerenders_bit_equal(bundle)


ELEMENTS = WORLDS / "elements_v1_2.yml"


def test_10_every_v1_2_element_renders_equal_to_layered_and_on_both_grids():
    """One class per v1.2 element: world render = layered, 40x40 = 80x80."""
    import math

    from sensoryforge.stimuli.layered import render_layers

    world = load_world(ELEMENTS)
    names = sorted(n for n in world.classes if world.classes[n].kind != "quiet")
    assert len(names) == 11
    canvas = Canvas.from_grid(rows=24, cols=24, spacing_mm=0.05)
    coarse = Canvas.from_grid(rows=41, cols=41, spacing_mm=0.15)
    fine = Canvas.from_grid(rows=81, cols=81, spacing_mm=0.075)
    # xx runs down the rows and yy along the columns (Canvas.from_grid)
    ir = torch.stack([(fine.xx[:, 0] - x).abs().argmin() for x in coarse.xx[:, 0]])
    ic = torch.stack([(fine.yy[0] - y).abs().argmin() for y in coarse.yy[0]])
    for name in names:
        draws = sample(world, n=2, seed=21, classes=[name])
        total = math.ceil(max(d.end_ms for d in draws)) + 5.0
        times = movie_times(1.0, total)
        frames = render(draws, canvas, times, dtype=torch.float64)
        assert torch.count_nonzero(frames) > 0, name
        for i, draw in enumerate(draws):
            ref = render_layers(
                [draw.to_layer()],
                canvas.xx.float(),
                canvas.yy.float(),
                dt_ms=1.0,
                total_ms=total,
            ).double()
            torch.testing.assert_close(frames[i], ref, atol=1e-5, rtol=0)
        draw = draws[0]
        times = movie_times(1.0, math.ceil(draw.end_ms) + 2.0)
        a = render([draw], coarse, times, dtype=torch.float64)[0]
        b = render([draw], fine, times, dtype=torch.float64)[0]
        torch.testing.assert_close(a, b[:, ir][:, :, ic], atol=1e-12, rtol=0)


_CONTACT = ("touch", "hold", "slide", "release")


def _episode_contact_ms(draw):
    return sum(b - a for phase, a, b in draw.timeline if phase in _CONTACT)


def test_11_model_sessions_meet_their_declared_fraction_and_gaps_are_zero():
    """A ``sessions:`` session whose episodes end before its duration ``D``
    has a contact share within ``D`` of at least ``f``, exceeding it by less
    than its last episode's contact time over ``D``; its gaps render 0."""
    world = load_world(ELEMENTS)
    canvas = Canvas.from_grid(8, 8, 0.15)
    gaps_seen = covered = 0
    for index in range(8):
        s = session(world, seed=5, index=index)
        target = s.contact_fraction
        assert target is not None and s.session_type is not None
        length = sum(d.end_ms for _, d in s.items)
        realised = s.contact_ms / s.duration_ms
        assert realised == pytest.approx(1.0 - s.quiet_fraction)
        if length < s.duration_ms:
            covered += 1
            last = _episode_contact_ms(s.items[-1][1]) / s.duration_ms
            assert target - 1e-9 <= realised <= target + last + 1e-9, index
        holes = []
        t = 0.0
        for start, draw in s.items:
            holes.append((t, start))
            t = start + draw.end_ms
        holes.append((t, s.duration_ms))
        holes = [(a, b) for a, b in holes if b - a > 1e-9]
        gaps_seen += len(holes)
        times = movie_times(1.0, s.duration_ms)
        frames = render_movie(s, canvas, 1.0, s.duration_ms, dtype=torch.float64)
        for a, b in holes:
            inside = (times >= a) & (times < b)
            assert torch.count_nonzero(frames[inside]) == 0
    assert gaps_seen > 0 and covered > 0


def test_11_a_session_cut_at_its_duration_is_outside_the_guarantee():
    """Guarantee 11 covers sessions whose episodes end before ``D``; a
    session whose last episode runs past ``D`` is cut there, and its contact
    share within ``D`` can fall below ``f`` even when its episodes' own
    contact share is at least ``f``.

    The same case as ``session(<the other repo's charter-like world>,
    seed=17, index=586)`` (f = 0.895, episodes' share 0.900, share within D
    0.863), in a placeholder world: two episodes that pause between contacts
    (contact share 40/130), then a long hold that runs past D.
    """
    gauss = {"shape": {"kind": "gaussian", "sigma_mm": 0.2}}
    world = load_world(
        {
            "world": {
                "name": "cut_at_d",
                "classes": {
                    "paused": {
                        "layer": gauss,
                        "axes": {
                            "hold_ms": {"value": 10.0},
                            "contacts": {"value": 4},
                            "pause_ms": {"value": 30.0},
                        },
                    },
                    "long": {
                        "layer": gauss,
                        "axes": {"hold_ms": {"range": [200, 400]}},
                    },
                },
                "sessions": {
                    "duration_ms": {"value": 400.0},
                    "contact_fraction": {"value": 0.6},
                    "gap_mean_ms": 50.0,
                },
            }
        }
    )
    s = session(world, seed=0, index=4)
    assert [d.class_name for _, d in s.items] == ["paused", "paused", "long"]
    length = sum(d.end_ms for _, d in s.items)
    episodes_share = sum(_episode_contact_ms(d) for _, d in s.items) / length
    realised = s.contact_ms / s.duration_ms
    assert length > s.duration_ms and s.truncated  # not covered: no gap
    assert episodes_share >= s.contact_fraction  # v1.2.0's wording claimed f
    assert realised == pytest.approx(0.55)  # 80 ms + 140 ms of 400 ms
    assert realised < s.contact_fraction
