"""`sensoryforge batch --dataset`: bundles, index, tasks, resume, seeds, failures."""

import json
from pathlib import Path

import pytest
import yaml

from sensoryforge.cli import cmd_batch, create_parser
from sensoryforge.io.bundle import load_bundle
from sensoryforge.io.design import load_design
from sensoryforge.world import rng
from sensoryforge.world.dataset import build_dataset, load_dataset
from sensoryforge.world.runner import read_batch_index, task_slice

TESTS = Path(__file__).resolve().parents[1]
FIXTURE = TESTS / "fixtures" / "design_8x8"
WORLD = TESTS / "fixtures" / "worlds" / "tactile_small.yml"


@pytest.fixture
def dataset_file(tmp_path):
    spec = {
        "dataset": {
            "name": "tiny",
            "world": str(WORLD),
            "seed": 5,
            "duration_ms": 120,
            "splits": {
                "test": {"stratified": {"bins": 1, "per_bin": 1}},
                "fixed": {"draws": ["braille_H"]},
                "sessions": {"n": 1, "duration_ms": 200},
            },
        }
    }
    path = tmp_path / "tiny.yml"
    path.write_text(yaml.safe_dump(spec))
    return path


def _batch(*argv):
    return cmd_batch(create_parser().parse_args(["batch", *map(str, argv)]))


def test_batch_writes_one_bundle_per_entry(tmp_path, dataset_file):
    out = tmp_path / "out"
    assert _batch("--design", FIXTURE, "--dataset", dataset_file, "--output", out) == 0
    entries = build_dataset(load_dataset(dataset_file))
    rows = read_batch_index(out)
    assert [r["entry"] for r in rows] == [e.entry for e in entries]
    assert {r["status"] for r in rows} == {"ok"}
    base = load_design(FIXTURE)
    for e in entries:
        bundle = load_bundle(out / e.entry)
        cfg = bundle.meta["config_json"]
        assert cfg["world"]["entry"] == e.entry
        sim = cfg["config"]["simulation"]
        assert sim["receptor_noise_seed"] == e.seeds["noise"]
        # Every population gets its own 53-bit noise seed, whether or not the
        # design set one: no noise may hang on the 32-bit run seed.
        pops = cfg["config"]["populations"]
        assert len(pops) == len(base.populations)
        assert all(p.noise_seed is None for p in base.populations)
        for i, pop in enumerate(pops):
            assert pop["noise_seed"] == rng.seed53(e.seeds["noise"], "population", i)
        assert bundle.stimulus.shape[0] == int(round(e.duration_ms))
    info = json.loads((out / "batch.json").read_text())
    assert info["dataset_id"] == entries[0].dataset_id
    assert (
        info["design"]["design_id"]
        == json.loads((FIXTURE / "design.json").read_text())["design_id"]
    )
    partial = out / ".partial"
    assert not partial.exists() or not list(partial.rglob("data.h5"))


def test_splits_select_entries(tmp_path, dataset_file):
    out = tmp_path / "out"
    assert (
        _batch(
            "--design",
            FIXTURE,
            "--dataset",
            dataset_file,
            "--output",
            out,
            "--splits",
            "fixed",
        )
        == 0
    )
    assert [r["entry"] for r in read_batch_index(out)] == ["fixed/braille_H"]
    assert (
        _batch(
            "--design",
            FIXTURE,
            "--dataset",
            dataset_file,
            "--output",
            out,
            "--splits",
            "nope",
        )
        == 1
    )


def test_tasks_partition_the_entries(tmp_path, dataset_file):
    n = len(build_dataset(load_dataset(dataset_file)))
    covered = [i for t in range(3) for i in task_slice(n, 3, t)]
    assert covered == list(range(n))
    out = tmp_path / "out"
    for t in range(3):
        argv = ["--design", FIXTURE, "--dataset", dataset_file, "--output", out]
        assert _batch(*argv, "--tasks", 3, "--task-index", t) == 0
    names = sorted(p.name for p in (out / "index").iterdir())
    assert names == ["task_0000.jsonl", "task_0001.jsonl", "task_0002.jsonl"]
    assert len(read_batch_index(out)) == n


def test_print_tasks_gives_one_command_per_task(tmp_path, dataset_file, capsys):
    out = tmp_path / "o"
    argv = ["--design", FIXTURE, "--dataset", dataset_file, "--output", out]
    assert _batch(*argv, "--tasks", 4, "--print-tasks") == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4
    assert lines[2].startswith("sensoryforge batch ") and lines[2].endswith(
        "--tasks 4 --task-index 2"
    )
    assert all(str(FIXTURE.resolve()) in line for line in lines)
    assert not out.exists()


def test_rerun_replaces_resume_skips_and_partials_are_cleaned(tmp_path, dataset_file):
    out = tmp_path / "out"
    argv = [
        "--design",
        FIXTURE,
        "--dataset",
        dataset_file,
        "--output",
        out,
        "--entries",
        "0:2",
    ]
    assert _batch(*argv) == 0
    first = read_batch_index(out)[0]["entry"]
    marker = out / first / "marker.txt"
    marker.write_text("x")
    stale = out / ".partial" / first
    stale.mkdir(parents=True)
    (stale / "junk").write_text("x")
    assert _batch(*argv, "--resume") == 0
    assert marker.exists()  # resumed: the finished bundle was skipped
    assert _batch(*argv) == 0
    assert not marker.exists()  # rerun: the bundle was replaced
    assert not (stale / "junk").exists()


def test_a_failed_entry_is_recorded_and_the_exit_status_says_so(
    tmp_path, dataset_file, monkeypatch
):
    import sensoryforge.world.runner as runner

    real = runner.render_movie

    def flaky(item, *args, **kwargs):
        if getattr(item, "class_name", None) == "edges":
            raise ValueError("boom")
        return real(item, *args, **kwargs)

    monkeypatch.setattr(runner, "render_movie", flaky)
    out = tmp_path / "out"
    argv = [
        "--design",
        FIXTURE,
        "--dataset",
        dataset_file,
        "--output",
        out,
        "--entries",
        "2:5",  # fixed, sessions, then test/<class> by name: braille, dots, edges
    ]
    assert _batch(*argv) == 1
    rows = {r["entry"]: r for r in read_batch_index(out)}
    assert sorted(rows) == ["test/braille/0000", "test/dots/0000", "test/edges/0000"]
    assert rows["test/edges/0000"]["status"] == "failed"
    assert "boom" in rows["test/edges/0000"]["error"]
    assert rows["test/dots/0000"]["status"] == "ok"
    assert not (out / "test" / "edges" / "0000").exists()


def test_frames_render_on_cpu_whatever_the_engine_device(
    tmp_path, dataset_file, monkeypatch
):
    import torch

    import sensoryforge.world.runner as runner

    devices, received = [], []
    real = runner.render_movie

    def spy(item, *args, **kwargs):
        devices.append(torch.device(kwargs.get("device", "cpu")))
        return real(item, *args, **kwargs)

    class OtherDeviceEngine:
        """Stands in for an engine on another device ('meta' does no maths)."""

        def __init__(self, config):
            self.device = torch.device("meta")

        def run(self, frames, bundle_dir, **kwargs):
            received.append((frames.device, frames.dtype))
            Path(bundle_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(runner, "render_movie", spy)
    monkeypatch.setattr(runner, "SimulationEngine", OtherDeviceEngine)
    summary = runner.run_dataset(
        load_design(FIXTURE),
        load_dataset(dataset_file),
        tmp_path / "out",
        entry_range=slice(0, 3),
    )
    assert summary == {"ok": 3, "failed": 0, "skipped": 0}
    assert [d.type for d in devices] == ["cpu"] * 3
    assert received == [(torch.device("meta"), torch.float32)] * 3


def test_atomic_json_temp_names_differ_across_hosts_tasks_and_processes(
    tmp_path, monkeypatch
):
    import os
    import socket

    import sensoryforge.world.runner as runner

    moved = []
    real = os.replace
    monkeypatch.setattr(
        runner.os, "replace", lambda src, dst: (moved.append(src), real(src, dst))
    )
    monkeypatch.setattr(socket, "gethostname", lambda: "node-17.cluster/a b")
    target = tmp_path / "batch.json"
    runner._write_json_atomic(target, {"a": 1}, task_index=7)
    assert json.loads(target.read_text()) == {"a": 1}
    (tmp,) = moved
    name = Path(tmp).name
    assert name.startswith(".batch.json.") and name.endswith(".tmp")
    assert "node-17.cluster_a_b" in name and ".t7." in name
    assert f".p{os.getpid()}." in name


def test_a_sweep_batch_still_needs_its_config(capsys):
    assert _batch() == 1
    assert "batch needs a config" in capsys.readouterr().err
