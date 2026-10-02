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
        for i, pop in enumerate(cfg["config"]["populations"]):
            if base.populations[i].noise_seed is None:
                assert pop.get("noise_seed") is None
            else:
                assert pop["noise_seed"] == rng.seed53(
                    e.seeds["noise"], "population", i
                )
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
        "0:4",
    ]
    assert _batch(*argv) == 1
    rows = {r["entry"]: r for r in read_batch_index(out)}
    assert rows["test/edges/0000"]["status"] == "failed"
    assert "boom" in rows["test/edges/0000"]["error"]
    assert rows["test/dots/0000"]["status"] == "ok"
    assert not (out / "test" / "edges" / "0000").exists()


def test_a_sweep_batch_still_needs_its_config(capsys):
    assert _batch() == 1
    assert "batch needs a config" in capsys.readouterr().err
