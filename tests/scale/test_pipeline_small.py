"""The small scale test every CI job runs (roadmap A11b).

It is the large run in miniature: the same world file goes through the same pipeline to
a report folder, and the same invariants are checked at a size that fits a few seconds.
Beyond the stage-by-stage tests in ``tests/unit/test_eval_pipeline.py`` it checks
what only shows with many entries: every entry is carried through every stage, a
run that resumes changes nothing, and splitting the batch into tasks (as a cluster
does) gives the data set a single process gives.
"""

from __future__ import annotations

import json

from sensoryforge.evaluate.audit.run import run_audit
from sensoryforge.io.design import load_design
from sensoryforge.world.dataset import (
    build_dataset,
    load_dataset,
    write_dataset,
)
from sensoryforge.world.runner import run_dataset

from .conftest import DESIGN, WORLD
from .pipeline_runs import check_report, lines, run_world, throughput
from sensoryforge.evaluate.pipeline import dataset_mapping

DRAWS = 24
DURATION_MS = 80.0


def measured(root):
    """Every entry's audit values except the wall-clock ones."""
    rows = (root / "evaluate" / "audit_entries.jsonl").read_text().splitlines()
    out = []
    for row in map(json.loads, rows):
        row["values"] = [v for v in row["values"] if v["id"] != "core.run.seconds"]
        out.append(row)
    return out


def test_world_to_report_on_many_draws(tmp_path, capsys):
    doc = run_world(tmp_path, DRAWS, DURATION_MS, checks=True)
    check_report(doc, tmp_path, DRAWS, checks=True)
    assert doc["counts"]["compared_pairs"] > DRAWS  # several metrics per entry
    with capsys.disabled():
        print("\nsmall scale:", throughput(doc, DRAWS))


def test_a_resumed_run_changes_nothing(tmp_path):
    first = run_world(tmp_path, DRAWS, DURATION_MS)
    audit = (tmp_path / "evaluate" / "audit_entries.jsonl").read_bytes()
    again = run_world(tmp_path, DRAWS, DURATION_MS, resume=True)
    check_report(again, tmp_path, DRAWS, checks=False)
    assert again["batch"] == {"ok": 0, "failed": 0, "skipped": DRAWS}
    assert (tmp_path / "evaluate" / "audit_entries.jsonl").read_bytes() == audit
    assert first["inputs"]["dataset_id"] == again["inputs"]["dataset_id"]


def test_tasks_make_the_same_data_set_as_one_process(tmp_path):
    single = tmp_path / "single"
    run_world(single, DRAWS, DURATION_MS, packs=("core",))

    split = tmp_path / "split"
    spec = load_dataset(dataset_mapping(WORLD, DRAWS, 11, DURATION_MS))
    write_dataset(spec, build_dataset(spec), split)
    for index in range(3):
        summary = run_dataset(
            load_design(DESIGN), spec, split, tasks=3, task_index=index
        )
        assert summary["failed"] == 0
    run_audit(split, split / "evaluate", packs=["core"])

    assert lines(split / "evaluate" / "audit_entries.jsonl") == DRAWS
    assert sorted(p.name for p in single.glob("train")) == sorted(
        p.name for p in split.glob("train")
    )
    # Wall-clock seconds are the one value that differs between two runs.
    assert measured(single) == measured(split)
