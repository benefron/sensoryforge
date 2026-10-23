"""Helpers the scale tests share: run the pipeline on the tactile test world."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

from sensoryforge.evaluate.pipeline import (
    PIPELINE_FILE,
    REPORT_DIR,
    STAGES,
    dataset_mapping,
    run_pipeline,
)
from sensoryforge.evaluate.stage_io import file_sha256
from sensoryforge.io.design import load_design
from sensoryforge.world.dataset import load_dataset

from .conftest import DESIGN, WORLD


def run_world(
    out: Path,
    draws: int,
    duration_ms: float = 100.0,
    seed: int = 11,
    packs: Sequence[str] = ("core", "tactile"),
    checks: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    """Take the tactile test world to a report folder; return ``pipeline.json``."""
    spec = load_dataset(dataset_mapping(WORLD, draws, seed, duration_ms))
    return run_pipeline(
        load_design(DESIGN),
        spec,
        out,
        packs=list(packs),
        run_check_stage=checks,
        check_sample=2,
        resume=resume,
    )


def lines(path: Path) -> int:
    """The number of non-empty lines of a text file."""
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def check_report(doc: Dict[str, Any], out: Path, draws: int, checks: bool) -> None:
    """The invariants every run keeps at any size, whatever the draws are."""
    report = out / REPORT_DIR
    assert doc["status"] == "ok", doc["error"]
    assert [s["name"] for s in doc["stages"]] == list(STAGES)
    expected_checks = "ok" if checks else "skipped"
    assert [s["status"] for s in doc["stages"]] == ["ok"] * 5 + [expected_checks]
    assert doc["counts"]["entries"] == draws and doc["counts"]["audited"] == draws
    assert (
        doc["batch"]["failed"] == 0
        and doc["batch"]["ok"] + doc["batch"]["skipped"] == draws
    )

    ids = [
        json.loads(x)["entry"]
        for x in (out / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(ids) == draws and len(set(ids)) == draws
    assert all((out / i / "data.h5").is_file() for i in ids)
    for rows in ("audit_entries", "declared_entries", "compare_entries"):
        assert lines(report / f"{rows}.jsonl") == draws, rows
    assert json.loads((report / PIPELINE_FILE).read_text()) == doc
    for name, sha in doc["files"].items():
        assert file_sha256(report / name) == sha, name
    if checks:
        assert doc["counts"]["checks"]["fail"] == 0
        assert doc["counts"]["checks"]["error"] == 0


def throughput(doc: Dict[str, Any], draws: int) -> str:
    """One line: seconds per stage and draws per second of the batch stage."""
    seconds = {s["name"]: s["seconds"] for s in doc["stages"]}
    rate = draws / seconds["batch"] if seconds["batch"] else float("inf")
    stages = ", ".join(f"{k} {v:.1f} s" for k, v in seconds.items())
    return f"{draws} draws: {stages}; batch {rate:.1f} draws/s"
