"""Helpers the scale tests share: run the pipeline on the tactile test world."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pytest
import yaml

from sensoryforge.evaluate.pipeline import (
    PIPELINE_FILE,
    REPORT_DIR,
    STAGES,
    dataset_mapping,
    run_pipeline,
)
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.evaluate.stage_io import file_sha256
from sensoryforge.io.design import load_design
from sensoryforge.presets import load_preset
from sensoryforge.world.dataset import build_dataset, load_dataset

from .conftest import DESIGN, WORLD


def run_world(
    out: Path,
    draws: int,
    duration_ms: float = 100.0,
    seed: int = 11,
    packs: Sequence[str] = ("core", "tactile"),
    checks: bool = False,
    resume: bool = False,
    guard: bool = False,
) -> Dict[str, Any]:
    """Take the tactile test world to a report folder; return ``pipeline.json``.

    ``guard`` measures one entry first and fails if the run would not fit (see
    :func:`guard_disk`); the large test sets it, the small tests do not.
    """
    spec = load_dataset(dataset_mapping(WORLD, draws, seed, duration_ms))
    if guard:
        per_entry = measure_entry(load_design(DESIGN), WORLD, seed, duration_ms, out)
        guard_disk(
            len(build_dataset(spec)),
            per_entry,
            out,
            "tactile_small on the 8x8 test design",
        )
    return run_pipeline(
        load_design(DESIGN),
        spec,
        out,
        packs=list(packs),
        run_check_stage=checks,
        check_sample=2,
        resume=resume,
    )


def matrix_mapping(
    world: Path, draws: int, seed: int, duration_ms: float, held_out: bool
) -> Dict[str, Any]:
    """A data set that uses every split kind, sized to about ``draws`` entries.

    A ``train`` split takes most of the budget; a stratified ``test`` split, two
    ``sessions`` (longer than a draw) and, for a world with held-out classes, a
    stratified ``held_out`` split and a ``probes`` split take the rest.
    """
    # A stratified split holds per_bin draws in each of `bins` bins of each class, so
    # per_bin comes from the class count to keep the split near a tenth of the budget.
    world_doc = yaml.safe_load(Path(world).read_text())["world"]
    classes = len(world_doc["classes"])
    extra = max(draws // 10, 2)
    per_bin = max(extra // (2 * classes), 1)
    splits: Dict[str, Any] = {
        "train": {"n": draws - 2 * extra},
        "test": {"stratified": {"bins": 2, "per_bin": per_bin}},
        "sessions": {"n": 2, "duration_ms": 3 * duration_ms},
    }
    if held_out:
        splits["held_out"] = {"stratified": {"bins": 2, "per_bin": per_bin}}
        splits["probes"] = {"per_bin": 1}
    return {
        "dataset": {
            "name": f"{Path(world).stem}_matrix",
            "world": str(Path(world).resolve()),
            "seed": int(seed),
            "duration_ms": float(duration_ms),
            "splits": splits,
        }
    }


def dir_bytes(folder: Path) -> int:
    """The bytes of every file under ``folder``."""
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())


def guard_disk(entries: int, per_entry_bytes: int, out: Path, what: str) -> float:
    """Fail before simulating when ``entries`` bundles would not fit; return the GB.

    Two limits: ``SF_SCALE_MAX_GB`` (default 10) for what this case writes, and
    ``SF_SCALE_MIN_FREE_GB`` (default 20) that must stay free on the volume of ``out``.
    """
    need = entries * per_entry_bytes / 1e9
    cap = float(os.environ.get("SF_SCALE_MAX_GB", 10))
    keep = float(os.environ.get("SF_SCALE_MIN_FREE_GB", 20))
    probe = out
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free / 1e9
    line = (
        f"{what}: {entries} entries x {per_entry_bytes / 1e6:.1f} MB = {need:.1f} GB "
        f"(cap {cap:g} GB, {free:.0f} GB free at {probe}, keep {keep:g} GB free)"
    )
    if need > cap or need > free - keep:
        pytest.fail(
            f"{line}. Too large: lower SF_SCALE_DRAWS, raise SF_SCALE_MAX_GB, or point "
            "SF_SCALE_OUT at a bigger volume."
        )
    return need


def measure_entry(
    config: SensoryForgeConfig, world: Path, seed: int, duration_ms: float, out: Path
) -> int:
    """Bytes one entry takes: a one-draw data set is run, measured and removed."""
    out.parent.mkdir(parents=True, exist_ok=True)
    probe = Path(tempfile.mkdtemp(prefix="sf_probe_", dir=out.parent))
    try:
        one = load_dataset(dataset_mapping(world, 1, seed, duration_ms))
        run_pipeline(config, one, probe, packs=["core"], run_check_stage=False)
        return dir_bytes(probe) // len(build_dataset(one))
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def run_recipe(
    out: Path,
    world: Path,
    preset: str,
    draws: int,
    duration_ms: float = 100.0,
    seed: int = 11,
    held_out: bool = False,
    log=None,
) -> Dict[str, Any]:
    """Take a world to a report folder on a shipped tactile recipe (an 80x80 sensor)."""
    spec = load_dataset(matrix_mapping(world, draws, seed, duration_ms, held_out))
    entries = len(build_dataset(spec))
    per_entry = measure_entry(
        SensoryForgeConfig.from_dict(load_preset(preset)), world, seed, duration_ms, out
    )
    guard_disk(entries, per_entry, out, f"{world.stem} on {preset}")
    return run_pipeline(
        SensoryForgeConfig.from_dict(load_preset(preset)),
        spec,
        out,
        sensor_desc=f"--preset {preset}",
        packs=["core", "tactile"],
        check_sample=2,
        log=log,
    )


def lines(path: Path) -> int:
    """The number of non-empty lines of a text file."""
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def check_report(
    doc: Dict[str, Any], out: Path, draws: Optional[int], checks: bool
) -> None:
    """The invariants every run keeps at any size, whatever the draws are.

    ``draws`` is the expected entry count; ``None`` takes the data set's own count
    (a stratified or session split does not hold an exact number the caller knows).
    """
    report = out / REPORT_DIR
    if draws is None:
        draws = doc["counts"]["entries"]
        assert draws > 0
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
