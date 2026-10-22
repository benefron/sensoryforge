"""Fixtures the evaluation tests share: one small tactile data set, built once.

The data set is built through the engine from the tactile test world and run on the 8x8
test design; the audit and the declared side are written from it with both packs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
STAMP = "2026-10-22T14:03:11Z"


@pytest.fixture(scope="session")
def tactile_dataset(tmp_path_factory):
    """A small data set from the tactile test world, on the 8x8 test design."""
    from sensoryforge.io.design import load_design
    from sensoryforge.world.dataset import (
        build_dataset,
        load_dataset,
        write_dataset,
    )
    from sensoryforge.world.runner import run_dataset

    root = tmp_path_factory.mktemp("tactile_session")
    raw = {
        "dataset": {
            "name": "tiny",
            "world": str(FIXTURES / "worlds" / "tactile_small.yml"),
            "seed": 5,
            "duration_ms": 120,
            "splits": {
                "train": {"n": 4},
                "test": {"stratified": {"bins": 1, "per_bin": 1}},
                "sessions": {"n": 1, "duration_ms": 200},
            },
        }
    }
    path = root / "tiny.yml"
    path.write_text(yaml.safe_dump(raw))
    spec = load_dataset(path)
    entries = build_dataset(spec)
    data = root / "data"
    write_dataset(spec, entries, data)
    summary = run_dataset(load_design(FIXTURES / "design_8x8"), spec, data)
    assert summary["failed"] == 0, summary
    return data


@pytest.fixture(scope="session")
def audited(tactile_dataset, tmp_path_factory):
    """``(audit.json content, folder)`` of the audit with the core and tactile packs."""
    from sensoryforge.evaluate.audit.run import run_audit

    out = tmp_path_factory.mktemp("tactile_audit")
    audit = run_audit(tactile_dataset, out, packs=["core", "tactile"], created_at=STAMP)
    return audit, out
