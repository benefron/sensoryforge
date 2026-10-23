"""Fixtures the evaluation tests share: one small tactile data set, built once.

The data set is built through the engine from the tactile test world and run on the 8x8
test design; the audit and the declared side are written from it with both packs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
STAMP = "2026-10-22T14:03:11Z"


def build_tactile_data(root: Path, design: Path) -> Path:
    """Build and run the small tactile data set on a design; return its folder."""
    from sensoryforge.io.design import load_design
    from sensoryforge.world.dataset import (
        build_dataset,
        load_dataset,
        write_dataset,
    )
    from sensoryforge.world.runner import run_dataset

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
    summary = run_dataset(load_design(design), spec, data)
    assert summary["failed"] == 0, summary
    return data


@pytest.fixture(scope="session")
def tactile_dataset(tmp_path_factory):
    """A small data set from the tactile test world, on the 8x8 test design."""
    return build_tactile_data(
        tmp_path_factory.mktemp("tactile_session"), FIXTURES / "design_8x8"
    )


def converter_design(root: Path, refractory_ms: float) -> Path:
    """The 8x8 test design with ``level_crossing`` (ra) and ``sigma_delta`` (sa).

    Both converters run with ``filter_method: "none"``, as pressure-simulation runs
    them, and with the given refractory period.
    """
    design = root / "design"
    shutil.copytree(FIXTURES / "design_8x8", design)
    doc = json.loads((design / "design.json").read_text())
    models = {"ra": "level_crossing", "sa": "sigma_delta"}
    for pop in doc["populations"]:
        pop["neuron_model"] = models[pop["name"]]
        pop["model_params"] = {"theta": 0.5, "refractory_ms": refractory_ms}
        pop["filter_method"] = "none"
    (design / "design.json").write_text(json.dumps(doc))
    return design


@pytest.fixture(scope="session")
def converter_dataset(tmp_path_factory):
    """``make(refractory_ms)``: the small data set on the converter design (cached)."""
    built = {}

    def make(refractory_ms: float) -> Path:
        if refractory_ms not in built:
            root = tmp_path_factory.mktemp("converters")
            built[refractory_ms] = build_tactile_data(
                root, converter_design(root, refractory_ms)
            )
        return built[refractory_ms]

    return make


@pytest.fixture(scope="session")
def audited(tactile_dataset, tmp_path_factory):
    """``(audit.json content, folder)`` of the audit with the core and tactile packs."""
    from sensoryforge.evaluate.audit.run import run_audit

    out = tmp_path_factory.mktemp("tactile_audit")
    audit = run_audit(tactile_dataset, out, packs=["core", "tactile"], created_at=STAMP)
    return audit, out
