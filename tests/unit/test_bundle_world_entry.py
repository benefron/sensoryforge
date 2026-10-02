"""Bundle schema 2.2.0: the sha in every bundle; world entries carry their record."""

import json
from pathlib import Path

import h5py
import torch

from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.io.bundle import SCHEMA_VERSION, load_bundle
from sensoryforge.io.design import load_design
from sensoryforge.provenance import source_info

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "design_8x8"


def _run(tmp_path, stimulus_config):
    engine = SimulationEngine(load_design(FIXTURE))
    bundle = tmp_path / "bundle"
    engine.run(
        torch.zeros(1, 10, 8, 8),
        bundle_dir=bundle,
        stimulus_config=stimulus_config,
        seed=1,
    )
    return bundle


def test_every_bundle_records_sensoryforge_sha(tmp_path):
    bundle = _run(tmp_path, {"type": "gaussian"})
    sha = source_info()["sha"]
    cfg = json.loads((bundle / "config.json").read_text())
    assert SCHEMA_VERSION == "2.2.0" and cfg["schema_version"] == "2.2.0"
    assert cfg["sensoryforge_sha"] == sha and "world" not in cfg
    with h5py.File(bundle / "data.h5", "r") as f:
        assert f.attrs["sensoryforge_sha"] == sha
    assert load_bundle(bundle).meta["sensoryforge_sha"] == sha


def test_a_world_entry_bundle_carries_its_record(tmp_path):
    entry = {
        "entry": "test/dots/0001",
        "world_id": "w-abc",
        "dataset_id": "d-def",
        "class": "dots",
    }
    layer = {"shape": {"kind": "gaussian"}}
    bundle = _run(
        tmp_path, {"kind": "sensoryforge_world_entry", "entry": entry, "layer": layer}
    )
    cfg = json.loads((bundle / "config.json").read_text())
    assert cfg["world"] == {
        "world_id": "w-abc",
        "dataset_id": "d-def",
        "entry": "test/dots/0001",
    }
    payload = json.loads((bundle / "stimuli" / "stimulus.json").read_text())
    assert payload["kind"] == "sensoryforge_world_entry"
    assert payload["schema_version"] == "2.2.0"
    assert payload["entry"] == entry and payload["layer"] == layer
    assert payload["n_frames"] == 10
    assert payload["reconstructible_by_pressure_simulation"] is False
