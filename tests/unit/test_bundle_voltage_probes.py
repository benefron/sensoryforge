"""Voltage probes in bundles (roadmap A6p, bundle schema 2.4.0): opt-in, exact.

The data set is the toy non-tactile one (A1b); a probe run and an ordinary run of it
share every seed, so everything but the probe is expected to be bit for bit equal.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from sensoryforge.config.probes import check_voltage_probes, resolve_voltage_probes
from sensoryforge.config.schema import PopulationConfig, SimulationConfig
from sensoryforge.evaluate.audit.reader import DatasetReader
from sensoryforge.evaluate.server.resim import resimulate, verify_probes
from sensoryforge.io.bundle import PROBE_SCHEMA_VERSION, SCHEMA_VERSION, load_bundle
from sensoryforge.world.runner import run_dataset
from tests.fixtures import toy_vision

DURATION = 40.0
#: A thresholdless DSL model: its readout is its state, and it returns no voltage.
ANALOG = {
    "equations": "dv/dt = (-(v - v_rest) + R*I) / tau_m",
    "parameters": {"v_rest": -65.0, "R": 1.0, "tau_m": 10.0},
    "state_vars": {"v": -65.0},
}


def _run(root: Path, *, per_population=None, run_wide=None) -> Path:
    """The toy data set with optional ``voltage_probes`` keys, built in *root*."""
    spec = toy_vision.toy_spec(2, DURATION)
    entries = toy_vision.build_dataset(spec)
    data = Path(root) / "data"
    toy_vision.write_dataset(spec, entries, data)
    config = toy_vision.toy_config()
    config.simulation.voltage_probes = run_wide
    for i, pop in enumerate(config.populations):
        pop.seed = 11 + i  # the toy wiring is otherwise drawn afresh on every run
        pop.voltage_probes = (per_population or {}).get(pop.name)
    summary = run_dataset(config, spec, data)
    assert summary["failed"] == 0, summary
    return data


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("plain"))


@pytest.fixture(scope="module")
def probed(tmp_path_factory):
    return _run(
        tmp_path_factory.mktemp("probed"),
        per_population={"luma": 2},
        run_wide=[1, 5],
    )


def _first_bundle(data: Path) -> Path:
    rows = [json.loads(x) for x in (data / "manifest.jsonl").read_text().splitlines()]
    return data / rows[0]["entry"]


def _h5(path: Path) -> dict:
    out = {}
    with h5py.File(path / "data.h5", "r") as f:
        f.visititems(
            lambda n, o: (
                out.__setitem__(n, o[...]) if isinstance(o, h5py.Dataset) else None
            )
        )
    return out


# --------------------------------------------------------------------- the key


def test_the_key_is_invisible_until_used_and_round_trips():
    assert "voltage_probes" not in PopulationConfig(name="p").to_dict()
    assert "voltage_probes" not in SimulationConfig().to_dict()
    for spec in (3, [0, 4], "all"):
        pop = PopulationConfig(name="p", voltage_probes=spec)
        again = PopulationConfig.from_dict(pop.to_dict())
        assert again.voltage_probes == spec
        assert (
            SimulationConfig.from_dict(
                SimulationConfig(voltage_probes=spec).to_dict()
            ).voltage_probes
            == spec
        )


@pytest.mark.parametrize(
    "bad", [0, -2, True, [], [1, 1], [-1], [1.5], "some", 2.5, {"n": 2}]
)
def test_a_bad_value_is_refused_where_it_is_set(bad):
    with pytest.raises(ValueError, match="voltage_probes"):
        check_voltage_probes(bad, where="x")
    with pytest.raises(ValueError, match="voltage_probes"):
        PopulationConfig(name="p", voltage_probes=bad)
    with pytest.raises(ValueError, match="voltage_probes"):
        SimulationConfig(voltage_probes=bad)


def test_a_count_picks_evenly_spaced_neurons_in_the_middle_of_their_share():
    assert resolve_voltage_probes(1, 16) == [8]
    assert resolve_voltage_probes(2, 16) == [4, 12]
    assert resolve_voltage_probes(4, 16) == [2, 6, 10, 14]
    assert resolve_voltage_probes(99, 5) == [0, 1, 2, 3, 4]
    assert resolve_voltage_probes("all", 3) == [0, 1, 2]
    assert resolve_voltage_probes([7, 2], 10) == [2, 7]
    with pytest.raises(ValueError, match="out of range"):
        resolve_voltage_probes([10], 10)
    for n in range(1, 40):
        for k in range(1, n + 1):
            ids = resolve_voltage_probes(k, n)
            assert len(set(ids)) == k and ids == sorted(ids) and 0 <= ids[0]


# ------------------------------------------------------------- the bundle writer


def test_a_run_without_the_key_writes_the_bundle_it_always_did(plain):
    bundle = _first_bundle(plain)
    cfg = json.loads((bundle / "config.json").read_text())
    assert cfg["schema_version"] == SCHEMA_VERSION == "2.3.0"
    assert all("voltage_probe" not in p for p in cfg["populations"])
    assert "voltage_probes" not in json.dumps(cfg["config"])
    assert not [k for k in _h5(bundle) if "voltage_probe" in k]


def test_probes_are_added_and_nothing_else_changes(plain, probed):
    a, b = _first_bundle(plain), _first_bundle(probed)
    cfg = json.loads((b / "config.json").read_text())
    assert cfg["schema_version"] == PROBE_SCHEMA_VERSION == "2.4.0"
    by_name = {p["name"]: p for p in cfg["populations"]}
    assert by_name["luma"]["voltage_probe"] == {
        "dataset": "voltage_probe",
        "ids": [4, 12],
        "unit": "mV",
        "sampling": "bin end",
    }
    # luma has 16 neurons and its own count; chroma has 32 and takes the run-wide ids
    assert by_name["chroma"]["voltage_probe"]["ids"] == [1, 5]
    plain_h5, probed_h5 = _h5(a), _h5(b)
    extra = set(probed_h5) - set(plain_h5)
    assert extra == {
        "populations/luma/voltage_probe",
        "populations/chroma/voltage_probe",
    }
    for key, value in plain_h5.items():
        if key == "meta/config_yaml":
            continue
        assert np.array_equal(value, probed_h5[key]), key


def test_the_stored_probe_is_the_engines_voltage_bit_for_bit(tmp_path):
    from sensoryforge.core.simulation_engine import SimulationEngine

    config = toy_vision.toy_config()
    config.populations[0].voltage_probes = 3
    engine = SimulationEngine(config)
    torch.manual_seed(1)
    frames = torch.rand(1, 30, 3, toy_vision.ROWS, toy_vision.COLS)
    results = engine.run(
        frames, return_intermediates=True, bundle_dir=tmp_path / "b", seed=7
    )
    ids = resolve_voltage_probes(3, results["luma"]["voltages"].shape[-1])
    bundle = load_bundle(tmp_path / "b")
    stored = bundle.populations["luma"]["voltage_probe"]
    assert stored.dtype == torch.float32 and stored.shape == (30, 3)
    assert torch.equal(stored, results["luma"]["voltages"][0][:, ids])
    with h5py.File(tmp_path / "b" / "data.h5", "r") as f:
        ds = f["populations/luma/voltage_probe"]
        assert list(ds.attrs["ids"]) == ids and ds.attrs["unit"] == "mV"
    assert "voltage_probe" not in bundle.populations["chroma"]


def test_all_keeps_every_neurons_voltage(tmp_path):
    from sensoryforge.core.simulation_engine import SimulationEngine

    config = toy_vision.toy_config()
    config.simulation.voltage_probes = "all"
    engine = SimulationEngine(config)
    frames = torch.rand(1, 20, 3, toy_vision.ROWS, toy_vision.COLS)
    results = engine.run(frames, bundle_dir=tmp_path / "b", seed=7)
    assert set(results["luma"]) == {"spikes"}  # the caller's return is unchanged
    stored = load_bundle(tmp_path / "b").populations
    assert stored["luma"]["voltage_probe"].shape == (20, 16)
    assert stored["chroma"]["voltage_probe"].shape == (20, 32)


def test_a_population_that_asks_for_a_voltage_it_cannot_have_is_refused(tmp_path):
    from sensoryforge.core.simulation_engine import SimulationEngine

    config = toy_vision.toy_config()
    config.populations[0].neuron_model = "dsl"
    config.populations[0].dsl_config = ANALOG
    config.populations[0].voltage_probes = 1
    frames = torch.rand(1, 10, 3, toy_vision.ROWS, toy_vision.COLS)
    with pytest.raises(ValueError, match="no membrane voltage"):
        SimulationEngine(config).run(frames, bundle_dir=tmp_path / "b")
    # the run-wide default merely skips a population that has none
    config.populations[0].voltage_probes = None
    config.simulation.voltage_probes = 1
    SimulationEngine(config).run(frames, bundle_dir=tmp_path / "c")
    cfg = json.loads((tmp_path / "c" / "config.json").read_text())
    by_name = {p["name"]: p for p in cfg["populations"]}
    assert "voltage_probe" not in by_name["luma"]
    assert by_name["chroma"]["voltage_probe"]["ids"] == [16]


# ------------------------------------------------------------------ the checks


def test_the_blind_audit_reads_the_same_with_or_without_probes(plain, probed):
    reader_a, reader_b = DatasetReader(plain), DatasetReader(probed)
    entry = reader_a.rows[0].entry
    view_a, view_b = reader_a.read_entry(entry), reader_b.read_entry(entry)
    assert np.array_equal(view_a.frames, view_b.frames)
    for name, pop in view_a.populations.items():
        other = view_b.populations[name]
        assert pop.readout == other.readout and set(pop.attrs) == set(other.attrs)
        for field in ("data", "drive", "filtered"):
            assert np.array_equal(getattr(pop, field), getattr(other, field))


def test_the_re_simulation_equals_the_stored_probes(probed):
    bundle = _first_bundle(probed)
    for name in ("luma", "chroma"):
        run = resimulate(bundle, name, "spikes")
        assert run.verification["matches"] is True
        probes = run.verification["probes"]
        assert probes["matches"] is True and probes["n_differing"] == 0
        assert probes["max_abs_difference"] == 0.0
        assert probes["ids"] == ([4, 12] if name == "luma" else [1, 5])


def test_a_bundle_without_probes_says_nothing_about_them(plain):
    run = resimulate(_first_bundle(plain), "luma", "spikes")
    assert "probes" not in run.verification


def test_a_changed_probe_is_reported_with_its_size(probed, tmp_path):
    import shutil

    bundle = tmp_path / "b"
    shutil.copytree(_first_bundle(probed), bundle)
    with h5py.File(bundle / "data.h5", "r+") as f:
        ds = f["populations/luma/voltage_probe"]
        values = ds[...]
        values[3, 1] += 0.25
        ds[...] = values
    probes = resimulate(bundle, "luma", "spikes").verification["probes"]
    assert probes["matches"] is False and probes["n_differing"] == 1
    assert probes["max_abs_difference"] == pytest.approx(0.25, abs=1e-4)


def test_verify_probes_handles_shapes_and_nan():
    v = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert verify_probes(v[:, [1, 3]], [1, 3], v)["matches"]
    wrong = verify_probes(v[:, [1]], [1, 3], v)
    assert wrong["matches"] is False and "shapes differ" in wrong["note"]
    nan = v.copy()
    nan[0, 1] = np.nan
    assert verify_probes(nan[:, [1]], [1], nan)["matches"]
    assert not verify_probes(v[:, [1]], [1], nan)["matches"]


def test_the_voltage_endpoint_reports_the_probe_comparison(probed, plain):
    from sensoryforge.evaluate.server import Api, DatasetService, Request

    def voltage(data):
        api = Api(DatasetService(data))
        entry = api.service.rows[0].entry
        target = f"/api/draw/voltage?entry={entry}&population=luma&index=4"
        response = api.handle(Request.from_target("GET", target))
        assert response.status == 200
        return json.loads(response.body)["verification"]

    check = voltage(probed)
    assert check["matches"] is True
    assert check["probes"]["matches"] is True and check["probes"]["ids"] == [4, 12]
    assert "probes" not in voltage(plain)
