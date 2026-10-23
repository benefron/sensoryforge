"""Membrane voltage by re-simulation and population geometry (roadmap A6b).

The data set is the toy non-tactile one (A1b).
"""

from __future__ import annotations

import json
import shutil

import h5py
import numpy as np
import pytest
import torch

from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.evaluate.server import Api, DatasetService, Request
from sensoryforge.evaluate.server.resim import verify
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch


def get(api, target):
    response = api.handle(Request.from_target("GET", target))
    return response.status, json.loads(response.body)


@pytest.fixture(scope="module")
def api(toy_dataset):
    return Api(DatasetService(toy_dataset.data))


@pytest.fixture()
def damaged(toy_dataset, tmp_path):
    dest = tmp_path / "data"
    shutil.copytree(toy_dataset.data, dest, ignore=shutil.ignore_patterns(".partial"))
    return dest


def _first(api):
    entry = api.service.rows[0].entry
    _, body = get(api, f"/api/draw?entry={entry}")
    return entry, body["bundle"]["populations"][0]


# ------------------------------------------------------------ the engine's stage


@pytest.mark.parametrize("noisy", [False, True])
def test_the_neuron_stage_is_what_a_population_run_does(noisy):
    """Running the stage on a run's ``filtered`` reproduces its readout and voltage."""
    torch.manual_seed(0)
    drive = torch.randn(1, 40, 6) * 3.0
    kwargs = dict(dt_ms=1.0, integrate_dt_ms=0.05, input_floor=0.0)

    def neuron():
        return IzhikevichNeuronTorch(dt=0.05, noise_std=2.0 if noisy else 0.0)

    def generator():
        return torch.Generator(device="cpu").manual_seed(11) if noisy else None

    run = SimulationEngine._run_pop_from_drive(
        drive,
        None,
        neuron(),
        input_gain=2.0,
        noise_std=0.5 if noisy else 0.0,
        return_intermediates=True,
        noise_generator=generator(),
        **kwargs,
    )
    again = SimulationEngine._run_neuron_stage(
        run["filtered"], neuron(), noise_generator=generator(), **kwargs
    )
    assert torch.equal(again["spikes"], run["spikes"])
    assert torch.equal(again["voltages"], run["voltages"])


# ------------------------------------------------------------ verification rules


def test_verify_compares_neuron_by_neuron():
    stored = np.zeros((5, 3), dtype=np.int16)
    same = verify(stored, stored.astype(np.float32), "spikes")
    assert same["matches"] and same["n_differing"] == 0
    other = stored.copy()
    other[2, 1] = 1
    other[4, 1] = 1
    other[0, 2] = 3
    bad = verify(stored, other.astype(np.float32), "spikes")
    assert not bad["matches"] and bad["n_differing"] == 2
    assert bad["first_differences"] == [1, 2] and bad["max_abs_difference"] == 3.0
    state = np.ones((4, 2), dtype=np.float32)
    assert verify(state, state.copy(), "state")["matches"]
    assert not verify(state, state + 1e-7, "state")["matches"]
    assert not verify(state, state[:, :1], "state")["matches"]


# --------------------------------------------------------------- the endpoint


def test_the_re_run_matches_the_stored_run_and_is_cached(api, toy_dataset):
    entry, pop = _first(api)
    status, body = get(
        api, f"/api/draw/voltage?entry={entry}&population={pop['name']}&index=2,0"
    )
    assert status == 200 and body["cached"] is False
    check = body["verification"]
    assert check["matches"] is True and check["n_differing"] == 0
    assert check["n_units"] == pop["n_units"] and check["readout"] == "spikes"
    assert check["bundle_sha"] and check["code_sha"]
    assert [s["index"] for s in body["series"]] == [0, 2]
    n = body["n_frames"]
    assert all(len(s["values"]) == n for s in body["series"])
    values = np.array(body["series"][0]["values"])
    assert np.isfinite(values).all() and values.min() < values.max()
    # a spike is a bin whose sub-steps crossed threshold; the voltage there is reset
    with h5py.File(toy_dataset.data / entry / "data.h5", "r") as f:
        spikes = f[f"populations/{pop['name']}/spikes"][...]
    assert spikes.sum() > 0, "the toy draw should make the neuron fire"
    again = get(
        api, f"/api/draw/voltage?entry={entry}&population={pop['name']}&index=1"
    )[1]
    assert again["cached"] is True and again["verification"] == check


def test_a_stored_readout_that_differs_is_reported_as_differing(damaged):
    service = DatasetService(damaged)
    api = Api(service)
    entry = service.rows[0].entry
    pop = service.info(entry).populations[0]
    with h5py.File(damaged / entry / "data.h5", "r+") as f:
        ds = f[f"populations/{pop[0]}/spikes"]
        data = ds[...]
        data[3, 1] += 1
        ds[...] = data
    status, body = get(
        api, f"/api/draw/voltage?entry={entry}&population={pop[0]}&index=0"
    )
    assert status == 200
    check = body["verification"]
    assert check["matches"] is False and check["n_differing"] == 1
    assert check["first_differences"] == [1]


def test_voltage_queries_that_cannot_be_answered_say_why(api):
    entry, pop = _first(api)
    base = f"/api/draw/voltage?entry={entry}"
    cases = {
        base: 404,  # no population
        f"{base}&population=nope": 404,
        f"{base}&population={pop['name']}&index=999999": 400,
        f"{base}&population={pop['name']}&index=x": 400,
        "/api/draw/voltage?population=a": 400,
    }
    for target, want in cases.items():
        status, body = get(api, target)
        assert status == want and body["error"], target


def test_the_voltage_cache_stays_within_its_limit(toy_dataset):
    service = DatasetService(toy_dataset.data, voltage_cache_bytes=1)
    api = Api(service)
    entry, pop = _first(api)
    target = f"/api/draw/voltage?entry={entry}&population={pop['name']}"
    assert get(api, target)[1]["cached"] is False
    assert get(api, target)[1]["cached"] is False  # too big to keep: re-run
    assert service.cache_stats() is not None
    with pytest.raises(ValueError):
        DatasetService(toy_dataset.data, voltage_cache_bytes=-1)


# ---------------------------------------------------------- population geometry


def test_a_population_reports_its_neuron_positions_and_one_receptive_field(
    api, toy_dataset
):
    entry, pop = _first(api)
    status, body = get(
        api, f"/api/draw/population?entry={entry}&population={pop['name']}&index=3"
    )
    assert status == 200
    centers = np.array(body["centers"])
    assert centers.shape == (pop["n_units"], 2) and body["n_units"] == pop["n_units"]
    field = body["field"]
    n_sensors = len(
        get(api, f"/api/draw?entry={entry}")[1]["bundle"]["layout"]["coords"]
    )
    assert field["index"] == 3
    assert len(field["inputs"]) == body["n_receptors"] // n_sensors
    for block in field["inputs"]:
        assert len(block["sensors"]) == len(block["weights"]) > 0
        assert min(block["weights"]) > 0 and max(block["sensors"]) < n_sensors
    status, none = get(
        api, f"/api/draw/population?entry={entry}&population={pop['name']}"
    )
    assert status == 200 and none["field"] is None
    assert (
        get(
            api,
            f"/api/draw/population?entry={entry}&population={pop['name']}&index=999",
        )[0]
        == 400
    )
    assert get(api, f"/api/draw/population?entry={entry}&population=nope")[0] == 404


def test_a_sparse_population_file_gives_the_same_receptive_field(damaged):
    """The server reads either layout of ``population_NN_*.pt`` (A11d)."""
    import torch

    from sensoryforge.core.rf_bank import encode_weights

    service = DatasetService(damaged)
    api = Api(service)
    entry, pop = _first(api)
    target = f"/api/draw/population?entry={entry}&population={pop['name']}&index=3"
    before = get(api, target)[1]
    bundle = service.reader().bundle_path(entry)
    config = json.loads((bundle / "config.json").read_text())
    path = bundle / next(
        p["tensors"] for p in config["populations"] if p["name"] == pop["name"]
    )
    raw = torch.load(path, weights_only=False)
    raw.update(encode_weights(raw.pop("innervation_weights"), True))
    torch.save(raw, path)
    after = get(Api(DatasetService(damaged)), target)
    assert after[0] == 200 and after[1] == before
