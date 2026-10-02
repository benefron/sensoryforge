"""The event encoders through the engine, the bundle and ``load_design``.

* A ``level_crossing`` RA population (``filter_method: none``) returns its
  signed ON/OFF counts under ``"events"``, never ``"spikes"``; its drive is not
  floored at 0 mA, so a falling edge produces OFF events; the per-bin events
  times theta reconstruct the gained drive to within theta.
* A ``sigma_delta`` SA population returns ordinary counts under ``"spikes"``.
* The bundle (schema 2.1.0 and later) stores the signed counts in an ``events`` dataset
  (int16, ``signed`` attribute) and records model, params and readout in
  ``config.json``; a spike reader looking for ``spikes`` finds none.
* ``load_design`` accepts both models with their params.
"""

import json
import shutil
from pathlib import Path

import pytest
import torch

from sensoryforge.config.defaults import resolve_input_floor
from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.io.bundle import SCHEMA_VERSION, load_bundle, write_bundle
from sensoryforge.io.design import load_design

h5py = pytest.importorskip("h5py")

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "design_8x8"
THETA = 0.2
GAIN = 5.0


def _config():
    return SensoryForgeConfig(
        grids=[GridConfig(name="Main", arrangement="grid", rows=6, cols=6, spacing=0.2)],
        populations=[
            PopulationConfig(
                name="RA events",
                neuron_type="RA",
                neuron_model="level_crossing",
                filter_method="none",
                innervation_method="gaussian",
                neurons_per_row=2,
                input_gain=GAIN,
                model_params={"theta": THETA},
                seed=3,
            ),
            PopulationConfig(
                name="SA sigma-delta",
                neuron_type="SA",
                neuron_model="sigma_delta",
                filter_method="none",
                innervation_method="gaussian",
                neurons_per_row=2,
                input_gain=GAIN,
                model_params={"theta": 5.0},
                seed=4,
            ),
        ],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0, integrate_dt_ms=0.1),
    )


def _stimulus(steps: int = 120) -> torch.Tensor:
    """A uniform press: ramp up 0..40 ms, hold, ramp down 80..120 ms."""
    t = torch.arange(steps, dtype=torch.float32)
    env = torch.clamp(t / 40.0, max=1.0) * torch.clamp((steps - t) / 40.0, max=1.0)
    return env.view(1, steps, 1, 1).expand(1, steps, 6, 6).contiguous()


@pytest.fixture(scope="module")
def run():
    config = _config()
    engine = SimulationEngine(config)
    stimulus = _stimulus()
    results = engine.run(stimulus, return_intermediates=True)
    return config, engine, stimulus, results


def test_level_crossing_input_is_not_floored():
    assert resolve_input_floor("RA", "level_crossing", None) is None
    assert resolve_input_floor("SA", "sigma_delta", None) == 0.0
    assert resolve_input_floor("RA", "adex", None) == 0.0


def test_level_crossing_population_returns_signed_events(run):
    _, _, _, results = run
    ra = results["RA events"]
    assert "events" in ra and "spikes" not in ra
    ev = ra["events"][0]
    assert float((ev > 0).sum()) > 0, "ramp up must give ON events"
    assert float((ev < 0).sum()) > 0, "ramp down must give OFF events"
    # Hold (bins 41..79): the drive is constant, so no events.
    assert float(ev[41:80].abs().sum()) == 0
    # The signed sum times theta tracks the gained drive within theta.
    recon = torch.cumsum(ev.double(), 0) * THETA
    filtered = ra["filtered"][0].double()
    assert float((filtered - recon).abs().max()) < THETA * (1 + 1e-4)


def test_sigma_delta_population_returns_spike_counts(run):
    _, _, _, results = run
    sa = results["SA sigma-delta"]
    assert "spikes" in sa and "events" not in sa
    assert float(sa["spikes"].min()) >= 0
    assert float(sa["spikes"].sum()) > 0


def test_public_results_keep_events_key_with_bundle(tmp_path):
    engine = SimulationEngine(_config())
    out = engine.run(_stimulus(), bundle_dir=tmp_path / "b")
    assert set(out["RA events"]) == {"events"}
    assert set(out["SA sigma-delta"]) == {"spikes"}


def test_bundle_stores_signed_events_separately(run, tmp_path):
    config, engine, stimulus, results = run
    bundle_dir = write_bundle(tmp_path / "bundle", config, engine, results, stimulus)

    cfg = json.loads((bundle_dir / "config.json").read_text())
    assert cfg["schema_version"] == SCHEMA_VERSION == "2.2.0"
    entries = {p["name"]: p for p in cfg["populations"]}
    ra, sa = entries["RA events"], entries["SA sigma-delta"]
    assert ra["readout"] == "events"
    assert ra["event_encoding"]["signed"] is True
    assert ra["encoder"]["model"] == "level_crossing"
    assert ra["encoder"]["params"]["theta"] == pytest.approx(THETA)
    assert ra["encoder"]["params"]["dt"] == pytest.approx(0.1)
    assert sa["readout"] == "spikes"
    assert sa["encoder"]["model"] == "sigma_delta"
    assert sa["encoder"]["params"]["theta"] == pytest.approx(5.0)
    assert "event_encoding" not in sa

    with h5py.File(bundle_dir / "data.h5", "r") as f:
        grp = f["populations"]["RA events"]
        assert "spikes" not in grp, "signed events must never sit under 'spikes'"
        ds = grp["events"]
        assert ds.dtype.name == "int16"
        assert bool(ds.attrs["signed"]) is True
        assert float(ds.attrs["theta"]) == pytest.approx(THETA)
        assert "spikes" in f["populations"]["SA sigma-delta"]

    loaded = load_bundle(bundle_dir)
    ev = loaded.populations["RA events"]["events"]
    assert torch.equal(ev.to(torch.int16), results["RA events"]["events"][0].to(torch.int16))
    assert int(ev.min()) < 0
    assert "spikes" not in loaded.populations["RA events"]


def _write_event_design(tmp_path: Path) -> Path:
    design_dir = tmp_path / "design"
    shutil.copytree(FIXTURE_DIR, design_dir)
    manifest = json.loads((design_dir / "design.json").read_text())
    for prec in manifest["populations"]:
        prec["filter_method"] = "none"
        prec["filter_params"] = {}
        if prec["name"] == "ra":
            prec["neuron_model"] = "level_crossing"
            prec["model_params"] = {"theta": 0.5, "refractory_ms": 1.0}
        else:
            prec["neuron_model"] = "sigma_delta"
            prec["model_params"] = {"theta": 20.0}
    (design_dir / "design.json").write_text(json.dumps(manifest))
    return design_dir


def test_load_design_accepts_event_encoders(tmp_path):
    config = load_design(_write_event_design(tmp_path))
    by_name = {p.name: p for p in config.populations}
    assert by_name["ra"].neuron_model == "level_crossing"
    assert by_name["ra"].neuron_type == "RA"
    assert by_name["ra"].model_params == {"theta": 0.5, "refractory_ms": 1.0}
    assert by_name["sa"].neuron_model == "sigma_delta"
    assert by_name["sa"].neuron_type == "SA"
    results = SimulationEngine(config).run(torch.rand(1, 20, 8, 8))
    assert "events" in results["ra"] and "spikes" in results["sa"]


def test_load_design_rejects_bad_event_params(tmp_path):
    design_dir = _write_event_design(tmp_path)
    manifest = json.loads((design_dir / "design.json").read_text())
    manifest["populations"][0]["model_params"] = {"thta": 0.5}
    (design_dir / "design.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model_params"):
        load_design(design_dir)


def test_gui_results_view_shows_events_as_activity(run, tmp_path):
    """The Results screen's panels plot ``spikes``; an events population must
    reach them as |events| (ON and OFF alike) and keep its signed counts."""
    pytest.importorskip("PyQt5")
    from sensoryforge.gui.screens.results_data import from_bundle

    config, engine, stimulus, results = run
    bundle_dir = write_bundle(tmp_path / "bundle", config, engine, results, stimulus)
    view = from_bundle(load_bundle(bundle_dir))
    pops = {p.name: p for p in view.populations}
    ra = pops["RA events"]
    assert not ra.is_analog and ra.n_neurons == ra.events.shape[-1]
    assert torch.equal(ra.spikes, ra.events.abs().float())
    assert int(ra.events.min()) < 0
    assert pops["SA sigma-delta"].events is None
