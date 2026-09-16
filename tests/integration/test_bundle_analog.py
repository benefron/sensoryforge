"""An analog population survives a bundle round trip (Waves J and N join).

Wave J (the bundle) and Wave N (analog readouts) were developed in parallel on
separate branches, so neither could test the other. This is the seam between
them: a population whose neuron model has no spike condition produces a
``"state"`` trace and no ``"spikes"`` key at all, and the bundle writer has to
store that instead of assuming spikes, with the reader giving it back.

A bundle that silently dropped an analog population, or stored its state under
the spikes dataset, would look perfectly healthy from either side alone.
"""

import pytest
import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.io.bundle import load_bundle, write_bundle

h5py = pytest.importorskip("h5py")

# A leaky integrator with no threshold and no reset: the definition of an
# analog readout (Wave N, N1).
LEAKY_INTEGRATOR = {
    "equations": "dv/dt = (-(v - v_rest) + R*I) / tau_m",
    "parameters": {"v_rest": -65.0, "R": 1.0, "tau_m": 10.0},
    "state_vars": {"v": -65.0},
}


def _config(with_spiking: bool):
    populations = [
        PopulationConfig(
            name="Analog",
            neuron_type="SA",
            neuron_model="dsl",
            filter_method="sa",
            innervation_method="gaussian",
            neurons_per_row=2,
            dsl_config=LEAKY_INTEGRATOR,
            seed=5,
        )
    ]
    if with_spiking:
        populations.append(
            PopulationConfig(
                name="Spiking",
                neuron_type="RA",
                neuron_model="izhikevich",
                filter_method="ra",
                innervation_method="gaussian",
                neurons_per_row=2,
                seed=6,
            )
        )
    return SensoryForgeConfig(
        grids=[
            GridConfig(name="Main", arrangement="grid", rows=8, cols=8, spacing=0.2)
        ],
        populations=populations,
        simulation=SimulationConfig(device="cpu", dt_ms=1.0, integrate_dt_ms=1.0),
    )


def _run(tmp_path, with_spiking):
    config = _config(with_spiking)
    engine = SimulationEngine(config)
    stimulus = torch.rand(1, 10, 8, 8)
    results = engine.run(stimulus, return_intermediates=True)
    bundle_dir = write_bundle(
        tmp_path / "bundle",
        config,
        engine,
        results,
        stimulus,
        stimulus_config={"type": "gaussian", "spread": 1.0},
    )
    return bundle_dir, results


def test_analog_population_produces_state_not_spikes(tmp_path):
    """The premise: without this, the rest of the test proves nothing."""
    _, results = _run(tmp_path, with_spiking=False)
    assert "state" in results["Analog"]
    assert "spikes" not in results["Analog"]


def test_analog_state_survives_the_bundle_round_trip(tmp_path):
    bundle_dir, results = _run(tmp_path, with_spiking=False)

    with h5py.File(bundle_dir / "data.h5", "r") as f:
        pop = f["populations"]["Analog"]
        assert "state" in pop, (
            "the bundle has no state dataset for an analog population; "
            f"it holds {sorted(pop.keys())}"
        )
        assert "spikes" not in pop, (
            "the bundle stored a spikes dataset for a population that never "
            "spiked; an analog state trace must not be written as spikes"
        )
        stored = torch.from_numpy(pop["state"][()])

    expected = results["Analog"]["state"][0]
    assert stored.shape == expected.shape
    assert torch.allclose(stored, expected.to(stored.dtype), atol=0, rtol=0)


def test_loader_returns_the_analog_state(tmp_path):
    bundle_dir, results = _run(tmp_path, with_spiking=False)
    loaded = load_bundle(bundle_dir)
    assert "Analog" in loaded.populations
    assert "state" in loaded.populations["Analog"]
    assert torch.allclose(
        loaded.populations["Analog"]["state"],
        results["Analog"]["state"][0].to(loaded.populations["Analog"]["state"].dtype),
    )


def test_mixed_analog_and_spiking_bundle(tmp_path):
    """Both readouts in one bundle, each stored under its own key."""
    bundle_dir, results = _run(tmp_path, with_spiking=True)

    with h5py.File(bundle_dir / "data.h5", "r") as f:
        pops = f["populations"]
        assert "state" in pops["Analog"] and "spikes" not in pops["Analog"]
        assert "spikes" in pops["Spiking"] and "state" not in pops["Spiking"]

    loaded = load_bundle(bundle_dir)
    assert set(loaded.populations) == {"Analog", "Spiking"}
    assert "state" in loaded.populations["Analog"]
    assert "spikes" in loaded.populations["Spiking"]


def test_analog_population_still_gets_a_receptive_field_file(tmp_path):
    """The readout kind must not change how receptive fields are exported."""
    bundle_dir, _ = _run(tmp_path, with_spiking=False)
    banks = sorted(bundle_dir.glob("population_*_Analog.pt"))
    assert len(banks) == 1, f"expected one bank file, found {[p.name for p in banks]}"
    data = torch.load(banks[0], weights_only=False)
    assert data["innervation_weights"].ndim == 2
    assert data["neuron_centers"].shape[1] == 2
    assert data["receptor_coords"].shape[0] == 8 * 8
