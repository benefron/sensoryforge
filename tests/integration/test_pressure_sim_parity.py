"""Golden parity test against pressure-simulation (task E5).

Compares SimulationEngine against a fixture exported from
pressure-simulation's own ``encoding/encode_runner.run_encoding`` (see
``scripts/dev/export_pressure_sim_golden.py`` and
``tests/fixtures/pressure_sim_golden/case_small.npz``): an 8x8 grid, T=200
record bins, dt_ms=1.0, one SA and one RA population with fixed (seeded)
innervation weights, a positive trapezoid stimulus, noise_std=0, and
explicit filter/model parameters matching both repos' resolved defaults
(SA tau_r=5, tau_d=30, k1=0.05, k2=3.0; RA tau_RA=8, k3=2.0; SA
regular-spiking, RA fast-spiking Izhikevich).

The comparison is staged so a mismatch is localized: drive (1e-6), then
the filtered-and-gained response (1e-6), then spikes as ``counts > 0``
(exact, since pressure-simulation's own spike output is already binary --
"any spike within a dt_ms bin").

Limitation (F-037, ledger D-007): SensoryForge clamps Izhikevich voltage
at v_floor=-120 mV; pressure-simulation's neuron has no such clamp. The
golden stimulus is kept strictly non-negative throughout so neither side's
membrane potential should approach that clamp, but this test does not
independently prove it never does -- a stronger negative-going stimulus
could diverge between the two repos for that reason alone, unrelated to
any bug in the sub-stepping/parity logic under test here.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "pressure_sim_golden"
    / "case_small.npz"
)


@pytest.fixture(scope="module")
def golden():
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"golden fixture not found at {FIXTURE_PATH}; run "
            "scripts/dev/export_pressure_sim_golden.py from the "
            "pressure-simulation repo root first"
        )
    return np.load(FIXTURE_PATH)


@pytest.fixture(scope="module")
def engine_result(golden):
    H = int(golden["H"])
    W = int(golden["W"])
    dt_ms = float(golden["dt_ms"])
    input_gain = float(golden["input_gain"])

    config = SensoryForgeConfig(
        grids=[
            GridConfig(name="grid", arrangement="grid", rows=H, cols=W, spacing=1.0)
        ],
        populations=[
            PopulationConfig(
                name="SA Pop",
                target_grid="grid",
                neuron_type="SA",
                neuron_model="izhikevich",
                filter_method="sa",
                filter_params={"tau_r": 5.0, "tau_d": 30.0, "k1": 0.05, "k2": 3.0},
                model_params={"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
                innervation_method="gaussian",
                neuron_rows=2,
                neuron_cols=2,
                input_gain=input_gain,
                noise_std=0.0,
            ),
            PopulationConfig(
                name="RA Pop",
                target_grid="grid",
                neuron_type="RA",
                neuron_model="izhikevich",
                filter_method="ra",
                filter_params={"tau_RA": 8.0, "k3": 2.0},
                model_params={"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0},
                innervation_method="gaussian",
                neuron_rows=2,
                neuron_cols=2,
                input_gain=input_gain,
                noise_std=0.0,
            ),
        ],
        stimulus=StimulusConfig(type="gaussian", amplitude=1.0, sigma=1.0),
        # integrate_dt_ms left at its default (0.05 ms), matching
        # pressure-simulation's hard-coded native Izhikevich step exactly.
        simulation=SimulationConfig(device="cpu", dt_ms=dt_ms),
    )

    engine = SimulationEngine(config)

    # Replace each population's innervation weights with the golden,
    # fixed-seed weights pressure-simulation used, so both sides drive
    # the same neurons with the same receptor-to-neuron mapping.
    sa_module = engine.populations[0]["innervation"]
    ra_module = engine.populations[1]["innervation"]
    assert engine.populations[0]["name"] == "SA Pop"
    assert engine.populations[1]["name"] == "RA Pop"
    sa_module.innervation_weights = torch.from_numpy(golden["sa_weights"]).clone()
    ra_module.innervation_weights = torch.from_numpy(golden["ra_weights"]).clone()

    stimulus = torch.from_numpy(golden["stimulus"])
    return engine.run(stimulus, return_intermediates=True)


def test_drive_matches(golden, engine_result):
    sa_drive = engine_result["SA Pop"]["drive"].numpy()
    ra_drive = engine_result["RA Pop"]["drive"].numpy()
    np.testing.assert_allclose(sa_drive, golden["sa_drive"], atol=1e-6)
    np.testing.assert_allclose(ra_drive, golden["ra_drive"], atol=1e-6)


def test_filtered_and_gained_response_matches(golden, engine_result):
    sa_filtered = engine_result["SA Pop"]["filtered"].numpy()
    ra_filtered = engine_result["RA Pop"]["filtered"].numpy()
    np.testing.assert_allclose(sa_filtered, golden["sa_filtered"], atol=1e-6)
    np.testing.assert_allclose(ra_filtered, golden["ra_filtered"], atol=1e-6)


def test_spikes_match_as_binary_raster(golden, engine_result):
    """pressure-simulation's own spike output is already binary ("any spike
    in this dt_ms bin"); SensoryForge returns sub-step counts (F-008), so
    the comparison is counts > 0 against that binary raster, exactly.
    """
    sa_counts = engine_result["SA Pop"]["spikes"].numpy()[0]  # [T, N]
    ra_counts = engine_result["RA Pop"]["spikes"].numpy()[0]

    sa_binary = (sa_counts > 0).astype(np.float32)
    ra_binary = (ra_counts > 0).astype(np.float32)

    np.testing.assert_array_equal(sa_binary, golden["sa_spikes"])
    np.testing.assert_array_equal(ra_binary, golden["ra_spikes"])
