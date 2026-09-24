"""The tactile recipes' calibrated gains meet P5, and never reach the voltage floor.

Ledger D-ea0f017: each tactile recipe population has its own ``input_gain``,
chosen by ``scripts/calibrate_recipe_gains.py``. This test runs each recipe at
its own preset gains through the four benchmark stimuli and applies the same
pass rules the calibration used, so a change that moves a recipe out of P5's
bands fails here.

It also guards ledger F-037: SensoryForge clamps the membrane voltage at
``v_floor`` (-120 mV Izhikevich, -130 mV AdEx) and pressure-simulation does
not, so the two could differ -- but only if a run reaches the floor. At the
calibrated gains the lowest voltage on these stimuli must stay at least 20 mV
above it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.presets import load_preset

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V_FLOOR_MV = {"Izhikevich": -120.0, "AdEx": -130.0}
MIN_MARGIN_MV = 20.0


@pytest.fixture(scope="module")
def calibration():
    # calibrate_recipe_gains imports tune_adex_populations by name.
    import sys

    sys.path.insert(0, str(_SCRIPTS))
    try:
        return _load("calibrate_recipe_gains")
    finally:
        sys.path.remove(str(_SCRIPTS))


@pytest.fixture(scope="module")
def drive(calibration):
    T = calibration.T
    base = SensoryForgeConfig.from_dict(load_preset("tactile_sa1_ra1"))
    _, frames_by, filtered_by = T.measure_drive_range(base, False, "cpu")
    # Responsive sets are a fraction of each population's own maximum drive,
    # so they do not depend on the gain the drive was measured at.
    return frames_by, filtered_by


@pytest.mark.parametrize("model", ["Izhikevich", "AdEx"])
def test_recipe_gains_meet_p5_and_stay_above_the_voltage_floor(
    calibration, drive, model
):
    T = calibration.T
    frames_by, filtered_by = drive
    config = SensoryForgeConfig.from_dict(load_preset(calibration.RECIPES[model]))
    gains = {p.name: p.input_gain for p in config.populations}
    assert len(set(gains.values())) == 2, "SA and RA must have their own gains"

    rows = []
    v_min = float("inf")
    dt_ms = config.simulation.dt_ms
    for name in T.STIMULI:
        windows = T.windows_for(name, False)
        results = SimulationEngine(config).run(
            frames_by[name].unsqueeze(0),
            return_intermediates=True,
            seed=config.simulation.seed,
        )
        for pop_name, pop_results in results.items():
            v_min = min(v_min, float(pop_results["voltages"].min()))
            neuron_type = T.population_neuron_type(config, pop_name).upper()
            window = windows["hold"] if neuron_type == "SA" else windows["onset"]
            mask = T.responsive_mask(filtered_by[name][pop_name], dt_ms, window)
            m = T.spike_metrics(
                pop_results["spikes"], dt_ms, windows, neuron_type, mask
            )
            rows.append(
                {
                    "gain": 0,
                    "stimulus": name,
                    "population": neuron_type,
                    "hold_is_scored": windows["hold_is_scored"],
                    "sa_rate_hz": m.get("mean_rate_hz"),
                    "isi_cv": m.get("isi_cv"),
                    "peak_per_neuron_hz": m["peak_per_neuron_hz"],
                    "hold_count": m["hold_count"],
                }
            )

    sa = {
        r["stimulus"]: round(r["sa_rate_hz"], 1)
        for r in rows
        if r["population"] == "SA"
    }
    ra = {
        r["stimulus"]: (r["peak_per_neuron_hz"], r["hold_count"])
        for r in rows
        if r["population"] == "RA"
    }
    assert calibration.sa_passes(rows, 0), f"SA rates (Hz) out of P5's band: {sa}"
    assert calibration.ra_passes(rows, 0), f"RA (peak Hz, hold spikes) fails P5: {ra}"
    assert v_min > V_FLOOR_MV[model] + MIN_MARGIN_MV, (
        f"lowest voltage {v_min:.1f} mV is within {MIN_MARGIN_MV} mV of the "
        f"{V_FLOOR_MV[model]} mV floor: the clamp pressure-simulation lacks "
        "is now reachable (ledger F-037)"
    )
