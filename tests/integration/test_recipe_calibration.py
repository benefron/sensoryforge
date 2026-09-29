"""The tactile recipes' calibrated populations fire as their calibration intends.

Each tactile recipe population has its own ``input_gain``, chosen by
``scripts/calibrate_recipe_gains.py`` (ledger D-ea0f017, D-d9bd411,
D-8dde454). This test runs each recipe at its own preset gains through the
four benchmark stimuli:

* SA, Izhikevich recipe (rule ``p5_hold``): the held stimulus's SA rate lies
  in P5's 20-100 Hz band with an ISI CV below 0.5.
* SA, AdEx recipe (rule ``low_threshold``): SA fires on the held benchmark at
  10% of its amplitude, so weak pressures are not invisible to it. The drive
  is linear in pressure, so the rendered frames are scaled by 0.1.
* RA: silent during ``ramp_gaussian``'s static hold, from 30 ms after the ramp
  (P5's physiological requirement). RA's sensitivity is set by TouchSim and
  pinned by ``tests/validation/test_touchsim_comparison.py``.
* Voltage (D-ce22df3): the recipes' neurons have no voltage clamp by default.
  Without it, the voltage must stay in a physiological range: above -150 mV
  on every stimulus, and above -110 mV during the static hold, where no
  negative drive exists and only adaptation could pull it down.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from sensoryforge.config.defaults import resolve_neuron_params
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.presets import load_preset

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

V_MIN_MV = -150.0
V_MIN_STATIC_HOLD_MV = -110.0


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def calibration():
    # calibrate_recipe_gains imports tune_adex_populations and
    # compare_with_touchsim by name.
    added = [str(_SCRIPTS), str(_SCRIPTS / "validation")]
    sys.path[:0] = added
    try:
        return _load("calibrate_recipe_gains")
    finally:
        for path in added:
            sys.path.remove(path)


@pytest.fixture(scope="module")
def drive(calibration):
    T = calibration.T
    base = SensoryForgeConfig.from_dict(load_preset("tactile_sa1_ra1"))
    _, frames_by, filtered_by = T.measure_drive_range(base, False, "cpu")
    # Responsive sets are a fraction of each population's own maximum drive,
    # so they do not depend on the gain the drive was measured at.
    return frames_by, filtered_by


def _run(config, frames):
    return SimulationEngine(config).run(
        frames.unsqueeze(0), return_intermediates=True, seed=config.simulation.seed
    )


def _sa_hold_rate(calibration, config, frames, filtered_by):
    T = calibration.T
    windows = T.windows_for("ramp_gaussian", False)
    results = _run(config, frames)
    sa = next(p.name for p in config.populations if p.neuron_type == "SA")
    mask = T.responsive_mask(
        filtered_by["ramp_gaussian"][sa], config.simulation.dt_ms, windows["hold"]
    )
    m = T.spike_metrics(
        results[sa]["spikes"], config.simulation.dt_ms, windows, "SA", mask
    )
    return m["mean_rate_hz"]


@pytest.mark.parametrize("model", ["Izhikevich", "AdEx"])
def test_recipe_populations_fire_as_calibrated(calibration, drive, model):
    T = calibration.T
    frames_by, filtered_by = drive
    config = SensoryForgeConfig.from_dict(load_preset(calibration.RECIPES[model]))
    gains = {p.name: p.input_gain for p in config.populations}
    assert len(set(gains.values())) == 2, "SA and RA must have their own gains"
    for pop in config.populations:
        params = resolve_neuron_params(
            pop.neuron_model, pop.neuron_type, pop.model_params
        )
        assert params.get("v_floor") is None, f"{pop.name} clamps its voltage"

    rows = []
    v_min = float("inf")
    v_min_static_hold = float("inf")
    dt_ms = config.simulation.dt_ms
    for name in T.STIMULI:
        windows = T.windows_for(name, False)
        results = _run(config, frames_by[name])
        for pop_name, pop_results in results.items():
            voltages = pop_results["voltages"]
            v_min = min(v_min, float(voltages.min()))
            if windows["hold_is_scored"]:
                hold = T._bin_slice(dt_ms, windows["hold"], voltages.shape[1])
                v_min_static_hold = min(
                    v_min_static_hold, float(voltages[:, hold].min())
                )
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

    if calibration.SA_RULES[model] == "p5_hold":
        held = [r for r in rows if r["population"] == "SA" and r["hold_is_scored"]]
        assert calibration.sa_passes(rows, 0), f"SA fails P5 on the hold: {held}"
    else:
        weak = frames_by["ramp_gaussian"] * calibration.SA_THRESHOLD_FRACTION
        rate = _sa_hold_rate(calibration, config, weak, filtered_by)
        assert rate > 0.0, "SA is silent at 10% of the held benchmark pressure"

    ra_static_hold = [
        r["hold_count"] for r in rows if r["population"] == "RA" and r["hold_is_scored"]
    ]
    assert ra_static_hold == [0], f"RA fires during the static hold: {ra_static_hold}"
    assert v_min > V_MIN_MV, f"voltage falls to {v_min:.1f} mV without a clamp"
    assert v_min_static_hold > V_MIN_STATIC_HOLD_MV, (
        f"during the static hold the voltage reaches {v_min_static_hold:.1f} mV: "
        "adaptation is driving the neuron far below rest"
    )
