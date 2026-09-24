"""The tactile recipes' calibrated populations meet P5 and stay off the voltage floor.

Each tactile recipe population has its own ``input_gain``, chosen by
``scripts/calibrate_recipe_gains.py``: SA's against P5's rate band (ledger
D-ea0f017), RA's against TouchSim's RA afferent (D-d9bd411). This test runs
each recipe at its own preset gains through the four benchmark stimuli:

* SA must meet P5 (every stimulus in 20-100 Hz, and the hold's ISI CV below
  0.5), with the same rule the calibration used.
* RA must stay silent during ``ramp_gaussian``'s static hold, from 30 ms after
  the ramp. That is P5's physiological requirement for RA. RA's sensitivity
  (its onset rates) is now set by TouchSim, and
  ``tests/validation/test_touchsim_comparison.py`` pins it, so P5's
  150-400 Hz peak band is not asserted here.
* Izhikevich (ledger F-037): SensoryForge clamps the membrane voltage at
  ``v_floor`` and pressure-simulation's own Izhikevich neurons do not. With
  SA's SA1-like ramp response, the negative drive on a moving stimulus's
  trailing edge does reach the floor. So the test checks the thing F-037 is
  about: every population's spikes on every stimulus must be identical with
  the clamp and without it. pressure-simulation has no AdEx neurons of its
  own (its AdEx runs use SensoryForge), so this does not apply to AdEx.
* AdEx: in this AdEx form the adaptation variable enters dv/dt in mV, so a
  burst of spikes can drive the voltage into the clamp. During a static hold,
  where no negative drive exists, the voltage must stay at least 20 mV above
  -130 mV. At the calibrated gains it does not (the ledger's open entry on
  AdEx's voltage range), so that check is an expected failure until the
  model is fixed; it is strict, so it fails loudly once it passes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.presets import load_preset

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

V_FLOOR_MV = {"Izhikevich": -120.0, "AdEx": -130.0}
MIN_MARGIN_MV = 20.0


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


def _run(config, frames, v_floor=None):
    if v_floor is not None:
        for pop in config.populations:
            pop.model_params = {**pop.model_params, "v_floor": v_floor}
    return SimulationEngine(config).run(
        frames.unsqueeze(0), return_intermediates=True, seed=config.simulation.seed
    )


@pytest.mark.parametrize(
    "model",
    [
        "Izhikevich",
        pytest.param(
            "AdEx",
            marks=pytest.mark.xfail(
                strict=True,
                reason="AdEx adaptation drives the voltage into v_floor during "
                "the static hold at the calibrated gains (open ledger entry)",
            ),
        ),
    ],
)
def test_recipe_populations_meet_p5_and_stay_off_the_voltage_floor(
    calibration, drive, model
):
    T = calibration.T
    frames_by, filtered_by = drive
    config = SensoryForgeConfig.from_dict(load_preset(calibration.RECIPES[model]))
    gains = {p.name: p.input_gain for p in config.populations}
    assert len(set(gains.values())) == 2, "SA and RA must have their own gains"

    rows = []
    v_min_static_hold = float("inf")
    dt_ms = config.simulation.dt_ms
    for name in T.STIMULI:
        windows = T.windows_for(name, False)
        results = _run(config, frames_by[name])
        if model == "Izhikevich":
            unclamped = _run(
                SensoryForgeConfig.from_dict(load_preset(calibration.RECIPES[model])),
                frames_by[name],
                v_floor=-1e6,
            )
            for pop_name in results:
                assert torch.equal(
                    results[pop_name]["spikes"], unclamped[pop_name]["spikes"]
                ), (
                    f"{pop_name} on {name}: the voltage clamp changes the spikes, "
                    "so SensoryForge and pressure-simulation's unclamped "
                    "Izhikevich neurons now differ (ledger F-037)"
                )
        for pop_name, pop_results in results.items():
            voltages = pop_results["voltages"]
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

    sa = {
        r["stimulus"]: round(r["sa_rate_hz"], 1)
        for r in rows
        if r["population"] == "SA"
    }
    assert calibration.sa_passes(rows, 0), f"SA rates (Hz) out of P5's band: {sa}"
    ra_static_hold = [
        r["hold_count"] for r in rows if r["population"] == "RA" and r["hold_is_scored"]
    ]
    assert ra_static_hold == [0], f"RA fires during the static hold: {ra_static_hold}"

    floor = V_FLOOR_MV[model] + MIN_MARGIN_MV
    assert v_min_static_hold > floor, (
        f"during the static hold the voltage reaches {v_min_static_hold:.1f} mV, "
        f"within {MIN_MARGIN_MV} mV of the {V_FLOOR_MV[model]} mV floor: "
        "adaptation is driving the neuron into the clamp"
    )
