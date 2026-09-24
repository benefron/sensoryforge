"""Fit the SA/RA filter and neuron parameters to TouchSim's SA1/RA afferents.

Ledger D-f4d0967 (SA matches SA1's ramp response and its graded rise with
indentation) and D-d9bd411 (RA matches RA's sensitivity). This script records
how the shipped values were found: grid searches over the parameters that
shape each response, scored against TouchSim's afferents at the probe centre
on the ramp-and-hold protocol of ``compare_with_touchsim.py``.

For every grid point, the amplitude per mm of indentation is refitted so SA's
hold rate matches SA1's at 1.25 mm; the point is then scored by the RMS of
``log((SF + 10 Hz) / (TouchSim + 10 Hz))`` over 0.1, 0.2, 0.4, 0.7 and 1.25 mm:

    SA stages: SA's onset (ramp) and hold rates against SA1's.
    RA stages: RA's onset and release rates against RA's (the amplitude per
        mm is held at the SA stage's value; RA's gain and adaptation vary).

The model is linear up to the neuron, so the amplitude per mm absorbs SA's
gain: these fits fix the response shapes and RA's gain relative to SA's. They
run at the SA gains the recipes had when the fit was made (Izhikevich 220,
AdEx 55), so an RA gain found here is read as a ratio to those. The absolute
gains are set afterwards by ``scripts/calibrate_recipe_gains.py``.

Runs use a 31 x 31 receptor grid (the probe is centred, so the grid edge is
irrelevant and a receptor sits at the centre); the recipes' 80 x 80 grid
gives the same rates to within one spike per window.

A point is rejected when the voltage during the hold comes within 20 mV of
the neuron's ``v_floor`` (-120 mV Izhikevich, -130 mV AdEx). In this AdEx
form the adaptation variable ``w`` enters dv/dt in mV, so a large ``b``
drives the voltage into the clamp after every spike, and the clamp, a
numerical guard, then shapes the dynamics. An earlier AdEx search without
this rule found b = 56-120 "optimal" for exactly that reason.

What was found (results: ``benchmarks/results/afferent_fit/``):

    * The SA filter's ``k2`` (gain on the input's rate of change) sets SA's
      ramp response; 3.0 gave about half of SA1's. 8.0 fits both neuron
      models (Izhikevich's own best is 10, AdEx's 5; 8 costs each about one
      spike at one depth and keeps one SA filter for both recipes and for
      pressure-simulation, which builds its filters from the same defaults).
    * Spike-frequency adaptation sets the graded rise of SA's hold rate:
      Izhikevich ``d`` 8 -> 15; AdEx ``SA1_tonic`` b 0 -> 28 with tau_w
      200 -> 110 ms and v_reset -58 -> -70 mV.
    * RA needs about 4-5x its P5 gain relative to SA to fire where TouchSim's
      RA fires, plus adaptation so its rate grows gradually with ramp speed:
      Izhikevich ``d`` 2 -> 24, AdEx ``RA1_phasic`` b 20 -> 40 with v_reset
      -58 -> -70 mV.

Usage:
    python scripts/validation/fit_afferents.py izhikevich-sa
    python scripts/validation/fit_afferents.py --list
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np

import compare_with_touchsim as C

OUT = C.REPO / "benchmarks" / "results" / "afferent_fit"
DEPTHS_MM = (0.1, 0.2, 0.4, 0.7, 1.25)
GRID_ROWS = 31

V_FLOOR_MV = {"tactile_sa1_ra1": -120.0, "tactile_sa1_ra1_adex": -130.0}
FLOOR_MARGIN_MV = 20.0

#: stage -> (recipe, fixed overrides, {"population.kind.param": values}, fit).
#: ``fit`` is "SA" or "RA"; RA stages hold the amplitude per mm at ``a_per_mm``.
STAGES: Dict[str, Dict[str, Any]] = {
    "izhikevich-sa": {
        "preset": "tactile_sa1_ra1",
        "fixed": {"SA.gain": 220},
        "grid": {"SA.filter.k2": [3, 6, 8, 10], "SA.model.d": [8, 12, 15, 18]},
        "fit": "SA",
    },
    "adex-sa": {
        "preset": "tactile_sa1_ra1_adex",
        "fixed": {"SA.gain": 55, "SA.filter.k2": 8, "SA.model.v_reset": -70},
        "grid": {
            "SA.model.b": [16, 24, 28, 32],
            "SA.model.tau_w": [90, 110, 125, 150],
        },
        "fit": "SA",
    },
    "izhikevich-ra": {
        "preset": "tactile_sa1_ra1",
        "fixed": {"SA.gain": 220, "SA.filter.k2": 8, "SA.model.d": 15},
        "grid": {"RA.gain": [120, 200, 240, 280], "RA.model.d": [2, 16, 24, 32]},
        "fit": "RA",
    },
    "adex-ra": {
        "preset": "tactile_sa1_ra1_adex",
        "fixed": {
            "SA.gain": 55,
            "SA.filter.k2": 8,
            "SA.model.b": 28,
            "SA.model.tau_w": 110,
            "SA.model.v_reset": -70,
            "RA.model.v_reset": -70,
        },
        "grid": {"RA.gain": [50, 60, 70, 80, 110], "RA.model.b": [20, 30, 40]},
        "fit": "RA",
    },
}


def _apply(config, overrides: Dict[str, float]) -> None:
    for key, value in overrides.items():
        ntype, kind, *name = key.split(".")
        for pop in config.populations:
            if pop.neuron_type != ntype:
                continue
            if kind == "gain":
                pop.input_gain = float(value)
            elif kind == "filter":
                pop.filter_params = {**pop.filter_params, name[0]: float(value)}
            elif kind == "model":
                pop.model_params = {**pop.model_params, name[0]: float(value)}


def rates(preset: str, overrides: Dict[str, float], amplitude: float):
    """Onset/hold/release rates (Hz) of each population's most-driven neuron."""
    from sensoryforge.core.simulation_engine import SimulationEngine
    from sensoryforge.stimuli.render import render_for_config

    windows = C.load_touchsim()["windows_ms"]
    config = C._config(preset, amplitude, windows)
    config.grids[0].rows = config.grids[0].cols = GRID_ROWS
    _apply(config, overrides)
    dt_ms = config.simulation.dt_ms
    frames = render_for_config(config, duration_ms=C.RUN_MS, dt_ms=dt_ms)[0]
    results = SimulationEngine(config).run(
        frames, return_intermediates=True, seed=config.simulation.seed
    )

    def sl(w):
        return slice(int(round(w[0] / dt_ms)), int(round(w[1] / dt_ms)))

    out = {}
    for pop in config.populations:
        spikes = results[pop.name]["spikes"][0].numpy()
        drive = results[pop.name]["filtered"][0].numpy()
        scoring = windows["sustained"] if pop.neuron_type == "SA" else windows["onset"]
        neuron = int(np.argmax(drive[sl(scoring)].mean(axis=0)))
        out[pop.neuron_type] = {
            k: float(spikes[sl(w), neuron].sum()) / ((w[1] - w[0]) / 1000.0)
            for k, w in windows.items()
        }
        hold = results[pop.name]["voltages"][0][sl(windows["sustained"])]
        out[pop.neuron_type]["v_min_hold"] = float(hold.min())
    return out


def fit_amplitude(preset: str, overrides: Dict[str, float]) -> float:
    """Amplitude per mm matching SA's hold rate to SA1's at 1.25 mm."""
    target = C.load_touchsim()["rates"]["SA1"][1.25]["sustained"]
    lo, hi = 0.01, 20.0
    for _ in range(12):
        mid = (lo * hi) ** 0.5
        if rates(preset, overrides, mid * 1.25)["SA"]["sustained"] < target:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def score_point(
    preset: str, overrides: Dict[str, float], fit: str, a_per_mm: Optional[float]
):
    """Score one grid point; returns its rates per depth and its error."""
    touchsim = C.load_touchsim()["rates"]
    a = a_per_mm if a_per_mm is not None else fit_amplitude(preset, overrides)
    features = (
        [("SA", "onset", "SA1"), ("SA", "sustained", "SA1")]
        if fit == "SA"
        else [("RA", "onset", "RA"), ("RA", "offset", "RA")]
    )
    per_depth, errors = {}, []
    for depth in DEPTHS_MM:
        r = rates(preset, overrides, a * depth)
        per_depth[depth] = r
        for pop, key, ts_pop in features:
            theirs = touchsim[ts_pop][depth][key]
            errors.append(math.log((r[pop][key] + 10.0) / (theirs + 10.0)) ** 2)
    v_min_hold = min(r[p]["v_min_hold"] for r in per_depth.values() for p in r)
    return {
        "overrides": overrides,
        "amplitude_per_mm": a,
        "error": math.sqrt(sum(errors) / len(errors)),
        "v_min_hold": v_min_hold,
        "floor_safe": v_min_hold > V_FLOOR_MV[preset] + FLOOR_MARGIN_MV,
        "rates": {str(d): v for d, v in per_depth.items()},
    }


def run_stage(name: str, workers: int, a_per_mm: Optional[float] = None) -> List[dict]:
    stage = STAGES[name]
    keys = list(stage["grid"])
    points = [
        {**stage["fixed"], **dict(zip(keys, values))}
        for values in itertools.product(*(stage["grid"][k] for k in keys))
    ]
    if stage["fit"] == "RA" and a_per_mm is None:
        a_per_mm = fit_amplitude(stage["preset"], stage["fixed"])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(score_point, stage["preset"], p, stage["fit"], a_per_mm)
            for p in points
        ]
        results = [f.result() for f in futures]
    # Floor-safe points first, then by error.
    results.sort(key=lambda r: (not r["floor_safe"], r["error"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(results, indent=1))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=sorted(STAGES))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list or not args.stage:
        for name, stage in STAGES.items():
            print(name, stage["grid"])
        return
    results = run_stage(args.stage, args.workers)
    for r in results[:5]:
        safe = "" if r["floor_safe"] else "  (rejected: reaches the floor)"
        print(f"{r['error']:.3f}", r["overrides"], safe)


if __name__ == "__main__":
    main()
