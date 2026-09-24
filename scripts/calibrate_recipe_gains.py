"""Calibrate each tactile recipe population's ``input_gain`` against decision P5.

Ledger D-ea0f017: SensoryForge's tactile recipes give SA and RA their own
input gains, calibrated on the responsive-set rate against the P5 bands over
the four benchmark stimuli. One shared gain of 50 left the Izhikevich SA
baseline at 8.75 Hz (F-093) and AdEx RA silent on ``drifting_grating``
(F-092).

The measurement is ``scripts/tune_adex_populations.py``'s, reused unchanged:
the same four stimuli, the same windows, the same drive-derived responsive
set and the same per-afferent peak rate. Only ``input_gain`` varies. SA and
RA populations do not interact, so one run at gain ``g`` scores both.

Selection rules (stated here so a reader need not read the code):

    SA: the gain at which the geometric mean, over the four stimuli, of the
    responsive-set SA rate (the scored hold for ``ramp_gaussian``, the
    steady-drive interval for the three moving stimuli) equals the centre of
    P5's 20-100 Hz band on a log scale, sqrt(20 * 100) = 44.7 Hz. It is found
    by interpolating log(rate) against log(gain) between the two bracketing
    sweep points, then checked: every stimulus's rate must lie in 20-100 Hz
    and ``ramp_gaussian``'s ISI CV must be below 0.5.

    RA: a gain passes when (a) every stimulus's onset burst reaches a
    per-afferent peak rate (5 ms bins, responsive set) of 150-400 Hz and
    (b) ``ramp_gaussian``'s scored hold has 0 spikes. P5 says bursts reach
    "up to ~300 Hz"; 5 ms bins resolve rates only in 200 Hz steps, so
    "~300 Hz" is read as 200-400 Hz, with 150 Hz (the harness's existing
    pass bar) as the floor. The chosen gain is the geometric centre of the
    contiguous run of passing sweep gains.

Chosen gains are rounded to two significant figures and re-run to confirm
they pass. The P5 rates are the calibration target, so passing them is not
an independent validation of the model -- that is the TouchSim comparison's
job (ledger F-070).

Usage:
    python scripts/calibrate_recipe_gains.py                  # full sweep
    python scripts/calibrate_recipe_gains.py --out DIR --steps 28
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_adex_populations as T  # noqa: E402

from sensoryforge.config.schema import SensoryForgeConfig  # noqa: E402
from sensoryforge.presets import load_preset  # noqa: E402

RECIPES = {"Izhikevich": "tactile_sa1_ra1", "AdEx": "tactile_sa1_ra1_adex"}
SA_BAND_HZ = (20.0, 100.0)
SA_TARGET_HZ = math.sqrt(SA_BAND_HZ[0] * SA_BAND_HZ[1])
SA_MAX_ISI_CV = 0.5
RA_PEAK_BAND_HZ = (150.0, 400.0)


def sweep_gains(
    start: float = 30.0, ratio: float = 1.1, steps: int = 28
) -> List[float]:
    """Gains on a geometric grid: ``start * ratio**k`` for ``k < steps``."""
    return [round(start * ratio**k, 1) for k in range(steps)]


def score(
    config: SensoryForgeConfig,
    gain: float,
    frames_by: Dict[str, Any],
    filtered_by: Dict[str, Dict[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    """Run the four stimuli at ``gain`` and score every population.

    Args:
        config: A recipe config (its populations' gains are overridden for
            the run only).
        gain: ``input_gain`` applied to every population.
        frames_by: Rendered ``[T, H, W]`` stimulus per name.
        filtered_by: Drive ``[T, N]`` (mA) per stimulus and population, from
            which the responsive sets are derived (gain-independent: the set
            is a fraction of the population's own maximum).

    Returns:
        One row per (stimulus, population).
    """
    dt_ms = config.simulation.dt_ms
    rows = []
    for name in T.STIMULI:
        windows = T.windows_for(name, False)
        results = T.run_stimulus_engine(config, frames_by[name], gain, "cpu")
        for pop_name, pop_results in results.items():
            neuron_type = T.population_neuron_type(config, pop_name).upper()
            window = windows["hold"] if neuron_type == "SA" else windows["onset"]
            mask = T.responsive_mask(filtered_by[name][pop_name], dt_ms, window)
            m = T.spike_metrics(
                pop_results["spikes"], dt_ms, windows, neuron_type, mask
            )
            rows.append(
                {
                    "gain": gain,
                    "stimulus": name,
                    "population": neuron_type,
                    "hold_is_scored": windows["hold_is_scored"],
                    "responsive_n": m["responsive_n"],
                    "sa_rate_hz": m.get("mean_rate_hz"),
                    "isi_cv": m.get("isi_cv"),
                    "peak_per_neuron_hz": m["peak_per_neuron_hz"],
                    "hold_count": m["hold_count"],
                    "total_spikes": m["total_spikes"],
                }
            )
    return rows


def _rows(rows, population, gain=None, stimulus=None):
    return [
        r
        for r in rows
        if r["population"] == population
        and (gain is None or r["gain"] == gain)
        and (stimulus is None or r["stimulus"] == stimulus)
    ]


def sa_geomean(rows: List[Dict[str, Any]], gain: float) -> float:
    """Geometric mean over the four stimuli of the SA responsive-set rate (Hz)."""
    rates = [max(r["sa_rate_hz"], 1e-6) for r in _rows(rows, "SA", gain)]
    return float(np.exp(np.mean(np.log(rates))))


def choose_sa_gain(rows: List[Dict[str, Any]], gains: List[float]) -> Optional[float]:
    """Interpolated gain putting the SA geometric-mean rate at ``SA_TARGET_HZ``."""
    means = [sa_geomean(rows, g) for g in gains]
    for (g0, m0), (g1, m1) in zip(zip(gains, means), zip(gains[1:], means[1:])):
        if m0 <= SA_TARGET_HZ <= m1 and m1 > m0:
            frac = (math.log(SA_TARGET_HZ) - math.log(m0)) / (
                math.log(m1) - math.log(m0)
            )
            return math.exp(math.log(g0) + frac * (math.log(g1) - math.log(g0)))
    return None


def ra_passes(rows: List[Dict[str, Any]], gain: float) -> bool:
    """P5's RA criteria at ``gain`` (see the module docstring)."""
    lo, hi = RA_PEAK_BAND_HZ
    for r in _rows(rows, "RA", gain):
        if not lo <= r["peak_per_neuron_hz"] <= hi:
            return False
        if r["hold_is_scored"] and r["hold_count"] > 0:
            return False
    return True


def choose_ra_gain(
    rows: List[Dict[str, Any]], gains: List[float]
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """Geometric centre of the longest contiguous run of passing gains."""
    best: List[float] = []
    run: List[float] = []
    for g in gains:
        if ra_passes(rows, g):
            run.append(g)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    if not best:
        return None, None
    return math.sqrt(best[0] * best[-1]), (best[0], best[-1])


def two_sig(x: float) -> float:
    """Round to two significant figures."""
    return float(f"{x:.2g}")


def sa_passes(rows: List[Dict[str, Any]], gain: float) -> bool:
    """Every stimulus in the SA band, and the scored hold's ISI CV below 0.5."""
    lo, hi = SA_BAND_HZ
    for r in _rows(rows, "SA", gain):
        if not lo <= r["sa_rate_hz"] <= hi:
            return False
        if r["hold_is_scored"] and (
            r["isi_cv"] is None or r["isi_cv"] >= SA_MAX_ISI_CV
        ):
            return False
    return True


def calibrate(steps: int) -> Dict[str, Any]:
    """Sweep, choose and confirm the gains for both recipes."""
    gains = sweep_gains(steps=steps)
    configs = {
        model: SensoryForgeConfig.from_dict(load_preset(preset))
        for model, preset in RECIPES.items()
    }
    # Drive at the Izhikevich recipe's own gain; both recipes share grid,
    # receptive fields, filters and gain, so the responsive sets are shared.
    base = configs["Izhikevich"]
    _, frames_by, filtered_by = T.measure_drive_range(base, False, "cpu")
    out: Dict[str, Any] = {"gains": gains, "models": {}}
    for model, config in configs.items():
        rows: List[Dict[str, Any]] = []
        for g in gains:
            rows.extend(score(config, g, frames_by, filtered_by))
        sa = choose_sa_gain(rows, gains)
        ra, ra_interval = choose_ra_gain(rows, gains)
        chosen = {
            "SA": two_sig(sa) if sa is not None else None,
            "RA": two_sig(ra) if ra is not None else None,
        }
        confirm = []
        for pop, g in chosen.items():
            if g is None:
                continue
            confirm.extend(
                r
                for r in score(config, g, frames_by, filtered_by)
                if r["population"] == pop
            )
        out["models"][model] = {
            "preset": RECIPES[model],
            "rows": rows,
            "chosen": chosen,
            "sa_interpolated": sa,
            "ra_interval": ra_interval,
            "confirm": confirm,
            "sa_confirm_passes": chosen["SA"] is not None
            and sa_passes(confirm, chosen["SA"]),
            "ra_confirm_passes": chosen["RA"] is not None
            and ra_passes(confirm, chosen["RA"]),
        }
    return out


def write_report(result: Dict[str, Any], out_dir: Path) -> Path:
    """Write ``recipe_calibration.md`` and ``sweep.json`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(json.dumps(result, indent=1))
    stimuli = list(T.STIMULI)
    lines = [
        "# Recipe gain calibration (P5)",
        "",
        "Generated by `scripts/calibrate_recipe_gains.py` (ledger D-ea0f017). "
        "Selection rules are in the script's docstring. The scored hold starts "
        "30 ms after `ramp_gaussian`'s ramp ends.",
        "",
    ]
    for model, m in result["models"].items():
        lines += [
            f"## {model} (`{m['preset']}`)",
            "",
            f"- **SA gain {m['chosen']['SA']}** (interpolated "
            f"{m['sa_interpolated']:.1f}); confirmed in band: "
            f"{m['sa_confirm_passes']}",
            f"- **RA gain {m['chosen']['RA']}** (passing interval "
            f"{m['ra_interval']}); confirmed passing: {m['ra_confirm_passes']}",
            "",
            "At the chosen gains:",
            "",
            "| population | " + " | ".join(stimuli) + " |",
            "|---|" + "---|" * len(stimuli),
        ]
        for pop in ("SA", "RA"):
            cells = []
            for s in stimuli:
                r = _rows(m["confirm"], pop, stimulus=s)[0]
                if pop == "SA":
                    cv = "n/a" if r["isi_cv"] is None else f"{r['isi_cv']:.2f}"
                    cells.append(f"{r['sa_rate_hz']:.1f} Hz (CV {cv})")
                else:
                    cells.append(
                        f"peak {r['peak_per_neuron_hz']:.0f} Hz, "
                        f"hold {int(r['hold_count'])}"
                    )
            lines.append(f"| {pop} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Sweep (SA: geometric-mean rate over the four stimuli; RA: pass):",
            "",
            "| gain | SA geomean (Hz) | SA in band | RA passes |",
            "|---|---|---|---|",
        ]
        for g in result["gains"]:
            lines.append(
                f"| {g} | {sa_geomean(m['rows'], g):.1f} | "
                f"{sa_passes(m['rows'], g)} | {ra_passes(m['rows'], g)} |"
            )
        lines.append("")
    path = out_dir / "recipe_calibration.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/results/recipe_calibration")
    )
    parser.add_argument(
        "--steps", type=int, default=28, help="Sweep points (10% apart)."
    )
    args = parser.parse_args()
    t0 = time.time()
    result = calibrate(args.steps)
    path = write_report(result, args.out)
    for model, m in result["models"].items():
        print(
            model,
            m["chosen"],
            "SA ok",
            m["sa_confirm_passes"],
            "RA ok",
            m["ra_confirm_passes"],
        )
    print(f"Wrote {path} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
