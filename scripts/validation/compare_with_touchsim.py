"""Compare the tactile recipes' SA/RA responses with TouchSim's SA1/RA afferents.

Ledger F-070 / D-4aafcdc. TouchSim (Saal et al. 2017) was run once on a
punctate ramp-and-hold probe and its output committed as
``tests/fixtures/reference/touchsim_ramp_hold.json``: a 0.5 mm-radius pin,
50 ms ramp, 450 ms hold, 50 ms release, seven indentation depths. This script
runs the same protocol through SensoryForge's tactile recipes and compares
the two.

The models do not share an intensity scale: TouchSim indents skin in mm,
SensoryForge draws a pressure-like stimulus in unit amplitude and has no skin
mechanics. So exactly one parameter is fitted, the amplitude that corresponds
to 1 mm of indentation, chosen so the SA hold rate of the most-driven SA
neuron matches TouchSim's SA1 afferent at the probe centre at the deepest
level (1.25 mm). Everything else is a comparison at the matched levels
``amplitude = amplitude_per_mm * depth``:

    windows (TouchSim's fixture):  onset [0, 50] ms (the ramp), hold
    [150, 500] ms, release [500, 550] ms.

    afferent compared: TouchSim's afferent at the probe centre; in
    SensoryForge the neuron with the largest mean drive in the scoring
    window (hold for SA, onset for RA).

Criteria, fixed before the comparison was run (a feature agrees when):

    * a rate TouchSim reports as 0 is 0 (RA during the hold; SA at release);
    * a rate or a ratio of rates is within a factor of 2 of TouchSim's, at
      every matched depth where TouchSim's value is positive.

The features: SA hold rate against depth (the rate-intensity curve, after
the one fitted point); SA onset rate; SA adaptation (hold / onset); RA onset
and release rates; RA release / onset. The receptive fields' spatial extent
is not compared: the recipes' receptive fields are designed for a declared
resolvable distance (D-020), not fitted to afferent receptive fields, so
their spatial spread differing from TouchSim's is by construction.

Usage:
    python scripts/validation/compare_with_touchsim.py
    python scripts/validation/compare_with_touchsim.py --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sensoryforge.config.schema import SensoryForgeConfig, StimulusConfig
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.presets import load_preset
from sensoryforge.stimuli.layered import default_layer
from sensoryforge.stimuli.render import render_for_config

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "reference" / "touchsim_ramp_hold.json"
RECIPES = {"Izhikevich": "tactile_sa1_ra1", "AdEx": "tactile_sa1_ra1_adex"}
#: Depths compared: those where TouchSim's SA1 at the probe centre fires in
#: the hold (the shallower levels are silent in both classes).
DEPTHS_MM = (0.2, 0.4, 0.7, 1.25)
FIT_DEPTH_MM = 1.25
FACTOR = 2.0
RUN_MS = 600.0


def load_touchsim() -> Dict[str, Any]:
    """TouchSim's rates (Hz) at the probe centre, per class and depth."""
    data = json.loads(FIXTURE.read_text())
    windows = data["windows_ms"]
    centre = {
        a["affclass"]: str(a["id"])
        for a in data["afferents"]
        if a["distance_mm"] == 0.0
    }
    rates: Dict[str, Dict[float, Dict[str, float]]] = {"SA1": {}, "RA": {}}
    for level in data["by_depth"]:
        for cls in ("SA1", "RA"):
            rates[cls][level["depth_mm"]] = level["rates_hz"][centre[cls]]
    return {"windows_ms": windows, "rates": rates, "provenance": data["provenance"]}


def _config(preset: str, amplitude: float, windows: Dict[str, List[float]]):
    config = SensoryForgeConfig.from_dict(load_preset(preset))
    layer = default_layer("disc")
    layer["shape"].update({"amplitude": amplitude, "diameter_mm": 1.0, "edge_mm": 0.05})
    ramp = windows["onset"][1] - windows["onset"][0]
    layer["timing"] = {
        "onset_ms": 0.0,
        "ramp_up_ms": ramp,
        "hold_ms": windows["offset"][0] - ramp,
        "ramp_down_ms": windows["offset"][1] - windows["offset"][0],
    }
    stimulus = StimulusConfig(type="layered")
    stimulus.layers = [layer]
    config.stimulus = stimulus
    return config


def sensoryforge_rates(
    preset: str, amplitude: float, windows: Dict[str, List[float]]
) -> Dict[str, Dict[str, float]]:
    """Onset/hold/release rates (Hz) of each population's most-driven neuron.

    Args:
        preset: Recipe preset name.
        amplitude: Probe amplitude (unit peak).
        windows: TouchSim's ``onset``/``sustained``/``offset`` windows in ms.

    Returns:
        ``{"SA": {...}, "RA": {...}}``, each with ``onset``, ``sustained`` and
        ``offset`` rates in Hz.
    """
    config = _config(preset, amplitude, windows)
    dt_ms = config.simulation.dt_ms
    frames = render_for_config(config, duration_ms=RUN_MS, dt_ms=dt_ms)[0]
    results = SimulationEngine(config).run(
        frames, return_intermediates=True, seed=config.simulation.seed
    )

    def sl(window):
        return slice(int(round(window[0] / dt_ms)), int(round(window[1] / dt_ms)))

    out = {}
    for pop in config.populations:
        spikes = results[pop.name]["spikes"][0].cpu().numpy()
        drive = results[pop.name]["filtered"][0].cpu().numpy()
        scoring = windows["sustained"] if pop.neuron_type == "SA" else windows["onset"]
        neuron = int(np.argmax(drive[sl(scoring)].mean(axis=0)))
        out[pop.neuron_type] = {
            name: float(spikes[sl(w), neuron].sum()) / ((w[1] - w[0]) / 1000.0)
            for name, w in windows.items()
        }
    return out


def fit_amplitude_per_mm(
    preset: str, target_hz: float, windows: Dict[str, List[float]]
) -> float:
    """Amplitude per mm at which SA's hold rate at FIT_DEPTH_MM is ``target_hz``.

    Bisection on a log scale; SA's hold rate increases monotonically with
    amplitude.
    """
    lo, hi = 0.01, 20.0
    for _ in range(14):
        mid = (lo * hi) ** 0.5
        rate = sensoryforge_rates(preset, mid * FIT_DEPTH_MM, windows)["SA"][
            "sustained"
        ]
        if rate < target_hz:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def _ratio(a: float, b: float) -> Optional[float]:
    return a / b if b > 0 else None


def features(rates: Dict[str, Dict[str, float]]) -> Dict[str, Optional[float]]:
    """The compared features from one depth's onset/hold/release rates."""
    sa = rates["SA"] if "SA" in rates else rates["SA1"]
    ra = rates["RA"]
    return {
        "SA hold rate": sa["sustained"],
        "SA onset rate": sa["onset"],
        "SA release rate": sa["offset"],
        "SA hold / onset": _ratio(sa["sustained"], sa["onset"]),
        "RA onset rate": ra["onset"],
        "RA hold rate": ra["sustained"],
        "RA release rate": ra["offset"],
        "RA release / onset": _ratio(ra["offset"], ra["onset"]),
    }


def agrees(ours: Optional[float], theirs: Optional[float]) -> Optional[bool]:
    """The criterion: zero where TouchSim is zero, else within FACTOR."""
    if theirs is None or ours is None:
        return None
    if theirs == 0.0:
        return ours == 0.0
    if ours <= 0.0:
        return False
    return 1.0 / FACTOR <= ours / theirs <= FACTOR


def compare(preset: str, touchsim: Dict[str, Any], amplitude_per_mm: float):
    """Feature table for one recipe at the matched depths."""
    windows = touchsim["windows_ms"]
    rows = []
    for depth in DEPTHS_MM:
        theirs = features({k: v[depth] for k, v in touchsim["rates"].items()})
        ours = features(sensoryforge_rates(preset, amplitude_per_mm * depth, windows))
        for name in theirs:
            rows.append(
                {
                    "depth_mm": depth,
                    "feature": name,
                    "touchsim": theirs[name],
                    "sensoryforge": ours[name],
                    "agrees": agrees(ours[name], theirs[name]),
                }
            )
    return rows


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Optional[bool]]:
    """Per feature: True if it agrees at every depth where it is defined."""
    out: Dict[str, Optional[bool]] = {}
    for row in rows:
        if row["agrees"] is None:
            continue
        out[row["feature"]] = out.get(row["feature"], True) and row["agrees"]
    return out


def run(out_dir: Path) -> Dict[str, Any]:
    touchsim = load_touchsim()
    target = touchsim["rates"]["SA1"][FIT_DEPTH_MM]["sustained"]
    result: Dict[str, Any] = {"fit_depth_mm": FIT_DEPTH_MM, "target_sa_hold_hz": target}
    for model, preset in RECIPES.items():
        a_per_mm = fit_amplitude_per_mm(preset, target, touchsim["windows_ms"])
        rows = compare(preset, touchsim, a_per_mm)
        result[model] = {
            "preset": preset,
            "amplitude_per_mm": a_per_mm,
            "rows": rows,
            "summary": summarise(rows),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=1))
    (out_dir / "comparison.md").write_text(report(result, touchsim))
    return result


def _fmt(v):
    return "n/a" if v is None else f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"


def report(result: Dict[str, Any], touchsim: Dict[str, Any]) -> str:
    prov = touchsim["provenance"]
    lines = [
        "# SensoryForge against TouchSim: ramp-and-hold probe",
        "",
        "Generated by `scripts/validation/compare_with_touchsim.py` (ledger F-070). "
        f"TouchSim {prov.get('touchsim_repo_url')} at "
        f"`{prov.get('touchsim_commit_sha', '')[:10]}`; protocol and criteria are "
        "in the script's docstring. One parameter is fitted per recipe: the "
        f"amplitude per mm of indentation, matching SA's hold rate to TouchSim's "
        f"SA1 at {result['fit_depth_mm']} mm ({result['target_sa_hold_hz']:.1f} Hz). "
        f"A feature agrees when it is 0 where TouchSim's is 0, and otherwise within "
        f"a factor of {FACTOR:g} at every compared depth.",
        "",
    ]
    for model, preset in RECIPES.items():
        m = result[model]
        lines += [
            f"## {model} (`{preset}`): amplitude per mm = {m['amplitude_per_mm']:.3f}",
            "",
            "| feature | agrees | "
            + " | ".join(f"{d} mm (TouchSim / SF)" for d in DEPTHS_MM)
            + " |",
            "|---|---|" + "---|" * len(DEPTHS_MM),
        ]
        names = list(dict.fromkeys(r["feature"] for r in m["rows"]))
        for name in names:
            cells = []
            for d in DEPTHS_MM:
                r = next(
                    x for x in m["rows"] if x["feature"] == name and x["depth_mm"] == d
                )
                cells.append(f"{_fmt(r['touchsim'])} / {_fmt(r['sensoryforge'])}")
            verdict = m["summary"].get(name)
            label = "n/a" if verdict is None else "yes" if verdict else "**no**"
            lines.append(f"| {name} | {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "benchmarks" / "results" / "touchsim_comparison",
    )
    args = parser.parse_args()
    result = run(args.out)
    for model in RECIPES:
        m = result[model]
        print(model, f"amplitude/mm {m['amplitude_per_mm']:.3f}", m["summary"])


if __name__ == "__main__":
    main()
