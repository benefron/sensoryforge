"""Tune and characterize AdEx SA1/RA1 populations against project decision P5.

Phase 2b, T2. This script (a) measures the actual filtered-drive current
range (mA) the ``tactile_sa1_ra1`` recipe delivers to its SA and RA
populations for the four benchmark stimuli, (b) computes steady-state
(and, for the phasic RA model, transient) f-I curves for four neuron
models -- AdEx ``SA1_tonic``, AdEx ``RA1_phasic``, Izhikevich ``RS``,
Izhikevich ``FS`` -- over a current range chosen from that measurement,
(c) runs the four benchmark stimuli (``ramp_gaussian``, ``moving_edge``,
``braille``, ``drifting_grating``, rendered exactly as
``examples/pressure_simulation_recipe.py``) through the engine twice each,
once on ``tactile_sa1_ra1`` (Izhikevich) and once on
``tactile_sa1_ra1_adex`` (AdEx), and (d) scores the result against
project decision P5 (physiology-anchored, engineering framing).

P5 criteria (verbatim, to be recorded -- not fitted to):

    SA-I: tonic (sustained) firing during a held stimulus; mean rate
    roughly 20-100 Hz over the amplitude range the four benchmark stimuli
    use; monotone in amplitude; ISI coefficient of variation < 0.5 during
    the hold.

    RA-I: silent during a steady hold (0 spikes after the onset
    transient, measured over a 200 ms hold window); transient bursts at
    onset and offset reaching instantaneous rates up to ~300 Hz; no
    spikes for a static stimulus after the first ~30 ms.

Window definitions (per stimulus; all times are ms from stimulus start,
t=0):

    ramp_gaussian (ramps to full amplitude over ``ramp_ms`` then HOLDS at
    full amplitude for the rest of the recorded run -- no ramp-down is
    ever rendered, so there is no recorded offset event):
        onset  = [0, 30]
        hold   = [ramp_ms, min(ramp_ms + 200, total_ms)]  (a genuine static
                 hold -- scored against P5)
        offset = N/A (never recorded)

    moving_edge (``ramp_up_ms`` rise, then the edge SWEEPS across the grid
    for ``plateau_ms`` at full envelope amplitude, then ``ramp_down_ms``
    fall to zero): the "plateau" is not a spatial hold (the edge keeps
    moving), so the SA/RA hold criteria are N/A here; the middle of the
    plateau is reported as an informational "steady-drive interval"
    instead.
        onset            = [0, 30]
        steady-drive (N/A for P5) = middle 200 ms of the plateau (or the
                 whole plateau if it is shorter than 200 ms)
        offset           = [down_start, min(down_start + 30, total_ms)],
                 down_start = ramp_up_ms + plateau_ms

    braille (a braille cell continuously SLIDES along y for the whole
    run, with a symmetric ramp in/out of ``ramp_ms``): never static, so
    hold is N/A; reported as a steady-drive interval.
        onset  = [0, 30]
        steady-drive (N/A) = middle 200 ms of [ramp_ms, total_ms - ramp_ms]
                 (or that whole interior interval if shorter than 200 ms)
        offset = [total_ms - 30, total_ms]

    drifting_grating (a grating continuously DRIFTS along x for the whole
    run, with a symmetric ramp in/out of ``ramp_ms``): never static, so
    hold is N/A; reported as a steady-drive interval.
        onset  = [0, 30]
        steady-drive (N/A) = middle 200 ms of [ramp_ms, total_ms - ramp_ms]
                 (or that whole interior interval if shorter than 200 ms)
        offset = [total_ms - 30, total_ms]

Total-spike and peak-instantaneous-population-rate (5 ms bins, stated
below) are reported for every stimulus regardless of hold/N-A status.

Usage:
    conda run -n sensoryforge python scripts/tune_adex_populations.py
    conda run -n sensoryforge python scripts/tune_adex_populations.py --quick
    conda run -n sensoryforge python scripts/tune_adex_populations.py \
        --out benchmarks/results/adex_tuning --seed 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sensoryforge.config.schema import SensoryForgeConfig  # noqa: E402
from sensoryforge.core.grid import ReceptorGrid  # noqa: E402
from sensoryforge.core.simulation_engine import SimulationEngine  # noqa: E402
from sensoryforge.neurons.adex import AdExNeuronTorch  # noqa: E402
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch  # noqa: E402
from sensoryforge.presets import load_preset  # noqa: E402
from sensoryforge.stimuli.render import render_stimulus  # noqa: E402

# name -> (registered stimulus type, quick-mode param overrides) -- same
# four stimuli, same quick-mode shortening, as examples/pressure_simulation_recipe.py.
STIMULI: Dict[str, Tuple[str, Dict[str, float]]] = {
    "ramp_gaussian": ("ramp_gaussian", {"total_ms": 60.0, "ramp_ms": 10.0}),
    "moving_edge": ("moving_edge", {"total_ms": 60.0, "plateau_ms": 40.0}),
    "braille": ("braille", {"total_ms": 60.0, "ramp_ms": 10.0}),
    "drifting_grating": ("drifting_grating", {"total_ms": 60.0, "ramp_ms": 10.0}),
}

# Full-mode envelope parameters (defaults on the stimulus classes), needed
# here (not only inside render_stimulus) to define onset/hold/offset
# windows without re-deriving them from the rendered tensor.
_FULL_ENVELOPE: Dict[str, Dict[str, float]] = {
    "ramp_gaussian": {"total_ms": 1100.0, "ramp_ms": 50.0},
    "moving_edge": {
        "total_ms": 330.0,
        "ramp_up_ms": 20.0,
        "plateau_ms": 300.0,
        "ramp_down_ms": 10.0,
    },
    "braille": {"total_ms": 900.0, "ramp_ms": 75.0},
    "drifting_grating": {"total_ms": 1000.0, "ramp_ms": 100.0},
}
_QUICK_ENVELOPE: Dict[str, Dict[str, float]] = {
    "ramp_gaussian": {"total_ms": 60.0, "ramp_ms": 10.0},
    "moving_edge": {
        "total_ms": 60.0,
        "ramp_up_ms": 20.0,
        "plateau_ms": 40.0,
        "ramp_down_ms": 10.0,
    },
    "braille": {"total_ms": 60.0, "ramp_ms": 10.0},
    "drifting_grating": {"total_ms": 60.0, "ramp_ms": 10.0},
}

ONSET_MS = 30.0
HOLD_MS = 200.0
OFFSET_MS = 30.0
PEAK_RATE_BIN_MS = 5.0  # bin width used for the peak instantaneous population rate


def _envelope(name: str, quick: bool) -> Dict[str, float]:
    return dict((_QUICK_ENVELOPE if quick else _FULL_ENVELOPE)[name])


def windows_for(name: str, quick: bool) -> Dict[str, Any]:
    """Return this stimulus's onset/hold-or-steady/offset windows (ms) and status.

    Returns:
        Dict with ``total_ms``, ``onset`` (tuple), ``offset`` (tuple or
        ``None``), ``hold`` (tuple), and ``hold_is_scored`` (bool -- True
        only for ``ramp_gaussian``, where the window is a genuine static
        hold; the other three stimuli never stop moving, so their
        "hold" window is an informational steady-drive interval only).
    """
    env = _envelope(name, quick)
    total_ms = env["total_ms"]
    onset = (0.0, min(ONSET_MS, total_ms))

    if name == "ramp_gaussian":
        ramp_ms = env["ramp_ms"]
        hold_start = ramp_ms
        hold_end = min(ramp_ms + HOLD_MS, total_ms)
        return {
            "total_ms": total_ms,
            "onset": onset,
            "hold": (hold_start, hold_end),
            "hold_is_scored": True,
            "offset": None,
        }

    if name == "moving_edge":
        down_start = env["ramp_up_ms"] + env["plateau_ms"]
        plateau_start, plateau_end = env["ramp_up_ms"], down_start
        mid = 0.5 * (plateau_start + plateau_end)
        half = min(HOLD_MS, plateau_end - plateau_start) / 2.0
        return {
            "total_ms": total_ms,
            "onset": onset,
            "hold": (max(plateau_start, mid - half), min(plateau_end, mid + half)),
            "hold_is_scored": False,
            "offset": (down_start, min(down_start + OFFSET_MS, total_ms)),
        }

    # braille, drifting_grating: symmetric ramp in/out, moving interior.
    ramp_ms = env["ramp_ms"]
    interior_start, interior_end = ramp_ms, max(ramp_ms, total_ms - ramp_ms)
    mid = 0.5 * (interior_start + interior_end)
    half = min(HOLD_MS, max(interior_end - interior_start, 0.0)) / 2.0
    return {
        "total_ms": total_ms,
        "onset": onset,
        "hold": (max(interior_start, mid - half), min(interior_end, mid + half)),
        "hold_is_scored": False,
        "offset": (max(0.0, total_ms - OFFSET_MS), total_ms),
    }


def _bin_slice(dt_ms: float, window: Tuple[float, float], n_bins: int) -> slice:
    lo = max(0, int(round(window[0] / dt_ms)))
    hi = min(n_bins, int(round(window[1] / dt_ms)))
    if hi < lo:
        hi = lo
    return slice(lo, hi)


def measure_drive_range(
    izh_config: SensoryForgeConfig, quick: bool, device: str
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, torch.Tensor]]:
    """Render the four stimuli and measure the filtered (post-gain) drive (mA).

    Args:
        izh_config: The ``tactile_sa1_ra1`` config (drive is neuron-model
            agnostic -- filter + gain happen before the neuron).
        quick: Use the quick-mode envelope overrides.
        device: Torch device string.

    Returns:
        Tuple of (stats, frames_by_stimulus): ``stats[stimulus][population]``
        holds ``{"min", "median", "p95", "max"}`` in mA; ``frames_by_stimulus``
        holds each stimulus's rendered ``[T, H, W]`` tensor (reused by the
        SA-vs-RA engine runs so the stimulus is rendered exactly once).
    """
    grid_cfg = izh_config.grids[0]
    stim_grid = ReceptorGrid(
        grid_size=(grid_cfg.rows, grid_cfg.cols),
        spacing=grid_cfg.spacing,
        arrangement=grid_cfg.arrangement,
        center=(grid_cfg.center_x, grid_cfg.center_y),
        device=device,
        seed=grid_cfg.seed,
    )
    xx, yy = stim_grid.get_coordinates()

    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    frames_by_stimulus: Dict[str, torch.Tensor] = {}
    for name, (stim_type, quick_params) in STIMULI.items():
        params = dict(quick_params) if quick else {}
        frames, _ = render_stimulus(
            stim_type, params, xx, yy, dt_ms=izh_config.simulation.dt_ms, device=device
        )
        frames_by_stimulus[name] = frames

        engine = SimulationEngine(izh_config)
        results = engine.run(
            frames.unsqueeze(0),
            return_intermediates=True,
            seed=izh_config.simulation.seed,
        )
        stats[name] = {}
        for pop_name, pop_results in results.items():
            filtered = pop_results["filtered"].detach().cpu().numpy().ravel()
            stats[name][pop_name] = {
                "min": float(np.min(filtered)),
                "median": float(np.median(filtered)),
                "p95": float(np.percentile(filtered, 95)),
                "max": float(np.max(filtered)),
            }
    return stats, frames_by_stimulus


def choose_current_levels(stats: Dict[str, Dict[str, Dict[str, float]]]) -> np.ndarray:
    """Pick 20 constant-current levels (mA) spanning the measured drive.

    Args:
        stats: Output of :func:`measure_drive_range`.

    Returns:
        A ``[20]`` numpy array of current levels in mA, linearly spaced
        from 0 to 1.2x the largest measured max across every
        stimulus x population.
    """
    all_max = [
        pop_stats["max"]
        for stim_stats in stats.values()
        for pop_stats in stim_stats.values()
    ]
    hi = max(all_max) * 1.2 if all_max else 100.0
    return np.linspace(0.0, hi, 20)


def fi_curve(
    neuron: torch.nn.Module, levels: np.ndarray, integrate_dt_ms: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Steady-state and first-30-ms f-I rate (Hz) for one neuron model.

    Runs all 20 current levels in a single batched forward pass (batch =
    current level, features = 1) over 1000 ms at ``integrate_dt_ms``.
    Steady-state rate is measured over the last half (500 ms) of the run;
    transient rate over the first 30 ms.

    Args:
        neuron: An instantiated neuron model (``dt=integrate_dt_ms``).
        levels: ``[L]`` array of constant currents in mA.
        integrate_dt_ms: Neuron integration step in ms.

    Returns:
        Tuple ``(steady_hz, transient_hz)``, each ``[L]``.
    """
    duration_ms = 1000.0
    n_steps = int(round(duration_ms / integrate_dt_ms))
    current = torch.as_tensor(levels, dtype=torch.float32).view(-1, 1, 1)
    input_current = current.expand(-1, n_steps, 1).contiguous()
    with torch.no_grad():
        _, spikes = neuron(input_current)
    spikes = spikes[:, 1:, :].float().squeeze(-1)  # [L, n_steps]

    half = n_steps // 2
    steady_counts = spikes[:, half:].sum(dim=1).numpy()
    steady_hz = steady_counts / ((n_steps - half) * integrate_dt_ms / 1000.0)

    n_transient = max(1, int(round(30.0 / integrate_dt_ms)))
    transient_counts = spikes[:, :n_transient].sum(dim=1).numpy()
    transient_hz = transient_counts / (n_transient * integrate_dt_ms / 1000.0)

    return steady_hz, transient_hz


def run_fi_curves(
    levels: np.ndarray, integrate_dt_ms: float, seed: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute f-I curves for the four required models.

    Returns:
        Dict keyed by model label -> {"steady": [...], "transient": [...] or None}.
    """
    torch.manual_seed(seed)
    curves: Dict[str, Dict[str, np.ndarray]] = {}

    adex_sa = AdExNeuronTorch(preset="SA1_tonic", dt=integrate_dt_ms)
    steady, _ = fi_curve(adex_sa, levels, integrate_dt_ms)
    curves["AdEx SA1_tonic"] = {"steady": steady, "transient": None}

    adex_ra = AdExNeuronTorch(preset="RA1_phasic", dt=integrate_dt_ms)
    steady, transient = fi_curve(adex_ra, levels, integrate_dt_ms)
    curves["AdEx RA1_phasic (steady)"] = {"steady": steady, "transient": None}
    curves["AdEx RA1_phasic (first 30 ms)"] = {"steady": transient, "transient": None}

    izh_rs = IzhikevichNeuronTorch(preset="RS", dt=integrate_dt_ms)
    steady, _ = fi_curve(izh_rs, levels, integrate_dt_ms)
    curves["Izhikevich RS"] = {"steady": steady, "transient": None}

    izh_fs = IzhikevichNeuronTorch(preset="FS", dt=integrate_dt_ms)
    steady, _ = fi_curve(izh_fs, levels, integrate_dt_ms)
    curves["Izhikevich FS"] = {"steady": steady, "transient": None}

    return curves


def plot_fi_curves(
    levels: np.ndarray, curves: Dict[str, Dict[str, np.ndarray]], out_path: Path
) -> None:
    """Plot every f-I curve on one figure and save it as a PNG."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    for label, data in curves.items():
        style = "--" if "first 30 ms" in label else "-"
        ax.plot(levels, data["steady"], style, marker="o", markersize=3, label=label)
    ax.set_xlabel("Constant current (mA)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("f-I curves: AdEx SA1_tonic/RA1_phasic vs. Izhikevich RS/FS")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_stimulus_engine(
    config: SensoryForgeConfig, frames: torch.Tensor, input_gain: float, device: str
) -> Dict[str, Any]:
    """Run one rendered stimulus through the engine, with an optional gain override.

    Args:
        config: The population config to run (Izhikevich or AdEx preset).
        frames: Rendered stimulus ``[T, H, W]``.
        input_gain: If not ``None``, overrides every population's
            ``input_gain`` for this run only (the passed-in ``config``
            object is mutated and restored).
        device: Torch device string.

    Returns:
        The engine's ``run()`` result dict.
    """
    originals = []
    if input_gain is not None:
        for pop in config.populations:
            originals.append(pop.input_gain)
            pop.input_gain = input_gain
    try:
        engine = SimulationEngine(config)
        results = engine.run(
            frames.unsqueeze(0), return_intermediates=False, seed=config.simulation.seed
        )
    finally:
        if input_gain is not None:
            for pop, orig in zip(config.populations, originals):
                pop.input_gain = orig
    return results


def population_neuron_type(config: SensoryForgeConfig, pop_name: str) -> str:
    for pop in config.populations:
        if pop.name == pop_name:
            return pop.neuron_type
    raise ValueError(f"Population {pop_name!r} not found in config")


def spike_metrics(
    spikes: torch.Tensor,
    dt_ms: float,
    windows: Dict[str, Any],
    neuron_type: str,
) -> Dict[str, Any]:
    """Compute the per-population metrics table (c) requires.

    Args:
        spikes: ``[1, T, N]`` sub-step spike COUNT per record bin (float,
            not boolean -- greater-than-zero gives a binary raster).
        dt_ms: Record step in ms.
        windows: Output of :func:`windows_for`.
        neuron_type: "SA" or "RA" (case-insensitive).

    Returns:
        Dict of metrics (see module docstring for which apply to which
        neuron type).
    """
    s = spikes[0].detach().cpu().numpy()  # [T, N]
    n_bins, n_neurons = s.shape
    total_spikes = float(s.sum())

    onset_sl = _bin_slice(dt_ms, windows["onset"], n_bins)
    hold_sl = _bin_slice(dt_ms, windows["hold"], n_bins)
    offset_sl = (
        _bin_slice(dt_ms, windows["offset"], n_bins) if windows["offset"] else None
    )

    onset_count = float(s[onset_sl].sum())
    hold_count = float(s[hold_sl].sum())
    offset_count = float(s[offset_sl].sum()) if offset_sl is not None else None

    # Peak instantaneous population rate: bin the whole run at
    # PEAK_RATE_BIN_MS and take the max population-summed rate (Hz).
    bin_steps = max(1, int(round(PEAK_RATE_BIN_MS / dt_ms)))
    n_super_bins = n_bins // bin_steps
    peak_hz = 0.0
    if n_super_bins > 0:
        trimmed = s[: n_super_bins * bin_steps].reshape(
            n_super_bins, bin_steps, n_neurons
        )
        per_bin_counts = trimmed.sum(axis=(1, 2))
        peak_hz = float(per_bin_counts.max() / (bin_steps * dt_ms / 1000.0))

    metrics: Dict[str, Any] = {
        "total_spikes": total_spikes,
        "onset_count": onset_count,
        "hold_count": hold_count,
        "offset_count": offset_count,
        "peak_instantaneous_hz": peak_hz,
        "peak_rate_bin_ms": PEAK_RATE_BIN_MS,
    }

    if neuron_type.upper() == "SA":
        hold_duration_s = max(windows["hold"][1] - windows["hold"][0], 1e-9) / 1000.0
        mean_rate_hz = hold_count / n_neurons / hold_duration_s
        # Pooled ISI CV: pool every neuron's spike times inside the hold
        # window into one ISI list (stated in the docstring/report: pooled,
        # not per-neuron median).
        hold_bins = np.arange(n_bins)[hold_sl]
        isis: List[float] = []
        for neuron_idx in range(n_neurons):
            spike_bins = hold_bins[s[hold_sl, neuron_idx] > 0]
            if spike_bins.size >= 2:
                isis.extend(np.diff(spike_bins.astype(float)) * dt_ms)
        if len(isis) >= 2:
            isis_arr = np.asarray(isis)
            isi_cv = float(isis_arr.std() / max(isis_arr.mean(), 1e-9))
        else:
            isi_cv = None
        metrics["mean_rate_hz"] = mean_rate_hz
        metrics["isi_cv"] = isi_cv
        metrics["isi_cv_method"] = "pooled across neurons, within the hold window"

    return metrics


def build_table_rows(
    stimulus_name: str,
    windows: Dict[str, Any],
    izh_results: Dict[str, Any],
    adex_results: Dict[str, Any],
    izh_config: SensoryForgeConfig,
    dt_ms: float,
) -> List[Dict[str, Any]]:
    rows = []
    for pop_name in izh_results:
        neuron_type = population_neuron_type(izh_config, pop_name)
        for model_label, results in (
            ("Izhikevich", izh_results),
            ("AdEx", adex_results),
        ):
            spikes = results[pop_name]["spikes"]
            metrics = spike_metrics(spikes, dt_ms, windows, neuron_type)
            rows.append(
                {
                    "stimulus": stimulus_name,
                    "population": pop_name,
                    "neuron_type": neuron_type,
                    "model": model_label,
                    **metrics,
                }
            )
    return rows


def _fmt(v, nd=2):
    if v is None:
        return "N/A"
    return f"{v:.{nd}f}"


def write_report(
    out_dir: Path,
    drive_stats: Dict[str, Dict[str, Dict[str, float]]],
    levels: np.ndarray,
    fi_curves_data: Dict[str, Dict[str, np.ndarray]],
    rows: List[Dict[str, Any]],
    windows_by_stimulus: Dict[str, Dict[str, Any]],
    input_gain_used: float,
    grid_size: int,
    wall_clock_s: float,
    seed: int,
    quick: bool,
    png_total_bytes: int,
) -> str:
    lines = []
    lines.append("# AdEx population tuning -- Phase 2b T2\n")
    lines.append(
        f"Grid: {grid_size}x{grid_size} @ 0.15 mm | seed={seed} | quick={quick} | "
        f"input_gain used={input_gain_used} | wall-clock={wall_clock_s:.1f}s\n"
    )

    lines.append("## Measured filtered-drive current range (mA)\n")
    lines.append("| stimulus | population | min | median | p95 | max |")
    lines.append("|---|---|---|---|---|---|")
    for stim, pops in drive_stats.items():
        for pop, s in pops.items():
            lines.append(
                f"| {stim} | {pop} | {_fmt(s['min'])} | {_fmt(s['median'])} | "
                f"{_fmt(s['p95'])} | {_fmt(s['max'])} |"
            )
    lines.append("")
    lines.append(
        f"20 constant-current f-I levels chosen: linspace(0, {levels.max():.2f}, 20) mA "
        "(1.2x the largest measured max across every stimulus x population).\n"
    )

    lines.append("## f-I curve summary (steady-state, last 500 ms of a 1000 ms run)\n")
    lines.append("| model | min rate (Hz) | max rate (Hz) | rate at max current (Hz) |")
    lines.append("|---|---|---|---|")
    for label, data in fi_curves_data.items():
        r = data["steady"]
        lines.append(f"| {label} | {_fmt(r.min())} | {_fmt(r.max())} | {_fmt(r[-1])} |")
    lines.append("")

    lines.append("## Per-stimulus tables\n")
    for stim_name, windows in windows_by_stimulus.items():
        lines.append(f"### {stim_name}\n")
        lines.append(
            f"Windows (ms): onset={windows['onset']}, "
            f"hold/steady-drive={windows['hold']} "
            f"({'scored' if windows['hold_is_scored'] else 'N/A -- informational only'}), "
            f"offset={windows['offset']}\n"
        )
        lines.append(
            "| population | model | total spikes | onset count | hold count | "
            "offset count | peak inst. rate (Hz, 5ms bins) | SA mean rate (Hz) | SA ISI CV |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            if row["stimulus"] != stim_name:
                continue
            lines.append(
                f"| {row['population']} | {row['model']} | {_fmt(row['total_spikes'], 0)} | "
                f"{_fmt(row['onset_count'], 0)} | {_fmt(row['hold_count'], 0)} | "
                f"{_fmt(row['offset_count'], 0)} | {_fmt(row['peak_instantaneous_hz'])} | "
                f"{_fmt(row.get('mean_rate_hz'))} | {_fmt(row.get('isi_cv'), 3)} |"
            )
        lines.append("")

    # ---- PASS / FAIL / N/A against P5 -----------------------------------
    lines.append("## P5 criteria: PASS / FAIL / N-A\n")

    sa_rows = [
        r for r in rows if r["neuron_type"].upper() == "SA" and r["model"] == "AdEx"
    ]
    ra_rows = [
        r for r in rows if r["neuron_type"].upper() == "RA" and r["model"] == "AdEx"
    ]

    for row in sa_rows:
        windows = windows_by_stimulus[row["stimulus"]]
        if not windows["hold_is_scored"]:
            lines.append(
                f"- SA-I ({row['stimulus']}): **N/A** -- no static hold in this stimulus "
                "(moving/drifting); reported as an informational steady-drive interval "
                f"(mean rate {_fmt(row.get('mean_rate_hz'))} Hz, "
                f"ISI CV {_fmt(row.get('isi_cv'), 3)})."
            )
            continue
        rate = row.get("mean_rate_hz")
        cv = row.get("isi_cv")
        rate_ok = rate is not None and 20.0 <= rate <= 100.0
        cv_ok = cv is not None and cv < 0.5
        status = "PASS" if (rate_ok and cv_ok) else "FAIL"
        lines.append(
            f"- SA-I ({row['stimulus']}): **{status}** -- mean rate {_fmt(rate)} Hz "
            f"(target 20-100 Hz), ISI CV {_fmt(cv, 3)} (target < 0.5)."
        )

    for row in ra_rows:
        windows = windows_by_stimulus[row["stimulus"]]
        hold_count = row["hold_count"]
        peak = row["peak_instantaneous_hz"]
        if not windows["hold_is_scored"]:
            lines.append(
                f"- RA-I silent-hold ({row['stimulus']}): **N/A** -- no static hold in this "
                f"stimulus; steady-drive-interval spike count = {_fmt(hold_count, 0)}."
            )
        else:
            status = "PASS" if hold_count == 0 else "FAIL"
            lines.append(
                f"- RA-I silent-hold ({row['stimulus']}): **{status}** -- "
                f"{_fmt(hold_count, 0)} spikes in the {windows['hold']} ms hold window "
                "(target: 0)."
            )
        peak_status = (
            "PASS" if peak >= 300.0 * 0.5 else "FAIL"
        )  # generous: within striking distance
        lines.append(
            f"- RA-I transient burst ({row['stimulus']}): peak instantaneous rate "
            f"{_fmt(peak)} Hz (target up to ~300 Hz) -- {peak_status if peak > 0 else 'FAIL'}."
        )

    lines.append("")
    return "\n".join(lines)


def plot_stimulus_figure(
    stim_name: str,
    izh_results: Dict[str, Any],
    adex_results: Dict[str, Any],
    dt_ms: float,
    izh_config: SensoryForgeConfig,
    out_path: Path,
    max_raster_neurons: int = 100,
) -> None:
    """Raster + population-rate figure, Izhikevich vs AdEx, one stimulus.

    Subsamples the raster to at most ``max_raster_neurons`` per population
    to keep the PNG small.
    """
    pop_names = list(izh_results.keys())
    fig, axes = plt.subplots(
        len(pop_names),
        4,
        figsize=(12, 2.6 * len(pop_names)),
        dpi=100,
        squeeze=False,
    )
    for row_idx, pop_name in enumerate(pop_names):
        for col_idx, (model_label, results) in enumerate(
            (("Izhikevich", izh_results), ("AdEx", adex_results))
        ):
            spikes = results[pop_name]["spikes"][0].detach().cpu().numpy()  # [T, N]
            n_bins, n_neurons = spikes.shape
            step = max(1, n_neurons // max_raster_neurons)
            sub = spikes[:, ::step] > 0
            times, neurons = np.nonzero(sub)

            ax_raster = axes[row_idx, col_idx * 2]
            ax_raster.scatter(times * dt_ms, neurons, s=1, color="black")
            ax_raster.set_title(f"{pop_name} ({model_label}) raster")
            ax_raster.set_xlabel("time (ms)")
            ax_raster.set_ylabel("neuron (subsampled)")

            ax_rate = axes[row_idx, col_idx * 2 + 1]
            pop_rate = spikes.sum(axis=1) / n_neurons / (dt_ms / 1000.0)
            ax_rate.plot(np.arange(n_bins) * dt_ms, pop_rate)
            ax_rate.set_title(f"{pop_name} ({model_label}) mean pop. rate")
            ax_rate.set_xlabel("time (ms)")
            ax_rate.set_ylabel("Hz")

    fig.suptitle(stim_name)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(
    out_dir: Path,
    quick: bool,
    seed: int,
    input_gain_override: float,
) -> Dict[str, Any]:
    """Run the full AdEx tuning characterization and write artifacts.

    Args:
        out_dir: Output directory for the markdown report and PNGs.
        quick: Shorten stimuli and reduce work for a fast smoke run.
        seed: Seed for reproducibility (passed to the engine's ``run()``).
        input_gain_override: If not ``None``, overrides both AdEx
            populations' ``input_gain`` for the stimulus runs only (the
            f-I curves and the drive measurement are unaffected -- they
            use the raw current axis / the Izhikevich preset's own gain).

    Returns:
        Dict with every computed artifact's path and key numbers, used by
        the smoke test and by the CLI's own summary print.
    """
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    izh_dict = load_preset("tactile_sa1_ra1")
    adex_dict = load_preset("tactile_sa1_ra1_adex")
    izh_config = SensoryForgeConfig.from_dict(izh_dict)
    adex_config = SensoryForgeConfig.from_dict(adex_dict)
    izh_config.simulation.device = device
    adex_config.simulation.device = device

    torch.manual_seed(seed)

    # (a) Measured drive range + f-I curves.
    drive_stats, frames_by_stimulus = measure_drive_range(izh_config, quick, device)
    levels = choose_current_levels(drive_stats)
    fi_data = run_fi_curves(levels, izh_config.simulation.integrate_dt_ms, seed)
    fi_png = out_dir / "fi_curves.png"
    plot_fi_curves(levels, fi_data, fi_png)

    # (b) The four benchmark stimuli, Izhikevich vs AdEx.
    all_rows: List[Dict[str, Any]] = []
    windows_by_stimulus: Dict[str, Dict[str, Any]] = {}
    stim_pngs: List[Path] = []
    for stim_name, frames in frames_by_stimulus.items():
        izh_results = run_stimulus_engine(izh_config, frames, None, device)
        adex_results = run_stimulus_engine(
            adex_config, frames, input_gain_override, device
        )

        windows = windows_for(stim_name, quick)
        windows_by_stimulus[stim_name] = windows
        rows = build_table_rows(
            stim_name,
            windows,
            izh_results,
            adex_results,
            izh_config,
            izh_config.simulation.dt_ms,
        )
        all_rows.extend(rows)

        stim_png = out_dir / f"{stim_name}_raster.png"
        plot_stimulus_figure(
            stim_name,
            izh_results,
            adex_results,
            izh_config.simulation.dt_ms,
            izh_config,
            stim_png,
        )
        stim_pngs.append(stim_png)

    wall_clock_s = time.time() - start
    png_paths = [fi_png] + stim_pngs
    png_total_bytes = sum(p.stat().st_size for p in png_paths if p.exists())

    input_gain_used = input_gain_override if input_gain_override is not None else 50.0

    report = write_report(
        out_dir,
        drive_stats,
        levels,
        fi_data,
        all_rows,
        windows_by_stimulus,
        input_gain_used,
        izh_config.grids[0].rows,
        wall_clock_s,
        seed,
        quick,
        png_total_bytes,
    )
    md_path = out_dir / "adex_tuning.md"
    md_path.write_text(report, encoding="utf-8")

    return {
        "md_path": md_path,
        "png_paths": png_paths,
        "png_total_bytes": png_total_bytes,
        "wall_clock_s": wall_clock_s,
        "drive_stats": drive_stats,
        "rows": all_rows,
        "input_gain_used": input_gain_used,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/adex_tuning"),
        help="Output directory for the report and PNGs.",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Shorten stimuli for a fast smoke run."
    )
    parser.add_argument("--seed", type=int, default=0, help="Reproducibility seed.")
    parser.add_argument(
        "--input-gain",
        type=float,
        default=None,
        help=(
            "Override input_gain for the AdEx stimulus runs (default: the "
            "preset's own value, 50.0 -- no override)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = main(args.out, args.quick, args.seed, args.input_gain)
    print(f"Wrote {result['md_path']}")
    print(f"PNGs: {[str(p) for p in result['png_paths']]}")
    print(f"Total PNG bytes: {result['png_total_bytes']}")
    print(f"Wall-clock: {result['wall_clock_s']:.1f}s")
