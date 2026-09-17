"""S3: the reproducibility proof (Wave S, `docs/development/handover/phase4_tasks.md`).

Runs the pressure-simulation recipe from Wave K5
(``examples/pressure_simulation_recipe.py``), computes a small set of summary
statistics (per stimulus, per population: total spike count and mean firing
rate) and one figure (mean rate per population per stimulus), and checks
them against the versions committed under
``tests/fixtures/reference/reproducibility/``.

**What "reproduces" means here, on one platform and across platforms.**
The recipe is deterministic given its preset's seed: the grid, the weights
and the neuron order are the same every run, and nothing else in the CPU
path draws random numbers. So on the *same platform* -- the same operating
system, processor architecture and torch build -- two runs produce the
identical spike tensor, and this script requires spike counts to match
exactly.

Across platforms that guarantee does not hold, and pretending otherwise
would make this check fail on its first run somewhere new with nothing
changed. A different processor or BLAS library can round the last bits of
a floating-point sum differently, and a neuron sitting just at threshold can
then spike on one machine and not the other. So when the reference was
recorded on a different platform, spike counts must agree within the larger
of ``CROSS_PLATFORM_SPIKE_ABS`` spikes or ``CROSS_PLATFORM_SPIKE_REL`` of the
reference count. That absorbs a few threshold-edge spikes flipping, and it
still fails on a real regression, which moves counts by far more: a wrong
gain or preset changes them by factors, not by a handful.

The cross-platform tolerance is an estimate, not a measurement. The
reference was recorded on one machine and nothing has yet been run on
another, so the size of the real cross-platform spread is unknown. If the
first run elsewhere fails by a few spikes, widen the tolerance with that
evidence in the commit; if it fails by factors, that is a real difference.

Mean firing rates are computed from the spike counts, so they add no
independent check and are compared with the same rule.

The reference records the platform it was made on. One recorded without a
platform is treated as foreign, since there is no evidence it matches.

Usage:
    # Compare against the committed reference (what CI runs):
    python scripts/reproduce_figure.py --check

    # Regenerate the committed reference after a deliberate, reviewed
    # change to the recipe's own output (do this by hand, never in CI):
    python scripts/reproduce_figure.py --write-reference

By default this runs the recipe's ``--quick`` (shortened-duration) mode so
the whole proof finishes in a few seconds; the reference JSON records
``"quick": true`` so a mismatch in mode cannot silently compare apples to
oranges. ``scripts/reproduce_env.sh`` wraps this script with the "clean
checkout, fresh environment, install the package" steps the Wave S brief
asks for; this script itself assumes the package is already importable
(it is meant to be invoked by that wrapper, or directly in an environment
that already has SensoryForge installed, such as this project's own dev
environment or CI).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

REFERENCE_DIR = REPO_ROOT / "tests" / "fixtures" / "reference" / "reproducibility"
STATS_PATH = REFERENCE_DIR / "summary_stats.json"
FIGURE_PATH = REFERENCE_DIR / "mean_rates.png"

CROSS_PLATFORM_SPIKE_ABS = 3
CROSS_PLATFORM_SPIKE_REL = 0.02


def platform_signature() -> dict:
    """The properties that decide whether exact reproduction is expected."""
    import platform

    import torch

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
    }


def same_platform(reference: dict) -> bool:
    """Whether *reference* was recorded on a platform matching this one.

    A reference with no recorded platform is treated as foreign: there is
    no evidence it matches, and assuming it does would demand exact
    agreement that may be impossible.
    """
    recorded = reference.get("platform")
    return isinstance(recorded, dict) and recorded == platform_signature()


def spike_tolerance(reference_spikes: int) -> int:
    """Allowed absolute spike-count difference across platforms."""
    return max(
        CROSS_PLATFORM_SPIKE_ABS,
        int(round(CROSS_PLATFORM_SPIKE_REL * abs(reference_spikes))),
    )


def compute_summary_stats(quick: bool = True) -> dict:
    """Run the K5 recipe and reduce each population's spikes to summary stats.

    Returns:
        ``{"quick": quick, "stimuli": {name: {pop: {"spikes": int,
        "n_neurons": int, "mean_rate_hz": float}}}}``.
    """
    import pressure_simulation_recipe as recipe
    from sensoryforge.config.schema import SensoryForgeConfig
    from sensoryforge.core.grid import ReceptorGrid
    from sensoryforge.core.simulation_engine import SimulationEngine
    from sensoryforge.presets import load_preset
    from sensoryforge.stimuli.render import render_stimulus

    config_dict = load_preset("tactile_sa1_ra1")
    sf_config = SensoryForgeConfig.from_dict(config_dict)
    grid_cfg = sf_config.grids[0]

    stim_grid = ReceptorGrid(
        grid_size=(grid_cfg.rows, grid_cfg.cols),
        spacing=grid_cfg.spacing,
        arrangement=grid_cfg.arrangement,
        center=(grid_cfg.center_x, grid_cfg.center_y),
        device=sf_config.simulation.device,
        seed=grid_cfg.seed,
    )
    xx, yy = stim_grid.get_coordinates()

    stats: dict = {"quick": quick, "stimuli": {}}

    for name, (stim_type, quick_params) in recipe._STIMULI.items():
        engine = SimulationEngine(sf_config)
        params = dict(quick_params) if quick else {}
        frames, _ = render_stimulus(
            stim_type,
            params,
            xx,
            yy,
            dt_ms=sf_config.simulation.dt_ms,
            device=sf_config.simulation.device,
        )
        stimulus_tensor = frames.unsqueeze(0)
        results = engine.run(
            stimulus_tensor,
            return_intermediates=True,
            bundle_dir=None,
            stimulus_config={"type": stim_type, **params},
            seed=grid_cfg.seed,
        )
        duration_s = frames.shape[0] * sf_config.simulation.dt_ms / 1000.0

        pop_stats = {}
        for pop_name, pop_results in results.items():
            spikes = pop_results["spikes"]
            total = int(spikes.sum().item())
            n_neurons = int(spikes.shape[-1])
            mean_rate_hz = (
                total / max(n_neurons, 1) / max(duration_s, 1e-9)
                if duration_s > 0
                else 0.0
            )
            pop_stats[pop_name] = {
                "spikes": total,
                "n_neurons": n_neurons,
                "mean_rate_hz": mean_rate_hz,
            }
        stats["stimuli"][name] = pop_stats

    return stats


def render_figure(stats: dict, out_path: Path) -> None:
    """Bar chart of each population's mean rate for each stimulus."""
    stim_names = list(stats["stimuli"].keys())
    pop_names = sorted({p for s in stats["stimuli"].values() for p in s})

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(len(pop_names), 1)
    for i, pop in enumerate(pop_names):
        rates = [
            stats["stimuli"][s].get(pop, {}).get("mean_rate_hz", 0.0)
            for s in stim_names
        ]
        xs = [j + i * width for j in range(len(stim_names))]
        ax.bar(xs, rates, width=width, label=pop)
    ax.set_xticks(
        [j + width * (len(pop_names) - 1) / 2 for j in range(len(stim_names))]
    )
    ax.set_xticklabels(stim_names, rotation=20, ha="right")
    ax.set_ylabel("mean rate (Hz)")
    ax.set_title("Pressure-simulation recipe (Wave K5): mean rate per population")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _compare(actual: dict, reference: dict, *, exact: bool = True) -> list:
    """Return human-readable mismatch descriptions; empty means a match.

    Args:
        actual: Summary statistics from a fresh run.
        reference: The committed reference statistics.
        exact: Require identical spike counts (same platform). When
            ``False``, counts must agree within :func:`spike_tolerance`.

    Returns:
        One string per mismatch.
    """
    problems = []
    if actual["quick"] != reference["quick"]:
        problems.append(
            f"quick mode differs: {actual['quick']} vs {reference['quick']}"
        )
        return problems

    for stim in reference["stimuli"]:
        if stim not in actual["stimuli"]:
            problems.append(f"missing stimulus in fresh run: {stim}")
            continue
        for pop in reference["stimuli"][stim]:
            ref_pop = reference["stimuli"][stim][pop]
            act_pop = actual["stimuli"][stim].get(pop)
            if act_pop is None:
                problems.append(f"{stim}/{pop}: missing in fresh run")
                continue
            ref_spikes = int(ref_pop["spikes"])
            act_spikes = int(act_pop["spikes"])
            if exact:
                if act_spikes != ref_spikes:
                    problems.append(
                        f"{stim}/{pop}: spike count {act_spikes} != reference "
                        f"{ref_spikes} (same platform: exact match required)"
                    )
            else:
                allowed = spike_tolerance(ref_spikes)
                if abs(act_spikes - ref_spikes) > allowed:
                    problems.append(
                        f"{stim}/{pop}: spike count {act_spikes} vs reference "
                        f"{ref_spikes}, differs by {abs(act_spikes - ref_spikes)} "
                        f"(cross-platform tolerance {allowed})"
                    )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Compare a fresh run against the committed reference.",
    )
    mode.add_argument(
        "--write-reference",
        action="store_true",
        help="Overwrite the committed reference with a fresh run (do this by hand, review the diff).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the recipe's full (non---quick) durations instead of the fast smoke mode.",
    )
    args = parser.parse_args()

    quick = not args.full
    stats = compute_summary_stats(quick=quick)

    if args.write_reference:
        stats["platform"] = platform_signature()
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATS_PATH, "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
            f.write("\n")
        render_figure(stats, FIGURE_PATH)
        print(f"Wrote {STATS_PATH}")
        print(f"Wrote {FIGURE_PATH}")
        return 0

    if not STATS_PATH.exists():
        print(f"No committed reference at {STATS_PATH}; run --write-reference first.")
        return 2

    with open(STATS_PATH) as f:
        reference = json.load(f)

    exact = same_platform(reference)
    print(
        "Same platform as the reference: exact spike counts required."
        if exact
        else "Different platform from the reference (or none recorded): "
        "comparing within the cross-platform tolerance."
    )
    problems = _compare(stats, reference, exact=exact)
    if problems:
        print("REPRODUCIBILITY CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("REPRODUCIBILITY CHECK PASSED: fresh run matches the committed reference")
    print(
        f"  ({sum(1 for s in stats['stimuli'].values() for _ in s)} population/stimulus cells checked)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
