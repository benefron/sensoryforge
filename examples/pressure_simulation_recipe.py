"""The pressure-simulation recipe, end to end (Phase 2, Wave K, K5).

Loads the ``tactile_sa1_ra1`` preset (80x80 grid at 0.15 mm, one SA and one
RA population with ``template`` receptive fields at a single resolvable
distance ``d = 0.40`` mm, derived neuron counts), renders each of the four
ported pressure-simulation stimuli (K2: ``ramp_gaussian``, ``moving_edge``,
``braille``, ``drifting_grating``), runs each through
:class:`~sensoryforge.core.simulation_engine.SimulationEngine`, and writes
one Wave J bundle per stimulus under
``examples/output/pressure_simulation_recipe/<stimulus_name>/``.

Run with no arguments for the real recipe (the stimuli's own default
durations -- several hundred ms to just over a second each); pass
``--quick`` to shorten every stimulus to a few tens of ms for a fast
smoke run (same 80x80/900-neuron grid and populations -- ``--quick``
shortens durations, it does not shrink the grid).

Usage:
    python examples/pressure_simulation_recipe.py
    python examples/pressure_simulation_recipe.py --quick
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.presets import load_preset
from sensoryforge.stimuli.render import render_stimulus

OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "pressure_simulation_recipe"

# name -> (registered stimulus type, quick-mode param overrides)
_STIMULI = {
    "ramp_gaussian": ("ramp_gaussian", {"total_ms": 60.0, "ramp_ms": 10.0}),
    "moving_edge": ("moving_edge", {"total_ms": 60.0, "plateau_ms": 40.0}),
    "braille": ("braille", {"total_ms": 60.0, "ramp_ms": 10.0}),
    "drifting_grating": ("drifting_grating", {"total_ms": 60.0, "ramp_ms": 10.0}),
}


def main(quick: bool = False) -> dict:
    """Run the recipe and write one bundle per stimulus.

    Args:
        quick: Shorten every stimulus's duration for a fast smoke run.

    Returns:
        Dict mapping stimulus name to the bundle directory written.
    """
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

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    bundle_dirs: dict = {}

    for name, (stim_type, quick_params) in _STIMULI.items():
        # A fresh engine per stimulus: SimulationEngine has no reset_state()
        # between runs and this keeps each bundle's neuron state
        # independent, matching how the CLI runs one stimulus per process.
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

        bundle_dir = OUTPUT_ROOT / name
        results = engine.run(
            stimulus_tensor,
            return_intermediates=True,
            bundle_dir=bundle_dir,
            stimulus_config={"type": stim_type, **params},
            seed=grid_cfg.seed,
            bundle_overwrite=True,
        )
        bundle_dirs[name] = bundle_dir

        print(f"\n{name} (T={frames.shape[0]} bins):")
        for pop_name, pop_results in results.items():
            spikes = pop_results["spikes"]
            total = int(spikes.sum().item())
            duration_s = frames.shape[0] * sf_config.simulation.dt_ms / 1000.0
            n_neurons = spikes.shape[-1]
            mean_rate_hz = (
                total / max(n_neurons, 1) / max(duration_s, 1e-9)
                if duration_s > 0
                else 0.0
            )
            print(
                f"  {pop_name}: {total} spikes, {n_neurons} neurons, "
                f"mean rate {mean_rate_hz:.2f} Hz"
            )
        print(f"  bundle: {bundle_dir}")

    return bundle_dirs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorten every stimulus's duration for a fast smoke run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    start = time.time()
    main(quick=args.quick)
    print(f"\nDone in {time.time() - start:.1f}s")
