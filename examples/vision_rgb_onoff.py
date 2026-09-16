"""The vision ON/OFF RGB demo, end to end (Phase 2, Wave M, M4).

Loads the ``vision_onoff_rgb`` preset (one grid with three named channels
R/G/B), renders three independent Gaussian stimuli -- one per channel, each
at a different location -- composes them into one ``[T, 3, H, W]`` stimulus
tensor, runs it through :class:`~sensoryforge.core.simulation_engine.SimulationEngine`,
and writes one Wave J bundle, mirroring the pressure-simulation recipe
script (``examples/pressure_simulation_recipe.py``, Wave K5).

This is the concrete proof that SimulationEngine (Wave M1/M2) is no longer
tactile-only: "RG OnOff Population" reads R and G, each through an
:class:`~sensoryforge.core.processing.OnOffLayer` (Wave M3), and sums;
"RGB Concat Population" reads all three channels directly and concatenates.

Usage:
    python examples/vision_rgb_onoff.py
    python examples/vision_rgb_onoff.py --quick
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

OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "vision_rgb_onoff"

# channel -> Gaussian stimulus params (a different spot per channel, so the
# three planes are visibly different -- not a smoke test that would pass
# even if channels were silently swapped).
_CHANNEL_STIMULI = {
    "R": {"center_x": 1.0, "center_y": 1.0, "amplitude": 20.0, "sigma": 0.6},
    "G": {"center_x": -1.0, "center_y": 1.0, "amplitude": 20.0, "sigma": 0.6},
    "B": {"center_x": 0.0, "center_y": -1.0, "amplitude": 20.0, "sigma": 0.6},
}


def main(quick: bool = False) -> Path:
    """Run the demo and write one bundle.

    Args:
        quick: Shorten the stimulus duration for a fast smoke run.

    Returns:
        The bundle directory written.
    """
    config_dict = load_preset("vision_onoff_rgb")
    sf_config = SensoryForgeConfig.from_dict(config_dict)
    grid_cfg = sf_config.grids[0]
    channels = list(grid_cfg.channels)

    stim_grid = ReceptorGrid(
        grid_size=(grid_cfg.rows, grid_cfg.cols),
        spacing=grid_cfg.spacing,
        arrangement=grid_cfg.arrangement,
        center=(grid_cfg.center_x, grid_cfg.center_y),
        device=sf_config.simulation.device,
        seed=grid_cfg.seed,
    )
    xx, yy = stim_grid.get_coordinates()

    duration_ms = 30.0 if quick else 200.0

    # Render each channel's stimulus separately (each call returns
    # [T, 3, H, W] with only its own plane non-zero) and sum -- exactly the
    # "caller composes several render_stimulus calls" contract documented
    # in render_stimulus's own docstring (Wave L2).
    stimulus = None
    time_ms = None
    for channel, params in _CHANNEL_STIMULI.items():
        frames, time_ms = render_stimulus(
            "gaussian",
            {**params, "channel": channel},
            xx,
            yy,
            dt_ms=sf_config.simulation.dt_ms,
            duration_ms=duration_ms,
            device=sf_config.simulation.device,
            channels=channels,
        )
        stimulus = frames if stimulus is None else stimulus + frames
    stimulus_tensor = stimulus.unsqueeze(0)  # [1, T, 3, H, W]

    engine = SimulationEngine(sf_config)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    bundle_dir = OUTPUT_ROOT / "run"
    results = engine.run(
        stimulus_tensor,
        return_intermediates=True,
        bundle_dir=bundle_dir,
        stimulus_config={"type": "vision_rgb_onoff", "channels": _CHANNEL_STIMULI},
        seed=42,
        bundle_overwrite=True,
    )

    print(f"vision_rgb_onoff (T={stimulus_tensor.shape[1]} bins, channels={channels}):")
    for pop_name, pop_results in results.items():
        spikes = pop_results["spikes"]
        n_neurons = spikes.shape[-1]
        total = int(spikes.sum().item())
        print(f"  {pop_name}: {total} spikes, {n_neurons} neurons")
    print(f"  bundle: {bundle_dir}")

    return bundle_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorten the stimulus duration for a fast smoke run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    start = time.time()
    main(quick=args.quick)
    print(f"\nDone in {time.time() - start:.1f}s")
