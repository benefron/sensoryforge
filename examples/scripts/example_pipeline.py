"""A shipped preset, one stimulus, two spike rasters.

Loads the ``tactile_sa1_ra1`` preset, renders a Gaussian stimulus on its grid,
runs it through :class:`~sensoryforge.core.simulation_engine.SimulationEngine`
(the engine ``sensoryforge run`` uses) and plots the stimulus with each
population's spike raster.

    python example_pipeline.py                  # opens a window
    python example_pipeline.py --save out.png   # writes the figure instead
"""

import argparse

import matplotlib.pyplot as plt

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.presets import load_preset
from sensoryforge.stimuli.render import render_stimulus


def main(save=None):
    config = SensoryForgeConfig.from_dict(load_preset("tactile_sa1_ra1"))
    grid = config.grids[0]
    xx, yy = ReceptorGrid(
        grid_size=(grid.rows, grid.cols),
        spacing=grid.spacing,
        arrangement=grid.arrangement,
        center=(grid.center_x, grid.center_y),
        seed=grid.seed,
    ).get_coordinates()

    # A Gaussian with the type's own defaults (amplitude, sigma), held for 200 ms.
    frames, _ = render_stimulus(
        "gaussian", {}, xx, yy, dt_ms=config.simulation.dt_ms, duration_ms=200.0
    )
    results = SimulationEngine(config).run(frames.unsqueeze(0), seed=42)  # [1, T, H, W]

    fig, axes = plt.subplots(1 + len(results), 1, figsize=(8, 3 * (1 + len(results))))
    axes[0].imshow(frames[frames.shape[0] // 2].cpu().numpy(), cmap="viridis")
    axes[0].set_title("Stimulus (middle frame)")
    for ax, (name, out) in zip(axes[1:], results.items()):
        spikes = out["spikes"].reshape(-1, out["spikes"].shape[-1])  # [T, neurons]
        ax.imshow(spikes.T.cpu().numpy(), aspect="auto", cmap="binary")
        ax.set_title(f"{name}: {int(spikes.sum())} spikes")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("neuron")
    fig.tight_layout()
    if save:
        fig.savefig(save)
        print(f"wrote {save}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save", help="Write the figure to this file instead of showing it"
    )
    main(parser.parse_args().save)
