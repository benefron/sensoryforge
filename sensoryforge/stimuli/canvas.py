"""A regular stimulus-render canvas for any receptor grid config (F-076).

``sensoryforge.core.grid.ReceptorGrid.get_coordinates()`` only returns a
meshgrid for arrangements that actually build one (``"grid"``,
``"jittered_grid"``, ``"blue_noise"`` before jitter is applied) -- it
raises ``ValueError`` for ``"poisson"`` and ``"hex"``, which have no
lattice at all (``core/grid.py`` ~302-316). Before this module existed,
``sensoryforge/cli.py``, ``sensoryforge/core/batch_executor.py`` and the
old GUI's Circuit tab each built a throwaway
:class:`~sensoryforge.core.grid.ReceptorGrid` purely to call
``get_coordinates()`` for a render canvas, so a canonical config using
``"poisson"`` or ``"hex"`` could not run through any of the three
config-driven entry points (F-076) even though
:class:`~sensoryforge.core.simulation_engine.SimulationEngine` already
samples the stimulus at each receptor's real ``(x, y)`` position via
``grid_sample`` (Wave L3, F-010) and does not need the canvas resolution
to equal the receptor count.

:func:`stimulus_canvas` is the one helper all three call sites (and any
future GUI code) use instead. It returns a regular ``rows x cols`` canvas
for *every* arrangement:

* **``"grid"``:** built with :func:`sensoryforge.core.grid.create_grid_torch`
  from ``rows``/``cols``/``spacing``/``center`` -- bit-identical to
  ``ReceptorGrid.get_coordinates()``, so the engine's bit-identical reshape
  fast path (``SimulationEngine._stimulus_to_receptors``) and the golden
  fixtures (``tests/fixtures/rf_engine_golden_weights.pt``) are unaffected.
* **``"poisson"``, ``"hex"``, ``"jittered_grid"``, ``"blue_noise"``,
  ``"composite"``:** the *same* ``rows``/``cols``/``spacing``/``center``
  formula. This is not an approximation: ``ReceptorGrid.__init__`` computes
  its own ``xlim``/``ylim`` for every one of these arrangements with this
  exact formula *before* generating the (possibly irregular) receptor
  positions (``core/grid.py`` ~185-274), and
  ``SimulationEngine._build_grids``'s ``"composite"`` branch computes its
  composite grid's ``xlim``/``ylim`` the identical way. So the canvas
  always spans exactly the extent
  ``SimulationEngine._stimulus_to_receptors``/``_sample_stimulus_at_receptors``
  samples against, for every arrangement that takes ``rows``/``cols``.
* **``coords_file`` set (any arrangement):** ``SimulationEngine._build_grids``
  ignores ``rows``/``cols``/``spacing`` entirely and builds a single-layer
  composite grid from the imported coordinates' own bounding box, padded by
  0.5 mm on a degenerate axis (``_composite_from_coords``). This function
  mirrors that bounding box exactly, and -- since ``rows``/``cols`` are not
  meaningful for imported coordinates -- derives a canvas resolution from
  that extent and ``grid_cfg.spacing``, the same way
  ``ReceptorGrid._generate_poisson``/``_generate_hex`` turn a spacing into a
  point count.

See ``docs/concepts/units_and_shapes.md`` ("Receptor sampling") for how the
engine samples a canvas built this way at arbitrary receptor coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from sensoryforge.config.schema import GridConfig
from sensoryforge.core.grid import create_grid_torch, load_receptor_coords_file


@dataclass
class StimulusCanvas:
    """A regular ``rows x cols`` rendering canvas for a receptor grid.

    Attributes:
        xx: X-coordinate meshgrid ``[rows, cols]`` in mm, built with
            ``indexing="ij"`` like :func:`sensoryforge.core.grid.create_grid_torch`
            (so the frame's first axis is x, second axis is y).
        yy: Y-coordinate meshgrid ``[rows, cols]`` in mm.
        xlim: ``(x_min, x_max)`` mm spanned by ``xx``.
        ylim: ``(y_min, y_max)`` mm spanned by ``yy``.
        shape: ``(rows, cols)`` canvas resolution.
    """

    xx: torch.Tensor
    yy: torch.Tensor
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    shape: Tuple[int, int]


def stimulus_canvas(
    grid_cfg: GridConfig, device: torch.device | str = "cpu"
) -> StimulusCanvas:
    """Build the regular render canvas :class:`GridConfig` ``grid_cfg`` samples onto.

    See the module docstring for the extent/resolution rule used for each
    arrangement. ``"grid"`` reproduces
    :meth:`sensoryforge.core.grid.ReceptorGrid.get_coordinates` bit-for-bit;
    every other arrangement gets a regular canvas spanning the same extent
    :class:`~sensoryforge.core.simulation_engine.SimulationEngine` builds
    that grid with.

    Args:
        grid_cfg: The grid's config entry.
        device: Torch device for the returned tensors.

    Returns:
        A :class:`StimulusCanvas`.
    """
    if grid_cfg.coords_file:
        coords = load_receptor_coords_file(grid_cfg.coords_file, device=device)
        x_min = coords[:, 0].min().item()
        x_max = coords[:, 0].max().item()
        y_min = coords[:, 1].min().item()
        y_max = coords[:, 1].max().item()
        # Mirror SimulationEngine._composite_from_coords exactly: pad a
        # degenerate axis by 0.5 mm so xlim[0] < xlim[1] holds.
        if x_min == x_max:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        if y_min == y_max:
            y_min, y_max = y_min - 0.5, y_max + 0.5

        spacing = (
            grid_cfg.spacing if grid_cfg.spacing and grid_cfg.spacing > 0 else 0.15
        )
        width = x_max - x_min
        height = y_max - y_min
        # rows/cols aren't meaningful for imported coordinates -- derive a
        # resolution from the extent and spacing, the same style
        # ReceptorGrid._generate_poisson/_generate_hex use.
        rows = max(1, int(width / spacing) + 1)
        cols = max(1, int(height / spacing) + 1)

        x = torch.linspace(x_min, x_max, rows, device=device)
        y = torch.linspace(y_min, y_max, cols, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        return StimulusCanvas(
            xx=xx, yy=yy, xlim=(x_min, x_max), ylim=(y_min, y_max), shape=(rows, cols)
        )

    rows = grid_cfg.rows or 40
    cols = grid_cfg.cols or 40
    xx, yy, x, y = create_grid_torch(
        grid_size=(rows, cols),
        spacing=grid_cfg.spacing,
        center=(grid_cfg.center_x, grid_cfg.center_y),
        device=device,
    )
    xlim = (x[0].item(), x[-1].item())
    ylim = (y[0].item(), y[-1].item())
    return StimulusCanvas(xx=xx, yy=yy, xlim=xlim, ylim=ylim, shape=(rows, cols))
