"""Tests for the regular stimulus-render canvas helper (Phase 0, F-076).

``sensoryforge/stimuli/canvas.py::stimulus_canvas`` replaces the three
``ReceptorGrid(...).get_coordinates()`` render blocks in ``cli.py``,
``batch_executor.py`` and ``gui/circuit/run.py`` -- ``get_coordinates()``
raises ``ValueError`` for ``"poisson"``/``"hex"`` (no lattice), so a config
using either arrangement could not run through any config-driven entry
point (F-076). This module checks the helper reproduces ``ReceptorGrid``'s
own canvas bit-for-bit for ``"grid"``, and spans the true receptor extent
for every other arrangement.
"""

from __future__ import annotations

import pytest
import torch

from sensoryforge.config.schema import GridConfig
from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.stimuli.canvas import StimulusCanvas, stimulus_canvas


def test_grid_arrangement_matches_receptor_grid_exactly():
    """"grid" canvas must equal ReceptorGrid.get_coordinates() bit-for-bit."""
    grid_cfg = GridConfig(
        name="g", arrangement="grid", rows=12, cols=9, spacing=0.2,
        center_x=0.3, center_y=-0.1,
    )
    canvas = stimulus_canvas(grid_cfg)

    ref = ReceptorGrid(
        grid_size=(grid_cfg.rows, grid_cfg.cols),
        spacing=grid_cfg.spacing,
        arrangement="grid",
        center=(grid_cfg.center_x, grid_cfg.center_y),
    )
    ref_xx, ref_yy = ref.get_coordinates()

    assert torch.equal(canvas.xx, ref_xx)
    assert torch.equal(canvas.yy, ref_yy)
    assert canvas.xlim == (ref.xlim[0], ref.xlim[1])
    assert canvas.ylim == (ref.ylim[0], ref.ylim[1])
    assert canvas.shape == (12, 9)


def test_grid_arrangement_default_rows_cols():
    """Missing rows/cols default to 40, matching every call site's fallback."""
    grid_cfg = GridConfig(name="g", arrangement="grid", spacing=0.15)
    canvas = stimulus_canvas(grid_cfg)
    assert canvas.shape == (40, 40)


def test_hex_canvas_covers_receptor_extent_and_expected_shape():
    grid_cfg = GridConfig(
        name="g", arrangement="hex", rows=10, cols=10, spacing=0.15, seed=1,
    )
    canvas = stimulus_canvas(grid_cfg)

    ref = ReceptorGrid(
        grid_size=(grid_cfg.rows, grid_cfg.cols),
        spacing=grid_cfg.spacing,
        arrangement="hex",
        seed=1,
    )
    coords = ref.get_receptor_coordinates()

    assert canvas.xlim[0] <= coords[:, 0].min().item()
    assert canvas.xlim[1] >= coords[:, 0].max().item()
    assert canvas.ylim[0] <= coords[:, 1].min().item()
    assert canvas.ylim[1] >= coords[:, 1].max().item()
    assert canvas.shape == (10, 10)
    # hex's own bounds formula matches the canvas to float32 precision
    # (both derived from rows/cols/spacing/center, before the hex lattice
    # generation clips points to the bounding box) -- ReceptorGrid's own
    # xlim/ylim for "hex" is plain Python (float64) arithmetic, while the
    # canvas's is a float32 linspace, so they agree to ~1e-6, not bit-for-bit.
    assert canvas.xlim[0] == pytest.approx(ref.xlim[0], abs=1e-5)
    assert canvas.xlim[1] == pytest.approx(ref.xlim[1], abs=1e-5)
    assert canvas.ylim[0] == pytest.approx(ref.ylim[0], abs=1e-5)
    assert canvas.ylim[1] == pytest.approx(ref.ylim[1], abs=1e-5)


def test_poisson_canvas_covers_receptor_extent_seeded():
    grid_cfg = GridConfig(
        name="g", arrangement="poisson", rows=10, cols=10, spacing=0.15, seed=1,
    )
    canvas = stimulus_canvas(grid_cfg)

    ref = ReceptorGrid(
        grid_size=(grid_cfg.rows, grid_cfg.cols),
        spacing=grid_cfg.spacing,
        arrangement="poisson",
        seed=1,
    )
    coords = ref.get_receptor_coordinates()

    assert canvas.xlim[0] <= coords[:, 0].min().item()
    assert canvas.xlim[1] >= coords[:, 0].max().item()
    assert canvas.ylim[0] <= coords[:, 1].min().item()
    assert canvas.ylim[1] >= coords[:, 1].max().item()
    assert canvas.shape == (10, 10)
    assert canvas.xlim[0] == pytest.approx(ref.xlim[0], abs=1e-5)
    assert canvas.xlim[1] == pytest.approx(ref.xlim[1], abs=1e-5)
    assert canvas.ylim[0] == pytest.approx(ref.ylim[0], abs=1e-5)
    assert canvas.ylim[1] == pytest.approx(ref.ylim[1], abs=1e-5)


def test_composite_arrangement_uses_rows_cols_spacing_formula():
    """SimulationEngine's own "composite" xlim/ylim is a rows/cols/spacing
    formula identical to "grid", not a bounding box of the layers -- the
    canvas must match that, not the layers' actual extent."""
    grid_cfg = GridConfig(
        name="g",
        arrangement="composite",
        rows=8,
        cols=6,
        spacing=0.25,
        center_x=1.0,
        center_y=-2.0,
        layers=[{"name": "l1", "density": 20.0}],
    )
    canvas = stimulus_canvas(grid_cfg)

    total_x = (8 - 1) * 0.25
    total_y = (6 - 1) * 0.25
    assert canvas.xlim[0] == pytest.approx(1.0 - total_x / 2, abs=1e-5)
    assert canvas.xlim[1] == pytest.approx(1.0 + total_x / 2, abs=1e-5)
    assert canvas.ylim[0] == pytest.approx(-2.0 - total_y / 2, abs=1e-5)
    assert canvas.ylim[1] == pytest.approx(-2.0 + total_y / 2, abs=1e-5)
    assert canvas.shape == (8, 6)


def test_coords_file_canvas_spans_bounding_box(tmp_path):
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.5], [0.5, 2.0]])
    coords_path = tmp_path / "coords.pt"
    torch.save(coords, coords_path)

    grid_cfg = GridConfig(
        name="g", arrangement="grid", spacing=0.15, coords_file=str(coords_path)
    )
    canvas = stimulus_canvas(grid_cfg)

    assert canvas.xlim[0] <= 0.0 and canvas.xlim[1] >= 1.0
    assert canvas.ylim[0] <= 0.0 and canvas.ylim[1] >= 2.0
    rows, cols = canvas.shape
    assert rows >= 1 and cols >= 1
    assert canvas.xx.shape == (rows, cols)
    assert canvas.yy.shape == (rows, cols)


def test_returns_stimulus_canvas_dataclass():
    grid_cfg = GridConfig(name="g", arrangement="grid", rows=4, cols=4, spacing=0.1)
    canvas = stimulus_canvas(grid_cfg)
    assert isinstance(canvas, StimulusCanvas)


def test_hex_canvas_sampling_recovers_analytic_gaussian():
    """Render an analytic 2-D Gaussian on a hex grid's canvas, sample it at
    the receptor coordinates the same way SimulationEngine does, and check
    the result against the closed-form Gaussian at those coordinates
    (RMS error < 2% of the peak, per the task brief's analytic check)."""
    grid_cfg = GridConfig(
        name="g", arrangement="hex", rows=24, cols=24, spacing=0.1, seed=3
    )
    canvas = stimulus_canvas(grid_cfg)

    sigma = 0.5
    x0, y0 = 0.1, -0.2
    x = torch.linspace(canvas.xlim[0], canvas.xlim[1], canvas.shape[0], dtype=torch.float64)
    y = torch.linspace(canvas.ylim[0], canvas.ylim[1], canvas.shape[1], dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    frame = torch.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2)).to(
        torch.float32
    )

    ref = ReceptorGrid(
        grid_size=(grid_cfg.rows, grid_cfg.cols),
        spacing=grid_cfg.spacing,
        arrangement="hex",
        seed=3,
    )
    coords = ref.get_receptor_coordinates()
    # Keep receptors well inside the canvas bounds so edge padding
    # (padding_mode="zeros") doesn't contaminate the analytic comparison.
    margin = 0.2
    inside = (
        (coords[:, 0] > canvas.xlim[0] + margin)
        & (coords[:, 0] < canvas.xlim[1] - margin)
        & (coords[:, 1] > canvas.ylim[0] + margin)
        & (coords[:, 1] < canvas.ylim[1] - margin)
    )
    coords = coords[inside]

    frames = frame.unsqueeze(0).unsqueeze(0)  # [1, 1, rows, cols]
    sampled = SimulationEngine._sample_stimulus_at_receptors(
        frames, coords, canvas.xlim, canvas.ylim
    )[0, 0]

    expected = torch.exp(
        -((coords[:, 0] - x0) ** 2 + (coords[:, 1] - y0) ** 2) / (2 * sigma**2)
    )
    rms = torch.sqrt(torch.mean((sampled - expected) ** 2)).item()
    peak = 1.0  # the Gaussian's own peak amplitude
    assert rms < 0.02 * peak, f"RMS error {rms} >= 2% of peak"
