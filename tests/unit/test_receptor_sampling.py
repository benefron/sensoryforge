"""Real receptor sampling (Phase 2, Wave L3, F-010).

Before this wave, ``SimulationEngine`` fed the stimulus to the receptive-
field bank with a bare row-major reshape, assuming receptor index k equals
stimulus pixel index k. That is only true for a regular ``"grid"``
arrangement whose resolution matches the stimulus frame. Everywhere else
(hex, Poisson, jittered, blue-noise, imported or composite coordinates) it
produced a wrong answer with no warning (Fact L-a).

``SimulationEngine._sample_stimulus_at_receptors`` fixes this by sampling
the stimulus at each receptor's own ``(x, y)`` position with
``torch.nn.functional.grid_sample``. The axis mapping between
SensoryForge's ``(x, y)`` convention (first frame axis is x) and
``grid_sample``'s ``(width, height)`` convention is a swap, not the
identity -- see the method's docstring. A symmetric test stimulus cannot
detect a swapped implementation (it would sample the transposed location,
which for a symmetric field has the same value); every geometry test here
therefore uses a deliberately asymmetric Gaussian (different sigma in x
and y, off-centre), independently-computed receptor positions and an
independently-computed analytic reference.
"""

from __future__ import annotations

import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
)
from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.core.simulation_engine import SimulationEngine


def _asymmetric_gaussian_frame(
    h: int,
    w: int,
    xlim,
    ylim,
    sigma_x: float,
    sigma_y: float,
    x0: float,
    y0: float,
) -> torch.Tensor:
    """Build a [H, W] frame of an off-centre, anisotropic Gaussian.

    Uses SensoryForge's own meshgrid convention (``indexing="ij"``, frame
    axis 0 is x, frame axis 1 is y) so the frame is exactly what
    ``ReceptorGrid``/the engine would build.
    """
    x = torch.linspace(xlim[0], xlim[1], h, dtype=torch.float64)
    y = torch.linspace(ylim[0], ylim[1], w, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    frame = torch.exp(
        -((xx - x0) ** 2 / (2 * sigma_x**2) + (yy - y0) ** 2 / (2 * sigma_y**2))
    )
    return frame.to(torch.float32)


def _analytic_gaussian(coords, sigma_x, sigma_y, x0, y0) -> torch.Tensor:
    rx, ry = coords[:, 0], coords[:, 1]
    return torch.exp(
        -((rx - x0) ** 2 / (2 * sigma_x**2) + (ry - y0) ** 2 / (2 * sigma_y**2))
    )


class TestAsymmetricGaussianRecovery:
    """The heart of L3: prove the index algebra with an asymmetric field."""

    def _setup(self):
        xlim, ylim = (-2.0, 2.0), (-1.6, 1.6)
        h, w = 121, 97  # deliberately unequal -- a transposed sampler
        # would either mis-shape or silently sample the wrong axis.
        sigma_x, sigma_y = 0.30, 0.65
        x0, y0 = 0.5, -0.35
        frame = _asymmetric_gaussian_frame(h, w, xlim, ylim, sigma_x, sigma_y, x0, y0)
        return xlim, ylim, sigma_x, sigma_y, x0, y0, frame

    def test_scattered_receptors_recover_the_field_to_1pct_rms(self):
        xlim, ylim, sigma_x, sigma_y, x0, y0, frame = self._setup()

        gen = torch.Generator().manual_seed(0)
        margin = 0.15  # stay well inside bounds -- edge behaviour is tested
        # separately.
        n = 500
        rx = (
            torch.rand(n, generator=gen) * (xlim[1] - xlim[0] - 2 * margin)
            + xlim[0]
            + margin
        )
        ry = (
            torch.rand(n, generator=gen) * (ylim[1] - ylim[0] - 2 * margin)
            + ylim[0]
            + margin
        )
        coords = torch.stack([rx, ry], dim=1)

        frames = frame.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        sampled = SimulationEngine._sample_stimulus_at_receptors(
            frames, coords, xlim, ylim
        )
        assert sampled.shape == (1, 1, n)
        sampled = sampled[0, 0]

        expected = _analytic_gaussian(coords, sigma_x, sigma_y, x0, y0)
        rms = torch.sqrt(torch.mean((sampled - expected) ** 2)).item()
        assert rms < 0.01, f"RMS error {rms} >= 1% -- axis mapping is likely wrong"

    def test_naively_swapped_axes_fail_this_same_check(self):
        """Negative control: confirms the test above actually discriminates.

        Manually samples with the (x, y) -> (grid_x, grid_y) identity
        mapping instead of the swap the docstring proves is correct, and
        shows it produces a much larger error against the same analytic
        reference -- so the passing test above is not vacuous.
        """
        import torch.nn.functional as F

        xlim, ylim, sigma_x, sigma_y, x0, y0, frame = self._setup()
        gen = torch.Generator().manual_seed(0)
        margin = 0.15
        n = 500
        rx = (
            torch.rand(n, generator=gen) * (xlim[1] - xlim[0] - 2 * margin)
            + xlim[0]
            + margin
        )
        ry = (
            torch.rand(n, generator=gen) * (ylim[1] - ylim[0] - 2 * margin)
            + ylim[0]
            + margin
        )
        coords = torch.stack([rx, ry], dim=1)

        def _norm(v, lo, hi):
            return 2.0 * (v - lo) / (hi - lo) - 1.0

        norm_x = _norm(rx, xlim[0], xlim[1])
        norm_y = _norm(ry, ylim[0], ylim[1])
        # The naive (wrong) mapping: grid[..., 0] = x, grid[..., 1] = y.
        wrong_grid = torch.stack([norm_x, norm_y], dim=-1).view(1, 1, n, 2)
        frames = frame.unsqueeze(0).unsqueeze(0)
        wrong = F.grid_sample(
            frames,
            wrong_grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        )[0, 0, 0]

        expected = _analytic_gaussian(coords, sigma_x, sigma_y, x0, y0)
        rms = torch.sqrt(torch.mean((wrong - expected) ** 2)).item()
        assert rms > 0.05, (
            "the naive swapped mapping should fail the 1% RMS bar -- if it "
            "doesn't, this test stimulus is not asymmetric enough to "
            "discriminate the axis order"
        )


class TestOutOfBoundsReceptor:
    def test_receptor_outside_bounds_samples_zero(self):
        xlim, ylim = (-1.0, 1.0), (-1.0, 1.0)
        frame = torch.ones(11, 11)  # constant nonzero field
        frames = frame.unsqueeze(0).unsqueeze(0)
        coords = torch.tensor([[0.0, 0.0], [5.0, 5.0], [-5.0, 0.3]])
        sampled = SimulationEngine._sample_stimulus_at_receptors(
            frames, coords, xlim, ylim
        )[0, 0]
        assert sampled[0].item() == 1.0  # interior receptor: unclamped field
        assert sampled[1].item() == 0.0  # outside both axes
        assert sampled[2].item() == 0.0  # outside x only


class TestFastPathBitIdentical:
    """The regular-grid reshape path must stay exactly what it was."""

    def test_fast_path_matches_reshape_bit_for_bit(self):
        rows, cols = 9, 7
        grid = ReceptorGrid(
            grid_size=(rows, cols), spacing=0.15, arrangement="grid", device="cpu"
        )
        receptor_coords = grid.get_receptor_coordinates()
        torch.manual_seed(0)
        frame = torch.randn(1, 3, rows, cols)  # [batch, time, H, W]

        expected = frame.reshape(1, 3, rows * cols)

        engine_like = SimulationEngine.__new__(SimulationEngine)
        actual = engine_like._stimulus_to_receptors(frame, grid, receptor_coords, grid)
        assert torch.equal(actual, expected)

    def test_non_grid_arrangement_or_mismatched_resolution_takes_slow_path(self):
        rows, cols = 9, 7
        grid = ReceptorGrid(
            grid_size=(rows, cols), spacing=0.15, arrangement="grid", device="cpu"
        )
        receptor_coords = grid.get_receptor_coordinates()
        torch.manual_seed(0)
        # A resolution mismatch means the fast reshape is not valid; the
        # slow (sampling) path must still produce the receptor count, not
        # the mismatched pixel count.
        frame = torch.randn(1, 3, rows + 1, cols)
        engine_like = SimulationEngine.__new__(SimulationEngine)
        actual = engine_like._stimulus_to_receptors(frame, grid, receptor_coords, grid)
        assert actual.shape == (1, 3, rows * cols)


class TestEngineEndToEnd:
    """The full path through SimulationEngine.run for a hex grid."""

    def test_hex_arrangement_runs_and_recovers_a_pixel(self):
        cfg = SensoryForgeConfig(
            grids=[
                GridConfig(name="g", arrangement="hex", rows=10, cols=10, spacing=0.2)
            ],
            populations=[
                PopulationConfig(
                    name="SA",
                    neurons_per_row=2,
                    seed=3,
                    filter_method="none",
                    innervation_method="one_to_one",
                )
            ],
        )
        engine = SimulationEngine(cfg)
        grid = engine.grids[0]
        xlim, ylim = grid.xlim, grid.ylim
        h, w = 81, 81
        frame = _asymmetric_gaussian_frame(
            h, w, xlim, ylim, sigma_x=0.4, sigma_y=0.2, x0=0.1, y0=-0.2
        )
        out = engine.run(frame.unsqueeze(0), return_intermediates=True)["SA"]
        assert out["drive"].shape[-1] == engine.populations[0]["bank"].num_neurons
        # Sanity: drive isn't identically zero or NaN.
        assert torch.isfinite(out["drive"]).all()
        assert out["drive"].abs().sum() > 0
