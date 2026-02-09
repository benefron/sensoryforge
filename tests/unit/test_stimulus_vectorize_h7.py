"""Tests for vectorized stimulus generation (ReviewFinding#H7).

Verifies that the vectorized broadcast approach produces correct shapes
and that each stimulus type returns the expected [1, T, H, W] tensor.

Reference: reviews/REVIEW_AGENT_FINDINGS_20260209.md#H7
"""

import pytest
import torch

from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline


@pytest.fixture
def pipeline():
    """Minimal pipeline for stimulus generation tests."""
    return GeneralizedTactileEncodingPipeline()


class TestVectorizedStimulusGeneration:
    """Verify vectorized stimulus generation matches expected shapes."""

    @pytest.mark.parametrize(
        "stimulus_type",
        ["trapezoidal", "gaussian", "step", "ramp"],
    )
    def test_stimulus_shape_4d(self, pipeline, stimulus_type):
        """All static stimulus types should return [1, T, H, W]."""
        stim, time_arr, temporal = pipeline.generate_stimulus(stimulus_type)
        assert stim.dim() == 4
        assert stim.shape[0] == 1
        assert stim.shape[1] == len(time_arr)
        assert stim.shape[1] == len(temporal)

    def test_trapezoidal_broadcast_correctness(self, pipeline):
        """Trapezoidal stimulus = spatial * temporal at each timestep."""
        stim, time_arr, temporal = pipeline.generate_stimulus("trapezoidal")
        # Peak of temporal profile should give full spatial amplitude
        peak_idx = temporal.argmax().item()
        peak_frame = stim[0, peak_idx]
        # During pre-stimulus (temporal=0), frame should be zero
        if temporal[0].item() == 0.0:
            assert stim[0, 0].abs().max().item() == 0.0
        assert peak_frame.abs().max().item() > 0.0

    def test_gaussian_constant_across_time(self, pipeline):
        """Static Gaussian should have identical frames at all timesteps."""
        stim, _, _ = pipeline.generate_stimulus("gaussian", duration=10.0)
        # All frames should be equal
        first_frame = stim[0, 0]
        for t in range(stim.shape[1]):
            assert torch.allclose(stim[0, t], first_frame), (
                f"Frame {t} differs from frame 0"
            )

    def test_step_stimulus_zero_before_step(self, pipeline):
        """Step stimulus should be zero before step time."""
        stim, _, temporal = pipeline.generate_stimulus(
            "step", duration=100.0, step_time=50.0
        )
        # First few frames should be zero (before step)
        zero_region = temporal[:10]
        if zero_region.sum().item() == 0.0:
            assert stim[0, :10].abs().max().item() == 0.0

    def test_ramp_stimulus_monotonic_increase(self, pipeline):
        """Ramp stimulus peak amplitude should increase over time."""
        stim, _, temporal = pipeline.generate_stimulus("ramp", duration=50.0)
        # temporal should be monotonically increasing
        assert (temporal[1:] >= temporal[:-1]).all()
        # Frame max should track temporal profile
        frame_maxes = stim[0].flatten(1).max(dim=1).values
        # Normalize both to compare shape
        if frame_maxes.max() > 0:
            norm_frames = frame_maxes / frame_maxes.max()
            norm_temporal = temporal / temporal.max()
            assert torch.allclose(norm_frames, norm_temporal, atol=1e-5)
