"""Regression test for missing gabor_texture import (ReviewFinding#M3).

The generalized pipeline used ``gabor_texture`` without importing it,
causing a ``NameError`` at runtime for texture stimuli.

Reference: reviews/REVIEW_AGENT_FINDINGS_20260209.md#M3
"""

import pytest
import torch

from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline


class TestGaborTextureImport:
    """Verify texture stimulus generation no longer raises NameError."""

    def test_gabor_texture_no_name_error(self):
        """_generate_texture_stimulus should not raise NameError."""
        pipeline = GeneralizedTactileEncodingPipeline()
        stim, time_arr, temporal = pipeline.generate_stimulus(
            "texture", pattern="gabor", duration=10.0
        )
        assert stim.dim() == 4
        assert stim.shape[0] == 1

    def test_grating_texture_works(self):
        """Grating texture should also work (already imported inline)."""
        pipeline = GeneralizedTactileEncodingPipeline()
        stim, time_arr, temporal = pipeline.generate_stimulus(
            "texture", pattern="grating", duration=10.0
        )
        assert stim.dim() == 4
