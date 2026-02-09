"""Regression test for MechanoreceptorModule buffer shape mismatch (ReviewFinding#M7).

``update_parameters()`` was squeezing the kernel from 4D to 2D, breaking
subsequent ``F.conv2d`` calls that expect ``(out_ch, in_ch, kH, kW)``.

Reference: reviews/REVIEW_AGENT_FINDINGS_20260209.md#M7
"""

import pytest
import torch

from sensoryforge.core.mechanoreceptors import MechanoreceptorModule


class TestMechanoreceptorBufferShape:
    """Verify kernel buffer stays 4D after update_parameters()."""

    @pytest.fixture
    def module(self):
        return MechanoreceptorModule(
            sigma_x_mm=0.5, sigma_y_mm=0.5, grid_spacing_mm=0.25
        )

    def test_kernel_4d_after_init(self, module):
        """Kernel should be 4D (1, 1, K, K) after construction."""
        assert module.kernel.dim() == 4
        assert module.kernel.shape[0] == 1
        assert module.kernel.shape[1] == 1

    def test_kernel_4d_after_update(self, module):
        """Kernel should remain 4D after update_parameters()."""
        module.update_parameters(sigma_x_mm=1.0, sigma_y_mm=1.0)
        assert module.kernel.dim() == 4, (
            f"Expected 4D kernel after update, got {module.kernel.dim()}D"
        )

    def test_forward_after_update(self, module):
        """forward() should work after update_parameters()."""
        module.update_parameters(sigma_x_mm=0.8)
        stimulus = torch.randn(1, 10, 10)
        response = module(stimulus)
        assert response.shape == stimulus.shape
