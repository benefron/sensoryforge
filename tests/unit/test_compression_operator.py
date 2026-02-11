"""Unit tests for compression operator.

Tests for ReviewFindings#H1 and ReviewFindings#T1.
"""
import pytest
import torch
from sensoryforge.core.compression import CompressionOperator, build_compression_operator
from unittest.mock import Mock


class TestCompressionOperator:
    """Test suite for CompressionOperator."""

    @pytest.fixture
    def grid_shaped_weights(self):
        """Create grid-shaped innervation weights [num_neurons, H, W]."""
        sa_weights = torch.randn(50, 10, 10)  # 50 SA neurons, 10x10 grid
        ra_weights = torch.randn(50, 10, 10)  # 50 RA neurons, 10x10 grid
        return sa_weights, ra_weights

    @pytest.fixture
    def flat_weights(self):
        """Create flat innervation weights [num_neurons, num_receptors]."""
        sa_weights = torch.randn(50, 100)  # 50 SA neurons, 100 receptors
        ra_weights = torch.randn(50, 100)  # 50 RA neurons, 100 receptors
        return sa_weights, ra_weights

    def test_grid_shaped_innervation_backward_compatibility(self, grid_shaped_weights):
        """Test standard grid-shaped weights work as before.
        
        Ensures backward compatibility with existing grid-based pipelines.
        """
        sa_weights, ra_weights = grid_shaped_weights
        
        operator = CompressionOperator(
            sa_weights=sa_weights,
            ra_weights=ra_weights,
            grid_shape=(10, 10),
            _num_receptors=None,
        )
        
        assert operator.num_receptors == 100
        assert operator.grid_shape == (10, 10)
        assert operator.compression_ratio("sa") == 0.5
        assert operator.compression_ratio("ra") == 0.5
        assert operator.compression_ratio("combined") == 1.0

    def test_flat_innervation_num_receptors(self, flat_weights):
        """Regression test for ReviewFinding#H1.
        
        Verifies that flat innervation weights [num_neurons, num_receptors]
        are handled correctly and num_receptors is computed without IndexError.
        
        Reference: reviews/REVIEW_AGENT_FINDINGS_20260211.md#H1
        """
        sa_weights, ra_weights = flat_weights
        
        operator = CompressionOperator(
            sa_weights=sa_weights,
            ra_weights=ra_weights,
            grid_shape=None,
            _num_receptors=100,
        )
        
        # Should not raise IndexError
        assert operator.num_receptors == 100
        assert operator.grid_shape is None

    def test_flat_innervation_compression_ratio(self, flat_weights):
        """Test compression ratio calculation for flat innervation.
        
        Reference: reviews/REVIEW_AGENT_FINDINGS_20260211.md#H1
        """
        sa_weights, ra_weights = flat_weights
        
        operator = CompressionOperator(
            sa_weights=sa_weights,
            ra_weights=ra_weights,
            grid_shape=None,
            _num_receptors=100,
        )
        
        assert operator.compression_ratio("sa") == 0.5
        assert operator.compression_ratio("ra") == 0.5
        assert operator.compression_ratio("combined") == 1.0

    def test_build_compression_operator_with_grid_weights(self, grid_shaped_weights):
        """Test build_compression_operator with grid-shaped weights."""
        sa_weights, ra_weights = grid_shaped_weights
        
        # Mock pipeline
        pipeline = Mock()
        pipeline.sa_innervation.innervation_weights = sa_weights
        pipeline.ra_innervation.innervation_weights = ra_weights
        
        operator = build_compression_operator(pipeline)
        
        assert operator.grid_shape == (10, 10)
        assert operator.num_receptors == 100
        assert operator._num_receptors is None  # Uses grid_shape

    def test_build_compression_operator_with_flat_weights(self, flat_weights):
        """Test build_compression_operator detects flat innervation.
        
        Reference: reviews/REVIEW_AGENT_FINDINGS_20260211.md#H1
        """
        sa_weights, ra_weights = flat_weights
        
        # Mock pipeline with flat innervation
        pipeline = Mock()
        pipeline.sa_innervation.innervation_weights = sa_weights
        pipeline.ra_innervation.innervation_weights = ra_weights
        
        operator = build_compression_operator(pipeline)
        
        assert operator.grid_shape is None
        assert operator._num_receptors == 100
        assert operator.num_receptors == 100

    def test_project_with_flat_weights(self, flat_weights):
        """Test projection works with flat innervation."""
        sa_weights, ra_weights = flat_weights
        
        operator = CompressionOperator(
            sa_weights=sa_weights,
            ra_weights=ra_weights,
            grid_shape=None,
            _num_receptors=100,
        )
        
        # Create stimulus with correct spatial dimensions to match receptor count
        # For 100 receptors, use 10x10 spatial grid
        stimulus = torch.randn(2, 10, 10)  # [batch=2, H=10, W=10]
        
        # Project should work without error
        projected = operator.project(stimulus, population="sa")
        assert projected.shape == (2, 50)  # [batch, num_sa_neurons]

    def test_to_device_preserves_num_receptors(self, flat_weights):
        """Test that to(device) preserves _num_receptors for flat innervation."""
        sa_weights, ra_weights = flat_weights
        
        operator = CompressionOperator(
            sa_weights=sa_weights,
            ra_weights=ra_weights,
            grid_shape=None,
            _num_receptors=100,
        )
        
        # Move to same device (cpu)
        operator_moved = operator.to("cpu")
        
        assert operator_moved._num_receptors == 100
        assert operator_moved.grid_shape is None
        assert operator_moved.num_receptors == 100

    def test_combined_weights_shape_flat(self, flat_weights):
        """Test combined_weights property with flat innervation."""
        sa_weights, ra_weights = flat_weights
        
        operator = CompressionOperator(
            sa_weights=sa_weights,
            ra_weights=ra_weights,
            grid_shape=None,
            _num_receptors=100,
        )
        
        combined = operator.combined_weights
        assert combined.shape == (100, 100)  # [50 SA + 50 RA, 100 receptors]

    def test_num_receptors_raises_when_both_none(self):
        """Test that num_receptors raises when both grid_shape and _num_receptors are None."""
        sa_weights = torch.randn(10, 100)
        ra_weights = torch.randn(10, 100)
        
        operator = CompressionOperator(
            sa_weights=sa_weights,
            ra_weights=ra_weights,
            grid_shape=None,
            _num_receptors=None,
        )
        
        with pytest.raises(ValueError, match="Cannot determine num_receptors"):
            _ = operator.num_receptors
