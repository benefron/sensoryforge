"""Unit tests for protocol backend.

Regression tests for ReviewFindings#C1.
"""
import pytest
import torch
from unittest.mock import Mock, MagicMock
from sensoryforge.gui.protocol_backend import ProtocolWorker, ProtocolSpec


class TestProtocolWorker:
    """Test suite for ProtocolWorker."""

    @pytest.fixture
    def minimal_worker_setup(self):
        """Create minimal mocked setup for ProtocolWorker."""
        grid_manager = Mock()
        grid_manager.grid_shape = (10, 10)
        grid_manager.cell_centers = torch.randn(100, 2)
        
        population = Mock()
        population.name = "TestPopulation"
        
        population_configs = {"TestPopulation": {"tau_ms": 10.0}}
        
        protocol_specs = [
            ProtocolSpec(key="test_protocol", title="Test", description="Test protocol")
        ]
        
        device = torch.device("cpu")
        
        return {
            "grid_manager": grid_manager,
            "populations": [population],
            "population_configs": population_configs,
            "protocol_specs": protocol_specs,
            "base_dt_ms": 1.0,
            "device": device,
        }

    def test_worker_initialization_with_debug_enabled(self, minimal_worker_setup):
        """Regression test for ReviewFinding#C1.
        
        Verifies that ProtocolWorker can be instantiated with debug=True
        without raising AttributeError for _perform_fit.
        
        Reference: reviews/REVIEW_AGENT_FINDINGS_20260211.md#C1
        """
        worker = ProtocolWorker(
            **minimal_worker_setup,
            debug=True
        )
        
        # Verify the worker initialized correctly
        assert worker is not None
        assert worker._debug_enabled is True
        assert hasattr(worker, "_perform_fit")
        assert worker._perform_fit is False  # Default value

    def test_worker_initialization_with_perform_fit_true(self, minimal_worker_setup):
        """Test that perform_fit parameter can be set to True."""
        worker = ProtocolWorker(
            **minimal_worker_setup,
            debug=True,
            perform_fit=True
        )
        
        assert worker._perform_fit is True

    def test_worker_debug_call_does_not_crash(self, minimal_worker_setup):
        """Verify that _debug() can access _perform_fit without AttributeError."""
        worker = ProtocolWorker(
            **minimal_worker_setup,
            debug=True
        )
        
        # This should not raise AttributeError
        worker._debug(
            "Test debug message",
            perform_fit=worker._perform_fit
        )
