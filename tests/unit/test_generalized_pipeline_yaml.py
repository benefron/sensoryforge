"""Unit tests for generalized pipeline YAML loading.

Tests for ReviewFindings#M2.
"""
import pytest
import tempfile
import os
from pathlib import Path
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline


class TestGeneralizedPipelineYAML:
    """Test suite for YAML loading in GeneralizedTactileEncodingPipeline."""

    @pytest.fixture
    def minimal_yaml_config(self):
        """Create a minimal YAML config file."""
        config_content = """
pipeline:
  device: cpu
  grid_size: 32
  spacing: 0.15
  
neurons:
  sa_neurons: 10
  ra_neurons: 10
"""
        return config_content

    def test_from_yaml_loads_config(self, minimal_yaml_config, tmp_path):
        """Regression test for ReviewFinding#M2.
        
        Verifies that GeneralizedTactileEncodingPipeline.from_yaml()
        correctly loads and parses YAML configuration files.
        
        Reference: reviews/REVIEW_AGENT_FINDINGS_20260211.md#M2
        """
        # Create temporary YAML file
        config_path = tmp_path / "test_config.yml"
        config_path.write_text(minimal_yaml_config)
        
        # Should not raise AttributeError or any other exception
        pipeline = GeneralizedTactileEncodingPipeline.from_yaml(str(config_path))
        
        # Verify pipeline initialized correctly
        assert pipeline is not None
        assert pipeline.grid_manager is not None
        # grid_size can be an int or tuple depending on config
        grid_size = pipeline.grid_manager.grid_size
        if isinstance(grid_size, tuple):
            assert grid_size[0] == 32
        else:
            assert grid_size == 32

    def test_from_yaml_example_from_docs(self, minimal_yaml_config, tmp_path):
        """Test the exact example shown in documentation."""
        config_path = tmp_path / "config.yml"
        config_path.write_text(minimal_yaml_config)
        
        # This is the documented API (from docs/index.md and docs/user_guide/gui_phase2_access.md)
        pipeline = GeneralizedTactileEncodingPipeline.from_yaml(str(config_path))
        
        # Pipeline should be initialized and callable
        assert hasattr(pipeline, 'forward')
        assert callable(pipeline.forward)

    def test_from_config_still_works(self):
        """Verify from_config continues to work as before."""
        config = {
            'pipeline': {'device': 'cpu', 'grid_size': 32},
            'neurons': {'sa_neurons': 10, 'ra_neurons': 10}
        }
        
        pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
        assert pipeline is not None
        # grid_size can be an int or tuple
        grid_size = pipeline.grid_manager.grid_size
        if isinstance(grid_size, tuple):
            assert grid_size[0] == 32
        else:
            assert grid_size == 32
