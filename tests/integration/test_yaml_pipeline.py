"""Integration tests for YAML-based pipeline configuration.

Tests the full integration of Phase 2 features (CompositeGrid, DSL neurons,
extended stimuli, adaptive solvers) via YAML configuration and CLI.
"""

import pytest
import torch
import tempfile
from pathlib import Path
import yaml
import sys

from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.cli import main as cli_main, load_config_file, validate_config


class TestYAMLPipelineBasic:
    """Test basic YAML configuration loading and pipeline creation."""

    def test_from_config_with_minimal_config(self):
        """Test pipeline creation from minimal configuration."""
        config = {
            'pipeline': {
                'device': 'cpu',
                'grid_size': 40,
                'spacing': 0.15,
            },
            'neurons': {
                'sa_neurons': 10,
                'ra_neurons': 14,
            }
        }
        
        pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
        
        assert pipeline is not None
        assert pipeline.device == torch.device('cpu')
        info = pipeline.get_pipeline_info()
        assert info['grid_properties']['size'] == (40, 40)

    def test_from_config_with_file_path(self, tmp_path):
        """Test pipeline creation from YAML file."""
        config = {
            'pipeline': {
                'device': 'cpu',
                'grid_size': 30,
            }
        }
        
        config_file = tmp_path / "test_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        pipeline = GeneralizedTactileEncodingPipeline(config_path=str(config_file))
        
        assert pipeline is not None
        info = pipeline.get_pipeline_info()
        assert info['grid_properties']['size'] == (30, 30)

    def test_config_validation_passes_for_valid_config(self):
        """Test validation accepts valid configurations."""
        config = {
            'pipeline': {'device': 'cpu'},
            'neurons': {'sa_neurons': 10},
        }
        
        assert validate_config(config) is True

    def test_config_validation_fails_for_invalid_grid_type(self):
        """Test validation rejects invalid grid types."""
        config = {
            'grid': {
                'type': 'invalid_type'
            }
        }
        
        assert validate_config(config) is False

    def test_config_validation_fails_for_invalid_solver_type(self):
        """Test validation rejects invalid solver types."""
        config = {
            'solver': {
                'type': 'invalid_solver'
            }
        }
        
        assert validate_config(config) is False


class TestYAMLPipelineExecution:
    """Test pipeline execution with different configurations."""

    def test_pipeline_runs_with_default_stimulus(self):
        """Test running pipeline with default stimulus configuration."""
        config = {
            'pipeline': {'device': 'cpu', 'grid_size': 40},
            'neurons': {'sa_neurons': 5, 'ra_neurons': 7, 'dt': 0.5},
            'temporal': {'t_pre': 10, 't_ramp': 5, 't_plateau': 50, 't_post': 10}
        }
        
        pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
        results = pipeline.forward(stimulus_type='trapezoidal', amplitude=20.0)
        
        assert 'sa_spikes' in results
        assert 'ra_spikes' in results
        assert results['sa_spikes'].shape[0] == 1  # Batch dimension

    def test_pipeline_runs_with_gaussian_stimulus(self):
        """Test running pipeline with Gaussian stimulus."""
        config = {
            'pipeline': {'device': 'cpu', 'grid_size': 40},
            'neurons': {'sa_neurons': 5, 'ra_neurons': 7, 'dt': 0.5},
        }
        
        pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
        results = pipeline.forward(
            stimulus_type='gaussian',
            duration=100.0,
            amplitude=20.0,
            sigma=1.0
        )
        
        assert 'sa_spikes' in results
        assert results['stimulus_sequence'].shape[1] == 200  # duration/dt

    def test_pipeline_produces_spikes_with_sufficient_stimulus(self):
        """Test that pipeline generates spikes with strong stimulus."""
        config = {
            'pipeline': {'device': 'cpu', 'grid_size': 40, 'seed': 42},
            'neurons': {'sa_neurons': 10, 'ra_neurons': 14, 'dt': 0.5},
        }
        
        pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
        results = pipeline.forward(
            stimulus_type='gaussian',
            duration=200.0,
            amplitude=50.0,  # Strong stimulus
            sigma=1.5
        )
        
        # Check that some spikes are generated
        sa_spike_count = results['sa_spikes'].sum().item()
        ra_spike_count = results['ra_spikes'].sum().item()
        
        assert sa_spike_count > 0 or ra_spike_count > 0, \
            "Expected at least some spikes with strong stimulus"


class TestCLICommands:
    """Test CLI command functionality."""

    def test_cli_validate_command_accepts_valid_config(self, tmp_path):
        """Test CLI validate command with valid config."""
        config = {
            'pipeline': {'device': 'cpu', 'grid_size': 30},
            'neurons': {'sa_neurons': 5, 'ra_neurons': 7},
        }
        
        config_file = tmp_path / "valid_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Test load_config_file directly
        loaded = load_config_file(str(config_file))
        assert loaded == config

    def test_cli_validate_command_rejects_missing_file(self):
        """Test CLI validate rejects non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_config_file("nonexistent_config.yml")

    def test_cli_list_components_executes(self, capsys):
        """Test list-components command runs without error."""
        # Simulate command line arguments
        sys.argv = ['sensoryforge', 'list-components']
        
        exit_code = cli_main()
        
        assert exit_code == 0
        captured = capsys.readouterr()
        assert 'Available SensoryForge Components' in captured.out
        assert 'Filters:' in captured.out
        assert 'Neuron Models:' in captured.out

    def test_cli_run_command_executes(self, tmp_path):
        """Test run command executes successfully."""
        config = {
            'pipeline': {'device': 'cpu', 'grid_size': 30, 'seed': 42},
            'neurons': {'sa_neurons': 5, 'ra_neurons': 5, 'dt': 0.5},
            'temporal': {'t_pre': 5, 't_ramp': 5, 't_plateau': 20, 't_post': 5}
        }
        
        config_file = tmp_path / "run_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        output_file = tmp_path / "output.pt"
        
        # Simulate command line arguments
        sys.argv = [
            'sensoryforge', 'run',
            str(config_file),
            '--duration', '100',
            '--output', str(output_file)
        ]
        
        exit_code = cli_main()
        
        assert exit_code == 0
        assert output_file.exists()
        
        # Load and verify output
        results = torch.load(output_file)
        assert 'config' in results
        assert 'results' in results
        assert 'sa_spikes' in results['results']

    def test_cli_validate_command_executes(self, tmp_path):
        """Test validate command executes successfully."""
        config = {
            'pipeline': {'device': 'cpu', 'grid_size': 30},
            'neurons': {'sa_neurons': 5, 'ra_neurons': 7},
        }
        
        config_file = tmp_path / "validate_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Simulate command line arguments
        sys.argv = ['sensoryforge', 'validate', str(config_file)]
        
        exit_code = cli_main()
        
        assert exit_code == 0

    def test_cli_visualize_command_executes(self, tmp_path):
        """Test visualize command executes successfully."""
        config = {
            'pipeline': {'device': 'cpu', 'grid_size': 30},
            'neurons': {'sa_neurons': 5, 'ra_neurons': 7},
        }
        
        config_file = tmp_path / "viz_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Simulate command line arguments
        sys.argv = ['sensoryforge', 'visualize', str(config_file)]
        
        exit_code = cli_main()
        
        assert exit_code == 0


class TestYAMLPhase2Features:
    """Test Phase 2 feature configuration (future implementation)."""

    def test_composite_grid_config_structure(self):
        """Test that composite grid config is accepted (implementation pending)."""
        config = {
            'grid': {
                'type': 'composite',
                'shape': [64, 64],
                'populations': {
                    'sa1': {'density': 0.30, 'arrangement': 'poisson'},
                    'ra1': {'density': 0.20, 'arrangement': 'hex'},
                }
            }
        }
        
        # Should pass validation even if not yet implemented
        assert validate_config(config) is True

    def test_dsl_neuron_config_structure(self):
        """Test that DSL neuron config is validated correctly."""
        # Missing required fields should fail
        config_missing_fields = {
            'neurons': {
                'type': 'dsl',
                'equations': 'dv/dt = v + I'
                # Missing threshold, reset, parameters
            }
        }
        
        assert validate_config(config_missing_fields) is False
        
        # Complete DSL config should pass
        config_complete = {
            'neurons': {
                'type': 'dsl',
                'equations': 'dv/dt = v + I',
                'threshold': 'v >= 30',
                'reset': 'v = -65',
                'parameters': {'a': 0.02}
            }
        }
        
        assert validate_config(config_complete) is True

    def test_solver_config_validation(self):
        """Test solver configuration validation."""
        # Euler solver (valid)
        config_euler = {
            'solver': {
                'type': 'euler',
                'dt': 0.001
            }
        }
        assert validate_config(config_euler) is True
        
        # Adaptive solver (valid)
        config_adaptive = {
            'solver': {
                'type': 'adaptive',
                'config': {
                    'method': 'dopri5',
                    'rtol': 1e-5
                }
            }
        }
        assert validate_config(config_adaptive) is True

    def test_extended_stimuli_config_accepted(self):
        """Test that extended stimuli configs are accepted."""
        config = {
            'stimuli': [
                {'type': 'gaussian', 'config': {'sigma': 0.5}},
                {'type': 'texture', 'config': {'pattern': 'gabor'}},
                {'type': 'moving', 'config': {'direction': [1, 0]}}
            ]
        }
        
        # Should not fail validation
        assert validate_config(config) is True


class TestBackwardCompatibility:
    """Ensure backward compatibility with existing pipeline usage."""

    def test_existing_pipeline_usage_still_works(self):
        """Test that existing pipeline instantiation still works."""
        # Old style instantiation
        pipeline = GeneralizedTactileEncodingPipeline()
        
        assert pipeline is not None
        assert pipeline.device == torch.device('cpu')

    def test_config_path_argument_still_works(self, tmp_path):
        """Test that config_path argument still works."""
        config = {
            'pipeline': {'device': 'cpu', 'grid_size': 35}
        }
        
        config_file = tmp_path / "compat_config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Old style with config_path
        pipeline = GeneralizedTactileEncodingPipeline(config_path=str(config_file))
        
        assert pipeline is not None

    def test_config_dict_argument_still_works(self):
        """Test that config_dict argument still works."""
        config = {
            'pipeline': {'device': 'cpu', 'grid_size': 35}
        }
        
        # Old style with config_dict
        pipeline = GeneralizedTactileEncodingPipeline(config_dict=config)
        
        assert pipeline is not None
        info = pipeline.get_pipeline_info()
        assert info['grid_properties']['size'] == (35, 35)
