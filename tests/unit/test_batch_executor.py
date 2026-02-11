"""Unit tests for batch execution engine.

Tests cover:
- Parameter sweep expansion (Cartesian product)
- Stimulus ID generation
- Checkpoint save/load
- Batch execution flow
- PyTorch and HDF5 output formats
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import numpy as np

from sensoryforge.core.batch_executor import BatchExecutor


@pytest.fixture
def minimal_batch_config():
    """Minimal valid batch configuration for testing."""
    return {
        'metadata': {
            'batch_name': 'test_batch',
            'description': 'Test batch execution',
        },
        'base_config': {
            'pipeline': {
                'device': 'cpu',
                'seed': 42,
                'grid_size': 20,  # Small for fast tests
            },
            'neurons': {
                'sa_neurons': 5,
                'ra_neurons': 8,
                'sa2_neurons': 3,
                'dt': 0.5,
            },
            'temporal': {
                't_pre': 10,
                't_ramp': 5,
                't_plateau': 50,
                't_post': 10,
                'dt': 0.5,
            },
        },
        'batch': {
            'output_dir': None,  # Will be set to temp dir in tests
            'save_format': 'pytorch',
            'stimuli': [
                {
                    'type': 'gaussian_sweep',
                    'parameters': {
                        'amplitude': [10.0, 20.0],
                        'sigma': [0.5, 1.0],
                    },
                    'repetitions': 2,
                }
            ],
        },
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for batch results."""
    output_dir = tmp_path / 'batch_output'
    output_dir.mkdir()
    return output_dir


class TestBatchExecutor:
    """Test suite for BatchExecutor class."""
    
    def test_initialization_with_valid_config(self, minimal_batch_config, temp_output_dir):
        """Test executor initializes correctly with valid config."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        
        executor = BatchExecutor(minimal_batch_config)
        
        assert executor.config == minimal_batch_config
        assert executor.output_dir == temp_output_dir
        assert isinstance(executor.stimulus_configs, list)
        assert len(executor.stimulus_configs) > 0
    
    def test_initialization_requires_batch_section(self):
        """Test initialization fails without batch section."""
        config = {'base_config': {}}
        
        with pytest.raises(ValueError, match="must include 'batch' section"):
            BatchExecutor(config)
    
    def test_parameter_sweep_expansion(self, minimal_batch_config, temp_output_dir):
        """Test Cartesian product expansion of parameter sweeps."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        
        executor = BatchExecutor(minimal_batch_config)
        
        # Should have 2×2 combinations × 2 repetitions = 8 configs
        assert len(executor.stimulus_configs) == 8
        
        # Check first few configs
        config = executor.stimulus_configs[0]
        assert config['type'] == 'gaussian'
        assert 'amplitude' in config
        assert 'sigma' in config
        assert 'stimulus_id' in config
        assert 'seed' in config
    
    def test_parameter_sweep_with_single_values(self, minimal_batch_config, temp_output_dir):
        """Test sweep expansion with single-value parameters."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        minimal_batch_config['batch']['stimuli'] = [
            {
                'type': 'gaussian_sweep',
                'parameters': {
                    'amplitude': [30.0],  # Single value
                    'sigma': [0.7],       # Single value
                },
                'repetitions': 3,
            }
        ]
        
        executor = BatchExecutor(minimal_batch_config)
        
        # Should have 1×1 combination × 3 repetitions = 3 configs
        assert len(executor.stimulus_configs) == 3
        
        # All should have same params but different seeds
        configs = executor.stimulus_configs
        assert all(c['amplitude'] == 30.0 for c in configs)
        assert all(c['sigma'] == 0.7 for c in configs)
        assert len(set(c['seed'] for c in configs)) == 3  # Different seeds
    
    def test_stimulus_id_generation(self, minimal_batch_config, temp_output_dir):
        """Test unique stimulus ID generation."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        
        executor = BatchExecutor(minimal_batch_config)
        
        # Check all IDs are unique
        ids = [config['stimulus_id'] for config in executor.stimulus_configs]
        assert len(ids) == len(set(ids)), "Stimulus IDs should be unique"
        
        # Check ID format
        id_example = ids[0]
        assert isinstance(id_example, str)
        assert 'gaussian' in id_example
        assert 'rep' in id_example
    
    def test_batch_id_generation(self, minimal_batch_config, temp_output_dir):
        """Test batch ID generation includes name and timestamp."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        
        executor = BatchExecutor(minimal_batch_config)
        
        assert 'test_batch' in executor.batch_id
        # Should contain timestamp elements (digits)
        assert any(c.isdigit() for c in executor.batch_id)
    
    def test_checkpoint_save_and_load(self, minimal_batch_config, temp_output_dir):
        """Test checkpoint persistence for resume capability."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        
        executor = BatchExecutor(minimal_batch_config)
        
        # Save checkpoint
        completed = {0, 1, 2}
        failed = [3]
        current = 4
        
        executor._save_checkpoint(completed, failed, current)
        
        # Check file exists
        assert executor.checkpoint_path.exists()
        
        # Load and verify
        loaded = executor._load_checkpoint(str(executor.checkpoint_path))
        
        assert set(loaded['completed_stimuli']) == completed
        assert loaded['failed_stimuli'] == failed
        assert loaded['current_stimulus'] == current
        assert loaded['batch_id'] == executor.batch_id
    
    @patch('sensoryforge.core.batch_executor.GeneralizedTactileEncodingPipeline')
    def test_execute_single_stimulus(self, mock_pipeline_class, minimal_batch_config, temp_output_dir):
        """Test single stimulus execution."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        
        # Mock pipeline instance
        mock_pipeline = Mock()
        mock_pipeline.forward.return_value = {
            'sa_spikes': torch.zeros(100, 5),
            'ra_spikes': torch.zeros(100, 8),
            'sa2_spikes': torch.zeros(100, 3),
        }
        mock_pipeline_class.from_config.return_value = mock_pipeline
        
        executor = BatchExecutor(minimal_batch_config)
        executor.pipeline = mock_pipeline
        
        # Execute single stimulus
        stim_config = executor.stimulus_configs[0]
        result = executor._execute_single_stimulus(stim_config, save_intermediates=False)
        
        # Verify pipeline was called
        mock_pipeline.forward.assert_called_once()
        
        # Verify result structure
        assert 'sa_spikes' in result
        assert 'ra_spikes' in result
        assert 'sa2_spikes' in result
    
    @patch('sensoryforge.core.batch_executor.GeneralizedTactileEncodingPipeline')
    def test_execute_batch_pytorch_format(self, mock_pipeline_class, minimal_batch_config, temp_output_dir):
        """Test full batch execution with PyTorch output format."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        minimal_batch_config['batch']['save_format'] = 'pytorch'
        
        # Reduce number of stimuli for faster test
        minimal_batch_config['batch']['stimuli'][0]['repetitions'] = 1
        # This gives 2×2×1 = 4 stimuli
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.forward.return_value = {
            'sa_spikes': torch.randint(0, 2, (100, 5)),
            'ra_spikes': torch.randint(0, 2, (100, 8)),
            'sa2_spikes': torch.randint(0, 2, (100, 3)),
        }
        mock_pipeline.get_pipeline_info.return_value = {
            'grid_properties': {'size': 20},
            'neuron_counts': {'sa_neurons': 5, 'ra_neurons': 8, 'sa2_neurons': 3},
        }
        mock_pipeline_class.from_config.return_value = mock_pipeline
        
        executor = BatchExecutor(minimal_batch_config)
        executor.pipeline = mock_pipeline
        
        # Execute batch
        results = executor.execute(save_format='pytorch', save_intermediates=False)
        
        # Verify execution completed
        assert results['num_stimuli'] == 4
        assert results['batch_id'] == executor.batch_id
        assert 'output_path' in results
        assert 'duration_seconds' in results
        
        # Verify output file was created
        output_path = Path(results['output_path'])
        assert output_path.exists()
        assert output_path.suffix == '.pt'
        
        # Verify can load results
        loaded = torch.load(output_path)
        assert 'metadata' in loaded
        assert 'results' in loaded
        assert len(loaded['results']) == 4
    
    @patch('sensoryforge.core.batch_executor.GeneralizedTactileEncodingPipeline')
    def test_execute_batch_hdf5_format(self, mock_pipeline_class, minimal_batch_config, temp_output_dir):
        """Test full batch execution with HDF5 output format."""
        # Skip if h5py not available
        pytest.importorskip('h5py')
        
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        minimal_batch_config['batch']['save_format'] = 'hdf5'
        minimal_batch_config['batch']['stimuli'][0]['repetitions'] = 1
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.forward.return_value = {
            'sa_spikes': torch.randint(0, 2, (100, 5)),
            'ra_spikes': torch.randint(0, 2, (100, 8)),
            'sa2_spikes': torch.randint(0, 2, (100, 3)),
        }
        mock_pipeline.get_pipeline_info.return_value = {
            'grid_properties': {'size': 20, 'spacing': 0.15},
            'neuron_counts': {'sa_neurons': 5, 'ra_neurons': 8, 'sa2_neurons': 3},
        }
        mock_pipeline_class.from_config.return_value = mock_pipeline
        
        executor = BatchExecutor(minimal_batch_config)
        executor.pipeline = mock_pipeline
        
        # Execute batch
        results = executor.execute(save_format='hdf5', save_intermediates=False)
        
        # Verify output file
        output_path = Path(results['output_path'])
        assert output_path.exists()
        assert output_path.suffix == '.h5'
        
        # Verify HDF5 structure
        import h5py
        with h5py.File(output_path, 'r') as f:
            assert 'metadata' in f
            assert 'grid' in f
            assert 'stimuli' in f
            assert 'responses' in f
            
            # Check at least one stimulus
            assert 'stim_0000' in f['responses']
            assert 'sa_spikes' in f['responses/stim_0000']
    
    @patch('sensoryforge.core.batch_executor.GeneralizedTactileEncodingPipeline')
    def test_resume_from_checkpoint(self, mock_pipeline_class, minimal_batch_config, temp_output_dir):
        """Test resuming batch execution from checkpoint."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        minimal_batch_config['batch']['stimuli'][0]['repetitions'] = 1
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.forward.return_value = {
            'sa_spikes': torch.zeros(100, 5),
            'ra_spikes': torch.zeros(100, 8),
            'sa2_spikes': torch.zeros(100, 3),
        }
        mock_pipeline.get_pipeline_info.return_value = {
            'grid_properties': {'size': 20},
            'neuron_counts': {'sa_neurons': 5},
        }
        mock_pipeline_class.from_config.return_value = mock_pipeline
        
        executor = BatchExecutor(minimal_batch_config)
        executor.pipeline = mock_pipeline
        
        # Create a fake checkpoint (2 out of 4 completed)
        checkpoint_data = {
            'batch_id': executor.batch_id,
            'completed_stimuli': [0, 1],
            'failed_stimuli': [],
            'total_stimuli': 4,
            'current_stimulus': 2,
        }
        
        checkpoint_file = temp_output_dir / 'test_checkpoint.json'
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Execute with resume
        results = executor.execute(resume_from=str(checkpoint_file))
        
        # Should have skipped first 2 stimuli
        # Pipeline should be called only 2 times (for stimuli 2 and 3)
        assert mock_pipeline.forward.call_count == 2
    
    def test_metadata_saved(self, minimal_batch_config, temp_output_dir):
        """Test batch metadata is saved to JSON file."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        
        executor = BatchExecutor(minimal_batch_config)
        executor._save_metadata(temp_output_dir)
        
        metadata_file = temp_output_dir / 'batch_metadata.json'
        assert metadata_file.exists()
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['batch_id'] == executor.batch_id
        assert 'config' in metadata
        assert 'num_stimuli' in metadata
    
    def test_stimulus_index_saved(self, minimal_batch_config, temp_output_dir):
        """Test stimulus index is saved for lookup."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        
        executor = BatchExecutor(minimal_batch_config)
        executor._save_stimulus_index(temp_output_dir)
        
        index_file = temp_output_dir / 'stimulus_index.json'
        assert index_file.exists()
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        # Should have entries for all stimuli
        assert len(index) == len(executor.stimulus_configs)
        
        # Check format
        first_key = list(index.keys())[0]
        assert first_key.startswith('stim_')
        assert isinstance(index[first_key], dict)
    
    def test_from_yaml_classmethod(self, minimal_batch_config, temp_output_dir, tmp_path):
        """Test creating executor from YAML file."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        
        # Write config to YAML file
        yaml_file = tmp_path / 'test_config.yml'
        
        # Manual YAML writing (simple dict to YAML)
        import yaml
        with open(yaml_file, 'w') as f:
            yaml.dump(minimal_batch_config, f)
        
        # Create executor from YAML
        executor = BatchExecutor.from_yaml(str(yaml_file))
        
        assert executor.config == minimal_batch_config
        assert len(executor.stimulus_configs) > 0
    
    def test_multiple_sweep_types(self, minimal_batch_config, temp_output_dir):
        """Test batch with multiple different sweep types."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        minimal_batch_config['batch']['stimuli'] = [
            {
                'type': 'gaussian_sweep',
                'parameters': {'amplitude': [10.0], 'sigma': [0.5]},
                'repetitions': 2,
            },
            {
                'type': 'step_sweep',
                'parameters': {'amplitude': [20.0], 'step_time': [50.0]},
                'repetitions': 1,
            },
        ]
        
        executor = BatchExecutor(minimal_batch_config)
        
        # Should have 2 + 1 = 3 total stimuli
        assert len(executor.stimulus_configs) == 3
        
        # Check types
        types = [c['type'] for c in executor.stimulus_configs]
        assert types.count('gaussian') == 2
        assert types.count('step') == 1
    
    def test_seed_reproducibility(self, minimal_batch_config, temp_output_dir):
        """Test that stimulus seeds are deterministic and unique."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        
        executor1 = BatchExecutor(minimal_batch_config)
        executor2 = BatchExecutor(minimal_batch_config)
        
        # Same configuration should generate same seeds
        seeds1 = [c['seed'] for c in executor1.stimulus_configs]
        seeds2 = [c['seed'] for c in executor2.stimulus_configs]
        
        assert seeds1 == seeds2
        
        # All seeds should be unique within a batch
        assert len(set(seeds1)) == len(seeds1)


class TestBatchExecutorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_stimuli_list(self, minimal_batch_config, temp_output_dir):
        """Test handling of empty stimuli list."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        minimal_batch_config['batch']['stimuli'] = []
        
        executor = BatchExecutor(minimal_batch_config)
        
        assert len(executor.stimulus_configs) == 0
    
    def test_zero_repetitions(self, minimal_batch_config, temp_output_dir):
        """Test handling of zero repetitions."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        minimal_batch_config['batch']['stimuli'][0]['repetitions'] = 0
        
        executor = BatchExecutor(minimal_batch_config)
        
        assert len(executor.stimulus_configs) == 0
    
    def test_hdf5_import_error_handling(self, minimal_batch_config, temp_output_dir):
        """Test graceful failure when h5py not installed."""
        minimal_batch_config['batch']['output_dir'] = str(temp_output_dir)
        
        executor = BatchExecutor(minimal_batch_config)
        
        # Mock h5py import to raise ImportError
        import sys
        with patch.dict(sys.modules, {'h5py': None}):
            with pytest.raises(ImportError, match="h5py is required"):
                executor._save_results_hdf5([], temp_output_dir / 'test.h5')
