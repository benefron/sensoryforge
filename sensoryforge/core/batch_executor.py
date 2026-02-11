"""Batch execution engine for running stimulus batteries.

This module provides infrastructure for executing large-scale stimulus sweeps,
generating datasets suitable for machine learning training and neuromorphic 
hardware evaluation.

The batch executor supports:
- Parameter sweep expansion (Cartesian product)
- Progress tracking with checkpointing
- Structured data persistence (HDF5/PyTorch)
- Reproducibility via deterministic seeding
- Resume from interruption

Example:
    >>> config = {
    ...     'base_config': {'pipeline': {'device': 'cpu'}},
    ...     'batch': {
    ...         'output_dir': './results',
    ...         'stimuli': [{
    ...             'type': 'gaussian_sweep',
    ...             'parameters': {
    ...                 'amplitude': [10, 20, 30],
    ...                 'sigma': [0.5, 1.0]
    ...             },
    ...             'repetitions': 2
    ...         }]
    ...     }
    ... }
    >>> executor = BatchExecutor(config)
    >>> results = executor.execute()
"""

import json
import itertools
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib

import torch
import numpy as np

from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline


class BatchExecutor:
    """Orchestrates batch execution of stimulus configurations.
    
    Expands parameter sweeps into individual stimulus configurations,
    executes them through the tactile encoding pipeline, and persists
    results in a structured format.
    
    Attributes:
        config: Full batch configuration dictionary
        base_config: Base pipeline configuration applied to all runs
        batch_config: Batch-specific settings (output_dir, save_format, etc.)
        pipeline: Initialized GeneralizedTactileEncodingPipeline instance
        output_dir: Path to output directory
        checkpoint_path: Path to checkpoint file for resume capability
    
    Example:
        >>> config = load_yaml('batch_config.yml')
        >>> executor = BatchExecutor(config)
        >>> results = executor.execute(save_format='hdf5')
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize batch executor from configuration.
        
        Args:
            config: Batch configuration dictionary with keys:
                - 'base_config': Base pipeline settings
                - 'batch': Batch execution settings
                - 'metadata': Optional batch metadata
        
        Raises:
            ValueError: If required configuration keys are missing
        """
        self.config = config
        self.base_config = config.get('base_config', {})
        self.batch_config = config.get('batch', {})
        
        if not self.batch_config:
            raise ValueError("Configuration must include 'batch' section")
        
        # Initialize pipeline with base configuration
        self.pipeline = GeneralizedTactileEncodingPipeline.from_config(
            self.base_config
        )
        
        # Setup output directory
        output_dir = self.batch_config.get('output_dir', './batch_results')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file for resume capability
        self.checkpoint_path = self.output_dir / 'checkpoint.json'
        
        # Expand stimulus configurations
        self.stimulus_configs = self._expand_all_sweeps()
        
        # Metadata
        self.metadata = config.get('metadata', {})
        self.batch_id = self._generate_batch_id()
    
    def _generate_batch_id(self) -> str:
        """Generate unique batch identifier.
        
        Returns:
            Batch ID string in format: {name}_{timestamp}
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_name = self.metadata.get('batch_name', 'batch')
        # Sanitize name (remove special characters)
        safe_name = ''.join(c if c.isalnum() or c in '_-' else '_' 
                           for c in batch_name)
        return f"{safe_name}_{timestamp}"
    
    def _expand_all_sweeps(self) -> List[Dict[str, Any]]:
        """Expand all stimulus sweep configurations.
        
        Processes the 'batch.stimuli' list, expanding each sweep specification
        into individual stimulus configurations via Cartesian product.
        
        Returns:
            List of individual stimulus configuration dictionaries
        
        Example:
            Input sweep:
                {'type': 'gaussian_sweep', 
                 'parameters': {'amplitude': [10, 20], 'sigma': [0.5]},
                 'repetitions': 2}
            
            Expands to 4 configs:
                [{'type': 'gaussian', 'amplitude': 10, 'sigma': 0.5, 'rep': 0},
                 {'type': 'gaussian', 'amplitude': 10, 'sigma': 0.5, 'rep': 1},
                 {'type': 'gaussian', 'amplitude': 20, 'sigma': 0.5, 'rep': 0},
                 {'type': 'gaussian', 'amplitude': 20, 'sigma': 0.5, 'rep': 1}]
        """
        all_configs = []
        stimuli_specs = self.batch_config.get('stimuli', [])
        
        for spec in stimuli_specs:
            sweep_configs = self._expand_parameter_sweep(spec)
            all_configs.extend(sweep_configs)
        
        return all_configs
    
    def _expand_parameter_sweep(
        self, 
        sweep_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Expand a single parameter sweep into individual configurations.
        
        Args:
            sweep_spec: Sweep specification with keys:
                - 'type': Stimulus type (e.g., 'gaussian_sweep')
                - 'parameters': Dict of parameter_name → list_of_values
                - 'repetitions': Number of repetitions per combination
                - 'base_seed': Optional base seed for reproducibility
        
        Returns:
            List of individual stimulus configurations
        """
        stimulus_type = sweep_spec['type']
        # Remove '_sweep' suffix if present to get base stimulus type
        base_type = stimulus_type.replace('_sweep', '')
        
        parameters = sweep_spec.get('parameters', {})
        repetitions = sweep_spec.get('repetitions', 1)
        base_seed = sweep_spec.get('base_seed', 42)
        
        # Extract parameter names and value lists
        param_names = list(parameters.keys())
        param_values = [parameters[name] if isinstance(parameters[name], list) 
                       else [parameters[name]] 
                       for name in param_names]
        
        # Generate Cartesian product
        combinations = list(itertools.product(*param_values))
        
        # Create individual configs with repetitions
        configs = []
        for combo_idx, combo in enumerate(combinations):
            for rep in range(repetitions):
                config = {
                    'type': base_type,
                    'stimulus_id': None,  # Will be generated
                    'combo_idx': combo_idx,
                    'rep_idx': rep,
                    'seed': base_seed + combo_idx * 10000 + rep,
                }
                
                # Add parameter values
                for param_name, param_value in zip(param_names, combo):
                    config[param_name] = param_value
                
                # Generate unique stimulus ID
                config['stimulus_id'] = self._generate_stimulus_id(config)
                
                configs.append(config)
        
        return configs
    
    def _generate_stimulus_id(self, config: Dict[str, Any]) -> str:
        """Generate unique identifier for a stimulus configuration.
        
        Args:
            config: Stimulus configuration dictionary
        
        Returns:
            Unique stimulus ID string
        
        Example:
            >>> config = {'type': 'gaussian', 'amplitude': 30, 'sigma': 0.5, 
            ...          'rep_idx': 0}
            >>> _generate_stimulus_id(config)
            'gaussian_a30.0_s0.5_rep0'
        """
        stim_type = config['type']
        rep_idx = config.get('rep_idx', 0)
        
        # Create parameter hash from relevant parameters
        param_parts = []
        for key, value in sorted(config.items()):
            if key not in ['type', 'stimulus_id', 'combo_idx', 'rep_idx', 'seed']:
                # Format value based on type
                if isinstance(value, float):
                    param_parts.append(f"{key[0]}{value:.1f}")
                elif isinstance(value, int):
                    param_parts.append(f"{key[0]}{value}")
                else:
                    param_parts.append(f"{key[0]}{value}")
        
        param_str = '_'.join(param_parts) if param_parts else 'default'
        return f"{stim_type}_{param_str}_rep{rep_idx}"
    
    def execute(
        self,
        save_format: str = 'pytorch',
        save_intermediates: bool = False,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute full batch with progress tracking.
        
        Args:
            save_format: Output format - 'pytorch' (default) or 'hdf5'
            save_intermediates: If True, save filtered currents and voltages
            resume_from: Optional checkpoint file path to resume from
        
        Returns:
            Dictionary with batch results and metadata:
                - 'batch_id': Unique batch identifier
                - 'num_stimuli': Total number of stimuli executed
                - 'output_path': Path to saved results file
                - 'duration_seconds': Total execution time
                - 'failed_stimuli': List of failed stimulus indices
        
        Example:
            >>> executor = BatchExecutor(config)
            >>> results = executor.execute(save_format='pytorch')
            >>> print(f"Saved to {results['output_path']}")
        """
        print(f"Starting batch execution: {self.batch_id}")
        print(f"Total stimuli: {len(self.stimulus_configs)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Save format: {save_format}")
        
        start_time = time.time()
        
        # Load checkpoint if resuming
        completed_indices = set()
        failed_indices = []
        
        if resume_from:
            checkpoint_data = self._load_checkpoint(resume_from)
            completed_indices = set(checkpoint_data['completed_stimuli'])
            failed_indices = checkpoint_data.get('failed_stimuli', [])
            print(f"Resuming from checkpoint: {len(completed_indices)} completed")
        
        # Determine output file path
        if save_format == 'hdf5':
            output_file = self.output_dir / f'{self.batch_id}.h5'
        else:  # pytorch
            output_file = self.output_dir / f'{self.batch_id}.pt'
        
        # Execute stimuli
        all_results = []
        
        for idx, stim_config in enumerate(self.stimulus_configs):
            # Skip if already completed
            if idx in completed_indices:
                print(f"[{idx+1}/{len(self.stimulus_configs)}] Skipping "
                      f"{stim_config['stimulus_id']} (already completed)")
                continue
            
            try:
                print(f"[{idx+1}/{len(self.stimulus_configs)}] Executing "
                      f"{stim_config['stimulus_id']}...")
                
                # Execute stimulus through pipeline
                result = self._execute_single_stimulus(
                    stim_config, 
                    save_intermediates
                )
                
                # Store result
                result['stimulus_config'] = stim_config
                result['stimulus_index'] = idx
                all_results.append(result)
                
                # Update checkpoint
                completed_indices.add(idx)
                self._save_checkpoint(
                    completed_indices, 
                    failed_indices, 
                    idx + 1
                )
                
            except Exception as e:
                print(f"ERROR executing stimulus {idx}: {e}")
                failed_indices.append(idx)
                # Save checkpoint even on failure
                self._save_checkpoint(
                    completed_indices, 
                    failed_indices, 
                    idx + 1
                )
                continue
        
        # Save consolidated results
        print(f"\nSaving results to {output_file}...")
        
        if save_format == 'hdf5':
            self._save_results_hdf5(all_results, output_file)
        else:
            self._save_results_pytorch(all_results, output_file)
        
        # Save metadata
        self._save_metadata(output_file.parent)
        
        # Save stimulus index
        self._save_stimulus_index(output_file.parent)
        
        duration = time.time() - start_time
        
        print(f"\nBatch execution completed!")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Successful: {len(all_results)}/{len(self.stimulus_configs)}")
        print(f"Failed: {len(failed_indices)}")
        print(f"Output: {output_file}")
        
        return {
            'batch_id': self.batch_id,
            'num_stimuli': len(all_results),
            'output_path': str(output_file),
            'duration_seconds': duration,
            'failed_stimuli': failed_indices,
        }
    
    def _execute_single_stimulus(
        self, 
        stim_config: Dict[str, Any],
        save_intermediates: bool
    ) -> Dict[str, Any]:
        """Execute a single stimulus configuration.
        
        Args:
            stim_config: Stimulus configuration dictionary
            save_intermediates: Whether to include intermediate results
        
        Returns:
            Dictionary with stimulus results including spikes and optionally
            intermediate states
        """
        # Extract stimulus parameters (remove metadata keys)
        stimulus_params = {
            k: v for k, v in stim_config.items() 
            if k not in ['stimulus_id', 'combo_idx', 'rep_idx', 'seed', 'type']
        }
        
        # Set seed for reproducibility
        seed = stim_config['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Execute through pipeline
        results = self.pipeline.forward(
            stimulus_type=stim_config['type'],
            return_intermediates=save_intermediates,
            **stimulus_params
        )
        
        return results
    
    def _save_checkpoint(
        self,
        completed: set,
        failed: list,
        current_idx: int
    ) -> None:
        """Save checkpoint for resume capability.
        
        Args:
            completed: Set of completed stimulus indices
            failed: List of failed stimulus indices
            current_idx: Current stimulus index being processed
        """
        checkpoint = {
            'batch_id': self.batch_id,
            'completed_stimuli': list(completed),
            'failed_stimuli': failed,
            'total_stimuli': len(self.stimulus_configs),
            'current_stimulus': current_idx,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint JSON file
        
        Returns:
            Checkpoint data dictionary
        """
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    
    def _save_results_pytorch(
        self, 
        results: List[Dict[str, Any]], 
        output_path: Path
    ) -> None:
        """Save batch results in PyTorch format.
        
        Args:
            results: List of result dictionaries
            output_path: Path to output .pt file
        """
        # Consolidate results
        consolidated = {
            'metadata': {
                'batch_id': self.batch_id,
                'config': self.config,
                'num_stimuli': len(results),
                'timestamp': datetime.now().isoformat(),
            },
            'results': results,
            'pipeline_info': self.pipeline.get_pipeline_info(),
        }
        
        # Move tensors to CPU before saving
        for result in consolidated['results']:
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.cpu()
        
        torch.save(consolidated, output_path)
    
    def _save_results_hdf5(
        self, 
        results: List[Dict[str, Any]], 
        output_path: Path
    ) -> None:
        """Save batch results in HDF5 format.
        
        Args:
            results: List of result dictionaries
            output_path: Path to output .h5 file
        
        Raises:
            ImportError: If h5py is not installed
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 output. Install with: pip install h5py"
            )
        
        with h5py.File(output_path, 'w') as f:
            # Save metadata
            meta_grp = f.create_group('metadata')
            meta_grp.attrs['batch_id'] = self.batch_id
            meta_grp.attrs['num_stimuli'] = len(results)
            meta_grp.attrs['timestamp'] = datetime.now().isoformat()
            meta_grp.attrs['config'] = json.dumps(self.config)
            
            # Save grid information
            grid_grp = f.create_group('grid')
            pipeline_info = self.pipeline.get_pipeline_info()
            for key, value in pipeline_info['grid_properties'].items():
                if isinstance(value, (int, float, str)):
                    grid_grp.attrs[key] = value
                elif isinstance(value, (list, tuple)):
                    grid_grp.create_dataset(key, data=np.array(value))
            
            # Save stimuli and responses
            stimuli_grp = f.create_group('stimuli')
            responses_grp = f.create_group('responses')
            
            for idx, result in enumerate(results):
                stim_id = f"stim_{idx:04d}"
                
                # Save stimulus configuration
                stim_subgrp = stimuli_grp.create_group(stim_id)
                config = result.get('stimulus_config', {})
                for key, value in config.items():
                    if isinstance(value, (int, float, str)):
                        stim_subgrp.attrs[key] = value
                
                # Save responses
                resp_subgrp = responses_grp.create_group(stim_id)
                
                # Save spike data
                for key in ['sa_spikes', 'ra_spikes', 'sa2_spikes']:
                    if key in result:
                        tensor = result[key]
                        if isinstance(tensor, torch.Tensor):
                            data = tensor.cpu().numpy()
                        else:
                            data = np.array(tensor)
                        
                        resp_subgrp.create_dataset(
                            key,
                            data=data,
                            compression='gzip',
                            compression_opts=4
                        )
                
                # Save intermediate results if present
                for key in ['sa_currents', 'ra_currents', 'sa_voltages', 
                           'ra_voltages', 'stimulus']:
                    if key in result:
                        tensor = result[key]
                        if isinstance(tensor, torch.Tensor):
                            data = tensor.cpu().numpy()
                        else:
                            data = np.array(tensor)
                        
                        resp_subgrp.create_dataset(
                            key,
                            data=data,
                            compression='gzip',
                            compression_opts=4
                        )
    
    def _save_metadata(self, output_dir: Path) -> None:
        """Save batch metadata to JSON file.
        
        Args:
            output_dir: Directory to save metadata file
        """
        metadata_file = output_dir / 'batch_metadata.json'
        
        metadata = {
            'batch_id': self.batch_id,
            'config': self.config,
            'num_stimuli': len(self.stimulus_configs),
            'timestamp': datetime.now().isoformat(),
            'pipeline_info': self.pipeline.get_pipeline_info(),
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_stimulus_index(self, output_dir: Path) -> None:
        """Save stimulus index mapping to JSON file.
        
        Args:
            output_dir: Directory to save index file
        """
        index_file = output_dir / 'stimulus_index.json'
        
        index = {
            f"stim_{idx:04d}": config
            for idx, config in enumerate(self.stimulus_configs)
        }
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BatchExecutor':
        """Create BatchExecutor from YAML configuration file.
        
        Args:
            yaml_path: Path to YAML configuration file
        
        Returns:
            Initialized BatchExecutor instance
        
        Example:
            >>> executor = BatchExecutor.from_yaml('batch_config.yml')
            >>> results = executor.execute()
        """
        from sensoryforge.config.yaml_utils import load_yaml
        
        with open(yaml_path, 'r') as f:
            config = load_yaml(f)
        
        return cls(config)
