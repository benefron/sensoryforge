"""Command-line interface for SensoryForge.

This module provides CLI commands for running simulations, validating configurations,
visualizing pipelines, and listing available components.

Example:
    $ sensoryforge run config.yml --duration 1000 --output result.h5
    $ sensoryforge validate config.yml
    $ sensoryforge list-components
    $ sensoryforge visualize config.yml --save output.png
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

import torch

from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.config.yaml_utils import load_yaml


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file.
    
    Returns:
        Parsed configuration dictionary.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config = load_yaml(f)
    
    if not config:
        raise ValueError(f"Empty or invalid config file: {config_path}")
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure and required fields.
    
    Args:
        config: Configuration dictionary to validate.
    
    Returns:
        True if valid, False otherwise.
    
    Note:
        Prints validation errors to stderr.
    """
    errors = []
    
    # Check for basic structure (lenient validation)
    # Pipeline section is optional (has defaults)
    
    # Validate grid if specified
    if 'grid' in config:
        grid_cfg = config['grid']
        if 'type' in grid_cfg:
            if grid_cfg['type'] not in ['standard', 'composite']:
                errors.append(f"Invalid grid type: {grid_cfg['type']}")
            
            if grid_cfg['type'] == 'composite':
                if 'populations' not in grid_cfg:
                    errors.append("Composite grid requires 'populations' field")
    
    # Validate neurons if specified
    if 'neurons' in config:
        neuron_cfg = config['neurons']
        if isinstance(neuron_cfg, dict) and 'type' in neuron_cfg:
            neuron_type = neuron_cfg['type']
            if neuron_type == 'dsl':
                required_dsl_fields = ['equations', 'threshold', 'reset', 'parameters']
                for field in required_dsl_fields:
                    if field not in neuron_cfg:
                        errors.append(f"DSL neuron requires '{field}' field")
    
    # Validate solver if specified
    if 'solver' in config:
        solver_cfg = config['solver']
        if 'type' in solver_cfg:
            if solver_cfg['type'] not in ['euler', 'adaptive']:
                errors.append(f"Invalid solver type: {solver_cfg['type']}")
    
    # Print errors
    if errors:
        for error in errors:
            print(f"Validation error: {error}", file=sys.stderr)
        return False
    
    return True


def cmd_run(args: argparse.Namespace) -> int:
    """Run simulation from YAML config.
    
    Args:
        args: Command-line arguments with config, duration, output, device.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        # Load configuration
        config = load_config_file(args.config)
        
        # Validate
        if not validate_config(config):
            print("Configuration validation failed", file=sys.stderr)
            return 1
        
        # Override device if specified
        if args.device:
            if 'pipeline' not in config:
                config['pipeline'] = {}
            config['pipeline']['device'] = args.device
        
        # Create pipeline
        print(f"Loading pipeline from {args.config}...")
        pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
        
        # Run simulation
        print(f"Running simulation (duration: {args.duration}ms)...")
        
        # Generate stimulus based on config or default
        stimulus_type = 'trapezoidal'
        if 'stimuli' in config and isinstance(config['stimuli'], list) and config['stimuli']:
            # Use first stimulus definition
            stimulus_cfg = config['stimuli'][0]
            stimulus_type = stimulus_cfg.get('type', 'trapezoidal')
        
        # Build stimulus parameters from CLI args and config
        # (resolves ReviewFinding#M10)
        stimulus_params = {}
        if 'stimuli' in config and isinstance(config['stimuli'], list) and config['stimuli']:
            stimulus_cfg = config['stimuli'][0]
            stimulus_params = {k: v for k, v in stimulus_cfg.items() if k != 'type'}
        # For non-trapezoidal types, pass duration from CLI
        if stimulus_type != 'trapezoidal' and args.duration:
            stimulus_params['duration'] = args.duration
        
        results = pipeline.forward(
            stimulus_type=stimulus_type,
            return_intermediates=True,
            **stimulus_params,
        )
        
        # Save output if requested
        if args.output:
            output_path = Path(args.output)
            print(f"Saving results to {output_path}...")
            
            # Save as PyTorch checkpoint
            save_dict = {
                'config': config,
                'results': {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                           for k, v in results.items()},
                'pipeline_info': pipeline.get_pipeline_info()
            }
            torch.save(save_dict, output_path)
            print(f"Results saved successfully")
        else:
            # Print summary
            print("\nSimulation completed successfully!")
            print(f"SA spikes: {results['sa_spikes'].sum().item()}")
            print(f"RA spikes: {results['ra_spikes'].sum().item()}")
            if 'sa2_spikes' in results:
                print(f"SA2 spikes: {results['sa2_spikes'].sum().item()}")
        
        return 0
        
    except Exception as e:
        print(f"Error running simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate YAML configuration without running.
    
    Args:
        args: Command-line arguments with config path.
    
    Returns:
        Exit code (0 for valid, 1 for invalid).
    """
    try:
        # Load configuration
        config = load_config_file(args.config)
        
        # Validate structure
        if not validate_config(config):
            print(f"❌ Configuration validation failed: {args.config}", file=sys.stderr)
            return 1
        
        # Try to instantiate pipeline (catches additional errors)
        print(f"Validating {args.config}...")
        try:
            pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
            pipeline_info = pipeline.get_pipeline_info()
            
            print(f"✓ Configuration is valid!")
            print(f"\nPipeline info:")
            print(f"  Device: {pipeline_info['config']['pipeline']['device']}")
            print(f"  Grid size: {pipeline_info['grid_properties']['size']}")
            print(f"  SA neurons: {pipeline_info['neuron_counts']['sa_neurons']}")
            print(f"  RA neurons: {pipeline_info['neuron_counts']['ra_neurons']}")
            
            return 0
        except Exception as e:
            print(f"❌ Pipeline instantiation failed: {e}", file=sys.stderr)
            return 1
            
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error validating config: {e}", file=sys.stderr)
        return 1


def cmd_list_components(args: argparse.Namespace) -> int:
    """List available components (filters, neurons, stimuli, solvers).
    
    Args:
        args: Command-line arguments (unused).
    
    Returns:
        Exit code (0 for success).
    """
    print("Available SensoryForge Components:")
    print("=" * 50)
    
    print("\n📊 Filters:")
    print("  - SA (Slowly Adapting)")
    print("  - RA (Rapidly Adapting)")
    print("  - center_surround (for vision)")
    
    print("\n🧠 Neuron Models:")
    print("  - izhikevich (hand-written)")
    print("  - adex (Adaptive Exponential)")
    print("  - mqif (Multi-Quadratic Integrate-and-Fire)")
    print("  - dsl (Equation DSL - custom models)")
    
    print("\n🎯 Stimuli:")
    print("  - gaussian (Static Gaussian blob)")
    print("  - texture (Gabor, edge grating, perlin noise)")
    print("  - moving (Linear, circular motion)")
    print("  - trapezoidal (Ramp-plateau-ramp)")
    print("  - step (Step function)")
    print("  - ramp (Linear ramp)")
    
    print("\n⚙️  Solvers:")
    print("  - euler (Forward Euler - default)")
    print("  - adaptive (Adaptive stepping - requires torchdiffeq/torchode)")
    
    print("\n🌐 Grid Types:")
    print("  - standard (Single population)")
    print("  - composite (Multi-population mosaic)")
    
    print("\n💡 Use 'sensoryforge run --help' for usage examples")
    
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    """Visualize pipeline structure from YAML config.
    
    Args:
        args: Command-line arguments with config and save path.
    
    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        # Load configuration
        config = load_config_file(args.config)
        
        # Validate
        if not validate_config(config):
            print("Configuration validation failed", file=sys.stderr)
            return 1
        
        # Create pipeline
        print(f"Loading pipeline from {args.config}...")
        pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
        info = pipeline.get_pipeline_info()
        
        # Print text visualization
        print("\n" + "=" * 60)
        print("PIPELINE STRUCTURE")
        print("=" * 60)
        
        print(f"\n📍 Grid Configuration:")
        print(f"  Size: {info['grid_properties']['size']}")
        print(f"  Spacing: {info['grid_properties']['spacing']} mm")
        print(f"  Bounds: X={info['grid_properties']['xlim']}, Y={info['grid_properties']['ylim']}")
        
        print(f"\n🧠 Neuron Populations:")
        print(f"  SA neurons: {info['neuron_counts']['sa_neurons']}")
        print(f"  RA neurons: {info['neuron_counts']['ra_neurons']}")
        print(f"  SA2 neurons: {info['neuron_counts']['sa2_neurons']}")
        
        print(f"\n⚙️  Device: {info['config']['pipeline']['device']}")
        
        if 'stimuli' in config:
            print(f"\n🎯 Configured Stimuli:")
            for i, stim in enumerate(config['stimuli']):
                print(f"  {i+1}. {stim.get('type', 'unknown')}")
        
        # Save visualization if requested
        if args.save:
            print(f"\nNote: Graphical visualization not yet implemented.")
            print(f"Use --save for future PNG export support.")
        
        return 0
        
    except Exception as e:
        print(f"Error visualizing pipeline: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog='sensoryforge',
        description='SensoryForge: Modular sensory encoding framework'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser(
        'run',
        help='Run simulation from YAML config'
    )
    run_parser.add_argument(
        'config',
        help='Path to YAML configuration file'
    )
    run_parser.add_argument(
        '--duration',
        type=float,
        default=1000.0,
        help='Simulation duration in milliseconds (default: 1000)'
    )
    run_parser.add_argument(
        '--output',
        help='Output file path (PyTorch checkpoint .pt or .pth)'
    )
    run_parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'mps'],
        help='Override device from config'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate YAML config without running'
    )
    validate_parser.add_argument(
        'config',
        help='Path to YAML configuration file'
    )
    
    # List components command
    list_parser = subparsers.add_parser(
        'list-components',
        help='List available filters, neurons, stimuli, and solvers'
    )
    
    # Visualize command
    viz_parser = subparsers.add_parser(
        'visualize',
        help='Visualize pipeline structure from config'
    )
    viz_parser.add_argument(
        'config',
        help='Path to YAML configuration file'
    )
    viz_parser.add_argument(
        '--save',
        help='Save visualization to file (future: PNG export)'
    )
    
    return parser


def main() -> int:
    """Main CLI entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to command handlers
    commands = {
        'run': cmd_run,
        'validate': cmd_validate,
        'list-components': cmd_list_components,
        'visualize': cmd_visualize,
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
