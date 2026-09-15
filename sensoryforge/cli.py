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

import torch

from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.core.batch_executor import BatchExecutor
from sensoryforge.config.yaml_utils import load_config_file
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.registry import (
    NEURON_REGISTRY,
    FILTER_REGISTRY,
    INNERVATION_REGISTRY,
    STIMULUS_REGISTRY,
    SOLVER_REGISTRY,
    GRID_REGISTRY,
)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure and required fields.

    Supports both canonical schema format (grids/populations lists) and
    legacy format (pipeline/grid/neurons dicts).

    Args:
        config: Configuration dictionary to validate.

    Returns:
        True if valid, False otherwise.

    Note:
        Prints validation errors to stderr.
    """
    errors = []

    # Check if it's canonical format
    is_canonical = (
        isinstance(config.get("grids"), list)
        and isinstance(config.get("populations"), list)
        and "pipeline" not in config
    )

    if is_canonical:
        # Validate canonical schema format
        try:
            SensoryForgeConfig.from_dict(config)
        except Exception as e:
            errors.append(f"Canonical schema validation failed: {e}")

        # Validate population configs
        populations = config.get("populations", [])
        for i, pop in enumerate(populations):
            if not isinstance(pop, dict):
                errors.append(f"Population {i} must be a dict")
                continue

            # Validate innervation method
            innervation_method = pop.get("innervation_method", "gaussian")
            valid_methods = ["gaussian", "one_to_one", "uniform", "distance_weighted"]
            if innervation_method not in valid_methods:
                errors.append(
                    f"Population '{pop.get('name', i)}': invalid innervation_method "
                    f"'{innervation_method}'. Valid: {valid_methods}"
                )

            # Validate neuron arrangement
            arrangement = pop.get("neuron_arrangement", "grid")
            valid_arrangements = [
                "grid",
                "poisson",
                "hex",
                "jittered_grid",
                "blue_noise",
            ]
            if arrangement not in valid_arrangements:
                errors.append(
                    f"Population '{pop.get('name', i)}': invalid neuron_arrangement "
                    f"'{arrangement}'. Valid: {valid_arrangements}"
                )

            # Validate DSL config if neuron_model is DSL
            if pop.get("neuron_model") == "DSL (Custom)":
                dsl_cfg = pop.get("dsl_config")
                if not dsl_cfg or not isinstance(dsl_cfg, dict):
                    errors.append(
                        f"Population '{pop.get('name', i)}': DSL neuron requires dsl_config dict"
                    )
                else:
                    required = ["equations", "threshold", "reset"]
                    for field in required:
                        if not dsl_cfg.get(field):
                            errors.append(
                                f"Population '{pop.get('name', i)}': DSL config missing '{field}'"
                            )
    else:
        # Validate legacy format
        # Check for basic structure (lenient validation)
        # Pipeline section is optional (has defaults)

        # Validate grid if specified
        if "grid" in config:
            grid_cfg = config["grid"]
            if "type" in grid_cfg:
                if grid_cfg["type"] not in ["standard", "composite"]:
                    errors.append(f"Invalid grid type: {grid_cfg['type']}")

                if grid_cfg["type"] == "composite":
                    if "populations" not in grid_cfg:
                        errors.append("Composite grid requires 'populations' field")

        # Validate neurons if specified
        if "neurons" in config:
            neuron_cfg = config["neurons"]
            if isinstance(neuron_cfg, dict) and "type" in neuron_cfg:
                neuron_type = neuron_cfg["type"]
                if neuron_type == "dsl":
                    required_dsl_fields = [
                        "equations",
                        "threshold",
                        "reset",
                        "parameters",
                    ]
                    for field in required_dsl_fields:
                        if field not in neuron_cfg:
                            errors.append(f"DSL neuron requires '{field}' field")

        # Validate solver if specified
        if "solver" in config:
            solver_cfg = config["solver"]
            if "type" in solver_cfg:
                if solver_cfg["type"] not in ["euler", "adaptive"]:
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

        # Detect format
        is_canonical = (
            isinstance(config.get("grids"), list)
            and isinstance(config.get("populations"), list)
            and "pipeline" not in config
        )

        # Resolve stimulus type and params once (shared by both paths)
        stimulus_type = "trapezoidal"
        stimulus_params: Dict[str, Any] = {}
        if (
            "stimuli" in config
            and isinstance(config["stimuli"], list)
            and config["stimuli"]
        ):
            stimulus_cfg = config["stimuli"][0]
            stimulus_type = stimulus_cfg.get("type", "trapezoidal")
            stimulus_params = {k: v for k, v in stimulus_cfg.items() if k != "type"}
        if args.duration:
            # F-040: --duration now reaches every stimulus type, including
            # trapezoidal (it scales the plateau so the total length equals
            # the requested duration; see _generate_trapezoidal_stimulus).
            stimulus_params["duration"] = args.duration

        print(f"Loading pipeline from {args.config}...")

        if is_canonical:
            # ---------------------------------------------------------------
            # Canonical path — SimulationEngine
            # ---------------------------------------------------------------
            # Apply device override into the config dict before parsing
            if args.device:
                if "simulation" not in config:
                    config["simulation"] = {}
                config["simulation"]["device"] = args.device

            sf_config = SensoryForgeConfig.from_dict(config)

            # Stimulus generation: reuse GeneralizedTactileEncodingPipeline for now
            # (SimulationEngine does not yet have its own stimulus module)
            pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
            stimulus_tensor, _, _ = pipeline.generate_stimulus(
                stimulus_type=stimulus_type, **stimulus_params
            )

            print(f"Running simulation (duration: {args.duration}ms)...")
            engine = SimulationEngine(sf_config)
            results = engine.run(stimulus_tensor, return_intermediates=True)

            if args.output:
                output_path = Path(args.output)
                print(f"Saving results to {output_path}...")
                save_dict = {
                    "config": config,
                    "results": {
                        f"{pop_name}__{k}": (
                            v.cpu() if isinstance(v, torch.Tensor) else v
                        )
                        for pop_name, pop_results in results.items()
                        for k, v in pop_results.items()
                    },
                    "populations": list(results.keys()),
                }
                torch.save(save_dict, output_path)
                print("Results saved successfully")
            else:
                print("\nSimulation completed successfully!")
                for pop_name, pop_results in results.items():
                    total = int(pop_results["spikes"].sum().item())
                    print(f"{pop_name} spikes: {total}")

        else:
            # ---------------------------------------------------------------
            # Legacy path — GeneralizedTactileEncodingPipeline (unchanged)
            # ---------------------------------------------------------------
            if args.device:
                if "pipeline" not in config:
                    config["pipeline"] = {}
                config["pipeline"]["device"] = args.device

            pipeline = GeneralizedTactileEncodingPipeline.from_config(config)

            print(f"Running simulation (duration: {args.duration}ms)...")
            results = pipeline.forward(
                stimulus_type=stimulus_type,
                return_intermediates=True,
                **stimulus_params,
            )

            if args.output:
                output_path = Path(args.output)
                print(f"Saving results to {output_path}...")
                save_dict = {
                    "config": config,
                    "results": {
                        k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in results.items()
                    },
                    "pipeline_info": pipeline.get_pipeline_info(),
                }
                torch.save(save_dict, output_path)
                print("Results saved successfully")
            else:
                print("\nSimulation completed successfully!")
                print(f"SA spikes: {results['sa_spikes'].sum().item()}")
                print(f"RA spikes: {results['ra_spikes'].sum().item()}")
                if "sa2_spikes" in results:
                    print(f"SA2 spikes: {results['sa2_spikes'].sum().item()}")

        return 0

    except Exception as e:
        print(f"Error running simulation: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def cmd_batch(args: argparse.Namespace) -> int:
    """Run batch execution from YAML config.

    Args:
        args: Command-line arguments with config, output, device, dry_run, resume.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        # Load configuration
        config = load_config_file(args.config)

        # Override output directory if specified
        if args.output:
            if "batch" not in config:
                config["batch"] = {}
            config["batch"]["output_dir"] = args.output

        # Override device if specified
        if args.device:
            if "base_config" not in config:
                config["base_config"] = {}
            if "pipeline" not in config["base_config"]:
                config["base_config"]["pipeline"] = {}
            config["base_config"]["pipeline"]["device"] = args.device

        # Dry run - just validate and show plan
        if args.dry_run:
            print(f"Validating batch configuration: {args.config}")
            executor = BatchExecutor(config)

            print(f"\nBatch Execution Plan:")
            print(f"=" * 60)
            print(f"Batch ID: {executor.batch_id}")
            print(f"Total stimuli: {len(executor.stimulus_configs)}")
            print(f"Output directory: {executor.output_dir}")

            # Show first few stimulus configs as examples
            print(f"\nFirst 5 stimulus configurations:")
            for i, stim_config in enumerate(executor.stimulus_configs[:5]):
                print(f"  {i+1}. {stim_config['stimulus_id']}")
                params = {
                    k: v
                    for k, v in stim_config.items()
                    if k not in ["stimulus_id", "combo_idx", "rep_idx", "seed", "type"]
                }
                print(f"      Type: {stim_config['type']}, Params: {params}")

            if len(executor.stimulus_configs) > 5:
                print(f"  ... and {len(executor.stimulus_configs) - 5} more")

            print(f"\n✓ Batch configuration is valid (dry run complete)")
            return 0

        # Create batch executor
        print(f"Loading batch configuration from {args.config}...")
        executor = BatchExecutor(config)

        # Determine save format
        save_format = config.get("batch", {}).get("save_format", "pytorch")
        save_intermediates = config.get("batch", {}).get("save_intermediates", False)

        # Execute batch
        results = executor.execute(
            save_format=save_format,
            save_intermediates=save_intermediates,
            resume_from=args.resume if args.resume else None,
        )

        # Print summary
        print(f"\n" + "=" * 60)
        print(f"BATCH EXECUTION SUMMARY")
        print(f"=" * 60)
        print(f"Batch ID: {results['batch_id']}")
        print(f"Stimuli executed: {results['num_stimuli']}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Output file: {results['output_path']}")

        if results["failed_stimuli"]:
            print(f"\n⚠ Warning: {len(results['failed_stimuli'])} stimuli failed")
            print(f"Failed indices: {results['failed_stimuli'][:10]}")
            if len(results["failed_stimuli"]) > 10:
                print(f"... and {len(results['failed_stimuli']) - 10} more")
        else:
            print(f"\n✓ All stimuli completed successfully")

        return 0

    except Exception as e:
        print(f"Error running batch: {e}", file=sys.stderr)
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

        # Detect format (same rule as cmd_run: canonical has grids/populations
        # lists and no legacy 'pipeline' key).
        is_canonical = (
            isinstance(config.get("grids"), list)
            and isinstance(config.get("populations"), list)
            and "pipeline" not in config
        )

        # Try to instantiate the real engine for this format (catches
        # additional errors that structural validation above can't, e.g.
        # F-018: canonical configs are validated through SimulationEngine,
        # not just the legacy pipeline).
        print(f"Validating {args.config}...")
        try:
            if is_canonical:
                sf_config = SensoryForgeConfig.from_dict(config)
                engine = SimulationEngine(sf_config)

                print(f"✓ Configuration is valid!")
                print(f"\nPipeline info:")
                print(f"  Device: {sf_config.simulation.device}")
                print(f"  Populations: {len(engine.populations)}")
                for pop in engine.populations:
                    n_neurons = pop["neuron_centers"].shape[0]
                    print(f"    {pop['name']}: {n_neurons} neurons")
            else:
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

    Reads the live component registries (`sensoryforge.registry`) so this command can never
    drift out of sync with what's actually registered (F-018) -- any component registered via
    `ComponentRegistry.register()`, built-in or third-party, shows up here.

    Args:
        args: Command-line arguments (unused).

    Returns:
        Exit code (0 for success).
    """
    print("Available SensoryForge Components:")
    print("=" * 50)

    sections = [
        ("📊 Filters", FILTER_REGISTRY),
        ("🧠 Neuron Models", NEURON_REGISTRY),
        ("🎯 Stimuli", STIMULUS_REGISTRY),
        ("⚙️  Solvers", SOLVER_REGISTRY),
        ("🌐 Grid Types", GRID_REGISTRY),
        ("🔗 Innervation Methods", INNERVATION_REGISTRY),
    ]
    for title, registry in sections:
        print(f"\n{title}:")
        for name in registry.list_registered():
            print(f"  - {name}")

    print("\n💡 Use 'sensoryforge run --help' for usage examples")

    return 0


def cmd_new_component(args: argparse.Namespace) -> int:
    """Scaffold a new component (H2/F-047).

    Default mode writes a standalone, installable plugin package under
    ``--dest`` (current directory by default). ``--in-repo`` preserves the
    original contributor workflow: it writes directly into the core
    ``sensoryforge/`` package, locating the repository root from the
    current working directory (never from the installed package location).

    Args:
        args: Command-line arguments with ``kind``, ``name``, and the
            optional ``dest``/``in_repo`` flags.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from sensoryforge.scaffold import (
        available_kinds,
        find_repo_root,
        generate_in_repo_component,
        generate_plugin_package,
    )

    in_repo = getattr(args, "in_repo", False)
    dest = getattr(args, "dest", None)

    try:
        if in_repo:
            repo_root = find_repo_root()
            paths = generate_in_repo_component(
                args.kind, args.name, repo_root=repo_root
            )
            print(f"✓ Scaffolded new {args.kind} component in {repo_root}:")
            print(f"  Module: {paths['module']}")
            print(f"  Test:   {paths['test']}")
            print(f"  Docs:   {paths['docs']}")
            print(
                "\nNext step: register it in sensoryforge/register_components.py "
                "(see the printed docs stub for the exact lines to add)."
            )
        else:
            dest_dir = Path(dest) if dest else Path.cwd()
            paths = generate_plugin_package(args.kind, args.name, dest=dest_dir)
            print(f"✓ Scaffolded new {args.kind} plugin package:")
            print(f"  Package: {paths['package_root']}")
            print(f"  Module:  {paths['module']}")
            print(f"  Test:    {paths['test']}")
            print(f"  Readme:  {paths['readme']}")
            print(
                f"\nNext steps:\n"
                f"  cd {paths['package_root']}\n"
                f"  pip install -e .\n"
                f"  pytest"
            )
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        if "Unknown component kind" in str(e):
            print(f"Available kinds: {', '.join(available_kinds())}", file=sys.stderr)
        return 1
    except FileExistsError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

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
        print(
            f"  Bounds: X={info['grid_properties']['xlim']}, Y={info['grid_properties']['ylim']}"
        )

        print(f"\n🧠 Neuron Populations:")
        print(f"  SA neurons: {info['neuron_counts']['sa_neurons']}")
        print(f"  RA neurons: {info['neuron_counts']['ra_neurons']}")
        print(f"  SA2 neurons: {info['neuron_counts']['sa2_neurons']}")

        print(f"\n⚙️  Device: {info['config']['pipeline']['device']}")

        if "stimuli" in config:
            print(f"\n🎯 Configured Stimuli:")
            for i, stim in enumerate(config["stimuli"]):
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
        prog="sensoryforge",
        description="SensoryForge: Modular sensory encoding framework",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run simulation from YAML config")
    run_parser.add_argument("config", help="Path to YAML configuration file")
    run_parser.add_argument(
        "--duration",
        type=float,
        default=1000.0,
        help="Simulation duration in milliseconds (default: 1000)",
    )
    run_parser.add_argument(
        "--output", help="Output file path (PyTorch checkpoint .pt or .pth)"
    )
    run_parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], help="Override device from config"
    )

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch", help="Run batch execution from YAML config"
    )
    batch_parser.add_argument("config", help="Path to batch YAML configuration file")
    batch_parser.add_argument("--output", help="Override output directory from config")
    batch_parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], help="Override device from config"
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and show execution plan without running",
    )
    batch_parser.add_argument(
        "--resume", help="Resume from checkpoint file (path to checkpoint.json)"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate YAML config without running"
    )
    validate_parser.add_argument("config", help="Path to YAML configuration file")

    # List components command
    list_parser = subparsers.add_parser(
        "list-components", help="List available filters, neurons, stimuli, and solvers"
    )

    # Visualize command
    viz_parser = subparsers.add_parser(
        "visualize", help="Visualize pipeline structure from config"
    )
    viz_parser.add_argument("config", help="Path to YAML configuration file")
    viz_parser.add_argument(
        "--save", help="Save visualization to file (future: PNG export)"
    )

    # New-component scaffold command (G5)
    new_component_parser = subparsers.add_parser(
        "new-component",
        help="Scaffold a new neuron/filter/stimulus/solver/grid component",
    )
    new_component_parser.add_argument(
        "kind", choices=["neuron", "filter", "stimulus", "solver", "grid"]
    )
    new_component_parser.add_argument(
        "name", help="Component name, e.g. 'Bandpass' or 'my_cool_filter'"
    )
    new_component_parser.add_argument(
        "--dest",
        default=None,
        help=(
            "Directory to write the standalone plugin package under "
            "(default: current directory). Ignored with --in-repo."
        ),
    )
    new_component_parser.add_argument(
        "--in-repo",
        dest="in_repo",
        action="store_true",
        help=(
            "Write directly into this SensoryForge checkout's core package "
            "(sensoryforge/, tests/, docs/) instead of generating a "
            "standalone plugin package. Must be run from inside a "
            "SensoryForge git checkout."
        ),
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
        "run": cmd_run,
        "batch": cmd_batch,
        "validate": cmd_validate,
        "list-components": cmd_list_components,
        "visualize": cmd_visualize,
        "new-component": cmd_new_component,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
