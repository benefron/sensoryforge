"""
Example demonstrating the generalized pipeline.
Shows how to use defaults, YAML config, and multiple stimulus types.
"""
import torch
import matplotlib.pyplot as plt
from sensoryforge.core.generalized_pipeline import create_generalized_pipeline


def test_defaults():
    """Test pipeline with all default values"""
    print("=== Testing Default Configuration ===")

    # Create pipeline with all defaults
    pipeline = create_generalized_pipeline()

    # Get pipeline info
    info = pipeline.get_pipeline_info()
    print(f"Grid size: {info['grid_properties']['size']}")
    print(f"Neuron counts: {info['neuron_counts']}")
    print(f"Uses SA2 filter: {info['filter_info']['sa2_uses_filter']}")

    # Generate trapezoidal stimulus with defaults
    results = pipeline.forward()

    # Count spikes
    sa_spike_count = results["sa_spikes"].sum().item()
    ra_spike_count = results["ra_spikes"].sum().item()
    sa2_spike_count = results["sa2_spikes"].sum().item()

    print(
        "Spike counts - SA: {sa}, RA: {ra}, SA2: {sa2}".format(
            sa=sa_spike_count,
            ra=ra_spike_count,
            sa2=sa2_spike_count,
        )
    )
    return pipeline, results


def test_yaml_config():
    """Test pipeline with YAML configuration"""
    print("\n=== Testing YAML Configuration ===")

    # Create pipeline with archived notebook config baseline
    pipeline = create_generalized_pipeline(
        "config/archive/notebook_config.yml"
    )

    # Generate trapezoidal stimulus (same as notebook)
    results = pipeline.forward()

    # Count spikes
    sa_spike_count = results["sa_spikes"].sum().item()
    ra_spike_count = results["ra_spikes"].sum().item()
    sa2_spike_count = results["sa2_spikes"].sum().item()

    print(
        "Spike counts - SA: {sa}, RA: {ra}, SA2: {sa2}".format(
            sa=sa_spike_count,
            ra=ra_spike_count,
            sa2=sa2_spike_count,
        )
    )
    return pipeline, results


def test_config_overrides():
    """Test pipeline with configuration overrides"""
    print("\n=== Testing Configuration Overrides ===")

    # Override some parameters
    config_overrides = {
        "neurons": {"sa_neurons": 8, "ra_neurons": 12},
        "filters": {"sa2_k1": 0.1, "sa2_k2": 4.0},
        "temporal": {"t_plateau": 500},
    }

    pipeline = create_generalized_pipeline(config_dict=config_overrides)

    # Generate stimulus with custom parameters
    results = pipeline.forward(amplitude=25.0, sigma=1.2)

    # Count spikes
    sa_spike_count = results["sa_spikes"].sum().item()
    ra_spike_count = results["ra_spikes"].sum().item()
    sa2_spike_count = results["sa2_spikes"].sum().item()

    print(
        "Modified neuron counts: SA:{sa}, RA:{ra}".format(
            sa=pipeline.sa_innervation.num_neurons,
            ra=pipeline.ra_innervation.num_neurons,
        )
    )
    print(
        "Spike counts - SA: {sa}, RA: {ra}, SA2: {sa2}".format(
            sa=sa_spike_count,
            ra=ra_spike_count,
            sa2=sa2_spike_count,
        )
    )
    return pipeline, results


def test_multiple_stimuli():
    """Test multiple stimulus types"""
    print("\n=== Testing Multiple Stimulus Types ===")

    pipeline = create_generalized_pipeline()

    # Test different stimulus types
    stimulus_configs = [
        {"type": "trapezoidal", "amplitude": 20, "sigma": 1.5},
        {
            "type": "gaussian",
            "duration": 150,
            "amplitude": 30,
            "center_x": 1.0,
        },
        {"type": "step", "duration": 200, "step_time": 50, "amplitude": 25},
        {"type": "ramp", "duration": 180, "amplitude": 35},
    ]

    results_list = pipeline.generate_encoding_data(stimulus_configs)

    for i, results in enumerate(results_list):
        sa_spikes = results["sa_spikes"].sum().item()
        ra_spikes = results["ra_spikes"].sum().item()
        sa2_spikes = results["sa2_spikes"].sum().item()
        print(
            "Stimulus {idx}: SA:{sa}, RA:{ra}, SA2:{sa2}".format(
                idx=i + 1,
                sa=sa_spikes,
                ra=ra_spikes,
                sa2=sa2_spikes,
            )
        )

    return results_list


def test_sa2_filters():
    """Test SA2 filter configurations"""
    print("\n=== Testing SA2 Filter Configurations ===")

    # Test with SA2 filter enabled
    config_with_sa2_filter = {
        "filters": {
            "sa2_tau_r": 8.0,
            "sa2_tau_d": 40.0,
            "sa2_k1": 0.03,
            "sa2_k2": 2.5,
        }
    }

    pipeline_filtered = create_generalized_pipeline(
        config_dict=config_with_sa2_filter
    )
    results_filtered = pipeline_filtered.forward()

    # Test with SA2 scaling (remove SA2 filter parameters to force scaling)
    # Create pipeline and manually remove SA2 filter parameters
    pipeline_scaled = create_generalized_pipeline()
    # Override the filter config to force scaling mode
    pipeline_scaled.config["filters"] = {
        **pipeline_scaled.config["filters"],
        "sa2_scale": 0.01,
    }
    # Remove SA2 filter keys to force scaling
    for key in ["sa2_tau_r", "sa2_tau_d", "sa2_k1", "sa2_k2"]:
        if key in pipeline_scaled.config["filters"]:
            del pipeline_scaled.config["filters"][key]

    # Recreate filters with modified config
    pipeline_scaled._create_filters()

    results_scaled = pipeline_scaled.forward()

    print("With SA2 filter:")
    print(f"  Uses filter: {pipeline_filtered.use_sa2_filter}")
    print(f"  SA2 spikes: {results_filtered['sa2_spikes'].sum().item()}")

    print("With SA2 scaling:")
    print(f"  Uses filter: {pipeline_scaled.use_sa2_filter}")
    if (
        hasattr(pipeline_scaled, "sa2_scale")
        and pipeline_scaled.sa2_scale is not None
    ):
        print(f"  Scale factor: {pipeline_scaled.sa2_scale}")
    print(f"  SA2 spikes: {results_scaled['sa2_spikes'].sum().item()}")


def create_comparison_plot(results_list):
    """Create a comparison plot of different stimuli"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    stimulus_names = ["Trapezoidal", "Gaussian", "Step", "Ramp"]

    for i, (results, name) in enumerate(zip(results_list, stimulus_names)):
        time_array = results["time_array"].cpu().numpy()
        temporal_profile = results["temporal_profile"].cpu().numpy()

        axes[i].plot(time_array, temporal_profile, "b-", linewidth=2)
        axes[i].set_title(f"{name} Stimulus")
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Normalized Amplitude")
        axes[i].grid(True, alpha=0.3)

        # Add spike count info
        sa_spikes = results["sa_spikes"].sum().item()
        ra_spikes = results["ra_spikes"].sum().item()
        sa2_spikes = results["sa2_spikes"].sum().item()
        axes[i].text(
            0.02,
            0.98,
            f"SA:{sa_spikes}\nRA:{ra_spikes}\nSA2:{sa2_spikes}",
            transform=axes[i].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(
        "generalized_pipeline_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    return fig


if __name__ == "__main__":
    # Run all tests
    print("Testing Generalized Tactile Encoding Pipeline")
    print("=" * 50)

    # Test 1: Defaults
    default_pipeline, default_results = test_defaults()

    # Test 2: YAML config
    yaml_pipeline, yaml_results = test_yaml_config()

    # Test 3: Config overrides
    override_pipeline, override_results = test_config_overrides()

    # Test 4: Multiple stimuli
    multi_results = test_multiple_stimuli()

    # Test 5: SA2 filters
    test_sa2_filters()

    # Create comparison plot
    create_comparison_plot(multi_results)

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("The generalized pipeline supports:")
    print("- Default configuration values")
    print("- YAML configuration files")
    print("- Runtime parameter overrides")
    print("- Multiple stimulus types")
    print("- Configurable SA2 filters")
    print("- Batch data generation for decoding")
