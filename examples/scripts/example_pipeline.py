import torch
import matplotlib.pyplot as plt
from encoding.pipeline_torch import TactileEncodingPipelineTorch


def main():
    # Load canonical configuration shared by GUI/tests
    config_path = "config/pipeline_config.yml"
    pipeline = TactileEncodingPipelineTorch(config_path=config_path)

    # Generate trapezoid stimulus steps
    stimulus_configs = [
        {"type": "trapezoid", "amplitude": amp, "duration": 1.0} for amp in range(1, 10)
    ]

    # Process stimuli through pipeline
    results = pipeline.generate_and_process(
        stimulus_configs, time_steps=9, return_intermediates=True
    )

    # Extract outputs
    sa_spikes = results["sa_spikes"]
    ra_spikes = results["ra_spikes"]
    stimuli = results["stimuli"]
    sa2_spikes = results.get("sa2_spikes", None)

    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    # Plot stimulus
    axes[0].imshow(stimuli[0].cpu().numpy(), cmap="viridis")
    axes[0].set_title("Stimulus")

    # Plot SA neuron raster
    axes[1].imshow(sa_spikes.cpu().numpy(), aspect="auto", cmap="binary")
    axes[1].set_title("SA Neuron Raster")

    # Plot RA neuron raster
    axes[2].imshow(ra_spikes.cpu().numpy(), aspect="auto", cmap="binary")
    axes[2].set_title("RA Neuron Raster")

    # Plot SA2 neuron raster (if available)
    if sa2_spikes is not None:
        axes[3].imshow(sa2_spikes.cpu().numpy(), aspect="auto", cmap="binary")
        axes[3].set_title("SA2 Neuron Raster")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
