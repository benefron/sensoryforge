"""
Notebook-compatible PyTorch tactile encoding pipeline.
This version exactly matches the notebook pytorch_tactile_analysis.ipynb
"""
import torch
import torch.nn as nn
import yaml
from .grid_torch import GridManager
from .stimulus_torch import gaussian_pressure_torch
from .innervation_torch import create_sa_innervation, create_ra_innervation
from .filters_torch import SAFilterTorch, RAFilterTorch
from neurons.izhikevich import IzhikevichNeuronTorch
from .noise_torch import MembraneNoiseTorch


class NotebookTactileEncodingPipeline(nn.Module):
    """
    Tactile encoding pipeline that exactly matches pytorch_tactile_analysis.ipynb
    """

    def __init__(self, config_path="notebook_config.yml"):
        """
        Args:
            config_path: path to YAML configuration file
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        super().__init__()
        self.device = self.config["pipeline"]["device"]
        self.seed = self.config["pipeline"]["seed"]

        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Create grid manager
        self.grid_manager = GridManager(
            grid_size=self.config["pipeline"]["grid_size"],
            spacing=self.config["pipeline"]["spacing"],
            device=self.device,
        )

        # Create innervation modules with notebook parameters
        self._create_innervation()

        # Create filters with notebook parameters
        self._create_filters()

        # Create neuron models
        self._create_neurons()

        # Create noise module
        self._create_noise()

        # Move all modules to device
        self.to(self.device)

    def _create_innervation(self):
        """Create innervation modules with exact notebook parameters"""
        innervation_cfg = self.config["innervation"]

        self.sa_innervation = create_sa_innervation(
            self.grid_manager,
            neurons_per_row=self.config["neurons"]["sa_neurons"],
            connections_per_neuron=innervation_cfg["receptors_per_neuron"],
            sigma_d_mm=innervation_cfg["sa_spread"],
            weight_range=tuple(innervation_cfg["connection_strength"]),
            seed=33,  # Same as notebook
        )

        self.ra_innervation = create_ra_innervation(
            self.grid_manager,
            neurons_per_row=self.config["neurons"]["ra_neurons"],
            connections_per_neuron=innervation_cfg["receptors_per_neuron"],
            sigma_d_mm=innervation_cfg["ra_spread"],
            weight_range=tuple(innervation_cfg["connection_strength"]),
            seed=33,  # Same as notebook
        )

        self.sa2_innervation = create_sa_innervation(
            self.grid_manager,
            neurons_per_row=self.config["neurons"]["sa2_neurons"],
            connections_per_neuron=innervation_cfg["sa2_connections"],
            sigma_d_mm=innervation_cfg["sa2_spread"],
            weight_range=tuple(innervation_cfg["sa2_weights"]),
            seed=39,  # Same as notebook
        )

    def _create_filters(self):
        """Create filters with exact notebook parameters"""
        filter_cfg = self.config["filters"]

        self.sa_filter = SAFilterTorch(
            tau_r=filter_cfg["sa_tau_r"],
            tau_d=filter_cfg["sa_tau_d"],
            k1=filter_cfg["sa_k1"],
            k2=filter_cfg["sa_k2"],
            dt=self.config["temporal"]["dt"],
        )

        self.ra_filter = RAFilterTorch(
            tau_RA=filter_cfg["ra_tau_ra"],
            k3=filter_cfg["ra_k3"],
            dt=self.config["temporal"]["dt"],
        )

        self.sa2_scale = filter_cfg["sa2_scale"]

    def _create_neurons(self):
        """Create neuron models with exact notebook parameters"""
        neuron_cfg = self.config["neuron_params"]

        self.sa_neuron = IzhikevichNeuronTorch(
            a=neuron_cfg["a"],
            b=neuron_cfg["b"],
            c=neuron_cfg["c"],
            d=neuron_cfg["d"],
            dt=self.config["temporal"]["dt"],
        )

        self.ra_neuron = IzhikevichNeuronTorch(
            a=neuron_cfg["a"],
            b=neuron_cfg["b"],
            c=neuron_cfg["c"],
            d=neuron_cfg["d"],
            dt=self.config["temporal"]["dt"],
        )

        # Store variability parameters for simulation
        self.a_params = (
            tuple(neuron_cfg["a_params"]) if neuron_cfg["a_params"] else None
        )
        self.b_params = neuron_cfg["b_params"]
        self.c_params = (
            tuple(neuron_cfg["c_params"]) if neuron_cfg["c_params"] else None
        )
        self.d_params = neuron_cfg["d_params"]
        self.threshold_val = tuple(neuron_cfg["threshold_val"])

    def _create_noise(self):
        """Create noise module with exact notebook parameters"""
        noise_cfg = self.config["noise"]

        self.membrane_noise = MembraneNoiseTorch(
            std=noise_cfg["membrane_std"],
            mean=noise_cfg["membrane_mean"],
            seed=noise_cfg["membrane_seed"],
        )

    def create_trapezoidal_stimulus(self):
        """Create trapezoidal temporal stimulus exactly as in notebook"""
        temporal = self.config["temporal"]
        stimulus_cfg = self.config["stimulus"]

        # Calculate time parameters (all in ms)
        T_PRE = temporal["t_pre"]
        T_RAMP = temporal["t_ramp"]
        T_PLATEAU = temporal["t_plateau"]
        T_POST = temporal["t_post"]
        DT = temporal["dt"]

        T_TOTAL = T_PRE + T_RAMP + T_PLATEAU + T_RAMP + T_POST

        # Create time array
        n_timesteps = int(T_TOTAL / DT)
        time_array = (
            torch.arange(n_timesteps, dtype=torch.float32, device=self.device) * DT
        )

        # Create temporal profile
        temporal_profile = torch.zeros(n_timesteps, device=self.device)
        idx = 0

        # Pre-stimulus (already zero)
        idx += int(T_PRE / DT)

        # Ramp up
        ramp_up_steps = int(T_RAMP / DT)
        temporal_profile[idx : idx + ramp_up_steps] = torch.linspace(
            0, 1, ramp_up_steps, device=self.device
        )
        idx += ramp_up_steps

        # Plateau
        plateau_steps = int(T_PLATEAU / DT)
        temporal_profile[idx : idx + plateau_steps] = 1.0
        idx += plateau_steps

        # Ramp down
        temporal_profile[idx : idx + ramp_up_steps] = torch.linspace(
            1, 0, ramp_up_steps, device=self.device
        )

        # Get grid coordinates
        xx, yy = self.grid_manager.get_coordinates()
        n_x, n_y = self.grid_manager.grid_size

        # Create spatial Gaussian stimulus
        spatial_stimulus = gaussian_pressure_torch(
            xx,
            yy,
            center_x=stimulus_cfg["center_x"],
            center_y=stimulus_cfg["center_y"],
            amplitude=stimulus_cfg["amplitude"],
            sigma=stimulus_cfg["sigma"],
        )

        # Create full temporal sequence [batch=1, time_steps, height, width]
        stimulus_sequence = torch.zeros((1, n_timesteps, n_x, n_y), device=self.device)
        for t_idx in range(n_timesteps):
            stimulus_sequence[0, t_idx] = spatial_stimulus * temporal_profile[t_idx]

        return stimulus_sequence, time_array, temporal_profile

    def forward(self, stimulus_sequence=None, return_intermediates=True):
        """
        Process stimuli through the notebook pipeline exactly.

        Args:
            stimulus_sequence: input stimuli tensor or None to generate
            return_intermediates: return all intermediate results

        Returns:
            dict with all pipeline stages and results
        """
        if stimulus_sequence is None:
            (
                stimulus_sequence,
                time_array,
                temporal_profile,
            ) = self.create_trapezoidal_stimulus()
        else:
            # If stimulus provided, create dummy time arrays
            n_timesteps = stimulus_sequence.shape[1]
            dt = self.config["temporal"]["dt"]
            time_array = torch.arange(n_timesteps, device=self.device) * dt
            temporal_profile = torch.ones(n_timesteps, device=self.device)

        # Step 4: Mechanoreceptor responses (identical to stimulus)
        mechanoreceptor_responses = stimulus_sequence.clone()

        # Step 5: Compute neural inputs through innervation
        with torch.no_grad():
            sa_inputs = self.sa_innervation(mechanoreceptor_responses)
            ra_inputs = self.ra_innervation(mechanoreceptor_responses)
            sa2_inputs = self.sa2_innervation(mechanoreceptor_responses)

        # Step 6: Apply SA/RA temporal filters
        with torch.no_grad():
            sa_filtered = self.sa_filter(sa_inputs, reset_states=True)
            ra_filtered = self.ra_filter(ra_inputs, reset_states=True)
            sa2_filtered = self.sa2_scale * sa2_inputs  # SA2 not filtered, scaled

        # Step 6.5: Add membrane noise
        with torch.no_grad():
            sa_filtered_noisy = self.membrane_noise(sa_filtered)
            ra_filtered_noisy = self.membrane_noise(ra_filtered)
            sa2_filtered_noisy = self.membrane_noise(sa2_filtered)

        # Step 8: Apply spiking neuron models
        with torch.no_grad():
            sa_v_trace, sa_spikes = self.sa_neuron(
                sa_filtered_noisy,
                a=self.a_params,
                b=self.b_params,
                c=self.c_params,
                d=self.d_params,
                threshold=self.threshold_val,
            )
            ra_v_trace, ra_spikes = self.ra_neuron(
                ra_filtered_noisy,
                a=self.a_params,
                b=self.b_params,
                c=self.c_params,
                d=self.d_params,
                threshold=self.threshold_val,
            )
            sa2_v_trace, sa2_spikes = self.sa_neuron(
                sa2_filtered_noisy,
                a=self.a_params,
                b=self.b_params,
                c=self.c_params,
                d=self.d_params,
                threshold=self.threshold_val,
            )

        results = {
            "stimulus_sequence": stimulus_sequence,
            "time_array": time_array,
            "temporal_profile": temporal_profile,
            "mechanoreceptor_responses": mechanoreceptor_responses,
            "sa_inputs": sa_inputs,
            "ra_inputs": ra_inputs,
            "sa2_inputs": sa2_inputs,
            "sa_filtered": sa_filtered,
            "ra_filtered": ra_filtered,
            "sa2_filtered": sa2_filtered,
            "sa_filtered_noisy": sa_filtered_noisy,
            "ra_filtered_noisy": ra_filtered_noisy,
            "sa2_filtered_noisy": sa2_filtered_noisy,
            "sa_v_trace": sa_v_trace,
            "ra_v_trace": ra_v_trace,
            "sa2_v_trace": sa2_v_trace,
            "sa_spikes": sa_spikes,
            "ra_spikes": ra_spikes,
            "sa2_spikes": sa2_spikes,
        }

        return results

    def create_notebook_visualization(self, results):
        """Create the exact visualization from the notebook"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Extract data
        time_array = results["time_array"]
        temporal_profile = results["temporal_profile"]
        sa_spikes = results["sa_spikes"]
        ra_spikes = results["ra_spikes"]
        sa2_spikes = results["sa2_spikes"]

        # Convert to numpy
        time_ms = time_array.cpu().numpy()
        sa_spikes_bin = sa_spikes[0].cpu().numpy()
        ra_spikes_bin = ra_spikes[0].cpu().numpy()
        sa2_spikes_bin = sa2_spikes[0].cpu().numpy()

        # Get neuron centers for spatial ordering
        sa_centers = self.sa_innervation.neuron_centers.cpu().numpy()
        ra_centers = self.ra_innervation.neuron_centers.cpu().numpy()
        sa2_centers = self.sa2_innervation.neuron_centers.cpu().numpy()

        # Sort by y (descending), then x (ascending)
        sa_order = np.lexsort((sa_centers[:, 0], -sa_centers[:, 1]))
        ra_order = np.lexsort((ra_centers[:, 0], -ra_centers[:, 1]))
        sa2_order = np.lexsort((sa2_centers[:, 0], -sa2_centers[:, 1]))

        sa_spikes_sorted = sa_spikes_bin[:, sa_order]
        ra_spikes_sorted = ra_spikes_bin[:, ra_order]
        sa2_spikes_sorted = sa2_spikes_bin[:, sa2_order]

        # Ensure same length
        min_len = min(
            len(time_ms),
            sa_spikes_sorted.shape[0],
            ra_spikes_sorted.shape[0],
            sa2_spikes_sorted.shape[0],
        )
        time_ms = time_ms[:min_len]
        sa_spikes_sorted = sa_spikes_sorted[:min_len]
        ra_spikes_sorted = ra_spikes_sorted[:min_len]
        sa2_spikes_sorted = sa2_spikes_sorted[:min_len]

        # Create figure with 4 rows exactly as notebook
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        # Row 1: Stimulus timeline
        axes[0].plot(time_ms, temporal_profile.cpu().numpy(), "purple", linewidth=2)
        axes[0].set_ylabel("Stimulus Amplitude")
        axes[0].set_title("Stimulus Timeline")
        axes[0].grid(True, alpha=0.3)

        # Create raster plots
        def create_raster(spikes_sorted, ax, color, title):
            spike_times = []
            spike_neurons = []
            n_neurons = spikes_sorted.shape[1]

            for neuron_idx in range(n_neurons):
                times = time_ms[spikes_sorted[:, neuron_idx].astype(bool)]
                spike_times.extend(times)
                spike_neurons.extend([neuron_idx + 1] * len(times))

            if spike_times:
                ax.scatter(spike_times, spike_neurons, c=color, s=1, alpha=0.8)

            ax.set_ylabel(f"{title} Neuron")
            ax.set_ylim(0.5, n_neurons + 0.5)
            ax.set_title(f"{title} Neurons Raster (spatially ordered)")
            ax.grid(True, alpha=0.3)

        # Row 2: SA raster (blue)
        create_raster(sa_spikes_sorted, axes[1], "blue", "SA")

        # Row 3: RA raster (orange)
        create_raster(ra_spikes_sorted, axes[2], "orange", "RA")

        # Row 4: SA2 raster (green)
        create_raster(sa2_spikes_sorted, axes[3], "green", "SA2")

        # Set x-axis label only on bottom plot
        axes[3].set_xlabel("Time (ms)")

        # Add vertical lines for stimulus phases
        temporal_cfg = self.config["temporal"]
        t_ramp_start = temporal_cfg["t_pre"]
        t_plateau_start = t_ramp_start + temporal_cfg["t_ramp"]
        t_plateau_end = t_plateau_start + temporal_cfg["t_plateau"]
        t_ramp_end = t_plateau_end + temporal_cfg["t_ramp"]

        for ax in axes:
            ax.axvline(x=t_ramp_start, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=t_plateau_start, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=t_plateau_end, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=t_ramp_end, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.suptitle(
            "Tactile Encoding Pipeline: Stimulus → Spikes (Notebook Recreation)",
            fontsize=14,
            y=0.98,
        )
        plt.show()

        # Print summary statistics
        sa_total_spikes = sa_spikes_sorted.sum()
        ra_total_spikes = ra_spikes_sorted.sum()
        sa2_total_spikes = sa2_spikes_sorted.sum()

        print(f"\nSpike Summary:")
        print(f"  SA neurons: {sa_total_spikes} total spikes")
        print(f"  RA neurons: {ra_total_spikes} total spikes")
        print(f"  SA2 neurons: {sa2_total_spikes} total spikes")

        return fig

    def run_complete_notebook_pipeline(self):
        """Run the complete notebook pipeline and create visualization"""
        print("Running complete notebook pipeline...")

        # Step 1-3: Grid, innervation, stimulus (handled in forward)
        print("Steps 1-3: Grid, innervation, stimulus generation...")

        # Run the pipeline
        results = self.forward()

        # Step 9: Create visualization
        print("Step 9: Creating visualization...")
        fig = self.create_notebook_visualization(results)

        print("✓ Pipeline complete! Figure should match notebook exactly.")
        return results, fig


def create_notebook_pipeline(config_path="notebook_config.yml"):
    """Factory function to create notebook-compatible pipeline"""
    return NotebookTactileEncodingPipeline(config_path)
