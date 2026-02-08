"""
Generalized PyTorch tactile encoding pipeline.

Configurable pipeline for generating tactile encoding data with
defaults. Compatible with notebook but generalizable for different stimuli
and parameters.
"""
import torch
import torch.nn as nn
from .grid import GridManager
from sensoryforge.config.yaml_utils import load_yaml
from sensoryforge.stimuli.stimulus import gaussian_pressure_torch, StimulusGenerator
from .innervation import create_sa_innervation, create_ra_innervation
from sensoryforge.filters.sa_ra import SAFilterTorch, RAFilterTorch
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
from sensoryforge.filters.noise import MembraneNoiseTorch


class GeneralizedTactileEncodingPipeline(nn.Module):
    """
    Generalized tactile encoding pipeline with configurable parameters
    and defaults. Supports multiple stimulus types and flexible
    configuration.
    """

    # Default configuration values
    DEFAULT_CONFIG = {
        "pipeline": {
            "device": "cpu",
            "seed": 42,
            "grid_size": 80,
            "spacing": 0.15,
            "center": [0.0, 0.0],
        },
        "neurons": {"sa_neurons": 10, "ra_neurons": 14, "sa2_neurons": 5, "dt": 0.5},
        "innervation": {
            "receptors_per_neuron": 28,
            "sa_spread": 0.3,
            "ra_spread": 0.39,
            "connection_strength": [0.05, 1.0],
            "sa2_connections": 500,
            "sa2_spread": 2.0,
            "sa2_weights": [0.4, 0.75],
            "sa_seed": 33,
            "ra_seed": 33,
            "sa2_seed": 39,
        },
        "filters": {
            "sa_tau_r": 5.0,
            "sa_tau_d": 30.0,
            "sa_k1": 0.05,
            "sa_k2": 3.0,
            "ra_tau_ra": 30.0,
            "ra_k3": 2.0,
            "sa2_tau_r": 8.0,  # New SA2 filter parameter
            "sa2_tau_d": 40.0,  # New SA2 filter parameter
            "sa2_k1": 0.03,  # New SA2 filter parameter
            "sa2_k2": 2.5,  # New SA2 filter parameter
            "sa2_scale": 0.005,  # Fallback if no SA2 filter
        },
        "neuron_params": {
            # SA neuron Izhikevich parameters
            "sa_a": 0.02,
            "sa_b": 0.2,
            "sa_c": -65.0,
            "sa_d": 8.0,
            "sa_v_init": -65.0,
            "sa_threshold": 30.0,
            "sa_a_std": 0.005,  # Variability in 'a' parameter
            "sa_b_std": 0.0,  # Variability in 'b' parameter
            "sa_c_std": 5.0,  # Variability in 'c' parameter
            "sa_d_std": 0.0,  # Variability in 'd' parameter
            "sa_threshold_std": 3.0,  # Variability in threshold
            # RA neuron Izhikevich parameters
            "ra_a": 0.02,
            "ra_b": 0.2,
            "ra_c": -65.0,
            "ra_d": 8.0,
            "ra_v_init": -65.0,
            "ra_threshold": 30.0,
            "ra_a_std": 0.005,
            "ra_b_std": 0.0,
            "ra_c_std": 5.0,
            "ra_d_std": 0.0,
            "ra_threshold_std": 3.0,
            # SA2 neuron Izhikevich parameters
            "sa2_a": 0.02,
            "sa2_b": 0.2,
            "sa2_c": -65.0,
            "sa2_d": 8.0,
            "sa2_v_init": -65.0,
            "sa2_threshold": 30.0,
            "sa2_a_std": 0.005,
            "sa2_b_std": 0.0,
            "sa2_c_std": 5.0,
            "sa2_d_std": 0.0,
            "sa2_threshold_std": 3.0,
            # Legacy parameters (for backward compatibility)
            "a_params": [0.02, 0.005],
            "b_params": None,
            "c_params": [-65.0, 5.0],
            "d_params": None,
            "threshold_val": [30.0, 3.0],
        },
        "noise": {
            # SA membrane noise
            "sa_membrane_std": 3.0,
            "sa_membrane_mean": 0.0,
            "sa_membrane_seed": 42,
            # RA membrane noise
            "ra_membrane_std": 3.0,
            "ra_membrane_mean": 0.0,
            "ra_membrane_seed": 43,
            # SA2 membrane noise
            "sa2_membrane_std": 3.0,
            "sa2_membrane_mean": 0.0,
            "sa2_membrane_seed": 44,
            # Receptor noise (applied to mechanoreceptor responses)
            "use_receptor_noise": False,
            "receptor_std": 5.0,
            "receptor_mean": 0.0,
            "receptor_seed": 123,
            # Legacy parameters (for backward compatibility)
            "membrane_std": 3.0,
            "membrane_mean": 0.0,
            "membrane_seed": 42,
        },
        "temporal": {
            "t_ramp": 10,
            "t_plateau": 800,
            "t_pre": 25,
            "t_post": 200,
            "dt": 0.5,
        },
    }

    def __init__(self, config_path=None, config_dict=None):
        """
        Args:
            config_path: path to YAML configuration file (optional)
            config_dict: configuration dictionary (optional)
            If both None, uses defaults. If both provided, config_dict takes
            precedence.
        """
        super().__init__()

        # Load and merge configuration with defaults
        self.config = self._load_config(config_path, config_dict)

        # Setup basic properties
        self.device = self.config["pipeline"]["device"]
        self.seed = self.config["pipeline"]["seed"]

        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Create grid manager
        self.grid_manager = GridManager(
            grid_size=self.config["pipeline"]["grid_size"],
            spacing=self.config["pipeline"]["spacing"],
            center=tuple(self.config["pipeline"]["center"]),
            device=self.device,
        )

        # Create stimulus generator
        self.stimulus_generator = StimulusGenerator(self.grid_manager)

        # Create pipeline components
        self._create_innervation()
        self._create_filters()
        self._create_neurons()
        self._create_noise()

        # Move all modules to device
        self.to(self.device)

    def _load_config(self, config_path, config_dict):
        """Load configuration with defaults fallback"""
        # Start with defaults
        config = self._deep_copy_dict(self.DEFAULT_CONFIG)

        # Load from file if provided
        if config_path:
            try:
                with open(config_path, "r", encoding="utf-8") as file:
                    file_config = load_yaml(file)
                if file_config:
                    config = self._deep_merge_dict(config, file_config)
            except FileNotFoundError:
                print(f"Config file {config_path} not found, using defaults")

        # Override with dict if provided
        if config_dict:
            config = self._deep_merge_dict(config, config_dict)

        return config

    def _deep_copy_dict(self, d):
        """Deep copy a dictionary"""
        import copy

        return copy.deepcopy(d)

    def _deep_merge_dict(self, base, override):
        """Deep merge two dictionaries"""
        result = self._deep_copy_dict(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value
        return result

    def _create_innervation(self):
        """Create innervation modules with configuration"""
        innervation_cfg = self.config["innervation"]
        neuron_cfg = self.config["neurons"]

        self.sa_innervation = create_sa_innervation(
            self.grid_manager,
            neurons_per_row=neuron_cfg["sa_neurons"],
            connections_per_neuron=innervation_cfg["receptors_per_neuron"],
            sigma_d_mm=innervation_cfg["sa_spread"],
            weight_range=tuple(innervation_cfg["connection_strength"]),
            seed=innervation_cfg["sa_seed"],
        )

        self.ra_innervation = create_ra_innervation(
            self.grid_manager,
            neurons_per_row=neuron_cfg["ra_neurons"],
            connections_per_neuron=innervation_cfg["receptors_per_neuron"],
            sigma_d_mm=innervation_cfg["ra_spread"],
            weight_range=tuple(innervation_cfg["connection_strength"]),
            seed=innervation_cfg["ra_seed"],
        )

        self.sa2_innervation = create_sa_innervation(
            self.grid_manager,
            neurons_per_row=neuron_cfg["sa2_neurons"],
            connections_per_neuron=innervation_cfg["sa2_connections"],
            sigma_d_mm=innervation_cfg["sa2_spread"],
            weight_range=tuple(innervation_cfg["sa2_weights"]),
            seed=innervation_cfg["sa2_seed"],
        )

    def _create_filters(self):
        """Create filters with configuration"""
        filter_cfg = self.config["filters"]
        dt = self.config["neurons"]["dt"]

        self.sa_filter = SAFilterTorch(
            tau_r=filter_cfg["sa_tau_r"],
            tau_d=filter_cfg["sa_tau_d"],
            k1=filter_cfg["sa_k1"],
            k2=filter_cfg["sa_k2"],
            dt=dt,
        )

        self.ra_filter = RAFilterTorch(
            tau_RA=filter_cfg["ra_tau_ra"], k3=filter_cfg["ra_k3"], dt=dt
        )

        # SA2 is just a scaling factor (not a filter)
        self.sa2_scale = filter_cfg["sa2_scale"]

    def _create_neurons(self):
        """Create neuron models with configuration"""
        neuron_cfg = self.config["neuron_params"]
        dt = self.config["neurons"]["dt"]

        # SA neurons with individual parameters
        self.sa_neuron = IzhikevichNeuronTorch(
            a=neuron_cfg["sa_a"],
            b=neuron_cfg["sa_b"],
            c=neuron_cfg["sa_c"],
            d=neuron_cfg["sa_d"],
            v_init=neuron_cfg["sa_v_init"],
            threshold=neuron_cfg["sa_threshold"],
            a_std=neuron_cfg["sa_a_std"],
            b_std=neuron_cfg["sa_b_std"],
            c_std=neuron_cfg["sa_c_std"],
            d_std=neuron_cfg["sa_d_std"],
            threshold_std=neuron_cfg["sa_threshold_std"],
            dt=dt,
        )

        # RA neurons with individual parameters
        self.ra_neuron = IzhikevichNeuronTorch(
            a=neuron_cfg["ra_a"],
            b=neuron_cfg["ra_b"],
            c=neuron_cfg["ra_c"],
            d=neuron_cfg["ra_d"],
            v_init=neuron_cfg["ra_v_init"],
            threshold=neuron_cfg["ra_threshold"],
            a_std=neuron_cfg["ra_a_std"],
            b_std=neuron_cfg["ra_b_std"],
            c_std=neuron_cfg["ra_c_std"],
            d_std=neuron_cfg["ra_d_std"],
            threshold_std=neuron_cfg["ra_threshold_std"],
            dt=dt,
        )

        # SA2 neurons with individual parameters
        self.sa2_neuron = IzhikevichNeuronTorch(
            a=neuron_cfg["sa2_a"],
            b=neuron_cfg["sa2_b"],
            c=neuron_cfg["sa2_c"],
            d=neuron_cfg["sa2_d"],
            v_init=neuron_cfg["sa2_v_init"],
            threshold=neuron_cfg["sa2_threshold"],
            a_std=neuron_cfg["sa2_a_std"],
            b_std=neuron_cfg["sa2_b_std"],
            c_std=neuron_cfg["sa2_c_std"],
            d_std=neuron_cfg["sa2_d_std"],
            threshold_std=neuron_cfg["sa2_threshold_std"],
            dt=dt,
        )

        # Store legacy variability parameters for backward compatibility
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
        """Create noise modules with individual configurations"""
        noise_cfg = self.config["noise"]

        # Individual membrane noise for each neuron type
        self.sa_membrane_noise = MembraneNoiseTorch(
            std=noise_cfg["sa_membrane_std"],
            mean=noise_cfg["sa_membrane_mean"],
            seed=noise_cfg["sa_membrane_seed"],
        )

        self.ra_membrane_noise = MembraneNoiseTorch(
            std=noise_cfg["ra_membrane_std"],
            mean=noise_cfg["ra_membrane_mean"],
            seed=noise_cfg["ra_membrane_seed"],
        )

        self.sa2_membrane_noise = MembraneNoiseTorch(
            std=noise_cfg["sa2_membrane_std"],
            mean=noise_cfg["sa2_membrane_mean"],
            seed=noise_cfg["sa2_membrane_seed"],
        )

        # Legacy membrane noise for backward compatibility
        self.membrane_noise = MembraneNoiseTorch(
            std=noise_cfg["membrane_std"],
            mean=noise_cfg["membrane_mean"],
            seed=noise_cfg["membrane_seed"],
        )

    def generate_stimulus(self, stimulus_type="trapezoidal", **stimulus_params):
        """
        Generate stimuli of various types.

        Args:
            stimulus_type: Type of stimulus to generate
                - 'trapezoidal': Trapezoidal temporal profile with Gaussian spatial
                - 'gaussian': Static Gaussian stimulus
                - 'step': Step function stimulus
                - 'ramp': Ramp stimulus
                - 'custom': Custom stimulus from provided tensor
            **stimulus_params: Parameters specific to stimulus type

        Returns:
            stimulus_sequence: (1, n_timesteps, grid_h, grid_w) tensor
            time_array: (n_timesteps,) time array
            temporal_profile: (n_timesteps,) temporal profile

        Stimulus Format Requirements:
        - All stimuli should have shape (1, n_timesteps, grid_h, grid_w)
        - Values represent pressure amplitude
        - Spatial coordinates match grid_manager coordinates
        - Time should be in milliseconds with dt spacing
        """
        if stimulus_type == "trapezoidal":
            return self._generate_trapezoidal_stimulus(**stimulus_params)
        elif stimulus_type == "gaussian":
            return self._generate_gaussian_stimulus(**stimulus_params)
        elif stimulus_type == "step":
            return self._generate_step_stimulus(**stimulus_params)
        elif stimulus_type == "ramp":
            return self._generate_ramp_stimulus(**stimulus_params)
        elif stimulus_type == "custom":
            return self._generate_custom_stimulus(**stimulus_params)
        else:
            raise ValueError(f"Unknown stimulus type: {stimulus_type}")

    def _generate_trapezoidal_stimulus(self, **params):
        """Generate trapezoidal stimulus using config defaults"""
        # Merge with defaults
        temporal_cfg = self.config["temporal"].copy()
        stimulus_cfg = self.config.get("stimulus", {})

        # Override with provided parameters
        temporal_cfg.update(
            {k: v for k, v in params.items() if k.startswith("t_") or k == "dt"}
        )
        stimulus_cfg.update(
            {
                k: v
                for k, v in params.items()
                if k in ["center_x", "center_y", "amplitude", "sigma"]
            }
        )

        # Set defaults if not provided
        center_x = stimulus_cfg.get("center_x", 0.0)
        center_y = stimulus_cfg.get("center_y", 0.0)
        amplitude = stimulus_cfg.get("amplitude", 30.0)
        sigma = stimulus_cfg.get("sigma", 1.0)

        # Calculate time parameters
        T_PRE = temporal_cfg["t_pre"]
        T_RAMP = temporal_cfg["t_ramp"]
        T_PLATEAU = temporal_cfg["t_plateau"]
        T_POST = temporal_cfg["t_post"]
        DT = temporal_cfg["dt"]

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

        # Create spatial stimulus
        xx, yy = self.grid_manager.get_coordinates()
        spatial_stimulus = gaussian_pressure_torch(
            xx, yy, center_x, center_y, amplitude, sigma
        )

        # Combine spatial and temporal
        n_x, n_y = self.grid_manager.grid_size
        stimulus_sequence = torch.zeros((1, n_timesteps, n_x, n_y), device=self.device)
        for t_idx in range(n_timesteps):
            stimulus_sequence[0, t_idx] = spatial_stimulus * temporal_profile[t_idx]

        return stimulus_sequence, time_array, temporal_profile

    def _generate_gaussian_stimulus(self, duration=None, **params):
        """Generate static Gaussian stimulus"""
        # Set defaults
        duration = duration or 100.0  # ms
        dt = params.get("dt", self.config["temporal"]["dt"])
        center_x = params.get("center_x", 0.0)
        center_y = params.get("center_y", 0.0)
        amplitude = params.get("amplitude", 30.0)
        sigma = params.get("sigma", 1.0)

        # Create time arrays
        n_timesteps = int(duration / dt)
        time_array = (
            torch.arange(n_timesteps, dtype=torch.float32, device=self.device) * dt
        )
        temporal_profile = torch.ones(n_timesteps, device=self.device)

        # Create spatial stimulus
        xx, yy = self.grid_manager.get_coordinates()
        spatial_stimulus = gaussian_pressure_torch(
            xx, yy, center_x, center_y, amplitude, sigma
        )

        # Combine
        n_x, n_y = self.grid_manager.grid_size
        stimulus_sequence = torch.zeros((1, n_timesteps, n_x, n_y), device=self.device)
        for t_idx in range(n_timesteps):
            stimulus_sequence[0, t_idx] = spatial_stimulus

        return stimulus_sequence, time_array, temporal_profile

    def _generate_step_stimulus(self, step_time=None, **params):
        """Generate step function stimulus"""
        # Set defaults
        duration = params.get("duration", 200.0)  # ms
        step_time = step_time or duration / 2  # Step occurs at half duration
        dt = params.get("dt", self.config["temporal"]["dt"])
        center_x = params.get("center_x", 0.0)
        center_y = params.get("center_y", 0.0)
        amplitude = params.get("amplitude", 30.0)
        sigma = params.get("sigma", 1.0)

        # Create time arrays
        n_timesteps = int(duration / dt)
        time_array = (
            torch.arange(n_timesteps, dtype=torch.float32, device=self.device) * dt
        )

        # Create step temporal profile
        step_idx = int(step_time / dt)
        temporal_profile = torch.zeros(n_timesteps, device=self.device)
        temporal_profile[step_idx:] = 1.0

        # Create spatial stimulus
        xx, yy = self.grid_manager.get_coordinates()
        spatial_stimulus = gaussian_pressure_torch(
            xx, yy, center_x, center_y, amplitude, sigma
        )

        # Combine
        n_x, n_y = self.grid_manager.grid_size
        stimulus_sequence = torch.zeros((1, n_timesteps, n_x, n_y), device=self.device)
        for t_idx in range(n_timesteps):
            stimulus_sequence[0, t_idx] = spatial_stimulus * temporal_profile[t_idx]

        return stimulus_sequence, time_array, temporal_profile

    def _generate_ramp_stimulus(self, **params):
        """Generate ramp stimulus"""
        # Set defaults
        duration = params.get("duration", 200.0)  # ms
        dt = params.get("dt", self.config["temporal"]["dt"])
        center_x = params.get("center_x", 0.0)
        center_y = params.get("center_y", 0.0)
        amplitude = params.get("amplitude", 30.0)
        sigma = params.get("sigma", 1.0)

        # Create time arrays
        n_timesteps = int(duration / dt)
        time_array = (
            torch.arange(n_timesteps, dtype=torch.float32, device=self.device) * dt
        )

        # Create ramp temporal profile
        temporal_profile = torch.linspace(0, 1, n_timesteps, device=self.device)

        # Create spatial stimulus
        xx, yy = self.grid_manager.get_coordinates()
        spatial_stimulus = gaussian_pressure_torch(
            xx, yy, center_x, center_y, amplitude, sigma
        )

        # Combine
        n_x, n_y = self.grid_manager.grid_size
        stimulus_sequence = torch.zeros((1, n_timesteps, n_x, n_y), device=self.device)
        for t_idx in range(n_timesteps):
            stimulus_sequence[0, t_idx] = spatial_stimulus * temporal_profile[t_idx]

        return stimulus_sequence, time_array, temporal_profile

    def _generate_custom_stimulus(
        self, stimulus_tensor=None, time_array=None, **params
    ):
        """Use custom provided stimulus"""
        if stimulus_tensor is None:
            raise ValueError("Custom stimulus requires 'stimulus_tensor' parameter")

        # Ensure proper shape
        if len(stimulus_tensor.shape) == 3:
            stimulus_tensor = stimulus_tensor.unsqueeze(0)  # Add batch dimension

        # Create time array if not provided
        if time_array is None:
            dt = params.get("dt", self.config["temporal"]["dt"])
            n_timesteps = stimulus_tensor.shape[1]
            time_array = (
                torch.arange(n_timesteps, dtype=torch.float32, device=self.device) * dt
            )

        # Create temporal profile (normalized stimulus amplitude over time)
        temporal_profile = stimulus_tensor[0].max(dim=-1)[0].max(dim=-1)[0]
        if temporal_profile.max() > 0:
            temporal_profile = temporal_profile / temporal_profile.max()

        return (
            stimulus_tensor.to(self.device),
            time_array.to(self.device),
            temporal_profile,
        )

    def forward(
        self,
        stimulus_sequence=None,
        stimulus_type="trapezoidal",
        return_intermediates=True,
        **stimulus_params,
    ):
        """
        Process stimuli through the pipeline.

        Args:
            stimulus_sequence: Pre-generated stimulus or None to generate
            stimulus_type: Type of stimulus to generate if stimulus_sequence is None
            return_intermediates: Return all intermediate results
            **stimulus_params: Parameters for stimulus generation

        Returns:
            dict with pipeline results
        """
        if stimulus_sequence is None:
            stimulus_sequence, time_array, temporal_profile = self.generate_stimulus(
                stimulus_type, **stimulus_params
            )
        else:
            # Extract time info from provided stimulus
            n_timesteps = stimulus_sequence.shape[1]
            dt = self.config["neurons"]["dt"]
            time_array = torch.arange(n_timesteps, device=self.device) * dt
            temporal_profile = stimulus_sequence[0].max(dim=-1)[0].max(dim=-1)[0]
            if temporal_profile.max() > 0:
                temporal_profile = temporal_profile / temporal_profile.max()

        # Step 4: Mechanoreceptor responses (identical to stimulus)
        mechanoreceptor_responses = stimulus_sequence.clone()

        # Step 5: Compute neural inputs through innervation
        with torch.no_grad():
            sa_inputs = self.sa_innervation(mechanoreceptor_responses)
            ra_inputs = self.ra_innervation(mechanoreceptor_responses)
            sa2_inputs = self.sa2_innervation(mechanoreceptor_responses)

        # Step 6: Apply temporal filters
        with torch.no_grad():
            sa_filtered = self.sa_filter(sa_inputs, reset_states=True)
            ra_filtered = self.ra_filter(ra_inputs, reset_states=True)

            # SA2 scaling (not filtering)
            sa2_filtered = self.sa2_scale * sa2_inputs

        # Step 6.5: Add individual membrane noise per neuron type
        with torch.no_grad():
            sa_filtered_noisy = self.sa_membrane_noise(sa_filtered)
            ra_filtered_noisy = self.ra_membrane_noise(ra_filtered)
            sa2_filtered_noisy = self.sa2_membrane_noise(sa2_filtered)

        # Step 8: Apply individual spiking neuron models
        with torch.no_grad():
            # SA neurons use their own parameters (no override needed)
            sa_v_trace, sa_spikes = self.sa_neuron(sa_filtered_noisy)

            # RA neurons use their own parameters (no override needed)
            ra_v_trace, ra_spikes = self.ra_neuron(ra_filtered_noisy)

            # SA2 neurons use their own parameters (no override needed)
            sa2_v_trace, sa2_spikes = self.sa2_neuron(sa2_filtered_noisy)

        results = {
            "stimulus_sequence": stimulus_sequence,
            "time_array": time_array,
            "temporal_profile": temporal_profile,
            "sa_spikes": sa_spikes,
            "ra_spikes": ra_spikes,
            "sa2_spikes": sa2_spikes,
        }

        if return_intermediates:
            results.update(
                {
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
                }
            )

        return results

    def generate_encoding_data(self, stimulus_configs, return_intermediates=False):
        """
        Generate encoding data for multiple stimuli (useful for decoding preparation).

        Args:
            stimulus_configs: List of stimulus configuration dicts
                Each dict should have 'type' and parameters for that stimulus type
            return_intermediates: Return intermediate pipeline stages

        Returns:
            List of results dictionaries, one per stimulus

        Example:
            stimulus_configs = [
                {'type': 'trapezoidal', 'amplitude': 20, 'sigma': 1.5},
                {'type': 'gaussian', 'duration': 150, 'center_x': 1.0},
                {'type': 'step', 'step_time': 50, 'amplitude': 25}
            ]
        """
        results_list = []

        for config in stimulus_configs:
            stimulus_type = config.pop("type", "trapezoidal")
            results = self.forward(
                stimulus_type=stimulus_type,
                return_intermediates=return_intermediates,
                **config,
            )
            results_list.append(results)

        return results_list

    def get_pipeline_info(self):
        """Get comprehensive information about the pipeline configuration"""
        return {
            "config": self.config,
            "grid_properties": {
                "size": self.grid_manager.grid_size,
                "spacing": self.grid_manager.spacing,
                "center": self.grid_manager.center,
                "xlim": self.grid_manager.xlim,
                "ylim": self.grid_manager.ylim,
            },
            "neuron_counts": {
                "sa_neurons": self.sa_innervation.num_neurons,
                "ra_neurons": self.ra_innervation.num_neurons,
                "sa2_neurons": self.sa2_innervation.num_neurons,
            },
            "filter_info": {
                "sa2_uses_filter": self.use_sa2_filter,
                "sa2_scale": getattr(self, "sa2_scale", None),
            },
        }


def create_generalized_pipeline(config_path=None, **config_overrides):
    """
    Factory function to create a generalized pipeline.

    Args:
        config_path: Path to YAML config file (optional)
        **config_overrides: Direct config overrides as nested dict or flat keys

    Returns:
        GeneralizedTactileEncodingPipeline instance

    Example:
        # Using config file
        pipeline = create_generalized_pipeline('my_config.yml')

        # Using defaults with overrides
        pipeline = create_generalized_pipeline(
            config_dict={'neurons': {'sa_neurons': 20}, 'filters': {'sa2_k1': 0.1}}
        )

        # Mixed approach
        pipeline = create_generalized_pipeline('base_config.yml',
            config_dict={'stimulus': {'amplitude': 40}}
        )
    """
    config_dict = (
        config_overrides.pop("config_dict", None)
        if "config_dict" in config_overrides
        else config_overrides
    )
    return GeneralizedTactileEncodingPipeline(
        config_path=config_path, config_dict=config_dict
    )


# Convenience function for notebook compatibility
def create_notebook_pipeline(config_path="notebook_config.yml"):
    """Create pipeline compatible with notebook examples"""
    return GeneralizedTactileEncodingPipeline(config_path=config_path)
