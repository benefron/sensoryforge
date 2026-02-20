"""
Generalized PyTorch tactile encoding pipeline.

Configurable pipeline for generating tactile encoding data with
defaults. Compatible with notebook but generalizable for different stimuli
and parameters.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List

from .grid import GridManager
from .composite_grid import CompositeGrid, CompositeReceptorGrid
from .innervation import (
    create_sa_innervation,
    create_ra_innervation,
    InnervationModule,
    FlatInnervationModule,
)
from .processing import ProcessingPipeline
from sensoryforge.config.yaml_utils import load_yaml
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.stimuli.stimulus import gaussian_pressure_torch, StimulusGenerator
from sensoryforge.stimuli.texture import gabor_texture  # (resolves ReviewFinding#M3)
from sensoryforge.stimuli.builder import TimelineStimulus, RepeatedPatternStimulus
from sensoryforge.filters.sa_ra import SAFilterTorch, RAFilterTorch
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
from sensoryforge.neurons.model_dsl import NeuronModel
from sensoryforge.filters.noise import MembraneNoiseTorch
from sensoryforge.solvers.adaptive import AdaptiveSolver
from sensoryforge.register_components import register_all
from sensoryforge.registry import NEURON_REGISTRY, FILTER_REGISTRY

# Ensure components are registered
register_all()


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
        # Phase 3: Composite grid with per-population offset/color
        "grid": {
            "type": "standard",  # "standard" or "composite"
            "populations": {},   # name → {density, arrangement, offset, color, ...}
        },
        # Phase 3: Processing layers between receptor grid and innervation
        "processing_layers": [],  # list of {type: "identity", ...} dicts
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
        self.device = torch.device(self.config["pipeline"]["device"])
        self.seed = self.config["pipeline"]["seed"]

        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Create grid manager (Standard dense grid for stimulus)
        self.grid_manager = GridManager(
            grid_size=self.config["pipeline"]["grid_size"],
            spacing=self.config["pipeline"]["spacing"],
            center=tuple(self.config["pipeline"]["center"]),
            device=self.device,
        )

        # Handle CompositeGrid if configured
        self.composite_grid = None
        grid_cfg = self.config.get("grid", {})
        if grid_cfg.get("type") == "composite":
            self.composite_grid = CompositeReceptorGrid(
                xlim=self.grid_manager.xlim, 
                ylim=self.grid_manager.ylim, 
                device=self.device
            )
            
            # Add populations using Phase 3 add_layer() with offset/color
            populations = grid_cfg.get("populations", {})
            for name, pop_cfg in populations.items():
                self.composite_grid.add_layer(
                    name=name,
                    density=pop_cfg.get("density", 10.0),
                    arrangement=pop_cfg.get("arrangement", "poisson"),
                    offset=tuple(pop_cfg.get("offset", [0.0, 0.0])),
                    color=tuple(pop_cfg["color"]) if pop_cfg.get("color") else None,
                )

        # Create stimulus generator
        self.stimulus_generator = StimulusGenerator(self.grid_manager)

        # Create pipeline components
        self._create_processing_layers()
        self._create_innervation()
        self._create_filters()
        self._create_neurons()
        self._create_noise()

        # Move all modules to device
        self.to(self.device)

    def _load_config(self, config_path, config_dict):
        """Load configuration with defaults fallback.
        
        Supports both canonical schema (SensoryForgeConfig format) and
        legacy format. If canonical format is detected, converts it to
        legacy format for backward compatibility.
        """
        # Start with defaults
        config = self._deep_copy_dict(self.DEFAULT_CONFIG)

        # Load from file if provided
        if config_path:
            try:
                with open(config_path, "r", encoding="utf-8") as file:
                    file_config = load_yaml(file)
                if file_config:
                    # Check if it's canonical format
                    if self._is_canonical_config(file_config):
                        file_config = self._canonical_to_legacy_config(file_config)
                    config = self._deep_merge_dict(config, file_config)
            except FileNotFoundError:
                print(f"Config file {config_path} not found, using defaults")

        # Override with dict if provided
        if config_dict:
            # Check if it's canonical format
            if self._is_canonical_config(config_dict):
                config_dict = self._canonical_to_legacy_config(config_dict)
            config = self._deep_merge_dict(config, config_dict)

        return config
    
    def _is_canonical_config(self, config: dict) -> bool:
        """Check if config is in canonical schema format.
        
        Canonical format has 'grids' (list) and 'populations' (list) at top level,
        while legacy format has 'pipeline', 'grid', 'neurons', etc.
        """
        return (
            isinstance(config.get("grids"), list) and
            isinstance(config.get("populations"), list) and
            "pipeline" not in config
        )
    
    def _canonical_to_legacy_config(self, canonical: dict) -> dict:
        """Convert canonical schema config to legacy pipeline format.
        
        This adapter allows the pipeline to work with canonical configs
        while maintaining backward compatibility with legacy format.
        """
        legacy = self._deep_copy_dict(self.DEFAULT_CONFIG)
        
        # Extract simulation config
        sim_cfg = canonical.get("simulation", {})
        legacy["pipeline"]["device"] = sim_cfg.get("device", "cpu")
        legacy["pipeline"]["seed"] = canonical.get("metadata", {}).get("seed", 42)
        legacy["neurons"]["dt"] = sim_cfg.get("dt", 1.0)
        
        # Extract grids
        grids = canonical.get("grids", [])
        if grids:
            # Use first grid for main grid_manager
            first_grid = grids[0]
            legacy["pipeline"]["grid_size"] = first_grid.get("rows", 40) * first_grid.get("cols", 40)
            legacy["pipeline"]["spacing"] = first_grid.get("spacing", 0.15)
            center = first_grid.get("center", [0.0, 0.0])
            if isinstance(center, list):
                legacy["pipeline"]["center"] = center
            else:
                legacy["pipeline"]["center"] = [
                    first_grid.get("center_x", 0.0),
                    first_grid.get("center_y", 0.0)
                ]
            
            # If multiple grids, create composite grid config
            if len(grids) > 1:
                legacy["grid"]["type"] = "composite"
                legacy["grid"]["populations"] = {}
                for g in grids:
                    name = g.get("name", f"layer{len(legacy['grid']['populations'])}")
                    legacy["grid"]["populations"][name] = {
                        "density": g.get("density", 10.0),
                        "arrangement": g.get("arrangement", "grid"),
                        "offset": g.get("offset", [0.0, 0.0]) if "offset" in g else [
                            g.get("center_x", 0.0) - legacy["pipeline"]["center"][0],
                            g.get("center_y", 0.0) - legacy["pipeline"]["center"][1]
                        ],
                        "color": g.get("color", [66, 135, 245, 200]),
                    }
        
        # Extract populations - map to SA/RA/SA2 for legacy format
        populations = canonical.get("populations", [])
        sa_pop = None
        ra_pop = None
        sa2_pop = None
        
        for pop in populations:
            neuron_type = pop.get("neuron_type", "SA").upper()
            if neuron_type.startswith("SA2") or neuron_type == "SA2":
                sa2_pop = pop
            elif neuron_type.startswith("RA"):
                ra_pop = pop
            elif neuron_type.startswith("SA") or not sa_pop:
                sa_pop = pop
        
        # Map population configs to legacy format
        if sa_pop:
            legacy["neurons"]["sa_neurons"] = (
                sa_pop.get("neuron_rows", sa_pop.get("neurons_per_row", 10)) *
                sa_pop.get("neuron_cols", sa_pop.get("neurons_per_row", 10))
            )
            legacy["innervation"]["sa_spread"] = sa_pop.get("sigma_d_mm", 0.3)
            legacy["innervation"]["sa_method"] = sa_pop.get("innervation_method", "gaussian")
            legacy["innervation"]["sa_seed"] = sa_pop.get("seed", 33)
            legacy["innervation"]["receptors_per_neuron"] = sa_pop.get("connections_per_neuron", 28)
            legacy["innervation"]["connection_strength"] = sa_pop.get("weight_range", [0.05, 1.0])
            
            # Map neuron model params
            model = sa_pop.get("neuron_model", "Izhikevich")
            if model == "DSL (Custom)" or (sa_pop.get("dsl_config") and sa_pop["dsl_config"].get("equations")):
                legacy["neurons"]["type"] = "dsl"
                dsl_cfg = sa_pop.get("dsl_config", {})
                legacy["neurons"]["equations"] = dsl_cfg.get("equations", "")
                legacy["neurons"]["threshold"] = dsl_cfg.get("threshold", "")
                legacy["neurons"]["reset"] = dsl_cfg.get("reset", "")
                legacy["neurons"]["parameters"] = dsl_cfg.get("parameters", {})
            else:
                model_params = sa_pop.get("model_params", {})
                legacy["neuron_params"]["sa_a"] = model_params.get("a", 0.02)
                legacy["neuron_params"]["sa_b"] = model_params.get("b", 0.2)
                legacy["neuron_params"]["sa_c"] = model_params.get("c", -65.0)
                legacy["neuron_params"]["sa_d"] = model_params.get("d", 8.0)
                legacy["neuron_params"]["sa_v_init"] = model_params.get("v_init", -65.0)
                legacy["neuron_params"]["sa_threshold"] = model_params.get("threshold", 30.0)
            
            # Map filter params
            filter_method = sa_pop.get("filter_method", "none")
            if filter_method.lower() in ("sa", "safilter"):
                filter_params = sa_pop.get("filter_params", {})
                legacy["filters"]["sa_tau_r"] = filter_params.get("tau_r", 5.0)
                legacy["filters"]["sa_tau_d"] = filter_params.get("tau_d", 30.0)
                legacy["filters"]["sa_k1"] = filter_params.get("k1", 0.05)
                legacy["filters"]["sa_k2"] = filter_params.get("k2", 3.0)
            
            # Map noise params
            legacy["noise"]["sa_membrane_std"] = sa_pop.get("noise_std", 3.0)
            legacy["noise"]["sa_membrane_mean"] = sa_pop.get("noise_mean", 0.0)
            legacy["noise"]["sa_membrane_seed"] = sa_pop.get("noise_seed", 42)
        
        if ra_pop:
            legacy["neurons"]["ra_neurons"] = (
                ra_pop.get("neuron_rows", ra_pop.get("neurons_per_row", 14)) *
                ra_pop.get("neuron_cols", ra_pop.get("neurons_per_row", 14))
            )
            legacy["innervation"]["ra_spread"] = ra_pop.get("sigma_d_mm", 0.39)
            legacy["innervation"]["ra_method"] = ra_pop.get("innervation_method", "gaussian")
            legacy["innervation"]["ra_seed"] = ra_pop.get("seed", 33)
            
            model = ra_pop.get("neuron_model", "Izhikevich")
            if model == "DSL (Custom)" or (ra_pop.get("dsl_config") and ra_pop["dsl_config"].get("equations")):
                # Use DSL if RA also uses DSL
                if legacy["neurons"].get("type") != "dsl":
                    legacy["neurons"]["type"] = "dsl"
            else:
                model_params = ra_pop.get("model_params", {})
                legacy["neuron_params"]["ra_a"] = model_params.get("a", 0.02)
                legacy["neuron_params"]["ra_b"] = model_params.get("b", 0.2)
                legacy["neuron_params"]["ra_c"] = model_params.get("c", -65.0)
                legacy["neuron_params"]["ra_d"] = model_params.get("d", 8.0)
                legacy["neuron_params"]["ra_v_init"] = model_params.get("v_init", -65.0)
                legacy["neuron_params"]["ra_threshold"] = model_params.get("threshold", 30.0)
            
            filter_method = ra_pop.get("filter_method", "none")
            if filter_method.lower() in ("ra", "rafilter"):
                filter_params = ra_pop.get("filter_params", {})
                legacy["filters"]["ra_tau_ra"] = filter_params.get("tau_ra", 30.0)
                legacy["filters"]["ra_k3"] = filter_params.get("k3", 2.0)
            
            legacy["noise"]["ra_membrane_std"] = ra_pop.get("noise_std", 3.0)
            legacy["noise"]["ra_membrane_mean"] = ra_pop.get("noise_mean", 0.0)
            legacy["noise"]["ra_membrane_seed"] = ra_pop.get("noise_seed", 43)
        
        if sa2_pop:
            legacy["neurons"]["sa2_neurons"] = (
                sa2_pop.get("neuron_rows", sa2_pop.get("neurons_per_row", 5)) *
                sa2_pop.get("neuron_cols", sa2_pop.get("neurons_per_row", 5))
            )
            legacy["innervation"]["sa2_spread"] = sa2_pop.get("sigma_d_mm", 2.0)
            legacy["innervation"]["sa2_method"] = sa2_pop.get("innervation_method", "gaussian")
            legacy["innervation"]["sa2_seed"] = sa2_pop.get("seed", 39)
            legacy["innervation"]["sa2_connections"] = sa2_pop.get("connections_per_neuron", 500)
            legacy["innervation"]["sa2_weights"] = sa2_pop.get("weight_range", [0.4, 0.75])
            
            model_params = sa2_pop.get("model_params", {})
            legacy["neuron_params"]["sa2_a"] = model_params.get("a", 0.02)
            legacy["neuron_params"]["sa2_b"] = model_params.get("b", 0.2)
            legacy["neuron_params"]["sa2_c"] = model_params.get("c", -65.0)
            legacy["neuron_params"]["sa2_d"] = model_params.get("d", 8.0)
            legacy["neuron_params"]["sa2_v_init"] = model_params.get("v_init", -65.0)
            legacy["neuron_params"]["sa2_threshold"] = model_params.get("threshold", 30.0)
            
            legacy["noise"]["sa2_membrane_std"] = sa2_pop.get("noise_std", 3.0)
            legacy["noise"]["sa2_membrane_mean"] = sa2_pop.get("noise_mean", 0.0)
            legacy["noise"]["sa2_membrane_seed"] = sa2_pop.get("noise_seed", 44)
        
        # Map solver config
        solver_cfg = sim_cfg.get("solver", {})
        if solver_cfg:
            legacy["solver"] = {
                "type": solver_cfg.get("type", "euler"),
                "config": {
                    "method": solver_cfg.get("method", "dopri5"),
                    "rtol": solver_cfg.get("rtol", 1e-5),
                    "atol": solver_cfg.get("atol", 1e-7),
                }
            }
        
        return legacy

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

    def _create_processing_layers(self):
        """Create processing layer pipeline from configuration.

        Reads the ``processing_layers`` list from config (each entry is a
        dict with at least a ``type`` key).  If empty or absent, a single
        :class:`IdentityLayer` is used (zero-overhead pass-through).
        """
        layer_configs = self.config.get("processing_layers", [])
        self.processing_pipeline = ProcessingPipeline.from_config(layer_configs)

    def _sample_stimulus_at_receptors(
        self, stimulus: torch.Tensor
    ) -> torch.Tensor:
        """Sample grid-based stimulus at composite receptor coordinates.

        Uses bilinear interpolation via ``torch.nn.functional.grid_sample``
        to evaluate the ``[B, T, H, W]`` stimulus field at each receptor
        position from the composite grid.

        Args:
            stimulus: ``[B, T, H, W]`` stimulus on the regular grid.

        Returns:
            ``[B, T, N_receptors]`` sampled values.
        """
        import torch.nn.functional as F

        B, T, H, W = stimulus.shape
        coords = self.composite_grid.get_all_coordinates()  # [N, 2]

        # Map physical coordinates → normalized [-1, 1] for grid_sample
        xlim = self.grid_manager.xlim
        ylim = self.grid_manager.ylim
        norm_x = 2.0 * (coords[:, 0] - xlim[0]) / (xlim[1] - xlim[0]) - 1.0
        norm_y = 2.0 * (coords[:, 1] - ylim[0]) / (ylim[1] - ylim[0]) - 1.0

        # grid_sample expects grid [B, T, N, 2] with (x_norm, y_norm)
        N = coords.shape[0]
        sample_grid = torch.stack([norm_x, norm_y], dim=-1)  # [N, 2]
        sample_grid = sample_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        sample_grid = sample_grid.expand(B, T, N, 2)

        # grid_sample needs input as [B, C, H, W] — treat T as batch dimension
        stim_flat = stimulus.reshape(B * T, 1, H, W)
        grid_flat = sample_grid.reshape(B * T, 1, N, 2)

        sampled = F.grid_sample(
            stim_flat,
            grid_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # [B*T, 1, 1, N]

        return sampled.reshape(B, T, N)

    def _create_innervation(self):
        """Create innervation modules with configuration.

        When a composite grid is configured **and** the ``innervation.method``
        key is ``"flat"``, :class:`FlatInnervationModule` is used instead of the
        default :class:`InnervationModule`.  This enables irregular (poisson /
        hex) receptor arrangements from :class:`CompositeReceptorGrid`.

        The grid-based :class:`InnervationModule` remains the default for
        backward compatibility.
        """
        innervation_cfg = self.config["innervation"]
        neuron_cfg = self.config["neurons"]
        use_flat = (
            self.composite_grid is not None
            and innervation_cfg.get("method") == "flat"
        )

        # Handle CompositeGrid inputs
        sa_centers, ra_centers, sa2_centers = None, None, None
        
        if self.composite_grid:
            pop_map = {k.lower(): k for k in self.composite_grid.layers.keys()}
            
            # Map canonical names to user population names
            if 'sa1' in pop_map:
                sa_centers = self.composite_grid.get_layer_coordinates(pop_map['sa1'])
            elif 'sa' in pop_map:
                sa_centers = self.composite_grid.get_layer_coordinates(pop_map['sa'])
            
            if 'ra1' in pop_map:
                ra_centers = self.composite_grid.get_layer_coordinates(pop_map['ra1'])
            elif 'ra' in pop_map:
                ra_centers = self.composite_grid.get_layer_coordinates(pop_map['ra'])
            
            if 'sa2' in pop_map:
                sa2_centers = self.composite_grid.get_layer_coordinates(pop_map['sa2'])

        if use_flat:
            # Phase 3: Flat innervation using composite grid coordinates
            all_coords = self.composite_grid.get_all_coordinates()
            xlim = self.composite_grid.xlim
            ylim = self.composite_grid.ylim

            self.sa_innervation = FlatInnervationModule(
                neuron_type="SA",
                receptor_coords=all_coords,
                neuron_centers=sa_centers,
                neurons_per_row=neuron_cfg["sa_neurons"],
                xlim=xlim,
                ylim=ylim,
                innervation_method=innervation_cfg.get("sa_method", "gaussian"),
                connections_per_neuron=innervation_cfg["receptors_per_neuron"],
                sigma_d_mm=innervation_cfg["sa_spread"],
                weight_range=tuple(innervation_cfg["connection_strength"]),
                seed=innervation_cfg["sa_seed"],
                device=self.device,
            )

            self.ra_innervation = FlatInnervationModule(
                neuron_type="RA",
                receptor_coords=all_coords,
                neuron_centers=ra_centers,
                neurons_per_row=neuron_cfg["ra_neurons"],
                xlim=xlim,
                ylim=ylim,
                innervation_method=innervation_cfg.get("ra_method", "gaussian"),
                connections_per_neuron=innervation_cfg["receptors_per_neuron"],
                sigma_d_mm=innervation_cfg["ra_spread"],
                weight_range=tuple(innervation_cfg["connection_strength"]),
                seed=innervation_cfg["ra_seed"],
                device=self.device,
            )

            self.sa2_innervation = FlatInnervationModule(
                neuron_type="SA2",
                receptor_coords=all_coords,
                neuron_centers=sa2_centers,
                neurons_per_row=neuron_cfg["sa2_neurons"],
                xlim=xlim,
                ylim=ylim,
                innervation_method=innervation_cfg.get("sa2_method", "gaussian"),
                connections_per_neuron=innervation_cfg.get("sa2_connections", 500),
                sigma_d_mm=innervation_cfg["sa2_spread"],
                weight_range=tuple(innervation_cfg["sa2_weights"]),
                seed=innervation_cfg["sa2_seed"],
                device=self.device,
            )
        else:
            # Default: Grid-based innervation (backward compatible)
            self.sa_innervation = InnervationModule(
                neuron_type="SA",
                grid_manager=self.grid_manager,
                neurons_per_row=neuron_cfg["sa_neurons"],
                connections_per_neuron=innervation_cfg["receptors_per_neuron"],
                sigma_d_mm=innervation_cfg["sa_spread"],
                weight_range=tuple(innervation_cfg["connection_strength"]),
                seed=innervation_cfg["sa_seed"],
                neuron_centers=sa_centers,
            )

            self.ra_innervation = InnervationModule(
                neuron_type="RA",
                grid_manager=self.grid_manager,
                neurons_per_row=neuron_cfg["ra_neurons"],
                connections_per_neuron=innervation_cfg["receptors_per_neuron"],
                sigma_d_mm=innervation_cfg["ra_spread"],
                weight_range=tuple(innervation_cfg["connection_strength"]),
                seed=innervation_cfg["ra_seed"],
                neuron_centers=ra_centers,
            )

            self.sa2_innervation = InnervationModule(
                neuron_type="SA2" if sa2_centers is not None else "SA",
                grid_manager=self.grid_manager,
                neurons_per_row=neuron_cfg["sa2_neurons"],
                connections_per_neuron=innervation_cfg.get("sa2_connections", 500),
                sigma_d_mm=innervation_cfg["sa2_spread"],
                weight_range=tuple(innervation_cfg["sa2_weights"]),
                seed=innervation_cfg["sa2_seed"],
                neuron_centers=sa2_centers,
            )

    def _create_filters(self):
        """Create filters with configuration"""
        filter_cfg = self.config["filters"]
        dt = self.config["neurons"]["dt"]

        # Use registry to create filters (with fallback for backward compatibility)
        try:
            sa_filter_cls = FILTER_REGISTRY.get_class("sa")
            self.sa_filter = sa_filter_cls(
                tau_r=filter_cfg["sa_tau_r"],
                tau_d=filter_cfg["sa_tau_d"],
                k1=filter_cfg["sa_k1"],
                k2=filter_cfg["sa_k2"],
                dt=dt,
            )
        except KeyError:
            self.sa_filter = SAFilterTorch(
                tau_r=filter_cfg["sa_tau_r"],
                tau_d=filter_cfg["sa_tau_d"],
                k1=filter_cfg["sa_k1"],
                k2=filter_cfg["sa_k2"],
                dt=dt,
            )

        try:
            ra_filter_cls = FILTER_REGISTRY.get_class("ra")
            self.ra_filter = ra_filter_cls(
                tau_RA=filter_cfg["ra_tau_ra"], k3=filter_cfg["ra_k3"], dt=dt
            )
        except KeyError:
            self.ra_filter = RAFilterTorch(
                tau_RA=filter_cfg["ra_tau_ra"], k3=filter_cfg["ra_k3"], dt=dt
            )

        # SA2 is just a scaling factor (not a filter)
        self.use_sa2_filter = False  # SA2 uses simple scaling, not a filter
        self.sa2_scale = filter_cfg["sa2_scale"]

    def _create_neurons(self):
        """Create neuron models with configuration"""
        neuron_cfg = self.config["neuron_params"]
        neuron_top_cfg = self.config.get("neurons", {})
        dt = neuron_top_cfg.get("dt", 0.5)

        # Handle Equation DSL
        if neuron_top_cfg.get("type") == "dsl":
            solver_cfg = self.config.get("solver", {})
            solver_type = solver_cfg.get("type", "euler")
            solver_args = solver_cfg.get("config", {})

            # Instantiate adaptive solver if requested
            solver = solver_type
            if solver_type == "adaptive":
                solver = AdaptiveSolver(
                    method=solver_args.get("method", "dopri5"),
                    rtol=solver_args.get("rtol", 1e-5),
                    atol=solver_args.get("atol", 1e-7),
                    dt=dt  # Initial hint
                )

            model = NeuronModel(
                equations=neuron_top_cfg["equations"],
                threshold=neuron_top_cfg["threshold"],
                reset=neuron_top_cfg["reset"],
                parameters=neuron_top_cfg.get("parameters", {}),
                state_vars=neuron_top_cfg.get("state_vars", {"v": -65.0, "u": 0.0})
            )
            
            # Compile separate instances for each population
            self.sa_neuron = model.compile(solver=solver, dt=dt, device=self.device)
            self.ra_neuron = model.compile(solver=solver, dt=dt, device=self.device)
            self.sa2_neuron = model.compile(solver=solver, dt=dt, device=self.device)
            
            # Initialize legacy params to None/safe values to avoid attribute errors if accessed
            self.a_params = None
            self.b_params = None
            self.c_params = None
            self.d_params = None
            self.threshold_val = (30.0, 0.0)
            return

        # Use registry to create neurons (with fallback for backward compatibility)
        try:
            neuron_cls = NEURON_REGISTRY.get_class("izhikevich")
        except KeyError:
            neuron_cls = IzhikevichNeuronTorch

        # SA neurons with individual parameters
        self.sa_neuron = neuron_cls(
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
        self.ra_neuron = neuron_cls(
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
        self.sa2_neuron = neuron_cls(
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
        elif stimulus_type == "texture":
            return self._generate_texture_stimulus(**stimulus_params)
        elif stimulus_type == "moving":
            return self._generate_moving_stimulus(**stimulus_params)
        elif stimulus_type == "timeline":
            return self._generate_timeline_stimulus(**stimulus_params)
        elif stimulus_type == "repeated_pattern":
            return self._generate_repeated_pattern_stimulus(**stimulus_params)
        else:
            raise ValueError(f"Unknown stimulus type: {stimulus_type}")

    def _generate_texture_stimulus(self, **params):
        """Generate static texture stimulus (Gabor or Gratings)"""
        texture_type = params.get("pattern", "gabor")
        duration = params.get("duration", 200.0)
        dt = params.get("dt", self.config["neurons"]["dt"])
        
        # Default spatial params
        center_x = params.get("center_x", 0.0)
        center_y = params.get("center_y", 0.0)
        amplitude = params.get("amplitude", 30.0)
        
        xx, yy = self.grid_manager.get_coordinates()
        
        if texture_type == "gabor":
            spatial_stimulus = gabor_texture(
                xx, yy, 
                center_x=center_x, center_y=center_y,
                amplitude=amplitude,
                wavelength=params.get("wavelength", 2.0),
                orientation=params.get("orientation", 0.0), # rad
                phase=params.get("phase", 0.0),
                sigma=params.get("sigma", 2.0),
                device=self.device
            )
        elif texture_type == "grating":
             # Use edge_grating if available, else fallback
             from sensoryforge.stimuli.texture import edge_grating
             spatial_stimulus = edge_grating(
                 xx, yy, 
                 orientation=params.get("orientation", 0.0),
                 spacing=params.get("spacing", 2.0),
                 count=params.get("count", 5),
                 edge_width=params.get("edge_width", 0.05),
                 amplitude=amplitude
             )
        else:
             spatial_stimulus = gaussian_pressure_torch(xx, yy, center_x, center_y, amplitude, 1.0)

        # Create time arrays
        n_timesteps = int(duration / dt)
        time_array = torch.arange(n_timesteps, dtype=torch.float32, device=self.device) * dt
        
        # Add temporal envelope
        temporal_profile = torch.ones(n_timesteps, device=self.device)
        
        # Vectorized: broadcast spatial [H, W] × temporal [T] → [1, T, H, W]
        # (resolves ReviewFinding#H7)
        stimulus_sequence = (
            spatial_stimulus.unsqueeze(0).unsqueeze(0)
            * temporal_profile.view(1, -1, 1, 1)
        )

        return stimulus_sequence, time_array, temporal_profile

    def _generate_moving_stimulus(self, **params):
        """Generate moving stimulus (blob following trajectory)"""
        from sensoryforge.stimuli.moving import linear_motion, circular_motion
        
        motion_type = params.get("motion_type", "linear")
        duration = params.get("duration", 200.0)
        dt = params.get("dt", self.config["neurons"]["dt"])
        n_timesteps = int(duration / dt)
        
        # Probe parameters
        amplitude = params.get("amplitude", 30.0)
        sigma = params.get("sigma", 1.0)
        
        # Generate trajectory
        if motion_type == "linear":
            start = params.get("start", (-2.0, 0.0))
            end = params.get("end", (2.0, 0.0))
            trajectory = linear_motion(start, end, n_timesteps, device=self.device)
        elif motion_type == "circular":
            center = params.get("center", (0.0, 0.0))
            radius = params.get("radius", 2.0)
            trajectory = circular_motion(center, radius, n_timesteps, device=self.device)
        else:
             trajectory = torch.zeros((n_timesteps, 2), device=self.device)
             
        # Generate stimulus sequence
        xx, yy = self.grid_manager.get_coordinates()
        n_x, n_y = self.grid_manager.grid_size
        stimulus_sequence = torch.zeros((1, n_timesteps, n_x, n_y), device=self.device)
        time_array = torch.arange(n_timesteps, dtype=torch.float32, device=self.device) * dt
        temporal_profile = torch.ones(n_timesteps, device=self.device)
        
        for t in range(n_timesteps):
            cx, cy = trajectory[t, 0], trajectory[t, 1]
            stimulus_sequence[0, t] = gaussian_pressure_torch(xx, yy, cx, cy, amplitude, sigma)
            
        return stimulus_sequence, time_array, temporal_profile

    def _generate_timeline_stimulus(self, **params):
        """Generate a timeline stimulus composed of sub-stimuli at different onsets.

        Delegates to :class:`TimelineStimulus`.  The caller provides either a
        pre-built :class:`TimelineStimulus` object or a list of sub-stimulus
        specification dicts.

        Args:
            timeline_stimulus: Pre-built :class:`TimelineStimulus` instance.
                If provided, ``sub_stimuli`` is ignored.
            sub_stimuli: List of dicts, each with ``'stimulus'`` (a
                :class:`BaseStimulus`), ``'onset_ms'``, ``'duration_ms'``,
                and optionally ``'envelope'``.
            duration: Total timeline duration in ms (default 500).
            dt: Time step in ms (default from config).

        Returns:
            Tuple of ``(stimulus_sequence, time_array, temporal_profile)``.
        """
        duration = params.get("duration", 500.0)
        dt = params.get("dt", self.config["neurons"]["dt"])
        n_timesteps = int(duration / dt)

        timeline = params.get("timeline_stimulus")
        if timeline is None:
            sub_stimuli = params.get("sub_stimuli", [])
            timeline = TimelineStimulus(
                sub_stimuli=sub_stimuli,
                total_time_ms=duration,
                dt_ms=dt,
                device=self.device,
            )

        xx, yy = self.grid_manager.get_coordinates()
        grid_h, grid_w = xx.shape

        # Step through the timeline building each frame
        timeline.reset_state()
        frames = []
        for _ in range(n_timesteps):
            frame = timeline(xx, yy)  # [H, W]
            frames.append(frame)
            timeline.step()

        # Stack into [1, T, H, W]
        stimulus_sequence = torch.stack(frames, dim=0).unsqueeze(0)
        time_array = torch.arange(
            n_timesteps, dtype=torch.float32, device=self.device
        ) * dt
        temporal_profile = stimulus_sequence[0].amax(dim=(-2, -1))
        if temporal_profile.max() > 0:
            temporal_profile = temporal_profile / temporal_profile.max()

        return stimulus_sequence, time_array, temporal_profile

    def _generate_repeated_pattern_stimulus(self, **params):
        """Generate a repeated-pattern (N×M tiled) stimulus.

        Delegates to :class:`RepeatedPatternStimulus`.

        Args:
            repeated_stimulus: Pre-built :class:`RepeatedPatternStimulus`.
                If provided, other pattern params are ignored.
            base_stimulus: A :class:`BaseStimulus` instance for the base
                pattern.  If absent, a default Gaussian blob is created.
            copies_x: Number of horizontal copies (default 3).
            copies_y: Number of vertical copies (default 2).
            spacing_x: Horizontal spacing in mm (default 0.5).
            spacing_y: Vertical spacing in mm (default 0.5).
            duration: Duration in ms (default 200).
            dt: Time step in ms (default from config).

        Returns:
            Tuple of ``(stimulus_sequence, time_array, temporal_profile)``.
        """
        from sensoryforge.stimuli.builder import StaticStimulus

        duration = params.get("duration", 200.0)
        dt = params.get("dt", self.config["neurons"]["dt"])
        n_timesteps = int(duration / dt)

        repeated = params.get("repeated_stimulus")
        if repeated is None:
            base_stim = params.get("base_stimulus")
            if base_stim is None:
                base_stim = StaticStimulus(
                    stim_type='gaussian',
                    params={
                        'amplitude': params.get("amplitude", 30.0),
                        'sigma': params.get("sigma", 0.5),
                        'center_x': 0.0,
                        'center_y': 0.0,
                    },
                    device=self.device,
                )

            repeated = RepeatedPatternStimulus(
                base_stimulus=base_stim,
                copies_x=params.get("copies_x", 3),
                copies_y=params.get("copies_y", 2),
                spacing_x=params.get("spacing_x", 0.5),
                spacing_y=params.get("spacing_y", 0.5),
                device=self.device,
            )

        xx, yy = self.grid_manager.get_coordinates()
        spatial = repeated(xx, yy)  # [H, W]

        # Expand to time dimension
        time_array = torch.arange(
            n_timesteps, dtype=torch.float32, device=self.device
        ) * dt
        temporal_profile = torch.ones(n_timesteps, device=self.device)
        stimulus_sequence = (
            spatial.unsqueeze(0).unsqueeze(0)
            * temporal_profile.view(1, -1, 1, 1)
        )

        return stimulus_sequence, time_array, temporal_profile

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

        # Vectorized: broadcast spatial [H, W] × temporal [T] → [1, T, H, W]
        # (resolves ReviewFinding#H7)
        stimulus_sequence = (
            spatial_stimulus.unsqueeze(0).unsqueeze(0)
            * temporal_profile.view(1, -1, 1, 1)
        )

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

        # Vectorized: expand constant spatial to all timesteps [1, T, H, W]
        # (resolves ReviewFinding#H7)
        stimulus_sequence = spatial_stimulus.unsqueeze(0).unsqueeze(0).expand(
            1, n_timesteps, -1, -1
        ).clone()

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

        # Vectorized: broadcast spatial [H, W] × temporal [T] → [1, T, H, W]
        # (resolves ReviewFinding#H7)
        stimulus_sequence = (
            spatial_stimulus.unsqueeze(0).unsqueeze(0)
            * temporal_profile.view(1, -1, 1, 1)
        )

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

        # Vectorized: broadcast spatial [H, W] × temporal [T] → [1, T, H, W]
        # (resolves ReviewFinding#H7)
        stimulus_sequence = (
            spatial_stimulus.unsqueeze(0).unsqueeze(0)
            * temporal_profile.view(1, -1, 1, 1)
        )

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

        # Step 4.5: Apply processing layers (Phase 3)
        mechanoreceptor_responses = self.processing_pipeline(
            mechanoreceptor_responses
        )

        # Step 5: Compute neural inputs through innervation
        # FlatInnervationModule expects [B, T, N_receptors] not [B, T, H, W].
        # When using flat innervation + composite grid, sample the grid-based
        # stimulus at each composite receptor coordinate via bilinear interpolation.
        innervation_input = mechanoreceptor_responses
        if isinstance(self.sa_innervation, FlatInnervationModule):
            if innervation_input.ndim == 4 and self.composite_grid is not None:
                innervation_input = self._sample_stimulus_at_receptors(
                    innervation_input
                )
            elif innervation_input.ndim == 4:
                # Fallback: simple flatten if no composite grid
                B, T, H, W = innervation_input.shape
                innervation_input = innervation_input.reshape(B, T, H * W)

        with torch.no_grad():
            sa_inputs = self.sa_innervation(innervation_input)
            ra_inputs = self.ra_innervation(innervation_input)
            sa2_inputs = self.sa2_innervation(innervation_input)

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

    @classmethod
    def from_config(cls, config: dict) -> 'GeneralizedTactileEncodingPipeline':
        """Create pipeline from configuration dictionary.
        
        This is the recommended way to instantiate pipelines from YAML configs,
        supporting all Phase 2 features: CompositeGrid, DSL neurons, extended
        stimuli, and adaptive solvers.
        
        Args:
            config: Configuration dictionary with keys like 'metadata', 'grid',
                'stimuli', 'filters', 'neurons', 'solver', 'gui'.
        
        Returns:
            Initialized GeneralizedTactileEncodingPipeline instance.
        
        Note:
            Currently uses the existing pipeline implementation. Full Phase 2
            support (CompositeGrid, DSL neurons, adaptive solvers) will be
            integrated in future updates while maintaining backward compatibility.
        
        Example:
            >>> config = {
            ...     'pipeline': {'device': 'cpu', 'grid_size': 64},
            ...     'neurons': {'sa_neurons': 20, 'ra_neurons': 30}
            ... }
            >>> pipeline = GeneralizedTactileEncodingPipeline.from_config(config)
        """
        # For now, delegate to __init__ with config_dict
        # Future enhancement: handle CompositeGrid, DSL neurons, adaptive solvers
        return cls(config_dict=config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'GeneralizedTactileEncodingPipeline':
        """Create pipeline from YAML configuration file.
        
        Convenience method that loads YAML and delegates to __init__.
        
        Args:
            yaml_path: Path to YAML configuration file.
        
        Returns:
            Initialized GeneralizedTactileEncodingPipeline instance.
        
        Example:
            >>> pipeline = GeneralizedTactileEncodingPipeline.from_yaml('config.yml')
            >>> results = pipeline.run(duration_ms=1000)
        
        Resolves ReviewFinding#M2.
        """
        # Pass the path directly to __init__'s config_path parameter
        return cls(config_path=yaml_path)

    def get_pipeline_info(self):
        """Get comprehensive information about the pipeline configuration"""
        info = {
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
                "sa2_scale": self.sa2_scale,
            },
        }
        # Phase 3 additions
        if self.composite_grid is not None:
            info["composite_grid"] = {
                "layers": self.composite_grid.list_layers(),
                "total_receptors": sum(
                    self.composite_grid.get_layer_count(n)
                    for n in self.composite_grid.list_layers()
                ),
            }
        info["processing_layers"] = self.processing_pipeline.to_dict()
        info["innervation_type"] = (
            "flat" if isinstance(self.sa_innervation, FlatInnervationModule)
            else "grid"
        )
        return info


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
