"""Canonical configuration schema for SensoryForge.

This module defines the unified configuration format that both GUI and CLI
consume. It ensures round-trip fidelity: GUI save → YAML → CLI load → same results.

The canonical schema supports:
- Multiple grid layers (standard, composite, Poisson, hexagonal)
- N populations with per-population innervation, filter, neuron, and solver config
- Stimulus definitions with all parameters
- Simulation settings (device, solver, dt)

Example:
    >>> from sensoryforge.config.schema import SensoryForgeConfig
    >>> config = SensoryForgeConfig.from_dict(yaml_dict)
    >>> yaml_str = config.to_yaml()
    >>> config2 = SensoryForgeConfig.from_yaml(yaml_str)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class GridConfig:
    """Configuration for a single receptor grid layer.

    Attributes:
        name: Unique identifier for this grid layer.
        arrangement: Grid arrangement type (grid, poisson, hex,
            jittered, blue_noise).
        rows: Number of rows (for grid arrangement).
        cols: Number of columns (for grid arrangement).
        spacing: Spacing between receptors in mm.
        density: Receptor density in receptors/mm² (for Poisson/hex).
        center_x: X-coordinate of grid center in mm.
        center_y: Y-coordinate of grid center in mm.
        color: RGBA color tuple [r, g, b, a] for visualization.
        visible: Whether this grid layer is visible in the GUI.
    """
    name: str
    arrangement: str = "grid"  # grid, poisson, hex, jittered, blue_noise
    rows: Optional[int] = None
    cols: Optional[int] = None
    spacing: float = 0.15  # mm
    density: Optional[float] = None  # receptors/mm²
    center_x: float = 0.0
    center_y: float = 0.0
    color: List[int] = field(
        default_factory=lambda: [66, 135, 245, 200]
    )
    visible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for YAML serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GridConfig:
        """Create from dict (e.g., from YAML).
        
        Handles both GridEntry format (center as [x, y] list) and GridConfig
        format (center_x, center_y).
        """
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in data:
                kwargs[field_name] = data[field_name]
        
        # Handle GridEntry format: center and offset as lists
        if "center" in data and isinstance(data["center"], list):
            kwargs["center_x"] = data["center"][0]
            kwargs["center_y"] = data["center"][1]
        if "offset" in data and isinstance(data["offset"], list):
            # Apply offset to center
            if "center_x" not in kwargs:
                kwargs["center_x"] = 0.0
            if "center_y" not in kwargs:
                kwargs["center_y"] = 0.0
            kwargs["center_x"] += data["offset"][0]
            kwargs["center_y"] += data["offset"][1]
        
        return cls(**kwargs)


@dataclass
class PopulationConfig:
    """Configuration for a single neuron population.

    This carries all parameters needed to instantiate a population:
    - Innervation method and parameters
    - Neuron model and parameters
    - Filter method and parameters
    - Solver configuration (for DSL neurons)
    - Grid arrangement and layout

    Attributes:
        name: Unique identifier for this population.
        neuron_type: Type identifier (SA, RA, SA2, or custom).
        target_grid: Name of the grid layer this population connects to.
        innervation_method: Method (gaussian, one_to_one, uniform,
            distance_weighted).
        connections_per_neuron: Number of receptor connections per neuron.
        sigma_d_mm: Gaussian spread in mm (for gaussian method).
        distance_weight_randomness_pct: Randomness percentage (0-100).
        use_distance_weights: Whether to use distance-based weighting.
        far_connection_fraction: Fraction of "far" connections.
        far_sigma_factor: Sigma multiplier for far connections.
        max_distance_mm: Maximum connection distance in mm.
        decay_function: Distance decay function (exponential, linear, etc.).
        decay_rate: Decay rate parameter.
        weight_range: [min, max] weight range.
        edge_offset: Edge offset in mm.
        neuron_arrangement: Arrangement (grid, poisson, hex, jittered,
            blue_noise).
        neurons_per_row: Neurons per row (for grid arrangement).
        neuron_rows: Number of rows (independent of neurons_per_row).
        neuron_cols: Number of columns (independent of neurons_per_row).
        neuron_jitter_factor: Jitter amount for jittered arrangements.
        neuron_model: Model type (Izhikevich, AdEx, MQIF, FA, SA, DSL).
        model_params: Model-specific parameters dict.
        dsl_config: DSL configuration dict (equations, threshold, reset,
            parameters).
        filter_method: Filter type (SA, RA, none/identity).
        filter_params: Filter-specific parameters dict.
        solver_config: Solver configuration dict (type, method, rtol, atol).
        noise_std: Membrane noise standard deviation.
        noise_mean: Membrane noise mean.
        noise_seed: Random seed for noise.
        color: RGBA color tuple [r, g, b, a].
        visible: Whether this population is visible in the GUI.
        enabled: Whether this population is enabled for simulation.
        input_gain: Input gain multiplier.
        seed: Random seed for innervation generation.
    """
    name: str
    neuron_type: str = "SA"
    target_grid: Optional[str] = None

    # Innervation parameters
    innervation_method: str = "gaussian"  # gaussian, one_to_one, etc.
    connections_per_neuron: int = 28
    sigma_d_mm: float = 0.3
    distance_weight_randomness_pct: float = 0.0
    use_distance_weights: bool = False
    far_connection_fraction: float = 0.0
    far_sigma_factor: float = 5.0
    max_distance_mm: float = 1.0
    decay_function: str = "exponential"
    decay_rate: float = 2.0
    weight_range: List[float] = field(
        default_factory=lambda: [0.05, 1.0]
    )
    edge_offset: float = 0.0

    # Neuron layout
    neuron_arrangement: str = "grid"  # grid, poisson, hex, etc.
    neurons_per_row: int = 10
    neuron_rows: Optional[int] = None
    neuron_cols: Optional[int] = None
    neuron_jitter_factor: float = 0.0

    # Neuron model
    neuron_model: str = "Izhikevich"  # Izhikevich, AdEx, MQIF, FA, SA, DSL
    model_params: Dict[str, Any] = field(default_factory=dict)
    dsl_config: Optional[Dict[str, Any]] = None

    # Filter
    filter_method: str = "none"  # SA, RA, none
    filter_params: Dict[str, Any] = field(default_factory=dict)

    # Solver (for DSL neurons)
    solver_config: Optional[Dict[str, Any]] = None

    # Noise
    noise_std: float = 0.0
    noise_mean: float = 0.0
    noise_seed: Optional[int] = None

    # Visualization
    color: List[int] = field(
        default_factory=lambda: [66, 135, 245, 255]
    )
    visible: bool = True

    # Simulation control
    enabled: bool = True
    input_gain: float = 1.0
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for YAML serialization."""
        result = asdict(self)
        # Remove None values for cleaner YAML
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PopulationConfig:
        """Create from dict (e.g., from YAML)."""
        # Handle missing optional fields
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in data:
                kwargs[field_name] = data[field_name]
        return cls(**kwargs)


@dataclass
class StimulusConfig:
    """Configuration for stimulus generation.

    Attributes:
        name: Stimulus name/identifier.
        type: Stimulus type (gaussian, texture, moving, timeline,
            repeated_pattern).
        motion: Motion type (static, moving).
        composition_mode: Composition mode for multi-stimulus.
        target_layer: Target grid layer name.
        stimuli: List of sub-stimulus config dicts (for timeline/composition).
        start: [x, y] start position in mm.
        end: [x, y] end position in mm.
        spread: Spatial spread in mm.
        orientation_deg: Orientation in degrees.
        amplitude: Stimulus amplitude.
        speed_mm_s: Speed for moving stimuli.
        ramp_up_ms: Ramp-up duration in ms.
        plateau_ms: Plateau duration in ms.
        ramp_down_ms: Ramp-down duration in ms.
        pattern: Pattern type (gabor, grating).
        wavelength: Wavelength for texture patterns.
        phase: Phase offset.
        sigma: Gaussian sigma for gabor.
        motion_type: Motion type (linear, circular).
        center: Center point for circular motion.
        radius: Radius for circular motion.
    """
    name: str = "Stimulus"
    type: str = "gaussian"  # gaussian, texture, moving, timeline, repeated_pattern
    motion: str = "static"  # static, moving
    composition_mode: str = "single"
    target_layer: Optional[str] = None
    stimuli: List[Dict[str, Any]] = field(default_factory=list)
    
    # Spatial parameters
    start: List[float] = field(default_factory=lambda: [0.0, 0.0])
    end: List[float] = field(default_factory=lambda: [0.0, 0.0])
    spread: float = 1.0
    orientation_deg: float = 0.0
    amplitude: float = 30.0
    
    # Temporal parameters
    speed_mm_s: float = 10.0
    ramp_up_ms: float = 10.0
    plateau_ms: float = 800.0
    ramp_down_ms: float = 10.0
    
    # Texture-specific
    pattern: str = "gabor"  # gabor, grating
    wavelength: float = 2.0
    phase: float = 0.0
    sigma: float = 2.0
    
    # Moving-specific
    motion_type: str = "linear"  # linear, circular
    center: List[float] = field(default_factory=lambda: [0.0, 0.0])
    radius: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for YAML serialization."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StimulusConfig:
        """Create from dict (e.g., from YAML)."""
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in data:
                kwargs[field_name] = data[field_name]
        return cls(**kwargs)


@dataclass
class SimulationConfig:
    """Configuration for simulation execution.

    Attributes:
        device: Device to run on (cpu, cuda, mps).
        dt: Time step in ms.
        solver: Global solver config (type, method, rtol, atol).
        duration_ms: Simulation duration in ms (optional, can be inferred
            from stimulus).
    """
    device: str = "cpu"
    dt: float = 1.0  # ms
    solver: Dict[str, Any] = field(default_factory=lambda: {"type": "euler"})
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for YAML serialization."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimulationConfig:
        """Create from dict (e.g., from YAML)."""
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in data:
                kwargs[field_name] = data[field_name]
        return cls(**kwargs)


@dataclass
class SensoryForgeConfig:
    """Canonical configuration schema for SensoryForge.

    This is the single source of truth for configuration. Both GUI and CLI
    consume this format, ensuring round-trip fidelity.

    Attributes:
        grids: List of grid layer configurations.
        populations: List of population configurations.
        stimulus: Stimulus configuration.
        simulation: Simulation configuration.
        metadata: Optional metadata dict (version, created timestamp, etc.).
    """
    grids: List[GridConfig] = field(default_factory=list)
    populations: List[PopulationConfig] = field(default_factory=list)
    stimulus: StimulusConfig = field(default_factory=StimulusConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for YAML serialization.

        Returns:
            Dictionary suitable for yaml.dump().
        """
        return {
            "metadata": self.metadata,
            "grids": [g.to_dict() for g in self.grids],
            "populations": [p.to_dict() for p in self.populations],
            "stimulus": self.stimulus.to_dict(),
            "simulation": self.simulation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SensoryForgeConfig:
        """Create from dict (e.g., from YAML).

        Args:
            data: Dictionary loaded from YAML or GUI get_config().

        Returns:
            SensoryForgeConfig instance.
        """
        grids = [GridConfig.from_dict(g) for g in data.get("grids", [])]
        populations = [
            PopulationConfig.from_dict(p)
            for p in data.get("populations", [])
        ]
        stimulus = StimulusConfig.from_dict(data.get("stimulus", {}))
        simulation = SimulationConfig.from_dict(data.get("simulation", {}))
        metadata = data.get("metadata", {})

        return cls(
            grids=grids,
            populations=populations,
            stimulus=stimulus,
            simulation=simulation,
            metadata=metadata,
        )

    def to_yaml(self) -> str:
        """Serialize to YAML string.

        Returns:
            YAML-formatted string.
        """
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> SensoryForgeConfig:
        """Load from YAML string.

        Args:
            yaml_str: YAML-formatted string.

        Returns:
            SensoryForgeConfig instance.
        """
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("YAML did not produce a dict")
        return cls.from_dict(data)
