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

import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from sensoryforge.config.defaults import DEFAULT_INTEGRATE_DT_MS


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
        seed: Seed for the random jitter of the ``jittered_grid``,
            ``blue_noise`` and ``poisson`` arrangements (F-050). ``None``
            draws from the global RNG (not reproducible).
        channels: Named sensor channels/planes carried by this grid (Phase 2,
            Wave L1). ``["value"]`` (the default) means a single, unnamed
            channel and is omitted from :meth:`to_dict` output so existing
            single-channel configs are unchanged byte for byte. Names must
            be non-empty, unique, valid Python identifiers.
        coords_file: Optional path to an ``[M, 2]`` CSV or ``.pt`` file of
            receptor coordinates in mm (Wave L1). When set, the grid is
            built from these coordinates (via
            ``CompositeReceptorGrid.add_layer_with_coords``) instead of
            ``rows``/``cols``/``spacing``.
        layers: For ``arrangement == "composite"`` (Wave L4), the ordered
            list of layer specs building a :class:`CompositeReceptorGrid`.
            Each entry is a dict with a required ``name`` and either
            ``density`` (+ optional ``arrangement``, ``offset``, ``seed``,
            ``color``) or ``coordinates`` (an ``[n, 2]`` list) or
            ``coords_file``. Layer order is the receptor-index contract.
    """

    name: str
    arrangement: str = "grid"  # grid, poisson, hex, jittered, blue_noise
    rows: Optional[int] = None
    cols: Optional[int] = None
    spacing: float = 0.15  # mm
    density: Optional[float] = None  # receptors/mm²
    center_x: float = 0.0
    center_y: float = 0.0
    color: List[int] = field(default_factory=lambda: [66, 135, 245, 200])
    visible: bool = True
    seed: Optional[int] = None
    channels: List[str] = field(default_factory=lambda: ["value"])
    coords_file: Optional[str] = None
    layers: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate channel names (Wave L1).

        Raises:
            ValueError: If a channel name is empty, not a valid identifier,
                or repeated -- named with the grid and the offending entry.
        """
        seen: set = set()
        for entry in self.channels:
            if not isinstance(entry, str) or not entry:
                raise ValueError(
                    f"Grid {self.name!r}: channel names must be non-empty "
                    f"strings, got {entry!r}"
                )
            if not entry.isidentifier():
                raise ValueError(
                    f"Grid {self.name!r}: channel name {entry!r} is not a "
                    "valid identifier"
                )
            if entry in seen:
                raise ValueError(
                    f"Grid {self.name!r}: duplicate channel name {entry!r}"
                )
            seen.add(entry)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for YAML serialization.

        ``channels == ["value"]`` (the single-channel default) and
        ``coords_file is None`` and ``layers == []`` are omitted so
        pre-Wave-L configs round-trip byte for byte (Wave L1).
        """
        result = asdict(self)
        if result.get("channels") == ["value"]:
            del result["channels"]
        if result.get("coords_file") is None:
            del result["coords_file"]
        if result.get("layers") == []:
            del result["layers"]
        return result

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
            distance_weighted, template, imported, or a plugin's name).
        resolvable_distance_mm: ``d`` for the ``template`` builder (sigma =
            d/pi, pitch = d); ``None`` otherwise.
        innervation_params: Extra builder parameters passed through to the
            registered builder (``template``: k, normalize, weight_scale,
            edge_offset_mm, sigma_mm/pitch_mm; ``imported``: path; plugins:
            anything). Unknown keys for a builder are dropped.
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
        readout: How to read out a DSL population (Phase 2, N3): "auto"
            (default) infers analog when dsl_config has no threshold, else
            spiking; "spiking" or "analog" force it, raising a ValueError
            when the dsl_config is incompatible (e.g. "analog" with a
            threshold present, or "spiking" with none). Ignored for
            non-DSL neuron models.
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
    use_distance_weights: bool = True
    far_connection_fraction: float = 0.0
    far_sigma_factor: float = 5.0
    max_distance_mm: float = 1.0
    decay_function: str = "exponential"
    decay_rate: float = 2.0
    weight_range: List[float] = field(default_factory=lambda: [0.05, 1.0])
    edge_offset: float = 0.0
    resolvable_distance_mm: Optional[float] = None
    innervation_params: Dict[str, Any] = field(default_factory=dict)
    target_layers: Optional[List[str]] = None

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
    readout: str = "auto"  # auto, spiking, analog -- DSL populations only

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
    color: List[int] = field(default_factory=lambda: [66, 135, 245, 255])
    visible: bool = True

    # Simulation control
    enabled: bool = True
    input_gain: float = 50.0
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for YAML serialization.

        Returns:
            Dictionary representation suitable for YAML export.
            None values are removed for cleaner YAML output.

        Example:
            >>> config = PopulationConfig(name="SA", neurons_per_row=10)
            >>> config_dict = config.to_dict()
            >>> # Can be saved to YAML or passed to pipeline
        """
        result = asdict(self)
        # Remove None values for cleaner YAML
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PopulationConfig:
        """Create from dict (e.g., from YAML).

        Handles missing optional fields by using defaults from dataclass definition.

        Args:
            data: Dictionary with population configuration fields.
                Can include any subset of PopulationConfig fields.

        Returns:
            PopulationConfig instance with provided values and defaults.

        Example:
            >>> data = {
            ...     "name": "SA Population",
            ...     "neuron_model": "izhikevich",
            ...     "filter_method": "sa",
            ...     "neurons_per_row": 10,
            ... }
            >>> config = PopulationConfig.from_dict(data)
        """
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

    # Sensor channel (Phase 2, Wave L2): which named plane of the target
    # grid's `channels` this stimulus drives. `None` (default) means the
    # single/first channel -- existing single-channel configs are
    # unaffected. Several stimuli with different `channel` values compose
    # into one multi-channel tensor; planes with no stimulus are zero.
    channel: Optional[str] = None

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


def validate_dt_ms(dt_ms: float, integrate_dt_ms: float) -> None:
    """Reject a record step that isn't a whole multiple of the integration step.

    Shared by :meth:`SimulationConfig.__post_init__` and
    :meth:`~sensoryforge.core.simulation_engine.SimulationEngine._run_pop_from_drive`
    (for direct callers that bypass ``SimulationConfig``) so both raise
    identically (F-042).

    Args:
        dt_ms: Record step in ms.
        integrate_dt_ms: Neuron integration step in ms.

    Raises:
        ValueError: If ``dt_ms < integrate_dt_ms``, or ``dt_ms`` is not a
            whole multiple of ``integrate_dt_ms`` within ``1e-6``.
    """
    if dt_ms < integrate_dt_ms:
        raise ValueError(
            f"dt_ms ({dt_ms}) must be >= integrate_dt_ms ({integrate_dt_ms})"
        )
    ratio = dt_ms / integrate_dt_ms
    if abs(ratio - round(ratio)) > 1e-6:
        raise ValueError(
            f"dt_ms ({dt_ms}) must be a whole multiple of integrate_dt_ms "
            f"({integrate_dt_ms}); got dt_ms / integrate_dt_ms = {ratio}"
        )


@dataclass
class SimulationConfig:
    """Configuration for simulation execution.

    Attributes:
        device: Device to run on (cpu, cuda, mps).
        dt_ms: Record step in ms -- the time resolution of the filter,
            stimulus, and returned spike/voltage arrays. Was named ``dt``;
            ``dt`` is still accepted, as a deprecated constructor keyword
            and as a ``from_dict``/YAML key alias.
        integrate_dt_ms: Neuron integration step in ms (F-008). The neuron
            model is stepped at this (generally finer) resolution, holding
            the drive constant across ``round(dt_ms / integrate_dt_ms)``
            sub-steps per record bin, matching pressure-simulation's
            ``encoding/encode_runner.run_encoding`` exactly. Defaults to
            0.05 ms, its hard-coded native Izhikevich integration step.
        solver: Global solver config (type, method, rtol, atol).
        duration_ms: Simulation duration in ms (optional, can be inferred
            from stimulus).
        dt: Deprecated alias for ``dt_ms`` (F-008/E10). Emits
            ``DeprecationWarning`` when used; raises ``ValueError`` if both
            ``dt`` and a non-default, different ``dt_ms`` are given.
    """

    device: str = "cpu"
    dt_ms: float = 1.0  # ms
    integrate_dt_ms: float = DEFAULT_INTEGRATE_DT_MS  # ms
    solver: Dict[str, Any] = field(default_factory=lambda: {"type": "euler"})
    duration_ms: Optional[float] = None
    dt: Optional[float] = None  # deprecated alias for dt_ms

    def __post_init__(self) -> None:
        """Resolve the deprecated ``dt`` alias, then validate (F-042).

        A record step that is not a whole multiple of the integration step
        silently rescales neuron time in
        :meth:`SimulationEngine._run_pop_from_drive` (F-008's
        ``n = round(dt_ms / integrate_dt_ms)`` sub-step count): e.g. 0.12 ms
        record bins would integrate at 0.10 ms instead of the requested
        0.12 ms. Catch it at config-construction time instead.
        """
        if self.dt is not None:
            if self.dt_ms != 1.0 and self.dt_ms != self.dt:
                raise ValueError(
                    f"SimulationConfig got both dt_ms={self.dt_ms!r} and the "
                    f"deprecated dt={self.dt!r} with different values; pass "
                    "only dt_ms"
                )
            warnings.warn(
                "SimulationConfig(dt=...) is deprecated; use dt_ms=... instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self.dt_ms = self.dt
        validate_dt_ms(self.dt_ms, self.integrate_dt_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for YAML serialization."""
        result = asdict(self)
        result.pop("dt", None)  # deprecated alias; never round-tripped
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimulationConfig:
        """Create from dict (e.g., from YAML).

        Accepts the legacy ``dt`` key as an alias for ``dt_ms``: routed
        through the (deprecated) ``dt`` constructor keyword, so loading a
        config that still uses ``dt`` emits the same ``DeprecationWarning``
        as calling ``SimulationConfig(dt=...)`` directly.
        """
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
            PopulationConfig.from_dict(p) for p in data.get("populations", [])
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
            YAML-formatted string suitable for saving to file or CLI usage.

        Example:
            >>> config = SensoryForgeConfig(...)
            >>> yaml_str = config.to_yaml()
            >>> with open('config.yml', 'w') as f:
            ...     f.write(yaml_str)
        """
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    @classmethod
    def from_yaml(cls, yaml_str_or_path: str | Path) -> SensoryForgeConfig:
        """Load from a YAML string, or from a path to a YAML file.

        Args:
            yaml_str_or_path: Either YAML-formatted text (file contents or a
                direct string), or a path (``str``/``pathlib.Path``) to a
                ``.yml``/``.yaml`` file to read. A path is detected by
                checking whether it names an existing file; prefer
                :meth:`from_yaml_file` when the caller already knows it has
                a path, to avoid that filesystem check.

        Returns:
            SensoryForgeConfig instance.

        Raises:
            ValueError: If the YAML does not produce a dict.

        Example:
            >>> # From a path (str or Path)
            >>> config = SensoryForgeConfig.from_yaml('config.yml')
            >>>
            >>> # From file contents
            >>> with open('config.yml', 'r') as f:
            ...     config = SensoryForgeConfig.from_yaml(f.read())
            >>>
            >>> # From a string
            >>> yaml_str = '''
            ... grids:
            ...   - name: "Main Grid"
            ...     arrangement: "grid"
            ...     rows: 80
            ... '''
            >>> config = SensoryForgeConfig.from_yaml(yaml_str)
        """
        if isinstance(yaml_str_or_path, Path):
            return cls.from_yaml_file(yaml_str_or_path)
        if "\n" not in yaml_str_or_path:
            try:
                is_path = Path(yaml_str_or_path).is_file()
            except OSError:
                # A string too long for a filesystem path (e.g. ENAMETOOLONG)
                # is never a path — treat it as inline YAML text.
                is_path = False
            if is_path:
                return cls.from_yaml_file(Path(yaml_str_or_path))
        data = yaml.safe_load(yaml_str_or_path)
        if not isinstance(data, dict):
            raise ValueError("YAML did not produce a dict")
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> SensoryForgeConfig:
        """Load from a YAML file path.

        Args:
            path: Path to a ``.yml``/``.yaml`` configuration file.

        Returns:
            SensoryForgeConfig instance.

        Raises:
            ValueError: If the YAML does not produce a dict.

        Example:
            >>> config = SensoryForgeConfig.from_yaml_file('config.yml')
        """
        from sensoryforge.config.yaml_utils import load_config_file

        data = load_config_file(path)
        if not isinstance(data, dict):
            raise ValueError(f"YAML file {path} did not produce a dict")
        return cls.from_dict(data)
