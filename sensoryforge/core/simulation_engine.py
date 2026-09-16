"""Unified simulation execution engine for SensoryForge.

This module provides a single execution engine that works with canonical configs
and can be used by GUI, CLI, and Batch execution paths. It replaces the need
for separate execution logic in each path.

The SimulationEngine:
- Takes canonical SensoryForgeConfig
- Builds grids, populations, innervation, filters, neurons dynamically via registries
- Runs simulations and returns structured results
- Supports both single-stimulus and batch execution

Example:
    >>> from sensoryforge.config.schema import SensoryForgeConfig
    >>> from sensoryforge.core.simulation_engine import SimulationEngine
    >>> config = SensoryForgeConfig.from_yaml("config.yml")
    >>> engine = SimulationEngine(config)
    >>> results = engine.run(stimulus_data)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np

from sensoryforge.config.schema import SensoryForgeConfig, validate_dt_ms
from sensoryforge.config.defaults import resolve_filter_params, resolve_neuron_params
from sensoryforge.register_components import register_all
from sensoryforge.registry import (
    NEURON_REGISTRY,
    FILTER_REGISTRY,
    INNERVATION_REGISTRY,
    STIMULUS_REGISTRY,
    SOLVER_REGISTRY,
    GRID_REGISTRY,
)
from sensoryforge.core.grid import ReceptorGrid, GridManager
from sensoryforge.core.composite_grid import CompositeReceptorGrid, CompositeGrid
from sensoryforge.core.innervation import (
    _grid_lattice_coords,
    build_population_bank,
    create_neuron_centers,
)
from sensoryforge.core.rf_bank import ReceptiveFieldBank
from sensoryforge.neurons.model_dsl import NeuronModel

# Ensure components are registered
register_all()


class SimulationEngine:
    """Unified simulation execution engine.

    This engine takes a canonical SensoryForgeConfig and executes simulations
    using registry-based component creation. It supports N populations dynamically.

    Attributes:
        config: Canonical SensoryForgeConfig instance
        device: PyTorch device for computation
        grids: List of ReceptorGrid or CompositeReceptorGrid instances
        populations: List of population execution contexts; each holds the
            population's ReceptiveFieldBank under "innervation" (and "bank"),
            its filter, neuron model and neuron centres.
    """

    def __init__(
        self,
        config: SensoryForgeConfig,
        device: Optional[torch.device] = None,
    ):
        """Initialize simulation engine from canonical config.

        Args:
            config: Canonical SensoryForgeConfig instance
            device: PyTorch device (defaults to config.simulation.device)
        """
        self.config = config
        self.device = device or torch.device(config.simulation.device)

        # Build grids
        self.grids: List[Any] = []
        self.grid_names: Dict[str, Any] = {}  # Map grid name to grid object
        self.grid_managers: Dict[str, Any] = {}  # Map grid name to GridManager
        self._build_grids()

        # Build populations (innervation, filters, neurons)
        self.populations: List[Dict[str, Any]] = []
        self._build_populations()

    def _build_grids(self) -> None:
        """Build receptor grids from config."""
        for grid_cfg in self.config.grids:
            # Create grid based on arrangement
            arrangement = grid_cfg.arrangement
            grid_name = grid_cfg.name

            if arrangement == "composite":
                # Composite grid with multiple layers
                layers = {}
                # For now, composite grids need special handling
                # This is a placeholder - full implementation needed
                raise NotImplementedError(
                    "Composite grids not yet implemented in SimulationEngine"
                )
            else:
                # Single grid
                # ReceptorGrid takes grid_size as tuple (rows, cols) or int
                rows = grid_cfg.rows or 40
                cols = grid_cfg.cols or 40
                grid_size = (rows, cols)

                # Create ReceptorGrid for coordinate access
                grid = ReceptorGrid(
                    grid_size=grid_size,
                    spacing=grid_cfg.spacing,
                    arrangement=arrangement,
                    center=(grid_cfg.center_x, grid_cfg.center_y),
                    density=grid_cfg.density,
                    device=self.device,
                    seed=grid_cfg.seed,
                )
                self.grids.append(grid)
                self.grid_names[grid_name] = grid

                # Regular GridManager: the receptor lattice the bank is built on
                grid_manager = GridManager(
                    grid_size=grid_size,
                    spacing=grid_cfg.spacing,
                    center=(grid_cfg.center_x, grid_cfg.center_y),
                    device=self.device,
                )
                self.grid_managers[grid_name] = grid_manager

    def _build_populations(self) -> None:
        """Build population execution contexts (innervation, filters, neurons)."""
        for pop_cfg in self.config.populations:
            if not pop_cfg.enabled:
                continue

            # Find target grid
            target_grid_name = pop_cfg.target_grid or (
                self.config.grids[0].name if self.config.grids else None
            )
            if target_grid_name is None:
                raise ValueError(f"Population {pop_cfg.name} has no target grid")

            grid = self.grid_names.get(target_grid_name)
            if grid is None:
                # Use first grid as fallback
                grid = self.grids[0] if self.grids else None
                if grid is None:
                    raise ValueError(f"No grid available for population {pop_cfg.name}")

            # Get receptor coordinates
            if isinstance(grid, CompositeReceptorGrid):
                receptor_coords = grid.get_all_coordinates()
                use_flat = True
            else:
                receptor_coords = grid.get_receptor_coordinates()
                use_flat = False

            # Build neuron arrangement first (needed for innervation)
            # Use neuron_rows/neuron_cols if specified, otherwise use neurons_per_row for square layout
            neuron_rows = (
                pop_cfg.neuron_rows
                if pop_cfg.neuron_rows is not None
                else pop_cfg.neurons_per_row
            )
            neuron_cols = (
                pop_cfg.neuron_cols
                if pop_cfg.neuron_cols is not None
                else pop_cfg.neurons_per_row
            )
            neuron_arrangement = pop_cfg.neuron_arrangement or "grid"

            # Get grid bounds
            if isinstance(grid, CompositeReceptorGrid):
                xlim = grid.xlim
                ylim = grid.ylim
            elif hasattr(grid, "xlim") and hasattr(grid, "ylim"):
                xlim = grid.xlim
                ylim = grid.ylim
            else:
                # Fallback: compute from grid properties
                if hasattr(grid, "spacing") and hasattr(grid, "grid_size"):
                    spacing = grid.spacing
                    if isinstance(grid.grid_size, tuple):
                        n_x, n_y = grid.grid_size
                    else:
                        n_x = n_y = grid.grid_size
                    total_x = (n_x - 1) * spacing
                    total_y = (n_y - 1) * spacing
                    center = grid.center if hasattr(grid, "center") else (0.0, 0.0)
                    xlim = (center[0] - total_x / 2, center[0] + total_x / 2)
                    ylim = (center[1] - total_y / 2, center[1] + total_y / 2)
                else:
                    xlim = (-5.0, 5.0)
                    ylim = (-5.0, 5.0)

            # Receptive fields: one registered builder per population (I6,
            # F-051). The builder's own class decides which of the population
            # parameters it takes (filter_params) and whether it derives the
            # neuron lattice itself (DERIVES_NEURON_CENTERS).
            innervation_method = pop_cfg.innervation_method or "gaussian"
            try:
                builder_cls = INNERVATION_REGISTRY.get_class(innervation_method)
            except KeyError:
                raise ValueError(
                    f"Unknown innervation method: {innervation_method!r}. "
                    f"Registered methods: {sorted(INNERVATION_REGISTRY.list_registered())}"
                ) from None

            if not use_flat:
                # Ordinary grids: the receptor lattice of the matching
                # GridManager (meshgrid, row-major -> receptor k = i*cols + j).
                # Non-grid arrangements are built but their coordinates are
                # not sampled yet (F-010, Wave L) -- same behaviour as
                # before, now with a warning.
                grid_manager = self.grid_managers.get(
                    target_grid_name,
                    self.grid_managers[list(self.grid_managers.keys())[0]],
                )
                receptor_coords = _grid_lattice_coords(grid_manager).reshape(-1, 2)
                if getattr(grid, "arrangement", "grid") != "grid":
                    warnings.warn(
                        f"Population {pop_cfg.name!r}: receptor arrangement "
                        f"{grid.arrangement!r} is built but innervation still "
                        "samples a regular lattice over the grid bounds (F-010; "
                        "real receptor sampling arrives with Wave L).",
                        UserWarning,
                        stacklevel=2,
                    )

            if builder_cls.DERIVES_NEURON_CENTERS:
                warnings.warn(
                    f"Population {pop_cfg.name!r}: innervation_method "
                    f"{innervation_method!r} derives its own neuron lattice; "
                    f"neurons_per_row={pop_cfg.neurons_per_row}, "
                    f"neuron_rows={pop_cfg.neuron_rows}, "
                    f"neuron_cols={pop_cfg.neuron_cols} are ignored.",
                    UserWarning,
                    stacklevel=2,
                )
                neuron_centers = None
            else:
                neuron_centers = create_neuron_centers(
                    neurons_per_row=neuron_rows,  # Used if rows/cols not specified
                    xlim=xlim,
                    ylim=ylim,
                    device=self.device,
                    edge_offset=pop_cfg.edge_offset,
                    sigma=pop_cfg.sigma_d_mm,
                    rows=neuron_rows,
                    cols=neuron_cols,
                    arrangement=neuron_arrangement,
                    seed=pop_cfg.seed,
                    jitter_factor=(
                        pop_cfg.neuron_jitter_factor
                        if hasattr(pop_cfg, "neuron_jitter_factor")
                        else 1.0
                    ),
                )

            bank: ReceptiveFieldBank = build_population_bank(
                receptor_coords=receptor_coords,
                innervation_method=innervation_method,
                neuron_type=pop_cfg.neuron_type,
                neuron_centers=neuron_centers,
                device=self.device,
                **self.builder_params(pop_cfg, grid_path=not use_flat),
            )
            neuron_centers = bank.neuron_centers

            # Build filter -- parameters resolved from the single shared
            # default table (sensoryforge.config.defaults) so the engine and
            # the GUI agree when a population supplies no overrides (F-026).
            filter_method = pop_cfg.filter_method or "none"
            filter_module = None
            if filter_method != "none":
                try:
                    filter_cls = FILTER_REGISTRY.get_class(filter_method)
                    if filter_method.lower() in ("sa", "ra"):
                        filter_params = resolve_filter_params(
                            filter_method, pop_cfg.filter_params
                        )
                    else:
                        filter_params = dict(pop_cfg.filter_params or {})
                    filter_params["dt"] = self.config.simulation.dt_ms
                    filter_module = filter_cls(**filter_params).to(self.device)
                except KeyError:
                    raise ValueError(f"Unknown filter method: {filter_method}")

            # Build neuron model -- F-004: RA/RA-I (Meissner) populations
            # resolve to the fast-spiking Izhikevich preset unless the
            # config already pins a preset or explicit a/b/c/d (SA/SA2 keep
            # the regular-spiking default). See resolve_neuron_params.
            neuron_model_name = pop_cfg.neuron_model or "izhikevich"
            try:
                neuron_cls = NEURON_REGISTRY.get_class(neuron_model_name)
            except KeyError:
                raise ValueError(f"Unknown neuron model: {neuron_model_name}")

            if neuron_cls is NeuronModel:
                # DSL model (F-010, N3): NeuronModel's constructor takes
                # equations/threshold/reset/..., not dt=/noise_std=, so it
                # is built from dsl_config and compiled instead of
                # constructed like the other neuron classes below.
                if not pop_cfg.dsl_config:
                    raise ValueError(
                        f"Population {pop_cfg.name!r} has neuron_model="
                        f"{neuron_model_name!r} (DSL) but no dsl_config. "
                        "Provide dsl_config with at least 'equations'."
                    )
                dsl_model = NeuronModel.from_config(pop_cfg.dsl_config)
                has_threshold = dsl_model.threshold_str is not None
                readout = (pop_cfg.readout or "auto").lower()
                if readout == "auto":
                    pass  # readout follows the model itself (N1/N2)
                elif readout == "analog":
                    if has_threshold:
                        raise ValueError(
                            f"Population {pop_cfg.name!r} readout='analog' "
                            "but its dsl_config defines a threshold; remove "
                            "the threshold or use readout='spiking'."
                        )
                elif readout == "spiking":
                    if not has_threshold:
                        raise ValueError(
                            f"Population {pop_cfg.name!r} readout='spiking' "
                            "but its dsl_config has no threshold; add one "
                            "or use readout='analog'."
                        )
                else:
                    raise ValueError(
                        f"Population {pop_cfg.name!r}: unknown readout "
                        f"{pop_cfg.readout!r}; choose 'auto', 'spiking', "
                        "or 'analog'."
                    )
                # F-008: the neuron integrates at integrate_dt_ms (finer,
                # default 0.05 ms), not the record step dt_ms; sub-stepping
                # happens in _run_pop_from_drive.
                neuron_model = dsl_model.compile(
                    dt=self.config.simulation.integrate_dt_ms,
                    device=str(self.device),
                    noise_std=pop_cfg.noise_std,
                )
            else:
                neuron_params = resolve_neuron_params(
                    neuron_model_name, pop_cfg.neuron_type, pop_cfg.model_params
                )
                # F-008: the neuron integrates at integrate_dt_ms (finer,
                # default 0.05 ms), not the record step dt_ms; sub-stepping
                # happens in _run_pop_from_drive.
                neuron_params["dt"] = self.config.simulation.integrate_dt_ms
                neuron_params["noise_std"] = pop_cfg.noise_std
                neuron_model = neuron_cls(**neuron_params).to(self.device)

            # Store population context
            self.populations.append(
                {
                    "name": pop_cfg.name,
                    "config": pop_cfg,
                    "grid": grid,
                    "innervation": bank,
                    "bank": bank,
                    "filter": filter_module,
                    "neuron": neuron_model,
                    "neuron_centers": neuron_centers,
                }
            )

    @staticmethod
    def builder_params(pop_cfg: Any, *, grid_path: bool = True) -> Dict[str, Any]:
        """Builder parameters a population config supplies (I6).

        The union of every method's parameters; each builder keeps its own
        subset through :meth:`BaseInnervation.filter_params`.
        ``resolvable_distance_mm`` is included when set, and
        ``innervation_params`` is merged last so it can override anything.

        Args:
            pop_cfg: A :class:`~sensoryforge.config.schema.PopulationConfig`.
            grid_path: ``True`` for ordinary grids. That path never applied
                a sigma cutoff (it used ``create_innervation_map_tensor``), so
                ``max_sigma_distance`` is 0 there to keep gaussian weights
                bit-identical to earlier releases; the flat/composite path
                keeps its 3-sigma cutoff.

        Returns:
            Keyword arguments for :func:`build_population_bank`.
        """
        params: Dict[str, Any] = {
            "connections_per_neuron": pop_cfg.connections_per_neuron,
            "sigma_d_mm": pop_cfg.sigma_d_mm,
            "max_sigma_distance": 0.0 if grid_path else 3.0,
            "weight_range": (
                tuple(pop_cfg.weight_range) if pop_cfg.weight_range else (0.1, 1.0)
            ),
            "use_distance_weights": pop_cfg.use_distance_weights,
            "far_connection_fraction": pop_cfg.far_connection_fraction,
            "far_sigma_factor": pop_cfg.far_sigma_factor,
            "max_distance_mm": pop_cfg.max_distance_mm,
            "decay_function": pop_cfg.decay_function,
            "decay_rate": pop_cfg.decay_rate,
            "distance_weight_randomness_pct": pop_cfg.distance_weight_randomness_pct,
            "seed": pop_cfg.seed,
        }
        resolvable = getattr(pop_cfg, "resolvable_distance_mm", None)
        if resolvable is not None:
            params["resolvable_distance_mm"] = resolvable
        params.update(getattr(pop_cfg, "innervation_params", None) or {})
        return params

    def run(
        self,
        stimulus: torch.Tensor,
        return_intermediates: bool = False,
        *,
        bundle_dir: Optional[Any] = None,
        stimulus_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        bundle_overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Run simulation with given stimulus.

        Processes stimulus through all configured populations (innervation → filter → neuron)
        and returns spike trains and optionally intermediate activations.

        Args:
            stimulus: Stimulus tensor.
                - Shape: `[time, height, width]` or `[batch, time, height, width]`
                - Units: Pressure/activation values (dimensionless or N/mm²)
                - Batch dimension is added automatically if missing
            return_intermediates: If True, return intermediate activations (drive, filtered, voltages)
            bundle_dir: If given, write a data bundle (J2, F-013) to this directory via
                :func:`sensoryforge.io.bundle.write_bundle` after the run. Internally forces
                intermediates on for every population (the bundle needs ``drive``/``filtered``)
                regardless of *return_intermediates*; the returned dict still only carries
                intermediates when *return_intermediates* is ``True``.
            stimulus_config: The stimulus's own config dict, written into the bundle's
                ``stimuli/stimulus.json`` (ignored unless *bundle_dir* is given).
            seed: The run's seed, recorded in the bundle (ignored unless *bundle_dir* is given).
            bundle_overwrite: Passed to :func:`~sensoryforge.io.bundle.write_bundle`.

        Returns:
            Dictionary with results for each population, keyed by population name. Each value is
            itself a dict of tensors shaped `[batch, time, num_neurons]` -- the sub-step spike
            count per record bin under the key spikes (F-008; use greater-than-zero for a binary
            raster), and, only when return_intermediates is True, drive and filtered (both mA)
            and voltages (mV). A population whose neuron model has no spike condition (an analog
            DSL model with no threshold, N1/N2) carries state (its readout trace) instead of
            spikes, and has no spikes key at all -- see `_run_pop_from_drive`.

        Examples:
            >>> from sensoryforge.config.schema import SensoryForgeConfig
            >>> from sensoryforge.core.simulation_engine import SimulationEngine
            >>> config = SensoryForgeConfig.from_yaml('config.yml')
            >>> engine = SimulationEngine(config)
            >>> stimulus = torch.randn(100, 80, 80)  # [time, height, width]
            >>> results = engine.run(stimulus)
            >>> sa_spikes = results['SA Population']['spikes']  # [batch, time, num_neurons]
            >>> print(f"Total spikes: {sa_spikes.sum().item()}")
        """
        want_intermediates = return_intermediates or bundle_dir is not None
        results = {}

        for pop in self.populations:
            pop_name = pop["name"]
            innervation = pop["innervation"]
            filter_module = pop["filter"]
            neuron_model = pop["neuron"]
            grid = pop["grid"]

            # Apply stimulus to receptors
            receptor_input = self._stimulus_to_receptors(stimulus, grid)

            # Receptive fields: the bank takes flattened receptor responses
            # [batch, time, M] (row-major over [grid_h, grid_w], receptor
            # k = i * cols + j) and raises ValueError naming both shapes when
            # M does not match its weights.
            if receptor_input.ndim == 4:
                batch, time, h, w = receptor_input.shape
                receptor_input = receptor_input.reshape(batch, time, h * w)
            elif receptor_input.ndim == 3:
                batch, h, w = receptor_input.shape
                receptor_input = receptor_input.reshape(batch, h * w)

            drive = innervation(receptor_input)
            if drive.ndim == 2:
                drive = drive.unsqueeze(1)

            pop_results = self._run_pop_from_drive(
                drive=drive,
                filter_module=filter_module,
                neuron_model=neuron_model,
                input_gain=pop["config"].input_gain,
                noise_std=pop["config"].noise_std,
                return_intermediates=want_intermediates,
                dt_ms=self.config.simulation.dt_ms,
                integrate_dt_ms=self.config.simulation.integrate_dt_ms,
            )
            results[pop_name] = pop_results

        if bundle_dir is not None:
            from sensoryforge.io.bundle import write_bundle

            write_bundle(
                bundle_dir,
                self.config,
                self,
                results,
                stimulus,
                stimulus_config=stimulus_config,
                seed=seed,
                overwrite=bundle_overwrite,
            )

        if not return_intermediates and bundle_dir is not None:
            # The bundle needed intermediates internally; the public return
            # value still honours the caller's own return_intermediates.
            results = {
                pop_name: {"spikes": pop_results["spikes"]}
                for pop_name, pop_results in results.items()
            }

        return results

    @staticmethod
    def _run_pop_from_drive(
        drive: "torch.Tensor",
        filter_module: Optional[Any],
        neuron_model: Any,
        input_gain: float = 1.0,
        noise_std: float = 0.0,
        return_intermediates: bool = False,
        dt_ms: float = 1.0,
        integrate_dt_ms: float = 0.05,
    ) -> Dict[str, Any]:
        """Run filter → gain → noise → sub-stepped neuron on a drive tensor.

        This is the shared backend kernel used by :meth:`run` and by the GUI
        simulation tab (C3-Step4 adapter). Callers that already have a drive
        tensor (from their own innervation module) can call this directly to
        obtain the same filter/gain/noise/neuron behaviour as the engine,
        without re-building innervation.

        Matches pressure-simulation's ``encoding/encode_runner.run_encoding``
        exactly (F-008): the filter integrates at the record step ``dt_ms``
        (one value per record bin); gain and noise are applied per record
        bin, on that filtered signal; the drive is then held constant across
        ``n = max(1, round(dt_ms / integrate_dt_ms))`` sub-steps per bin
        (``repeat_interleave``) and run through ``neuron_model`` (which must
        already be constructed with ``dt=integrate_dt_ms``) in a single
        forward pass, so its state carries continuously across bins. The
        model's initial sample is dropped (``[:, 1:, :]``, matching
        pressure-simulation and fixing the historical T+1-vs-T mismatch
        between spikes and drive/filtered, ledger F-013); each bin's
        sub-steps are then reduced to a spike **count** (not just any/binary
        -- pressure-simulation's own binary flag is exactly ``counts > 0``)
        and a bin-end voltage.

        Args:
            drive: Innervation output ``[batch, time, num_neurons]`` in mA.
            filter_module: Instantiated :class:`~sensoryforge.filters.base.BaseFilter`
                or ``None`` for no filtering.
            neuron_model: Instantiated neuron model (constructed with
                ``dt=integrate_dt_ms``) with ``forward()`` returning
                ``(v_trace, spikes)`` or just ``spikes``.
            input_gain: Multiplicative gain applied to the filtered drive before
                the neuron model.  Default ``1.0`` (no scaling).
            noise_std: Standard deviation of Gaussian noise added after gain.
                Default ``0.0`` (no noise).
            return_intermediates: If ``True``, include ``"drive"``, ``"filtered"``,
                and ``"voltages"`` in the returned dict.
            dt_ms: Record step (ms) -- the time resolution of ``drive`` and
                the returned ``"spikes"``/``"filtered"``/``"voltages"``.
            integrate_dt_ms: Neuron integration step (ms); must match the dt
                ``neuron_model`` was constructed with.

        Returns:
            Dictionary with at minimum ``"spikes"`` (integer sub-step spike
            counts per record bin, ``[batch, time, num_neurons]``) and, if
            *return_intermediates* is ``True``, also ``"drive"``,
            ``"filtered"``, and optionally ``"voltages"`` (at bin ends).
            When ``neuron_model`` has no spike condition (N1/N2 -- an analog
            DSL model with no threshold, ``forward()`` returns
            ``(state_trace, None)``), the dictionary carries ``"state"``
            (bin-end samples, ``[batch, time, num_neurons]``) instead, and
            has no ``"spikes"`` key at all.
        """
        import torch as _torch  # local import to keep signature clean

        # F-042: catch a record step that isn't a whole multiple of the
        # integration step here too, for callers that build dt_ms/
        # integrate_dt_ms directly instead of through SimulationConfig.
        validate_dt_ms(dt_ms, integrate_dt_ms)

        # Apply filter (resets state on each call via BaseFilter contract)
        if filter_module is not None:
            filtered = filter_module(drive)
        else:
            filtered = drive

        # Apply per-population input gain
        if input_gain != 1.0:
            filtered = filtered * input_gain

        # Additive Gaussian noise
        if noise_std > 0.0:
            filtered = filtered + _torch.randn_like(filtered) * noise_std

        filtered = filtered.float()

        # F-008: hold the (record-step) drive constant across n sub-steps
        # per bin, then run the neuron once over the whole sub-stepped
        # sequence so its state carries continuously across bins.
        n_substeps = max(1, round(dt_ms / integrate_dt_ms))
        filtered_sub = filtered.repeat_interleave(n_substeps, dim=1)

        neuron_output = neuron_model(filtered_sub)
        if isinstance(neuron_output, tuple):
            v_trace_sub, spikes_sub = neuron_output
        else:
            spikes_sub = neuron_output
            v_trace_sub = None

        batch, _, num_neurons = filtered.shape
        time_steps = filtered.shape[1]

        if spikes_sub is None:
            # Analog readout (N2): the neuron model has no spike condition
            # (e.g. a thresholdless DSL model, N1) and returned only a state
            # trace. Reduce it the same way "voltages" already is -- the
            # bin-end sample of each bin's n_substeps sub-steps -- and carry
            # it as "state" instead of "spikes"; no "spikes" key at all.
            if v_trace_sub is None:
                raise ValueError(
                    "neuron_model returned no spikes and no state trace; "
                    "expected forward() to return (state_trace, None) for "
                    "an analog readout."
                )
            v_trace_sub = v_trace_sub[:, 1:, :]
            state = v_trace_sub.view(batch, time_steps, n_substeps, num_neurons)[
                :, :, -1, :
            ]

            pop_results: Dict[str, Any] = {"state": state}
            if return_intermediates:
                pop_results["drive"] = drive
                pop_results["filtered"] = filtered
            return pop_results

        # Drop the initial sample (index 0), then collapse each bin's
        # n_substeps sub-steps: sum -> integer spike count per bin.
        spikes_sub = spikes_sub[:, 1:, :].float()
        spikes = spikes_sub.view(batch, time_steps, n_substeps, num_neurons).sum(dim=2)

        v_trace = None
        if v_trace_sub is not None:
            v_trace_sub = v_trace_sub[:, 1:, :]
            # Voltage at each bin's end: the last sub-step.
            v_trace = v_trace_sub.view(batch, time_steps, n_substeps, num_neurons)[
                :, :, -1, :
            ]

        pop_results: Dict[str, Any] = {"spikes": spikes}
        if return_intermediates:
            pop_results["drive"] = drive
            pop_results["filtered"] = filtered
            if v_trace is not None:
                pop_results["voltages"] = v_trace

        return pop_results

    def _stimulus_to_receptors(
        self,
        stimulus: torch.Tensor,
        grid: Any,
    ) -> torch.Tensor:
        """Map stimulus to receptor activations.

        This is a simplified implementation. Full implementation would
        properly sample stimulus at receptor locations.

        The bank expects flattened receptor responses; run() reshapes:
        - Grid-based: [batch, time, grid_h, grid_w] or [batch, grid_h, grid_w]
        - Flat-based: [batch, time, num_receptors] or [batch, num_receptors]
        """
        # Add batch dimension if missing
        if stimulus.ndim == 3:
            # [time, height, width] -> [1, time, height, width]
            stimulus = stimulus.unsqueeze(0)
        elif stimulus.ndim == 2:
            # [height, width] -> [1, 1, height, width]
            stimulus = stimulus.unsqueeze(0).unsqueeze(0)

        # Now stimulus is [batch, time, height, width] or [batch, height, width]
        return stimulus
