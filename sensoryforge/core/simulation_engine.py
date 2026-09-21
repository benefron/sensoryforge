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

import random
import warnings
from typing import Callable, Dict, List, Any, Optional, Tuple
import torch
import torch.nn.functional as F
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
from sensoryforge.core.grid import ReceptorGrid, GridManager, load_receptor_coords_file
from sensoryforge.core.composite_grid import CompositeReceptorGrid, CompositeGrid
from sensoryforge.core.innervation import (
    build_population_bank,
    create_neuron_centers,
)
from sensoryforge.core.rf_bank import ReceptiveFieldBank
from sensoryforge.core.processing import ProcessingPipeline
from sensoryforge.neurons.model_dsl import NeuronModel

# Ensure components are registered
register_all()


def _lattice_fields_set_by_user(pop_cfg: Any) -> List[str]:
    """Neuron-lattice size fields on *pop_cfg* that differ from their defaults.

    Reads the defaults from the ``PopulationConfig`` dataclass itself, so
    the rule stays right if a default changes.

    Args:
        pop_cfg: A ``PopulationConfig``.

    Returns:
        ``name=value`` strings for each of ``neurons_per_row``,
        ``neuron_rows`` and ``neuron_cols`` the user changed.
    """
    import dataclasses

    from sensoryforge.config.schema import PopulationConfig

    defaults = {
        f.name: f.default
        for f in dataclasses.fields(PopulationConfig)
        if f.name in ("neurons_per_row", "neuron_rows", "neuron_cols")
    }
    return [
        f"{name}={getattr(pop_cfg, name)!r}"
        for name, default in defaults.items()
        if getattr(pop_cfg, name) != default
    ]


def _composite_from_coords(
    layer_name: str, coords: torch.Tensor, *, device: torch.device
) -> CompositeReceptorGrid:
    """Wrap an ``[M, 2]`` coordinate tensor as a single-layer composite grid.

    Bounds are the coordinates' own bounding box (padded by 0.5 mm on a
    degenerate axis, so ``CompositeReceptorGrid``'s ``xlim[0] < xlim[1]``
    invariant holds for a single point or a co-linear set).

    Args:
        layer_name: Name of the single layer to add.
        coords: Receptor coordinates ``[M, 2]`` in mm.
        device: Device to build the composite grid on.

    Returns:
        A ``CompositeReceptorGrid`` with one layer named ``layer_name``.
    """
    x_min = coords[:, 0].min().item()
    x_max = coords[:, 0].max().item()
    y_min = coords[:, 1].min().item()
    y_max = coords[:, 1].max().item()
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_min == y_max:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    composite = CompositeReceptorGrid(
        xlim=(x_min, x_max), ylim=(y_min, y_max), device=device
    )
    composite.add_layer_with_coords(layer_name, coords)
    return composite


def build_grid(grid_cfg: Any, *, device: Any = "cpu") -> Any:
    """Build one receptor grid from a :class:`~sensoryforge.config.schema.GridConfig`.

    This is the exact construction ``SimulationEngine._build_grids()`` uses
    per grid entry (coords_file, composite arrangement, or an ordinary
    ``ReceptorGrid``), extracted so other callers (e.g. the GUI's
    ``GridPreview``) build grids the same way the engine does, rather than
    re-implementing arrangement logic.

    Args:
        grid_cfg: The grid configuration to build from.
        device: Device string (e.g. ``"cpu"``) or ``torch.device`` to build
            tensors on.

    Returns:
        A ``ReceptorGrid`` (ordinary arrangement) or ``CompositeReceptorGrid``
        (``arrangement == "composite"``, or ``grid_cfg.coords_file`` set).
        Composite grids carry a ``provenance`` dict describing their layers.

    Example:
        >>> from sensoryforge.config.schema import GridConfig
        >>> grid = build_grid(GridConfig(name="skin", rows=20, cols=20))
        >>> grid.get_all_coordinates().shape
        torch.Size([400, 2])
    """
    device = torch.device(device) if isinstance(device, str) else device
    grid_name = grid_cfg.name
    arrangement = grid_cfg.arrangement

    if grid_cfg.coords_file:
        # L1: an explicit [M, 2] coordinate file builds a single-layer
        # CompositeReceptorGrid from those exact positions, bypassing
        # rows/cols/spacing entirely.
        coords = load_receptor_coords_file(grid_cfg.coords_file, device=device)
        composite = _composite_from_coords(grid_name, coords, device=device)
        composite.provenance = {
            "layers": [{"name": grid_name, "count": coords.shape[0]}],
            "source": "coords_file",
            "coords_file": grid_cfg.coords_file,
        }
        return composite

    if arrangement == "composite":
        # L4: layers come from grid_cfg.layers, in declaration order --
        # that order is the receptor-index contract (get_all_coordinates()
        # concatenates layers in insertion order), so it is recorded
        # verbatim in provenance.
        if not grid_cfg.layers:
            raise ValueError(
                f"Grid {grid_name!r}: arrangement='composite' requires "
                "a non-empty 'layers' list"
            )
        rows = grid_cfg.rows or 40
        cols = grid_cfg.cols or 40
        total_x = (rows - 1) * grid_cfg.spacing
        total_y = (cols - 1) * grid_cfg.spacing
        xlim = (
            grid_cfg.center_x - total_x / 2,
            grid_cfg.center_x + total_x / 2,
        )
        ylim = (
            grid_cfg.center_y - total_y / 2,
            grid_cfg.center_y + total_y / 2,
        )
        composite = CompositeReceptorGrid(xlim=xlim, ylim=ylim, device=device)
        layer_order: List[str] = []
        for entry in grid_cfg.layers:
            lname = entry.get("name")
            if not lname:
                raise ValueError(
                    f"Grid {grid_name!r}: each composite layer entry "
                    f"needs a non-empty 'name', got {entry!r}"
                )
            if "coordinates" in entry:
                coords = torch.as_tensor(
                    entry["coordinates"],
                    dtype=torch.float32,
                    device=device,
                )
                composite.add_layer_with_coords(lname, coords, color=entry.get("color"))
            elif "coords_file" in entry:
                coords = load_receptor_coords_file(entry["coords_file"], device=device)
                composite.add_layer_with_coords(lname, coords, color=entry.get("color"))
            elif "density" in entry:
                composite.add_layer(
                    name=lname,
                    density=entry["density"],
                    arrangement=entry.get("arrangement", "grid"),
                    offset=tuple(entry.get("offset", (0.0, 0.0))),
                    color=entry.get("color"),
                    seed=entry.get("seed"),
                )
            else:
                raise ValueError(
                    f"Grid {grid_name!r} layer {lname!r}: needs one of "
                    "'density', 'coordinates' or 'coords_file'"
                )
            layer_order.append(lname)
        composite.provenance = {
            "layers": [
                {"name": n, "count": composite.get_layer_count(n)} for n in layer_order
            ]
        }
        return composite

    # Ordinary (non-composite) grid.
    rows = grid_cfg.rows or 40
    cols = grid_cfg.cols or 40
    grid_size = (rows, cols)
    return ReceptorGrid(
        grid_size=grid_size,
        spacing=grid_cfg.spacing,
        arrangement=arrangement,
        center=(grid_cfg.center_x, grid_cfg.center_y),
        density=grid_cfg.density,
        device=device,
        seed=grid_cfg.seed,
    )


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
        # Wave M2: grid configs by name, so run() can resolve a
        # PopulationInput's channel name against that grid's GridConfig.channels.
        self.grid_configs: Dict[str, Any] = {g.name: g for g in config.grids}
        self._build_grids()

        # Build populations (innervation, filters, neurons)
        self.populations: List[Dict[str, Any]] = []
        self._build_populations()

    def _build_grids(self) -> None:
        """Build receptor grids from config."""
        for grid_cfg in self.config.grids:
            grid_name = grid_cfg.name
            grid = build_grid(grid_cfg, device=self.device)
            self.grids.append(grid)
            self.grid_names[grid_name] = grid

            if grid_cfg.arrangement != "composite" and not grid_cfg.coords_file:
                # Regular GridManager: the receptor lattice the bank is built on
                rows = grid_cfg.rows or 40
                cols = grid_cfg.cols or 40
                grid_manager = GridManager(
                    grid_size=(rows, cols),
                    spacing=grid_cfg.spacing,
                    center=(grid_cfg.center_x, grid_cfg.center_y),
                    device=self.device,
                )
                self.grid_managers[grid_name] = grid_manager

    def _resolve_input_grid(self, pop_cfg: Any, pop_input: Any) -> Any:
        """Resolve a :class:`~sensoryforge.config.schema.PopulationInput`'s
        grid object, falling back to the first configured grid (M2)."""
        target_grid_name = pop_input.grid or (
            self.config.grids[0].name if self.config.grids else None
        )
        if target_grid_name is None:
            raise ValueError(f"Population {pop_cfg.name} has no target grid")
        grid = self.grid_names.get(target_grid_name)
        if grid is None:
            grid = self.grids[0] if self.grids else None
            if grid is None:
                raise ValueError(f"No grid available for population {pop_cfg.name}")
        return target_grid_name, grid

    def _build_input_bank(
        self,
        pop_cfg: Any,
        pop_input: Any,
        shared_neuron_centers: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Build one :class:`ReceptiveFieldBank` for one
        :class:`~sensoryforge.config.schema.PopulationInput` (Wave M2).

        Mirrors the single-input logic ``_build_populations`` used before
        Wave M exactly (same grid/neuron-lattice/builder-parameter
        resolution), so the sugar path (``effective_inputs()``'s one
        implicit input) is bit-identical to the pre-M behaviour.

        Args:
            pop_cfg: The population's config.
            pop_input: The input being built.
            shared_neuron_centers: Neuron centres already laid out for an
                earlier input in this population (``None`` for the first),
                reused so every non-deriving input shares one lattice.

        Returns:
            A dict with ``bank``, ``target_grid_name``, ``grid``,
            ``grid_manager``, ``receptor_coords`` (the *raw*, pre-processing
            receptor coordinates used at run time to sample the stimulus)
            and ``neuron_centers`` (the lattice actually used, for callers
            that want to cache it as ``shared_neuron_centers``).
        """
        target_grid_name, grid = self._resolve_input_grid(pop_cfg, pop_input)

        # Get receptor coordinates
        if isinstance(grid, CompositeReceptorGrid):
            # L4: a population may innervate a named subset of a
            # composite grid's layers instead of all of them. Layer
            # order (and therefore receptor index) is the contract --
            # get_all_coordinates() concatenates in insertion order, and
            # pop_input.layers preserves whatever order it's given in.
            if pop_input.layers:
                receptor_coords = torch.cat(
                    [grid.get_layer_coordinates(name) for name in pop_input.layers],
                    dim=0,
                )
            else:
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

        # Receptive fields: one registered builder per input (I6, F-051,
        # extended to per-input in M2). The builder's own class decides
        # which of the population parameters it takes (filter_params) and
        # whether it derives the neuron lattice itself
        # (DERIVES_NEURON_CENTERS).
        innervation_method = pop_input.rf.method or "gaussian"
        try:
            builder_cls = INNERVATION_REGISTRY.get_class(innervation_method)
        except KeyError:
            raise ValueError(
                f"Unknown innervation method: {innervation_method!r}. "
                f"Registered methods: {sorted(INNERVATION_REGISTRY.list_registered())}"
            ) from None

        if not use_flat:
            # Ordinary grids: the population's receptive-field bank is
            # built on the grid's *real* receptor coordinates --
            # get_receptor_coordinates() -- for every arrangement,
            # closing the build-time half of F-010 (Wave L). For a
            # "grid" arrangement this is bit-identical to the previous
            # `_grid_lattice_coords(grid_manager)` (same construction
            # args), since it was the same meshgrid either way; for
            # hex/poisson/jittered_grid/blue_noise it is now the actual
            # scattered positions instead of a synthetic regular
            # lattice standing in for them.
            receptor_coords = grid.get_receptor_coordinates()

        if builder_cls.DERIVES_NEURON_CENTERS:
            # Warn only about lattice sizes the user actually set (F-069).
            # These fields always exist on PopulationConfig, so warning
            # whenever a lattice-deriving builder is used told everyone
            # "neurons_per_row=10 ... ignored" -- including every user of
            # the shipped tactile preset, which never sets it. A warning that
            # fires on defaults teaches people to ignore warnings.
            ignored = _lattice_fields_set_by_user(pop_cfg)
            if ignored:
                warnings.warn(
                    f"Population {pop_cfg.name!r}: innervation_method "
                    f"{innervation_method!r} derives its own neuron lattice, "
                    f"so {', '.join(ignored)} "
                    f"{'is' if len(ignored) == 1 else 'are'} ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            neuron_centers = None
        elif shared_neuron_centers is not None:
            # M2: every input that lays out its own lattice (i.e. does not
            # derive one) shares the *same* lattice, computed once from the
            # population's first such input -- required for "sum" (every
            # input's drive must land on the same N neurons) and harmless
            # for "concat" (each block still gets its own bank/provenance).
            neuron_centers = shared_neuron_centers.to(self.device)
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

        builder_params = self.builder_params(pop_cfg, grid_path=not use_flat)
        builder_params.update(pop_input.rf.params)

        # M3: a non-empty processing pipeline changes the receptor axis the
        # bank is built on (e.g. OnOffLayer's ON+OFF planes double it); the
        # *raw* receptor_coords (returned below) are what run() samples the
        # stimulus at, before the pipeline runs.
        bank_receptor_coords = receptor_coords
        if pop_input.processing:
            bank_receptor_coords = ProcessingPipeline.expand_receptor_coords(
                pop_input.processing, receptor_coords
            )

        bank: ReceptiveFieldBank = build_population_bank(
            receptor_coords=bank_receptor_coords,
            innervation_method=innervation_method,
            neuron_type=pop_cfg.neuron_type,
            neuron_centers=neuron_centers,
            device=self.device,
            **builder_params,
        )
        return {
            "bank": bank,
            "target_grid_name": target_grid_name,
            "grid": grid,
            "grid_manager": self.grid_managers.get(target_grid_name),
            "receptor_coords": receptor_coords,
            "processing": list(pop_input.processing),
            "channel": pop_input.channel,
            "gain": pop_input.gain,
            "neuron_centers": bank.neuron_centers,
            "derives_neuron_centers": builder_cls.DERIVES_NEURON_CENTERS,
        }

    @staticmethod
    def _combine_banks(
        banks: List["ReceptiveFieldBank"],
        gains: List[float],
        combine: str,
        input_names: List[str],
    ) -> "ReceptiveFieldBank":
        """Combine one bank per input into the population's single bank (M2).

        The combined bank's ``forward()`` on the *concatenation* (along the
        receptor axis, same order as ``banks``) of each input's own
        (post-processing) receptor response reproduces the per-input
        combination exactly:

        - ``"sum"``: every input must have the same neuron count ``N``.
          The combined weights are ``hstack(gain_i * weights_i)`` (receptor
          axis concatenated, neuron axis shared) so one matmul against the
          concatenated receptor responses equals
          ``sum_i gain_i * bank_i(response_i)``.
        - ``"concat"``: the combined weights are block-diagonal (each
          input's ``[N_i, M_i]`` block placed at its own offset, zero
          elsewhere), so one matmul reproduces
          ``cat([gain_i * bank_i(response_i) for i], dim=-1)`` -- the
          neuron axis grows to ``sum_i N_i``.

        A single input needs no combination beyond its own gain.

        Raises:
            ValueError: If ``combine`` is unknown, or ``"sum"`` inputs
                disagree on neuron count.
        """
        if len(banks) == 1:
            bank = banks[0]
            gain = gains[0]
            if gain == 1.0:
                return bank
            weights = bank.weights * gain
            return ReceptiveFieldBank(
                weights,
                bank.neuron_centers,
                bank.receptor_coords,
                provenance=dict(bank.provenance),
            )

        if combine == "sum":
            n_counts = [b.num_neurons for b in banks]
            if len(set(n_counts)) != 1:
                raise ValueError(
                    "combine='sum' requires every input to produce the same "
                    f"neuron count N; got {n_counts} for inputs {input_names}"
                )
            weights = torch.cat([g * b.weights for g, b in zip(gains, banks)], dim=1)
            receptor_coords = torch.cat([b.receptor_coords for b in banks], dim=0)
            neuron_centers = banks[0].neuron_centers
            provenance = {
                "builder": "combine_sum",
                "inputs": [
                    {**dict(b.provenance), "input": name, "gain": g}
                    for b, name, g in zip(banks, input_names, gains)
                ],
            }
            return ReceptiveFieldBank(
                weights, neuron_centers, receptor_coords, provenance=provenance
            )

        if combine == "concat":
            n_total = sum(b.num_neurons for b in banks)
            m_total = sum(b.num_receptors for b in banks)
            weights = torch.zeros(
                n_total,
                m_total,
                dtype=banks[0].weights.dtype,
                device=banks[0].weights.device,
            )
            neuron_centers_list = []
            receptor_coords_list = []
            provenance_blocks = []
            n_off = 0
            m_off = 0
            for gain, bank, name in zip(gains, banks, input_names):
                n, m = bank.num_neurons, bank.num_receptors
                weights[n_off : n_off + n, m_off : m_off + m] = gain * bank.weights
                neuron_centers_list.append(bank.neuron_centers)
                receptor_coords_list.append(bank.receptor_coords)
                provenance_blocks.append(
                    {
                        **dict(bank.provenance),
                        "input": name,
                        "gain": gain,
                        "neuron_slice": [n_off, n_off + n],
                    }
                )
                n_off += n
                m_off += m
            neuron_centers = torch.cat(neuron_centers_list, dim=0)
            receptor_coords = torch.cat(receptor_coords_list, dim=0)
            provenance = {"builder": "combine_concat", "inputs": provenance_blocks}
            return ReceptiveFieldBank(
                weights, neuron_centers, receptor_coords, provenance=provenance
            )

        raise ValueError(
            f"Unknown combine mode: {combine!r}; expected 'sum' or 'concat'"
        )

    def _build_populations(self) -> None:
        """Build population execution contexts (innervation, filters, neurons)."""
        for pop_cfg in self.config.populations:
            if not pop_cfg.enabled:
                continue

            effective_inputs = pop_cfg.effective_inputs()
            input_ctxs: List[Dict[str, Any]] = []
            shared_neuron_centers: Optional[torch.Tensor] = None
            for pop_input in effective_inputs:
                ctx = self._build_input_bank(pop_cfg, pop_input, shared_neuron_centers)
                if shared_neuron_centers is None and not ctx["derives_neuron_centers"]:
                    shared_neuron_centers = ctx["neuron_centers"]
                input_ctxs.append(ctx)

            combine = pop_cfg.combine or "sum"
            gains = [ctx["gain"] for ctx in input_ctxs]
            input_names = [
                f"{ctx['target_grid_name']}:{ctx['channel']}" for ctx in input_ctxs
            ]
            bank = self._combine_banks(
                [ctx["bank"] for ctx in input_ctxs], gains, combine, input_names
            )
            neuron_centers = bank.neuron_centers

            grid = input_ctxs[0]["grid"]
            target_grid_name = input_ctxs[0]["target_grid_name"]

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

            # Store population context. "inputs" carries the per-input build
            # contexts (M2) run() needs to sample each input's own
            # grid/channel/processing at run time; "bank"/"innervation" is
            # the single combined bank (bit-identical to the pre-M bank for
            # a one-input population) that a concatenation of the inputs'
            # (post-processing) receptor responses, in the same order, is
            # matmul'd against.
            self.populations.append(
                {
                    "name": pop_cfg.name,
                    "config": pop_cfg,
                    "grid": grid,
                    "target_grid_name": target_grid_name,
                    "inputs": input_ctxs,
                    "combine": combine,
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
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
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
            seed: The run's seed (F-075). This is the single run-seed: when given, it is
                used as-is; when ``None``, it falls back to ``self.config.simulation.seed``.
                The value that resolves (either one, or ``None`` if both are unset) is used to
                seed ``torch``/``numpy``/``random`` at the start of the run, before stimulus
                sampling and the population loop, and is also what gets recorded in the bundle
                (ignored unless *bundle_dir* is given). Distinct from a population's own
                ``noise_seed`` (per-population membrane noise) and ``seed`` (innervation wiring,
                F-006 open).
            bundle_overwrite: Passed to :func:`~sensoryforge.io.bundle.write_bundle`.
            progress_cb: If given, called once per population as
                ``progress_cb(index, n_populations, population_name)``, immediately before that
                population's filter/neuron pass (so at call time ``results`` does not yet hold
                that population's entry). ``None`` (default) leaves behaviour unchanged.

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
        # F-075: `seed` resolves against `self.config.simulation.seed` and, once
        # resolved, is the single value used both to seed the RNGs below and to
        # record into the bundle -- see the docstring's precedence note.
        seed = seed if seed is not None else self.config.simulation.seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        want_intermediates = return_intermediates or bundle_dir is not None
        results = {}
        n_populations = len(self.populations)

        for index, pop in enumerate(self.populations):
            pop_name = pop["name"]
            innervation = pop["innervation"]
            filter_module = pop["filter"]
            neuron_model = pop["neuron"]

            if progress_cb is not None:
                progress_cb(index, n_populations, pop_name)

            # M2: sample each input's own grid/channel, run it through that
            # input's processing pipeline (if any), then concatenate along
            # the receptor axis in the same order the combined bank's
            # weights were built (_combine_banks) -- one matmul against
            # `innervation` then reproduces the population's sum/concat
            # combination exactly.
            input_responses = []
            for ctx in pop["inputs"]:
                grid_cfg = self.grid_configs.get(ctx["target_grid_name"])
                channels = (
                    list(grid_cfg.channels) if grid_cfg is not None else ["value"]
                )
                channel_stimulus = self._select_stimulus_channel(
                    stimulus, channels, ctx["channel"]
                )
                receptor_response = self._stimulus_to_receptors(
                    channel_stimulus,
                    ctx["grid"],
                    ctx["receptor_coords"],
                    ctx["grid_manager"],
                )
                # M3: an input's processing pipeline (empty by default --
                # no allocation at all on the sugar/no-processing path)
                # sits between receptor sampling and the receptive-field
                # bank.
                if ctx["processing"]:
                    pipeline = ProcessingPipeline.from_config(
                        ctx["processing"], receptor_coords=ctx["receptor_coords"]
                    )
                    receptor_response = pipeline(receptor_response)
                input_responses.append(receptor_response)

            receptor_input = (
                input_responses[0]
                if len(input_responses) == 1
                else torch.cat(input_responses, dim=-1)
            )

            drive = innervation(receptor_input)
            if drive.ndim == 2:
                drive = drive.unsqueeze(1)

            pop_cfg = pop["config"]
            noise_generator = None
            if pop_cfg.noise_seed is not None and pop_cfg.noise_std > 0:
                # F-075: a per-population torch.Generator, seeded independently
                # of the run seed, drives that population's membrane noise.
                # Not every device supports torch.Generator(device=...)
                # (e.g. an unsupported backend); fall back to a CPU generator
                # and let _run_pop_from_drive move the draw to drive's device.
                try:
                    noise_generator = torch.Generator(device=self.device).manual_seed(
                        pop_cfg.noise_seed
                    )
                except (RuntimeError, TypeError):
                    noise_generator = torch.Generator(device="cpu").manual_seed(
                        pop_cfg.noise_seed
                    )

            pop_results = self._run_pop_from_drive(
                drive=drive,
                filter_module=filter_module,
                neuron_model=neuron_model,
                input_gain=pop_cfg.input_gain,
                noise_std=pop_cfg.noise_std,
                return_intermediates=want_intermediates,
                dt_ms=self.config.simulation.dt_ms,
                integrate_dt_ms=self.config.simulation.integrate_dt_ms,
                noise_generator=noise_generator,
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
        noise_generator: Optional[torch.Generator] = None,
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
            noise_generator: If given, the membrane noise draw uses this
                ``torch.Generator`` instead of the global RNG (F-075), drawn on
                the generator's own device and moved to ``drive``'s device if
                they differ. ``None`` (default) draws from the global RNG via
                ``torch.randn_like``, exactly as before this parameter existed
                -- bit-identical to the pre-F-075 behaviour. When given, the
                global RNG is also reseeded from the generator's own seed for
                the neuron call (see the neuron-model note below), but its
                prior state is saved and restored around that call, so this
                does not leak into anything that draws from the global RNG
                afterwards -- e.g. a later population in the same ``run()``
                that has no ``noise_seed`` of its own. The CPU generator state
                is always saved/restored; the CUDA state is too when ``drive``
                is on CUDA; the MPS state is too whenever MPS is available
                (``torch.manual_seed()`` reseeds MPS's global generator
                regardless of ``drive``'s own device, so that state must be
                saved/restored unconditionally on MPS availability, not on
                whether the drive itself is on MPS).

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
            if noise_generator is None:
                noise = _torch.randn_like(filtered)
            else:
                # F-075: draw on the generator's own device; a generator
                # built on a device the drive isn't on (e.g. a CPU fallback
                # generator feeding an MPS/CUDA drive) still works.
                noise = _torch.randn(
                    filtered.shape,
                    generator=noise_generator,
                    device=noise_generator.device,
                    dtype=filtered.dtype,
                )
                if noise.device != filtered.device:
                    noise = noise.to(filtered.device)
            filtered = filtered + noise * noise_std

        filtered = filtered.float()

        # F-008: hold the (record-step) drive constant across n sub-steps
        # per bin, then run the neuron once over the whole sub-stepped
        # sequence so its state carries continuously across bins.
        n_substeps = max(1, round(dt_ms / integrate_dt_ms))
        filtered_sub = filtered.repeat_interleave(n_substeps, dim=1)

        if noise_generator is None:
            neuron_output = neuron_model(filtered_sub)
        else:
            # F-075: the built-in neuron models (Izhikevich/AdEx/MQIF/FA)
            # draw their own membrane (Langevin) noise from the *global* RNG
            # inside forward(), using this same noise_std -- there is no
            # generator parameter threaded into neuron_model. Reseed the
            # global RNG from noise_generator's own seed immediately before
            # the neuron call so that noise is reproducible too, matching
            # this population's noise_seed end to end -- but save/restore
            # the global RNG state around it so this population's seed does
            # not leak into whatever draws from the global RNG afterwards
            # (a later population with no noise_seed of its own, or
            # anything else in the process). Only happens when a generator
            # is given; the noise_generator=None branch above never touches
            # the global RNG. `torch.manual_seed()` reseeds *every* global
            # generator it knows about, not just the one for the drive's own
            # device: CUDA's when a CUDA device is available, and MPS's
            # whenever MPS is available -- regardless of whether the drive
            # itself is on that device -- so both must be saved/restored too,
            # unconditionally on availability (not on the drive's device).
            cpu_state = _torch.get_rng_state()
            cuda_state = None
            if filtered_sub.is_cuda:
                cuda_state = _torch.cuda.get_rng_state(filtered_sub.device)
            mps_state = None
            if getattr(_torch.backends, "mps", None) is not None and (
                _torch.backends.mps.is_available()
            ):
                mps_state = _torch.mps.get_rng_state()
            try:
                _torch.manual_seed(noise_generator.initial_seed())
                neuron_output = neuron_model(filtered_sub)
            finally:
                _torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    _torch.cuda.set_rng_state(cuda_state, filtered_sub.device)
                if mps_state is not None:
                    _torch.mps.set_rng_state(mps_state)

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

    @staticmethod
    def _select_stimulus_channel(
        stimulus: torch.Tensor,
        channels: List[str],
        channel_name: str,
    ) -> torch.Tensor:
        """Select one named channel plane from a multi-channel stimulus (M2).

        Mirrors :func:`sensoryforge.stimuli.render.render_stimulus`'s own
        channel convention: a stimulus with no channel axis (``[H, W]``,
        ``[T, H, W]`` or ``[batch, T, H, W]``) is returned unchanged --
        single/implicit channel, whatever ``channel_name`` is. A stimulus
        that *does* carry a channel axis is always 5-D,
        ``[batch, T, C, H, W]`` (see ``_stimulus_to_receptors``); its
        ``channel_name`` plane is selected by index into ``channels``
        (``GridConfig.channels``), defaulting to ``channels[0]``.

        Args:
            stimulus: The full simulation stimulus tensor.
            channels: The target grid's channel names.
            channel_name: The requesting input's channel
                (``PopulationInput.channel``).

        Returns:
            A tensor with the same shape as *stimulus* but no channel axis.

        Raises:
            ValueError: If *stimulus* is 5-D and ``channel_name`` is not one
                of *channels*.
        """
        if stimulus.ndim != 5:
            return stimulus
        if not channels or len(channels) <= 1:
            return stimulus[:, :, 0]
        target = channel_name if channel_name is not None else channels[0]
        if target not in channels:
            raise ValueError(f"stimulus channel {target!r} is not one of {channels}")
        index = channels.index(target)
        return stimulus[:, :, index]

    def _stimulus_to_receptors(
        self,
        stimulus: torch.Tensor,
        grid: Any,
        receptor_coords: torch.Tensor,
        grid_manager: Optional[Any],
    ) -> torch.Tensor:
        """Map stimulus frames to receptor responses ``[batch, time, M]``.

        Wave L3 (F-010): earlier releases assumed receptor index equals
        stimulus pixel index (a bare reshape); that is only true for a
        regular ``"grid"`` receptor lattice whose resolution matches the
        stimulus frame. Every other arrangement -- hex, Poisson, jittered,
        blue-noise, imported or composite coordinates -- needs the stimulus
        *sampled* at each receptor's own ``(x, y)`` position. This method
        picks the fast, bit-identical reshape when it is provably correct,
        and otherwise samples via
        :meth:`_sample_stimulus_at_receptors`.

        Args:
            stimulus: ``[height, width]``, ``[time, height, width]`` or
                ``[batch, time, height, width]`` (a missing batch/time axis
                is added).
            grid: The population's target grid (:class:`ReceptorGrid` or
                :class:`CompositeReceptorGrid`).
            receptor_coords: ``[M, 2]`` ``(x, y)`` mm -- the bank's own
                receptor coordinates, i.e. ``innervation.receptor_coords``.
            grid_manager: The :class:`GridManager` (alias of
                :class:`ReceptorGrid`) built alongside ``grid`` for the same
                config entry, or ``None`` when unavailable (composite/flat
                paths). Supplies ``grid_size`` and ``xlim``/``ylim`` for the
                fast-path check and the sampling bounds.

        Returns:
            ``[batch, time, M]`` receptor responses, same dtype/device as
            ``stimulus``.
        """
        # Add batch/time dimensions if missing, same as before Wave L.
        if stimulus.ndim == 2:
            # [height, width] -> [1, 1, height, width]
            stimulus = stimulus.unsqueeze(0).unsqueeze(0)
        elif stimulus.ndim == 3:
            # [time, height, width] -> [1, time, height, width]
            stimulus = stimulus.unsqueeze(0)
        elif stimulus.ndim not in (4, 5):
            raise ValueError(
                "stimulus must be [H, W], [T, H, W], [batch, T, H, W] or "
                f"[batch, T, C, H, W], got shape {list(stimulus.shape)}"
            )

        num_receptors = receptor_coords.shape[0]
        spatial = stimulus.shape[-2:]

        fast_path = (
            grid_manager is not None
            and getattr(grid, "arrangement", None) == "grid"
            and stimulus.ndim == 4
            and tuple(spatial) == tuple(grid_manager.grid_size)
            and num_receptors == spatial[0] * spatial[1]
        )
        if fast_path:
            batch, time, h, w = stimulus.shape
            return stimulus.reshape(batch, time, h * w)

        if grid_manager is not None:
            xlim, ylim = grid_manager.xlim, grid_manager.ylim
        else:
            xlim, ylim = grid.xlim, grid.ylim
        return self._sample_stimulus_at_receptors(stimulus, receptor_coords, xlim, ylim)

    @staticmethod
    def _sample_stimulus_at_receptors(
        frames: torch.Tensor,
        receptor_coords: torch.Tensor,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
    ) -> torch.Tensor:
        """Sample stimulus frames at arbitrary receptor ``(x, y)`` positions.

        Wave L3 (F-010): the receptive-field bank must receive the actual
        stimulus value under each receptor, not the value at the pixel that
        happens to share the receptor's index. This uses
        ``torch.nn.functional.grid_sample`` for bilinear interpolation.

        **The index algebra (read this before touching the axis order).**
        SensoryForge builds every meshgrid with ``indexing="ij"``
        (``core/grid.py:create_grid_torch``), so a frame tensor ``[..., H,
        W]`` has element ``frame[..., i, j]`` at physical position
        ``(x[i], y[j])``: **the frame's second-to-last axis (size H) is x,
        and its last axis (size W) is y.**

        ``grid_sample`` reads its sampling grid as ``grid[..., 0]`` /
        ``grid[..., 1]`` = normalised coordinates along the input's *last*
        axis / *second-to-last* axis respectively (its own docs call these
        "x" and "y", meaning "the width axis" and "the height axis" of the
        image tensor it was written for -- nothing about our physical x/y).
        Put in terms of our tensor axes: ``grid[..., 0]`` addresses the
        frame's last axis (W, which is our **y**), and ``grid[..., 1]``
        addresses the frame's second-to-last axis (H, which is our **x**).

        So the mapping is the swap, not the identity:

        ``grid[..., 0] = normalise(receptor_y, ylim)``
        ``grid[..., 1] = normalise(receptor_x, xlim)``

        A naive ``grid[..., 0] = normalise(receptor_x, ...)`` produces a
        smooth, plausible, **transposed** result -- correct on any
        symmetric test stimulus and wrong on an asymmetric one (see
        ``tests/unit/test_receptor_sampling.py``, which uses a Gaussian
        with different sigma in x and y specifically so a swap fails it).

        Normalisation uses ``align_corners=True`` (``v -> 2*(v-lo)/(hi-lo)
        - 1``), matching ``xlim``/``ylim`` being the coordinates of the
        first and last pixel centres (``torch.linspace`` bounds, not a
        half-pixel-padded extent). ``padding_mode="zeros"`` makes a
        receptor outside ``[xlim, ylim]`` sample exactly zero rather than
        the clamped edge value.

        Args:
            frames: ``[batch, time, H, W]`` or ``[batch, time, C, H, W]``.
            receptor_coords: ``[M, 2]`` ``(x, y)`` in mm.
            xlim: ``(x_min, x_max)`` mm spanned by the frame's H axis.
            ylim: ``(y_min, y_max)`` mm spanned by the frame's W axis.

        Returns:
            ``[batch, time, M]`` (input was 4-D) or ``[batch, time, C, M]``
            (input was 5-D).

        Raises:
            ValueError: If ``frames`` is not 4-D or 5-D.
        """
        if frames.ndim == 4:
            frames5 = frames.unsqueeze(2)
            had_channel = False
        elif frames.ndim == 5:
            frames5 = frames
            had_channel = True
        else:
            raise ValueError(
                "frames must be [batch, time, H, W] or [batch, time, C, H, W], "
                f"got shape {list(frames.shape)}"
            )

        batch, time, channels, h, w = frames5.shape
        device = frames5.device
        dtype = frames5.dtype

        coords = receptor_coords.to(device=device, dtype=dtype)
        num_receptors = coords.shape[0]
        recept_x = coords[:, 0]
        recept_y = coords[:, 1]

        def _normalize(v: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
            if hi == lo:
                return torch.zeros_like(v)
            return 2.0 * (v - lo) / (hi - lo) - 1.0

        norm_x = _normalize(recept_x, xlim[0], xlim[1])
        norm_y = _normalize(recept_y, ylim[0], ylim[1])

        # See the docstring: grid_sample's last axis is (along-W, along-H),
        # i.e. (our y, our x) here -- the swap, not the naive (x, y).
        sample_grid = torch.stack([norm_y, norm_x], dim=-1)  # [M, 2]
        sample_grid = sample_grid.view(1, 1, num_receptors, 2).expand(
            batch * time, 1, num_receptors, 2
        )

        frames_flat = frames5.reshape(batch * time, channels, h, w)
        sampled = F.grid_sample(
            frames_flat,
            sample_grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        )  # [batch*time, C, 1, M]
        sampled = sampled.squeeze(2).view(batch, time, channels, num_receptors)
        if not had_channel:
            sampled = sampled.squeeze(2)
        return sampled
