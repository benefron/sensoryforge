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

from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np

from sensoryforge.config.schema import SensoryForgeConfig
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
    InnervationModule,
    FlatInnervationModule,
    create_neuron_centers,
)

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
        populations: List of population execution contexts (innervation, filters, neurons)
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
                raise NotImplementedError("Composite grids not yet implemented in SimulationEngine")
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
                )
                self.grids.append(grid)
                self.grid_names[grid_name] = grid
                
                # Create GridManager for InnervationModule (needs grid_size, not ReceptorGrid)
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
            target_grid_name = pop_cfg.target_grid or (self.config.grids[0].name if self.config.grids else None)
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
            neuron_rows = pop_cfg.neuron_rows if pop_cfg.neuron_rows is not None else pop_cfg.neurons_per_row
            neuron_cols = pop_cfg.neuron_cols if pop_cfg.neuron_cols is not None else pop_cfg.neurons_per_row
            neuron_arrangement = pop_cfg.neuron_arrangement or "grid"
            
            # Get grid bounds
            if isinstance(grid, CompositeReceptorGrid):
                xlim = grid.xlim
                ylim = grid.ylim
            elif hasattr(grid, 'xlim') and hasattr(grid, 'ylim'):
                xlim = grid.xlim
                ylim = grid.ylim
            else:
                # Fallback: compute from grid properties
                if hasattr(grid, 'spacing') and hasattr(grid, 'grid_size'):
                    spacing = grid.spacing
                    if isinstance(grid.grid_size, tuple):
                        n_x, n_y = grid.grid_size
                    else:
                        n_x = n_y = grid.grid_size
                    total_x = (n_x - 1) * spacing
                    total_y = (n_y - 1) * spacing
                    center = grid.center if hasattr(grid, 'center') else (0.0, 0.0)
                    xlim = (center[0] - total_x/2, center[0] + total_x/2)
                    ylim = (center[1] - total_y/2, center[1] + total_y/2)
                else:
                    xlim = (-5.0, 5.0)
                    ylim = (-5.0, 5.0)
            
            # Generate neuron centers using the existing function
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
                jitter_factor=pop_cfg.neuron_jitter_factor if hasattr(pop_cfg, 'neuron_jitter_factor') else 1.0,
            )
            
            # Create innervation module
            if use_flat:
                innervation_module = FlatInnervationModule(
                    neuron_type=pop_cfg.neuron_type,
                    receptor_coords=receptor_coords,
                    neuron_centers=neuron_centers,
                    neurons_per_row=neuron_rows,
                    xlim=xlim,
                    ylim=ylim,
                    innervation_method=innervation_method,
                    connections_per_neuron=pop_cfg.connections_per_neuron,
                    sigma_d_mm=pop_cfg.sigma_d_mm,
                    weight_range=tuple(pop_cfg.weight_range) if pop_cfg.weight_range else (0.1, 1.0),
                    seed=pop_cfg.seed,
                    use_distance_weights=pop_cfg.use_distance_weights,
                    max_distance_mm=pop_cfg.max_distance_mm,
                    decay_function=pop_cfg.decay_function,
                    decay_rate=pop_cfg.decay_rate,
                    far_connection_fraction=pop_cfg.far_connection_fraction,
                    far_sigma_factor=pop_cfg.far_sigma_factor,
                    distance_weight_randomness_pct=pop_cfg.distance_weight_randomness_pct,
                    device=self.device,
                )
            else:
                # Grid-based innervation needs GridManager
                grid_manager = self.grid_managers.get(target_grid_name, self.grid_managers[list(self.grid_managers.keys())[0]])
                innervation_module = InnervationModule(
                    neuron_type=pop_cfg.neuron_type,
                    grid_manager=grid_manager,
                    neurons_per_row=neuron_rows,
                    neuron_rows=neuron_rows,
                    neuron_cols=neuron_cols,
                    neuron_arrangement=neuron_arrangement,
                    connections_per_neuron=pop_cfg.connections_per_neuron,
                    sigma_d_mm=pop_cfg.sigma_d_mm,
                    weight_range=tuple(pop_cfg.weight_range) if pop_cfg.weight_range else (0.1, 1.0),
                    seed=pop_cfg.seed,
                    neuron_centers=neuron_centers,
                    use_distance_weights=pop_cfg.use_distance_weights,
                    far_connection_fraction=pop_cfg.far_connection_fraction,
                    far_sigma_factor=pop_cfg.far_sigma_factor,
                    distance_weight_randomness_pct=pop_cfg.distance_weight_randomness_pct,
                    edge_offset=pop_cfg.edge_offset,
                    neuron_jitter_factor=pop_cfg.neuron_jitter_factor if hasattr(pop_cfg, 'neuron_jitter_factor') else 1.0,
                )
            
            # Build filter
            filter_method = pop_cfg.filter_method or "none"
            filter_module = None
            if filter_method != "none":
                try:
                    filter_cls = FILTER_REGISTRY.get_class(filter_method)
                    filter_params = pop_cfg.filter_params or {}
                    filter_params["dt"] = self.config.simulation.dt
                    filter_module = filter_cls(**filter_params).to(self.device)
                except KeyError:
                    raise ValueError(f"Unknown filter method: {filter_method}")
            
            # Build neuron model
            neuron_model_name = pop_cfg.neuron_model or "izhikevich"
            try:
                neuron_cls = NEURON_REGISTRY.get_class(neuron_model_name.lower())
                neuron_params = pop_cfg.model_params or {}
                neuron_params["dt"] = self.config.simulation.dt
                neuron_params["noise_std"] = pop_cfg.noise_std
                neuron_model = neuron_cls(**neuron_params).to(self.device)
            except KeyError:
                raise ValueError(f"Unknown neuron model: {neuron_model_name}")
            
            # Store population context
            self.populations.append({
                "name": pop_cfg.name,
                "config": pop_cfg,
                "grid": grid,
                "innervation": innervation_module,
                "filter": filter_module,
                "neuron": neuron_model,
                "neuron_centers": neuron_centers,
            })
    
    def run(
        self,
        stimulus: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """Run simulation with given stimulus.
        
        Args:
            stimulus: Stimulus tensor [time, height, width] or [batch, time, height, width]
            return_intermediates: If True, return intermediate activations
        
        Returns:
            Dictionary with results for each population:
            {
                "population_name": {
                    "spikes": torch.Tensor,
                    "voltages": torch.Tensor,  # if return_intermediates
                    "filtered": torch.Tensor,  # if return_intermediates
                    "drive": torch.Tensor,  # if return_intermediates
                }
            }
        """
        results = {}
        
        for pop in self.populations:
            pop_name = pop["name"]
            innervation = pop["innervation"]
            filter_module = pop["filter"]
            neuron_model = pop["neuron"]
            grid = pop["grid"]
            
            # Apply stimulus to receptors
            receptor_input = self._stimulus_to_receptors(stimulus, grid)
            
            # Apply innervation
            # InnervationModule expects [batch, time, grid_h, grid_w] for grid-based
            # FlatInnervationModule expects [batch, time, num_receptors] or [batch, num_receptors]
            if isinstance(innervation, FlatInnervationModule):
                # For flat innervation, need to convert [batch, time, h, w] to [batch, time, num_receptors]
                if receptor_input.ndim == 4:
                    batch, time, h, w = receptor_input.shape
                    receptor_input = receptor_input.view(batch, time, h * w)
                elif receptor_input.ndim == 3:
                    batch, h, w = receptor_input.shape
                    receptor_input = receptor_input.view(batch, h * w)
            
            drive = innervation(receptor_input)
            
            # Apply filter
            if filter_module is not None:
                filtered = filter_module(drive)
            else:
                filtered = drive
            
            # Apply neuron model (returns tuple: (v_trace, spikes) or just spikes)
            neuron_output = neuron_model(filtered)
            if isinstance(neuron_output, tuple):
                v_trace, spikes = neuron_output
            else:
                spikes = neuron_output
                v_trace = None
            
            # Store results
            pop_results = {"spikes": spikes}
            if return_intermediates:
                pop_results.update({
                    "drive": drive,
                    "filtered": filtered,
                })
                if v_trace is not None:
                    pop_results["voltages"] = v_trace
            results[pop_name] = pop_results
        
        return results
    
    def _stimulus_to_receptors(
        self,
        stimulus: torch.Tensor,
        grid: Any,
    ) -> torch.Tensor:
        """Map stimulus to receptor activations.
        
        This is a simplified implementation. Full implementation would
        properly sample stimulus at receptor locations.
        
        InnervationModule expects:
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
