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
from sensoryforge.core.composite_grid import CompositeReceptorGrid
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
        self._build_grids()
        
        # Build populations (innervation, filters, neurons)
        self.populations: List[Dict[str, Any]] = []
        self._build_populations()
    
    def _build_grids(self) -> None:
        """Build receptor grids from config."""
        for grid_cfg in self.config.grids:
            # Create grid based on arrangement
            arrangement = grid_cfg.arrangement
            
            if arrangement == "composite":
                # Composite grid with multiple layers
                layers = {}
                # For now, composite grids need special handling
                # This is a placeholder - full implementation needed
                raise NotImplementedError("Composite grids not yet implemented in SimulationEngine")
            else:
                # Single grid
                grid = ReceptorGrid(
                    rows=grid_cfg.rows or 40,
                    cols=grid_cfg.cols or 40,
                    spacing=grid_cfg.spacing,
                    arrangement=arrangement,
                    center=(grid_cfg.center_x, grid_cfg.center_y),
                    density=grid_cfg.density,
                    seed=grid_cfg.seed,
                    device=self.device,
                )
                self.grids.append(grid)
    
    def _build_populations(self) -> None:
        """Build population execution contexts (innervation, filters, neurons)."""
        for pop_cfg in self.config.populations:
            if not pop_cfg.enabled:
                continue
            
            # Find target grid
            target_grid_name = pop_cfg.target_grid or (self.config.grids[0].name if self.config.grids else None)
            if target_grid_name is None:
                raise ValueError(f"Population {pop_cfg.name} has no target grid")
            
            grid = next((g for g in self.grids if getattr(g, 'name', None) == target_grid_name), None)
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
            neuron_rows = pop_cfg.neuron_rows or int(np.sqrt(pop_cfg.num_neurons)) if pop_cfg.num_neurons else 10
            neuron_cols = pop_cfg.neuron_cols or int(np.sqrt(pop_cfg.num_neurons)) if pop_cfg.num_neurons else 10
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
                grid_manager = GridManager(grid)
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
            
            # Apply stimulus to receptors (simplified - assumes stimulus matches grid)
            # In full implementation, this would map stimulus to receptor locations
            receptor_input = self._stimulus_to_receptors(stimulus, pop["grid"])
            
            # Apply innervation
            drive = innervation(receptor_input)
            
            # Apply filter
            if filter_module is not None:
                filtered = filter_module(drive)
            else:
                filtered = drive
            
            # Apply neuron model
            spikes = neuron_model(filtered)
            
            # Store results
            pop_results = {"spikes": spikes}
            if return_intermediates:
                pop_results.update({
                    "drive": drive,
                    "filtered": filtered,
                    "voltages": getattr(neuron_model, "v", None),
                })
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
        """
        # Simplified: assume stimulus is already at receptor resolution
        # In full implementation, would interpolate/sample stimulus at receptor coords
        if stimulus.ndim == 3:
            # [time, height, width]
            return stimulus.flatten(start_dim=1)  # [time, num_receptors]
        elif stimulus.ndim == 4:
            # [batch, time, height, width]
            return stimulus.flatten(start_dim=2)  # [batch, time, num_receptors]
        else:
            raise ValueError(f"Unexpected stimulus shape: {stimulus.shape}")
