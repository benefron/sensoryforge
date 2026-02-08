"""Core module for bio-inspired sensory encoding.

This module provides a complete PyTorch-based implementation of sensory encoding
architectures with dense tensor operations, batch processing, GPU support, and
modular components.

Modules:
    grid: Grid management and coordinate systems
    composite_grid: Multi-population spatial substrates
    innervation: Receptive field generation and neural connectivity
    mechanoreceptors: Gaussian convolution mechanoreceptor simulation
    pipeline: Complete encoding pipeline orchestration
    tactile_network: Spiking neural network integration
    compression: Compression operators for dimensionality reduction
    visualization: Visualization utilities

The core module integrates with filters (SA/RA temporal dynamics) and neurons
(spiking models) to provide end-to-end sensory encoding capabilities.
"""

# PyTorch-based modules
from .grid import GridManager, create_grid_torch
from .composite_grid import CompositeGrid
from sensoryforge.stimuli.stimulus import (
    StimulusGenerator,
    gaussian_pressure_torch,
    point_pressure_torch,
)
from .mechanoreceptors import (
    MechanoreceptorModule,
    compute_mechanoreceptor_responses_torch,
)
from .innervation import (
    InnervationModule,
    create_sa_innervation,
    create_ra_innervation,
)
from sensoryforge.filters.sa_ra import SAFilterTorch, RAFilterTorch, CombinedSARAFilter
from .tactile_network import (
    TactileNeuronAdapter,
    TactileSpikingNetwork,
    create_complete_tactile_network,
)
from .pipeline import (
    TactileEncodingPipelineTorch,
    create_standard_pipeline,
    create_small_pipeline,
)
from .compression import CompressionOperator, build_compression_operator

__all__ = [
    # Grid management
    "GridManager",
    "create_grid_torch",
    "CompositeGrid",
    # Stimulus generation
    "StimulusGenerator",
    "gaussian_pressure_torch",
    "point_pressure_torch",
    # Mechanoreceptor responses
    "MechanoreceptorModule",
    "compute_mechanoreceptor_responses_torch",
    # Neural innervation
    "InnervationModule",
    "create_sa_innervation",
    "create_ra_innervation",
    # SA/RA filters
    "SAFilterTorch",
    "RAFilterTorch",
    "CombinedSARAFilter",
    # Neural networks
    "TactileNeuronAdapter",
    "TactileSpikingNetwork",
    "create_complete_tactile_network",
    # Complete pipeline
    "TactileEncodingPipelineTorch",
    "create_standard_pipeline",
    "create_small_pipeline",
    # Compression operators
    "CompressionOperator",
    "build_compression_operator",
]
