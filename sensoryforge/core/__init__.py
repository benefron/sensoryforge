"""
Encoding module for bio-inspired tactile encoding pipeline.

This module provides a complete PyTorch-based implementation of the
Parvizi-Fard tactile encoding architecture with dense tensor operations,
batch processing, GPU support, and enhanced filter processing.

PyTorch modules:
- grid_torch: Grid management and coordinate systems
- stimulus_torch: Stimulus generation with batch processing
- mechanoreceptors_torch: Gaussian convolution using conv2d
- innervation_torch: Dense 3D tensor innervation maps
- filters_torch: SA/RA differential equation filters with enhanced processing
- tactile_network: Integration with existing Izhikevich neurons
- pipeline_torch: Complete pipeline orchestration
"""

# PyTorch-based modules
from .grid_torch import GridManager, create_grid_torch
from .stimulus_torch import (
    StimulusGenerator,
    gaussian_pressure_torch,
    point_pressure_torch,
)
from .mechanoreceptors_torch import (
    MechanoreceptorModule,
    compute_mechanoreceptor_responses_torch,
)
from .innervation_torch import (
    InnervationModule,
    create_sa_innervation,
    create_ra_innervation,
)
from .filters_torch import SAFilterTorch, RAFilterTorch, CombinedSARAFilter
from .tactile_network import (
    TactileNeuronAdapter,
    TactileSpikingNetwork,
    create_complete_tactile_network,
)
from .pipeline_torch import (
    TactileEncodingPipelineTorch,
    create_standard_pipeline,
    create_small_pipeline,
)
from .compression import CompressionOperator, build_compression_operator

__all__ = [
    # Grid management
    "GridManager",
    "create_grid_torch",
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
