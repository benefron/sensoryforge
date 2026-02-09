"""SensoryForge: An extensible playground for sensory encoding and population activity generation.

SensoryForge is a GPU-accelerated, PyTorch-based toolkit for exploring sensory
encoding schemes across modalities (touch, vision, audition, fabricated). It enables
interactive experiment design (GUI), scalable batch execution (CLI/YAML), and
artificial dataset generation for ML and neuromorphic applications.

Key Components:
    - core: Spatial grids, receptive fields, pipeline orchestration, composite grids
    - filters: Temporal filtering (SA/RA dual-pathway)
    - neurons: Spiking neuron models (Izhikevich, AdEx, MQIF, FA, SA, Equation DSL)
    - stimuli: Stimulus generation (Gaussian, texture, moving)
    - solvers: ODE integration (Euler, adaptive via torchdiffeq)
    - gui: PyQt5 interactive workbench
    - cli: Command-line interface for batch execution

Workflow:
    GUI → Design and test experiments interactively
    CLI/YAML → Export for batch execution and scalability
    Python API → Programmatic access for custom analysis

Example:
    >>> from sensoryforge.core.pipeline import TactileEncodingPipelineTorch
    >>> from sensoryforge.stimuli import gaussian_pressure_torch
    >>> # Create and run sensory encoding pipeline
"""

__version__ = "0.2.0"
__author__ = "Sensory Forge Contributors"
__license__ = "MIT"

# Import key classes for top-level access
from sensoryforge.core.grid import GridManager, create_grid_torch

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "GridManager",
    "create_grid_torch",
]
