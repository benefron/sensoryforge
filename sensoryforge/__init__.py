"""SensoryForge: Modular, extensible framework for simulating sensory encoding across modalities.

SensoryForge is a GPU-accelerated, PyTorch-based toolkit for exploring sensory encoding
schemes inspired by neuroscience. The architecture is fully modality-agnostic and supports
touch, vision, audition, and multi-modal fusion.

Key Components:
    - core: Spatial grids, receptive fields, pipeline orchestration
    - filters: Temporal filtering (SA/RA, ON/OFF, custom)
    - neurons: Spiking neuron models (Izhikevich, AdEx, MQIF, custom)
    - stimuli: Stimulus generation and pattern synthesis
    - gui: Optional PyQt5 interactive interface

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
