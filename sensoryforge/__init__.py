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

# Import key classes for top-level access (resolves ReviewFinding#L4)
from sensoryforge.core.grid import GridManager, create_grid_torch
from sensoryforge.core.pipeline import TactileEncodingPipelineTorch
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
from sensoryforge.neurons.adex import AdExNeuronTorch
from sensoryforge.neurons.mqif import MQIFNeuronTorch
from sensoryforge.neurons.fa import FANeuronTorch
from sensoryforge.neurons.sa import SANeuronTorch
from sensoryforge.filters.sa_ra import SAFilterTorch, RAFilterTorch
from sensoryforge.stimuli.stimulus import StimulusGenerator

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "GridManager",
    "create_grid_torch",
    "TactileEncodingPipelineTorch",
    "GeneralizedTactileEncodingPipeline",
    "IzhikevichNeuronTorch",
    "AdExNeuronTorch",
    "MQIFNeuronTorch",
    "FANeuronTorch",
    "SANeuronTorch",
    "SAFilterTorch",
    "RAFilterTorch",
    "StimulusGenerator",
]
