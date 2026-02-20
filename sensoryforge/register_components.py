"""Auto-registration of all SensoryForge components.

This module registers all concrete component implementations with their
respective registries. Import this module to ensure all components are
registered before use.

Example:
    >>> from sensoryforge.register_components import register_all
    >>> register_all()
    >>> from sensoryforge.registry import NEURON_REGISTRY
    >>> neuron = NEURON_REGISTRY.create("izhikevich", **config)
"""

from sensoryforge.registry import (
    NEURON_REGISTRY,
    FILTER_REGISTRY,
    INNERVATION_REGISTRY,
    STIMULUS_REGISTRY,
    SOLVER_REGISTRY,
    GRID_REGISTRY,
    PROCESSING_REGISTRY,
)

# Neurons
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
from sensoryforge.neurons.adex import AdExNeuronTorch
from sensoryforge.neurons.mqif import MQIFNeuronTorch
from sensoryforge.neurons.fa import FANeuronTorch
from sensoryforge.neurons.sa import SANeuronTorch
from sensoryforge.neurons.model_dsl import NeuronModel

# Filters
from sensoryforge.filters.sa_ra import SAFilterTorch, RAFilterTorch
from sensoryforge.filters.base import BaseFilter

# Innervation
from sensoryforge.core.innervation import (
    GaussianInnervation,
    UniformInnervation,
    OneToOneInnervation,
    DistanceWeightedInnervation,
)

# Stimuli
from sensoryforge.stimuli.builder import (
    StaticStimulus,
    MovingStimulus,
    CompositeStimulus,
    TimelineStimulus,
    RepeatedPatternStimulus,
)
from sensoryforge.stimuli.gaussian import GaussianStimulus
from sensoryforge.stimuli.texture import GaborTexture, EdgeGrating
from sensoryforge.stimuli.moving import MovingStimulus as MovingStimulusLegacy

# Solvers
from sensoryforge.solvers.euler import EulerSolver
from sensoryforge.solvers.adaptive import AdaptiveSolver

# Processing
from sensoryforge.core.processing import IdentityLayer

# Grid arrangements (these are string identifiers, not classes)
# Grid creation is handled via ReceptorGrid with arrangement parameter


def register_all() -> None:
    """Register all SensoryForge components with their registries.
    
    This function should be called once at module import time or application
    startup to ensure all components are available via registry lookup.
    """
    # Register neurons
    NEURON_REGISTRY.register("izhikevich", IzhikevichNeuronTorch)
    NEURON_REGISTRY.register("Izhikevich", IzhikevichNeuronTorch)  # Alias
    NEURON_REGISTRY.register("adex", AdExNeuronTorch)
    NEURON_REGISTRY.register("AdEx", AdExNeuronTorch)  # Alias
    NEURON_REGISTRY.register("mqif", MQIFNeuronTorch)
    NEURON_REGISTRY.register("MQIF", MQIFNeuronTorch)  # Alias
    NEURON_REGISTRY.register("fa", FANeuronTorch)
    NEURON_REGISTRY.register("FA", FANeuronTorch)  # Alias
    NEURON_REGISTRY.register("sa", SANeuronTorch)
    NEURON_REGISTRY.register("SA", SANeuronTorch)  # Alias
    NEURON_REGISTRY.register("dsl", NeuronModel)
    NEURON_REGISTRY.register("DSL (Custom)", NeuronModel)  # GUI alias
    
    # Register filters
    # Note: SAFilterTorch and RAFilterTorch don't inherit BaseFilter yet
    # They will be refactored in a future update
    FILTER_REGISTRY.register("sa", SAFilterTorch)
    FILTER_REGISTRY.register("SA", SAFilterTorch)  # Alias
    FILTER_REGISTRY.register("safilter", SAFilterTorch)  # Alias
    FILTER_REGISTRY.register("ra", RAFilterTorch)
    FILTER_REGISTRY.register("RA", RAFilterTorch)  # Alias
    FILTER_REGISTRY.register("rafilter", RAFilterTorch)  # Alias
    FILTER_REGISTRY.register("none", type(None))  # No filter
    FILTER_REGISTRY.register("identity", type(None))  # No filter alias
    
    # Register innervation methods
    # These use factory functions since they're instantiated via create_innervation()
    def create_gaussian_innervation(**kwargs):
        from sensoryforge.core.innervation import GaussianInnervation
        receptor_coords = kwargs.pop("receptor_coords")
        neuron_centers = kwargs.pop("neuron_centers")
        device = kwargs.pop("device", "cpu")
        return GaussianInnervation(receptor_coords, neuron_centers, device=device, **kwargs)
    
    def create_uniform_innervation(**kwargs):
        from sensoryforge.core.innervation import UniformInnervation
        receptor_coords = kwargs.pop("receptor_coords")
        neuron_centers = kwargs.pop("neuron_centers")
        device = kwargs.pop("device", "cpu")
        return UniformInnervation(receptor_coords, neuron_centers, device=device, **kwargs)
    
    def create_one_to_one_innervation(**kwargs):
        from sensoryforge.core.innervation import OneToOneInnervation
        receptor_coords = kwargs.pop("receptor_coords")
        neuron_centers = kwargs.pop("neuron_centers")
        device = kwargs.pop("device", "cpu")
        return OneToOneInnervation(receptor_coords, neuron_centers, device=device, **kwargs)
    
    def create_distance_weighted_innervation(**kwargs):
        from sensoryforge.core.innervation import DistanceWeightedInnervation
        receptor_coords = kwargs.pop("receptor_coords")
        neuron_centers = kwargs.pop("neuron_centers")
        device = kwargs.pop("device", "cpu")
        return DistanceWeightedInnervation(receptor_coords, neuron_centers, device=device, **kwargs)
    
    INNERVATION_REGISTRY.register("gaussian", GaussianInnervation, create_gaussian_innervation)
    INNERVATION_REGISTRY.register("uniform", UniformInnervation, create_uniform_innervation)
    INNERVATION_REGISTRY.register("one_to_one", OneToOneInnervation, create_one_to_one_innervation)
    INNERVATION_REGISTRY.register("distance_weighted", DistanceWeightedInnervation, create_distance_weighted_innervation)
    
    # Register stimuli
    STIMULUS_REGISTRY.register("gaussian", GaussianStimulus)
    STIMULUS_REGISTRY.register("static", StaticStimulus)
    STIMULUS_REGISTRY.register("moving", MovingStimulus)
    STIMULUS_REGISTRY.register("composite", CompositeStimulus)
    STIMULUS_REGISTRY.register("timeline", TimelineStimulus)
    STIMULUS_REGISTRY.register("repeated_pattern", RepeatedPatternStimulus)
    STIMULUS_REGISTRY.register("texture", GaborTexture)  # Default texture type
    STIMULUS_REGISTRY.register("gabor", GaborTexture)
    STIMULUS_REGISTRY.register("edge_grating", EdgeGrating)
    
    # Register solvers
    SOLVER_REGISTRY.register("euler", EulerSolver)
    SOLVER_REGISTRY.register("adaptive", AdaptiveSolver)
    
    # Register processing layers
    PROCESSING_REGISTRY.register("identity", IdentityLayer)
    
    # Register grid arrangements (as string identifiers)
    # These are used by ReceptorGrid, not instantiated directly
    GRID_REGISTRY.register("grid", str)  # Placeholder - grid creation handled differently
    GRID_REGISTRY.register("poisson", str)
    GRID_REGISTRY.register("hex", str)
    GRID_REGISTRY.register("jittered_grid", str)
    GRID_REGISTRY.register("blue_noise", str)


# Auto-register on import
register_all()
