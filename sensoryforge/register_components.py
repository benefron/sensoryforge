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

# Innervation / receptive-field builders
from sensoryforge.core.innervation import (
    GaussianInnervation,
    UniformInnervation,
    OneToOneInnervation,
    DistanceWeightedInnervation,
)
from sensoryforge.core.rf_builders.imported import ImportedRFBuilder
from sensoryforge.core.rf_builders.template import TemplateRFBuilder

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
from sensoryforge.stimuli.tactile import (
    RampGaussianStimulus,
    MovingEdgeStimulus,
    BrailleStimulus,
    DriftingGratingStimulus,
)

# Solvers
from sensoryforge.solvers.euler import EulerSolver
from sensoryforge.solvers.adaptive import AdaptiveSolver

# Processing
from sensoryforge.core.processing import IdentityLayer, OnOffLayer

# Grid arrangements (G3): thin ReceptorGrid subclasses, one per arrangement
from sensoryforge.core.grid_arrangements import (
    GridArrangement,
    PoissonArrangement,
    HexArrangement,
    JitteredGridArrangement,
    BlueNoiseArrangement,
)


def register_all() -> None:
    """Register all SensoryForge components with their registries.

    This function should be called once at module import time or application
    startup to ensure all components are available via registry lookup.
    """
    # Register neurons
    # F-046: the registry itself is case-insensitive (ComponentRegistry
    # case-folds lookups), so pure case variants of the same name (e.g.
    # "izhikevich"/"Izhikevich") are no longer registered separately here --
    # only genuinely distinct names/aliases are.
    NEURON_REGISTRY.register("Izhikevich", IzhikevichNeuronTorch)
    NEURON_REGISTRY.register("AdEx", AdExNeuronTorch)
    NEURON_REGISTRY.register("MQIF", MQIFNeuronTorch)
    NEURON_REGISTRY.register("FA", FANeuronTorch)
    NEURON_REGISTRY.register("SA", SANeuronTorch)
    NEURON_REGISTRY.register("dsl", NeuronModel)
    NEURON_REGISTRY.register("DSL (Custom)", NeuronModel)  # GUI alias (distinct name)

    # Register filters
    # Note: SAFilterTorch and RAFilterTorch don't inherit BaseFilter yet
    # They will be refactored in a future update
    FILTER_REGISTRY.register("SA", SAFilterTorch)
    FILTER_REGISTRY.register("safilter", SAFilterTorch)  # Alias (distinct name)
    FILTER_REGISTRY.register("RA", RAFilterTorch)
    FILTER_REGISTRY.register("rafilter", RAFilterTorch)  # Alias (distinct name)
    FILTER_REGISTRY.register("none", type(None))  # No filter
    FILTER_REGISTRY.register("identity", type(None))  # No filter alias

    # Register innervation methods (receptive-field builders). The classes
    # are registered directly -- their keyword-only-friendly constructors and
    # from_config() take receptor_coords/neuron_centers, so no factory
    # closure is needed and the registry key maps to a contract-checked class
    # whose build() returns a ReceptiveFieldBank (Phase 2, I3).
    INNERVATION_REGISTRY.register("gaussian", GaussianInnervation)
    INNERVATION_REGISTRY.register("uniform", UniformInnervation)
    INNERVATION_REGISTRY.register("one_to_one", OneToOneInnervation)
    INNERVATION_REGISTRY.register("distance_weighted", DistanceWeightedInnervation)
    INNERVATION_REGISTRY.register("template", TemplateRFBuilder)
    INNERVATION_REGISTRY.register("imported", ImportedRFBuilder)

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

    # Pressure-simulation's four ported stimuli (Phase 2, K2)
    STIMULUS_REGISTRY.register("ramp_gaussian", RampGaussianStimulus)
    STIMULUS_REGISTRY.register("moving_edge", MovingEdgeStimulus)
    STIMULUS_REGISTRY.register("braille", BrailleStimulus)
    STIMULUS_REGISTRY.register("drifting_grating", DriftingGratingStimulus)

    # Register solvers
    SOLVER_REGISTRY.register("euler", EulerSolver)
    SOLVER_REGISTRY.register("adaptive", AdaptiveSolver)

    # Register processing layers
    PROCESSING_REGISTRY.register("identity", IdentityLayer)
    PROCESSING_REGISTRY.register("onoff", OnOffLayer)  # Wave M3

    # Register grid arrangements (G3): real ReceptorGrid subclasses, one per
    # arrangement, each constructible via from_config()/GRID_REGISTRY.create().
    GRID_REGISTRY.register("grid", GridArrangement)
    GRID_REGISTRY.register("poisson", PoissonArrangement)
    GRID_REGISTRY.register("hex", HexArrangement)
    GRID_REGISTRY.register("jittered_grid", JitteredGridArrangement)
    GRID_REGISTRY.register("blue_noise", BlueNoiseArrangement)

    # G2: discover third-party components advertised via the
    # "sensoryforge.components" entry-point group. A plugin that fails to
    # load produces a warning (see sensoryforge.plugins), never a crash.
    from sensoryforge.plugins import discover_entry_point_plugins

    discover_entry_point_plugins()


# Auto-register on import
register_all()
