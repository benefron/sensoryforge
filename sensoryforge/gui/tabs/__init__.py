"""Tab widgets for the tactile simulation GUI."""

from .mechanoreceptor_tab import MechanoreceptorTab, NeuronPopulation
from .stimulus_tab import StimulusDesignerTab
from .spiking_tab import SpikingNeuronTab
from .protocol_suite_tab import ProtocolSuiteTab

# NOTE: AnalyticalInversionTab excluded from v0.1.0 (requires decoding).
# Will be added in v0.2.0+ after Papers 2-3 publication.

__all__ = [
    "MechanoreceptorTab",
    "NeuronPopulation",
    "StimulusDesignerTab",
    "SpikingNeuronTab",
    "ProtocolSuiteTab",
]
