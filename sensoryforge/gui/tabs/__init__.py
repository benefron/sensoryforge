"""Tab widgets for the tactile simulation GUI."""

from .mechanoreceptor_tab import MechanoreceptorTab, NeuronPopulation
from .stimulus_tab import StimulusDesignerTab
from .spiking_tab import SpikingNeuronTab
from .protocol_suite_tab import ProtocolSuiteTab
from .analytical_inversion_tab import AnalyticalInversionTab

__all__ = [
    "MechanoreceptorTab",
    "NeuronPopulation",
    "StimulusDesignerTab",
    "SpikingNeuronTab",
    "ProtocolSuiteTab",
    "AnalyticalInversionTab",
]
