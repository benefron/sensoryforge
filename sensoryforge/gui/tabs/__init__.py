"""Tab widgets for the SensoryForge GUI.

Current tabs:
    MechanoreceptorTab: Spatial grid, receptor populations, receptive fields
    StimulusDesignerTab: Interactive stimulus creation and preview
    SpikingNeuronTab: Neuron model configuration, simulation, spike visualization
    ProtocolSuiteTab: Protocol library and batch execution queue
"""

from .mechanoreceptor_tab import MechanoreceptorTab, NeuronPopulation
from .stimulus_tab import StimulusDesignerTab
from .spiking_tab import SpikingNeuronTab
from .protocol_suite_tab import ProtocolSuiteTab

__all__ = [
    "MechanoreceptorTab",
    "NeuronPopulation",
    "StimulusDesignerTab",
    "SpikingNeuronTab",
    "ProtocolSuiteTab",
]
