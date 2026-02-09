"""Tab widgets for the SensoryForge GUI.

Current tabs:
    MechanoreceptorTab: Spatial grid, receptor populations, receptive fields
    StimulusDesignerTab: Interactive stimulus creation and preview
    SpikingNeuronTab: Neuron model configuration, simulation, spike visualization
"""

from .mechanoreceptor_tab import MechanoreceptorTab, NeuronPopulation
from .stimulus_tab import StimulusDesignerTab
from .spiking_tab import SpikingNeuronTab

__all__ = [
    "MechanoreceptorTab",
    "NeuronPopulation",
    "StimulusDesignerTab",
    "SpikingNeuronTab",
]
