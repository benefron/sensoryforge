"""Tab widgets for the SensoryForge GUI.

Current tabs:
    MechanoreceptorTab: Spatial grid, receptor populations, receptive fields
    StimulusDesignerTab: Interactive stimulus creation and preview
    SpikingNeuronTab: Neuron model configuration, simulation, spike visualization
    VisualizationTab: Time-synchronized, multi-panel simulation playback
    BatchTab: Batch execution and SLURM script export
"""

from .mechanoreceptor_tab import MechanoreceptorTab, NeuronPopulation
from .stimulus_tab import StimulusDesignerTab
from .spiking_tab import SpikingNeuronTab
from .visualization_tab import VisualizationTab
from .batch_tab import BatchTab

__all__ = [
    "MechanoreceptorTab",
    "NeuronPopulation",
    "StimulusDesignerTab",
    "SpikingNeuronTab",
    "VisualizationTab",
    "BatchTab",
]
