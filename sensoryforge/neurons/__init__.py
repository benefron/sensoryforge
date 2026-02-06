"""
Neurons Module

This module contains all spiking neuron model implementations for the
bio-inspired sensory encoding system.

Available Models:
- IzhikevichNeuron: Simple, efficient model with rich dynamics
- AdExNeuron: Adaptive exponential integrate-and-fire model
- MQIFNeuron: Modified quadratic integrate-and-fire model

Unified Interface:
- SpikeEncoder: Model-agnostic encoding interface
- MultiModelEncoder: Parallel multi-model comparison

Legacy Models (for reference):
- adex_legacy, izhikevich_legacy, mqif_legacy
"""

# Import PyTorch neuron models
from .izhikevich import IzhikevichNeuronTorch
from .adex import AdExNeuronTorch
from .mqif import MQIFNeuronTorch
from .fa import FANeuronTorch
from .sa import SANeuronTorch

# Import unified encoder - temporarily disabled for testing
# from .spike_encoder import (
#     SpikeEncoder,
#     MultiModelEncoder,
#     create_tactile_encoder,
#     create_comparison_encoder
# )

__version__ = "0.1.0"
__author__ = "Bio-Inspired Sensory Encoding Project"

__all__ = [
    # PyTorch neuron models
    "IzhikevichNeuronTorch",
    "AdExNeuronTorch",
    "MQIFNeuronTorch",
    "FANeuronTorch",
    "SANeuronTorch",
]
