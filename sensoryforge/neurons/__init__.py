"""Spiking neuron models for sensory encoding.

This module contains PyTorch-based spiking neuron model implementations
that convert filtered sensory signals into sparse spike trains.

Available Models:
    IzhikevichNeuronTorch: Simple, efficient model with rich dynamics
    AdExNeuronTorch: Adaptive exponential integrate-and-fire model
    MQIFNeuronTorch: Modified quadratic integrate-and-fire model
    FANeuronTorch: Fast adapting neuron model
    SANeuronTorch: Slowly adapting neuron model

All models inherit from torch.nn.Module and support batched processing,
GPU acceleration, and configurable parameters.

Example:
    >>> from sensoryforge.neurons import IzhikevichNeuronTorch
    >>> neuron = IzhikevichNeuronTorch(num_neurons=100)
    >>> spikes, state = neuron(input_current)
"""

# Import PyTorch neuron models
from .izhikevich import IzhikevichNeuronTorch
from .adex import AdExNeuronTorch
from .mqif import MQIFNeuronTorch
from .fa import FANeuronTorch
from .sa import SANeuronTorch

__all__ = [
    "IzhikevichNeuronTorch",
    "AdExNeuronTorch",
    "MQIFNeuronTorch",
    "FANeuronTorch",
    "SANeuronTorch",
]
