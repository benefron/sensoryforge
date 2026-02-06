"""Bridge the tactile encoding pipeline with Izhikevich spiking neurons."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

try:
    from neurons.izhikevich import IzhikevichNeuronTorch
except ImportError:  # pragma: no cover - fallback for script execution
    NEURON_PATH = os.path.join(os.path.dirname(__file__), "..", "neurons")
    if NEURON_PATH not in sys.path:
        sys.path.append(NEURON_PATH)
    from izhikevich import IzhikevichNeuronTorch  # type: ignore

if TYPE_CHECKING:
    from encoding.pipeline_torch import TactileEncodingPipelineTorch


class TactileNeuronAdapter(nn.Module):
    """Wrap :class:`IzhikevichNeuronTorch` for SA/RA population firing."""

    def __init__(
        self,
        neuron_type: str = "SA",
        num_neurons: Optional[int] = None,
        *,
        input_scaling: float = 1.0,
        noise_std: float = 0.0,
        neuron_params: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialise an adapter for one mechanoreceptor population."""
        super().__init__()

        self.neuron_type = neuron_type
        self.num_neurons = num_neurons
        self.input_scaling = input_scaling
        self.noise_std = noise_std

        # Default Izhikevich parameters for tactile encoding
        default_params = {
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 8.0,
            "v_init": -65.0,
            "dt": 1e-4,  # Match the filter dt
            "threshold": 30.0,
        }

        neuron_params = neuron_params or default_params

        # Create Izhikevich neuron
        self.izhikevich = IzhikevichNeuronTorch(**neuron_params)

    def forward(self, filter_outputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project filter responses into spike trains."""
        # Handle single time step by expanding to sequence
        if len(filter_outputs.shape) == 2:
            # (batch_size, num_neurons) -> (batch_size, 1, num_neurons)
            filter_outputs = filter_outputs.unsqueeze(1)

        batch_size, time_steps, num_neurons = filter_outputs.shape

        if self.num_neurons is None:
            self.num_neurons = num_neurons

        # Scale inputs
        scaled_inputs = self.input_scaling * filter_outputs

        # Add noise if specified
        if self.noise_std > 0:
            noise = torch.normal(
                0, self.noise_std, scaled_inputs.shape, device=scaled_inputs.device
            )
            scaled_inputs = scaled_inputs + noise

        # Process through Izhikevich neuron
        v_trace, spikes = self.izhikevich(scaled_inputs)

        return {"v_trace": v_trace, "spikes": spikes}


class TactileSpikingNetwork(nn.Module):
    """Compose tactile encoding filters with spiking neuron adapters."""

    def __init__(
        self,
        pipeline: "TactileEncodingPipelineTorch",
        neuron_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Wrap an encoding pipeline with SA/RA neuron populations."""
        super().__init__()

        self.pipeline = pipeline

        # Default neuron configuration
        default_config = {
            "sa_scaling": 1.0,
            "ra_scaling": 1.0,
            "noise_std": 0.5,
            "neuron_params": {
                "a": 0.02,
                "b": 0.2,
                "c": -65.0,
                "d": 8.0,
                "dt": 1e-4,
                "threshold": 30.0,
            },
        }

        config = {**default_config, **(neuron_config or {})}

        # Get neuron counts from pipeline
        neuron_counts = pipeline.get_neuron_counts()

        # Create SA neuron adapter
        self.sa_neurons = TactileNeuronAdapter(
            neuron_type="SA",
            num_neurons=neuron_counts["sa_neurons"],
            input_scaling=config["sa_scaling"],
            noise_std=config["noise_std"],
            neuron_params=config["neuron_params"],
        )

        # Create RA neuron adapter
        self.ra_neurons = TactileNeuronAdapter(
            neuron_type="RA",
            num_neurons=neuron_counts["ra_neurons"],
            input_scaling=config["ra_scaling"],
            noise_std=config["noise_std"],
            neuron_params=config["neuron_params"],
        )

    def forward(
        self,
        stimuli: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """Process stimuli through filters and produce spike outputs."""
        # Process through tactile encoding pipeline
        pipeline_results = self.pipeline(stimuli, return_intermediates)

        # Process SA/RA outputs through spiking neurons
        sa_spikes = self.sa_neurons(pipeline_results["sa_outputs"])
        ra_spikes = self.ra_neurons(pipeline_results["ra_outputs"])

        results = {
            "sa_spikes": sa_spikes,
            "ra_spikes": ra_spikes,
            "sa_outputs": pipeline_results["sa_outputs"],
            "ra_outputs": pipeline_results["ra_outputs"],
        }

        if return_intermediates:
            results.update(pipeline_results)

        return results

    def reset_pipeline_states(self) -> None:
        """Reset SA/RA filter states in the pipeline."""
        self.pipeline.reset_filter_states()

    def get_network_info(self) -> Dict[str, Any]:
        """Summarise neuron counts alongside pipeline metadata."""
        pipeline_info = self.pipeline.get_pipeline_info()

        return {
            "pipeline": pipeline_info,
            "sa_neuron_count": self.sa_neurons.num_neurons,
            "ra_neuron_count": self.ra_neurons.num_neurons,
            "total_neurons": (
                self.sa_neurons.num_neurons + self.ra_neurons.num_neurons
            ),
        }


def create_complete_tactile_network(
    grid_size: int = 80,
    device: torch.device | str = "cpu",
    seed: Optional[int] = None,
    neuron_config: Optional[Dict[str, Any]] = None,
) -> TactileSpikingNetwork:
    """Instantiate the full tactile network with default parameters."""
    # Import pipeline creation function
    from .pipeline_torch import create_standard_pipeline

    # Create pipeline
    pipeline = create_standard_pipeline(grid_size=grid_size, device=device, seed=seed)

    # Create complete network
    network = TactileSpikingNetwork(pipeline, neuron_config)

    return network


def create_small_tactile_network(
    grid_size: int = 40,
    device: torch.device | str = "cpu",
    seed: Optional[int] = None,
    neuron_config: Optional[Dict[str, Any]] = None,
) -> TactileSpikingNetwork:
    """Create a lighter-weight tactile network for tests and demos."""
    from .pipeline_torch import create_small_pipeline

    pipeline = create_small_pipeline(grid_size=grid_size, device=device, seed=seed)

    network = TactileSpikingNetwork(pipeline, neuron_config)

    return network
