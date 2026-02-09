"""encoding.pipeline_torch
=================================

Primary entry point for the **Parvizi–Fard tactile encoding pipeline** used
throughout the project.  The module wires together the grid, stimulus
generation, innervation, temporal filtering, stochastic perturbations, and the
spiking neuron adapters to produce spike rasters from tactile stimuli.  The
implementation mirrors the scientific workflow described in
``docs_root/SCIENTIFIC_HYPOTHESIS.md`` and serves as the canonical
spiking-accuracy baseline for both GUI-driven experiments and automated tests.

The :class:`~encoding.pipeline_torch.TactileEncodingPipelineTorch` class wraps
the entire encode → filter → spike pass and exposes convenience utilities for
stimulus synthesis, batch execution, and device handling.  Helper constructors
``create_standard_pipeline`` and ``create_small_pipeline`` provide common
configuration presets used by notebooks and pytest suites.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from torch import Tensor

from .grid import GridManager
from sensoryforge.stimuli.stimulus import StimulusGenerator
from sensoryforge.config.yaml_utils import load_yaml
from .innervation import create_sa_innervation, create_ra_innervation
from sensoryforge.filters.sa_ra import CombinedSARAFilter
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
from sensoryforge.filters.noise import MembraneNoiseTorch, ReceptorNoiseTorch


def _deep_update(
    base: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base``.

    The pipeline configuration is frequently assembled from multiple nested
    dictionaries (baseline YAML + user overrides + helper patches).  This
    helper maintains that behaviour without mutating the caller-provided
    dictionaries.

    Args:
        base: The base dictionary to update.  A deep copy is created internally
            so the original object can be safely re-used by the caller.
        overrides: Dictionary containing new values.  Nested dictionaries are
            merged recursively; sequences and scalars replace the existing
            values.

    Returns:
    A deep-merged dictionary containing ``base`` with ``overrides``
    applied.
    """

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(copy.deepcopy(base[key]), value)
        else:
            base[key] = value
    return base


class TactileEncodingPipelineTorch(nn.Module):
    """End-to-end tactile encoding pipeline implemented in PyTorch.

    The class encapsulates the canonical SA/RA tactile encoding stack used by
    analytical reconstruction experiments.  It loads a YAML configuration,
    allocates the grid, synthesises stimuli, builds innervation tensors,
    applies SA/RA filters, injects biologically inspired noise, and finally
    forwards the resulting currents through Izhikevich neuron models to obtain
    spike trains.

    Typical usage keeps a single instance alive for a batch of experiments and
    calls :meth:`forward` or :meth:`generate_and_process` repeatedly while
    varying stimuli or overrides.  The :mod:`tests/test_pytorch_pipeline.py`
    suite exercises this class directly.
    """

    def __init__(
        self,
        config_path: str = "sensoryforge/config/default_config.yml",
        *,
        config: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """Instantiate the pipeline from YAML and optional overrides.

        Args:
            config_path: Absolute or workspace-relative path to the baseline
                pipeline YAML.  Ignored when ``config`` is supplied.
            config: In-memory configuration dictionary.  When provided it is
                deep-copied and used instead of reading ``config_path``.
            overrides: Optional nested dictionary applied on top of whichever
                configuration was loaded.  Useful for temporary parameter
                sweeps without mutating the source YAML file.
        """
        if config is None:
            with open(config_path, "r", encoding="utf-8") as file:
                config = load_yaml(file)
        else:
            config = copy.deepcopy(config)

        if overrides:
            config = _deep_update(config, overrides)

        self.config = copy.deepcopy(config)

        pipeline_config = self.config["pipeline"]
        neuron_config = self.config.get("neurons", {})
        noise_config = self.config.get("noise", {})
        innervation_config = self.config.get("innervation", {})

        super().__init__()
        self.device = pipeline_config["device"]
        self.seed = pipeline_config["seed"]

        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Create grid manager
        self.grid_manager = GridManager(
            grid_size=pipeline_config["grid_size"],
            spacing=pipeline_config["spacing"],
            center=tuple(pipeline_config["center"]),
            device=self.device,
        )

        # Create stimulus generator
        self.stimulus_generator = StimulusGenerator(self.grid_manager)

        # Create innervation modules matching SA/RA populations
        sa_config = innervation_config.get("sa", {})
        self.sa_innervation = create_sa_innervation(
            self.grid_manager, seed=self.seed, **sa_config
        )

        ra_config = innervation_config.get("ra", {})
        self.ra_innervation = create_ra_innervation(
            self.grid_manager, seed=self.seed, **ra_config
        )

        # Create SA/RA filters
        self.filters = CombinedSARAFilter()

        # Neuron models
        self.sa_neurons = IzhikevichNeuronTorch(dt=neuron_config.get("dt", 0.1))
        self.ra_neurons = IzhikevichNeuronTorch(dt=neuron_config.get("dt", 0.1))

        # Optional input scaling/bias before neurons (helps ensure spiking)
        # Supports shared keys (input_gain, input_bias) or per-type variants
        self.sa_input_gain = float(
            neuron_config.get("input_gain_sa", neuron_config.get("input_gain", 1.0))
        )
        self.ra_input_gain = float(
            neuron_config.get("input_gain_ra", neuron_config.get("input_gain", 1.0))
        )
        self.sa_input_bias = float(
            neuron_config.get("input_bias_sa", neuron_config.get("input_bias", 0.0))
        )
        self.ra_input_bias = float(
            neuron_config.get("input_bias_ra", neuron_config.get("input_bias", 0.0))
        )

        # Noise modules
        self.membrane_noise = MembraneNoiseTorch(
            std=noise_config.get("membrane_amplitude", 0.1)
        )
        self.receptor_noise = ReceptorNoiseTorch(
            std=noise_config.get("receptor_amplitude", 0.1)
        )

        # Move all modules to device
        self.to(self.device)

    def generate_stimulus(
        self,
        stimulus_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Synthesize a single stimulus batch using the configured grid.

        Args:
            stimulus_type: Identifier understood by
                :class:`~encoding.stimulus_torch.StimulusGenerator`.  Defaults
                to ``"gaussian"`` for parity with existing demos.
            **kwargs: Additional keyword arguments forwarded verbatim to the
                generator (e.g. ``center_x``, ``sigma``, ``time_steps``).

        Returns:
            Tensor: Stimulus batch with shape ``(1, …)`` ready to pass into
            :meth:`forward`.
        """
        stimulus_type = stimulus_type or "gaussian"
        config = {"type": stimulus_type, **kwargs}
        return self.stimulus_generator.generate_batch_stimuli([config])

    def apply_noise(self, inputs: Tensor) -> Tensor:
        """Apply receptor and membrane noise in sequence.

        The same noise stack is used for both SA and RA pathways to model
        receptor stochasticity followed by membrane fluctuations before
        spiking.
        """
        noisy_inputs = self.receptor_noise(inputs)
        return self.membrane_noise(noisy_inputs)

    def forward(
        self,
        stimuli: Tensor,
        return_intermediates: bool = False,
        filter_method: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        """Run the full pipeline for a pre-generated stimulus batch.

        Args:
            stimuli: Input tensor with either ``(batch, grid_h, grid_w)`` for
                static stimuli or ``(batch, time, grid_h, grid_w)`` for
                temporal sequences.
            return_intermediates: When ``True`` the response dictionary
                includes tensors from every stage (stimulus, mechanoreceptor
                currents, innervation outputs) to facilitate debugging and
                documentation exports.
            filter_method: Explicit override for
                :meth:`CombinedSARAFilter.forward_enhanced`.  Defaults to
                ``"multi_step"`` whenever a temporal dimension is absent so the
                filter can settle to steady state.

        Returns:
            Dictionary mapping stage name to tensors.  ``sa_spikes`` and
            ``ra_spikes`` contain the spike rasters used by downstream decoding
            experiments.
        """
        # Step 1: Mechanoreceptor responses are identical to the stimulus
        mech_responses = stimuli.clone()

        # Step 2: Apply innervation to get SA/RA neuron inputs
        sa_inputs = self.sa_innervation(mech_responses)
        ra_inputs = self.ra_innervation(mech_responses)

        # Apply biologically inspired receptor + membrane noise before
        # filtering so each pathway sees matched stochastic perturbations.
        sa_inputs = self.apply_noise(sa_inputs)
        ra_inputs = self.apply_noise(ra_inputs)

        # Step 3: Apply SA/RA filters with enhanced processing
        if len(stimuli.shape) == 3:  # Single time step
            filter_method = filter_method or "multi_step"
            sa_outputs, ra_outputs = self.filters.forward_enhanced(
                sa_inputs, ra_inputs, method=filter_method
            )
        else:  # Time sequence - use regular processing
            sa_outputs, ra_outputs = self.filters(sa_inputs, ra_inputs)

        # Izhikevich neuron expects [batch, steps, features] input
        # Add temporal dimension if needed
        if len(sa_outputs.shape) == 2:
            sa_outputs = sa_outputs.unsqueeze(1)  # Add time dimension
            ra_outputs = ra_outputs.unsqueeze(1)

        # Apply configurable gain/bias to drive neurons
        sa_drive = sa_outputs * self.sa_input_gain + self.sa_input_bias
        ra_drive = ra_outputs * self.ra_input_gain + self.ra_input_bias

        sa_result = self.sa_neurons(sa_drive)
        ra_result = self.ra_neurons(ra_drive)

        # Extract spikes from neuron results
        sa_spikes = sa_result[1] if isinstance(sa_result, tuple) else sa_result
        ra_spikes = ra_result[1] if isinstance(ra_result, tuple) else ra_result

        results = {
            "sa_outputs": sa_outputs,
            "ra_outputs": ra_outputs,
            "sa_drive": sa_drive,
            "ra_drive": ra_drive,
            "sa_spikes": sa_spikes,
            "ra_spikes": ra_spikes,
        }

        if return_intermediates:
            results.update(
                {
                    "stimuli": stimuli,
                    "mechanoreceptor_responses": mech_responses,
                    "sa_inputs": sa_inputs,
                    "ra_inputs": ra_inputs,
                }
            )

        return results

    def generate_and_process(
        self,
        stimulus_configs: list[Dict[str, Any]],
        time_steps: Optional[int] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, Tensor]:
        """Generate a batch of stimuli and immediately run the pipeline."""
        # Generate stimuli
        stimuli = self.stimulus_generator.generate_batch_stimuli(
            stimulus_configs, time_steps
        )

        # Process through pipeline
        return self.forward(stimuli, return_intermediates)

    def process_single_stimulus(
        self,
        stimulus_type: str = "gaussian",
        return_intermediates: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        """Convenience wrapper around :meth:`generate_and_process`."""
        config = {"type": stimulus_type, **kwargs}
        return self.generate_and_process(
            [config], return_intermediates=return_intermediates
        )

    def reset_filter_states(self) -> None:
        """Reset the internal state of the SA/RA filters.

        Call this before re-running the same pipeline instance with independent
        sequences to avoid temporal carryover, mirroring the behaviour used in
        notebooks and documentation demos.
        """
        # Use public clear_state() instead of touching private attrs
        # (resolves ReviewFinding#M6)
        self.filters.sa_filter.clear_state()
        self.filters.ra_filter.clear_state()

    def get_neuron_counts(self) -> Dict[str, int]:
        """Return the number of SA and RA neurons encoded in the pipeline."""
        return {
            "sa_neurons": self.sa_innervation.num_neurons,
            "ra_neurons": self.ra_innervation.num_neurons,
        }

    def get_innervation_info(self) -> Dict[str, Any]:
        """Expose diagnostic statistics about the innervation tensors."""
        return {
            "sa_density": self.sa_innervation.get_connection_density(),
            "ra_density": self.ra_innervation.get_connection_density(),
            "sa_connections_per_neuron": self.sa_innervation.get_weights_per_neuron(),
            "ra_connections_per_neuron": self.ra_innervation.get_weights_per_neuron(),
        }

    def visualize_neuron_receptive_field(
        self,
        neuron_type: str,
        neuron_idx: int,
    ):
        """Return a neuron's receptive field for plotting/debugging."""
        if neuron_type.upper() == "SA":
            return self.sa_innervation.visualize_neuron_connections(neuron_idx)
        elif neuron_type.upper() == "RA":
            return self.ra_innervation.visualize_neuron_connections(neuron_idx)
        else:
            raise ValueError(f"Unknown neuron_type: {neuron_type}")

    def to_device(self, device: str) -> nn.Module:
        """Move the pipeline and all child modules to ``device``."""
        self.device = device
        self.grid_manager.to_device(device)
        self.stimulus_generator.to_device(device)
        self.sa_innervation.to_device(device)
        self.ra_innervation.to_device(device)
        return self.to(device)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Return a summary covering grid, neuron, and device information."""
        grid_props = self.grid_manager.get_grid_properties()
        neuron_counts = self.get_neuron_counts()
        innervation_info = self.get_innervation_info()

        return {
            "grid": grid_props,
            "neurons": neuron_counts,
            "innervation": innervation_info,
            "device": self.device,
        }


def create_standard_pipeline(
    grid_size: Optional[int] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    config_path: str = "sensoryforge/config/default_config.yml",
    overrides: Optional[Dict[str, Any]] = None,
):
    """Factory for the default pipeline configuration used in experiments."""
    pipeline_overrides: Dict[str, Any] = {}

    if grid_size is not None or device is not None or seed is not None:
        pipeline_overrides.setdefault("pipeline", {})
        if grid_size is not None:
            pipeline_overrides["pipeline"]["grid_size"] = grid_size
        if device is not None:
            pipeline_overrides["pipeline"]["device"] = device
        if seed is not None:
            pipeline_overrides["pipeline"]["seed"] = seed

    merged_overrides: Dict[str, Any] = {}
    for patch in (overrides or {}, pipeline_overrides):
        if patch:
            merged_overrides = _deep_update(merged_overrides, patch)

    return TactileEncodingPipelineTorch(
        config_path=config_path,
        overrides=merged_overrides if merged_overrides else None,
    )


def create_small_pipeline(
    grid_size: int = 40,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    config_path: str = "sensoryforge/config/default_config.yml",
    overrides: Optional[Dict[str, Any]] = None,
):
    """Factory for a reduced pipeline footprint convenient in tests/visuals."""
    pipeline_overrides = {
        "pipeline": {
            "grid_size": grid_size,
        }
    }
    if device is not None:
        pipeline_overrides["pipeline"]["device"] = device
    if seed is not None:
        pipeline_overrides["pipeline"]["seed"] = seed

    merged_overrides: Dict[str, Any] = _deep_update(
        pipeline_overrides,
        overrides or {},
    )

    return TactileEncodingPipelineTorch(
        config_path=config_path,
        overrides=merged_overrides,
    )
