"""
TactilePipelineVisualizer: Interactive Plotly visualization for tactile
encoding pipeline outputs.

- Modular, extensible, and notebook-friendly
- Supports raster plots, PSTH, heatmaps, innervation, and aggregation
"""
from typing import Optional, Tuple

import plotly.graph_objects as go
import plotly.subplots as psub
import numpy as np
import torch


class TactilePipelineVisualizer:
    """
    Visualization module for tactile encoding pipeline outputs.
    All methods return Plotly Figure objects for notebook integration.
    """

    def __init__(self, pipeline, results):
        """
        Args:
            pipeline: GeneralizedTactileEncodingPipeline instance
            results: Output dict from
                ``pipeline.forward(..., return_intermediates=True)``
        """
        self.pipeline = pipeline
        self.results = results
        # Use neuron types based on spike keys in results
        self.neuron_types = []
        for k in results.keys():
            if k.endswith("_spikes"):
                base = k.split("_")[0]
                if base not in self.neuron_types:
                    self.neuron_types.append(base)
        self.grid_shape = self._get_grid_shape()
        self.time_array = self._get_time_array()

    def _get_grid_shape(self):
        """Infer the underlying stimulus grid shape from cached results."""
        # Prefer the stimulus tensor when available, since it captures
        # any dynamic grid overrides emitted at runtime.
        stim = self.results.get("stimulus_sequence", None)
        if stim is not None:
            if isinstance(stim, torch.Tensor):
                return stim.shape[-2:]
            elif isinstance(stim, np.ndarray):
                return stim.shape[-2:]
        # Fall back to pipeline metadata when the stimulus was not persisted.
        return getattr(self.pipeline.grid_manager, "grid_size", (10, 10))

    def _get_time_array(self):
        """Return the per-timestep timebase (ms) associated with results."""
        # Use cached time array when available to preserve custom sampling.
        arr = self.results.get("time_array", None)
        if arr is not None:
            return arr
        # Fall back to uniform sampling derived from temporal config.
        dt = self.pipeline.config["temporal"]["dt"]
        default_stimulus = torch.zeros(1, 10, *self.grid_shape)
        n_timesteps = self.results.get("stimulus_sequence", default_stimulus).shape[1]
        return np.arange(n_timesteps) * dt

    def plot_raster_and_stimulus(self, bin_size=10):
        """
        Build a publication-ready summary figure combining spike rasters,
        PSTHs, and the stimulus profile.

        Args:
            bin_size: Number of samples per PSTH bin when downsampling spikes.

        Returns:
            plotly.graph_objects.Figure: Five-row subplot combining rasters,
                PSTHs, and the stimulus trace.
        """
        psth_colors = ["red", "blue", "green", "orange", "purple", "black"]
        subplot_titles = [f"{ntype.upper()} Raster" for ntype in self.neuron_types] + [
            "Overlayed PSTH",
            "Overlayed Stimulus",
        ]
        fig = psub.make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            row_heights=[0.18, 0.18, 0.18, 0.23, 0.23],
        )
        n_time = self.time_array.shape[0]
        raster_color = "black"
        # Raster plots (rows 1-3)
        for row, ntype in enumerate(self.neuron_types, start=1):
            spikes = self.results.get(f"{ntype}_spikes", None)
            if spikes is not None:
                if isinstance(spikes, torch.Tensor):
                    spikes = spikes.cpu().numpy()
                if spikes.ndim == 3 and spikes.shape[0] == 1:
                    spikes = spikes[0].T
                elif spikes.ndim == 3:
                    spikes = spikes.transpose(2, 1, 0).squeeze(-1)
                if spikes.ndim == 1:
                    spikes = spikes[np.newaxis, :]
                # Emit a scatter trace per neuron so inter-spike gaps stay
                # visible.
                for neuron_idx in range(spikes.shape[0]):
                    spike_mask = spikes[neuron_idx][:n_time] > 0
                    spike_times = self.time_array[spike_mask]
                    fig.add_trace(
                        go.Scatter(
                            x=spike_times,
                            y=[neuron_idx] * len(spike_times),
                            mode="markers",
                            marker=dict(size=3, color=raster_color),
                            name=f"{ntype} Raster",
                            showlegend=False,
                        ),
                        row=row,
                        col=1,
                    )
        # Overlayed PSTH (row 4, line plots)
        for idx, ntype in enumerate(self.neuron_types):
            spikes = self.results.get(f"{ntype}_spikes", None)
            if spikes is not None:
                if isinstance(spikes, torch.Tensor):
                    spikes = spikes.cpu().numpy()
                if spikes.ndim == 3 and spikes.shape[0] == 1:
                    spikes = spikes[0].T
                elif spikes.ndim == 3:
                    spikes = spikes.transpose(2, 1, 0).squeeze(-1)
                if spikes.ndim == 1:
                    spikes = spikes[np.newaxis, :]
                psth = np.sum(spikes[:, :n_time], axis=0)
                bins = np.arange(0, len(psth), bin_size)
                binned = [np.sum(psth[b : b + bin_size]) for b in bins]
                fig.add_trace(
                    go.Scatter(
                        x=self.time_array[::bin_size],
                        y=binned,
                        mode="lines",
                        line=dict(
                            color=psth_colors[idx % len(psth_colors)],
                            width=3,
                        ),
                        name=f"{ntype} PSTH",
                        showlegend=True,
                    ),
                    row=4,
                    col=1,
                )
        # Overlayed stimulus (row 5)
        if "temporal_profile" in self.results:
            fig.add_trace(
                go.Scatter(
                    y=self.results["temporal_profile"],
                    x=self.time_array,
                    mode="lines",
                    name="Stimulus",
                    showlegend=True,
                ),
                row=5,
                col=1,
            )
        fig.update_layout(
            height=1200,
            width=900,
            title="Raster, PSTH, and Stimulus (Overlayed)",
        )
        return fig

    def plot_innervation_heatmap(self, neuron_type="sa"):
        """
        Visualise the aggregate innervation footprint for a single population.

        Args:
            neuron_type: Population key (`'sa'`, `'ra'`, or `'sa2'`).

        Returns:
            plotly.graph_objects.Figure containing a heatmap of summed
                innervation weights.
        """
        inn = self.results.get(f"{neuron_type}_innervation", None)
        if inn is None:
            inn = getattr(self.pipeline, f"{neuron_type}_innervation", None)
        if inn is None:
            raise ValueError(f"Innervation for {neuron_type} not found.")
        # Get weights tensor from module or dict
        if isinstance(inn, dict) and "weights" in inn:
            weights = inn["weights"]
        elif hasattr(inn, "innervation_weights"):
            weights = inn.innervation_weights
        else:
            weights = inn
        # Convert to numpy if tensor
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        # Aggregate over neurons if needed
        heat = np.sum(weights, axis=0) if weights.ndim == 3 else weights
        fig = go.Figure(go.Heatmap(z=heat, coloraxis="coloraxis"))
        fig.update_layout(
            title=f"{neuron_type.upper()} Innervation Heatmap",
            coloraxis={"colorscale": "Viridis"},
        )
        return fig

    def plot_stimulus_heatmap(self, time_idx=0):
        """
        Render the spatial stimulus snapshot at the requested time step.

        Args:
            time_idx: Index into the temporal axis of `stimulus_sequence`.

        Returns:
            plotly.graph_objects.Figure showing the stimulus amplitude across
                the grid.
        """
        stim = self.results.get("stimulus_sequence", None)
        if stim is None:
            raise ValueError("stimulus_sequence not found in results.")
        if isinstance(stim, torch.Tensor):
            stim = stim.cpu().numpy()
        stim_frame = stim[0, time_idx]
        fig = go.Figure(go.Heatmap(z=stim_frame, coloraxis="coloraxis"))
        fig.update_layout(
            title=(f"Stimulus Heatmap (t={self.time_array[time_idx]:.1f} ms)"),
            coloraxis={"colorscale": "Viridis"},
        )
        return fig

    def plot_sensory_response_heatmap(
        self,
        neuron_type: str = "sa",
        time_idx: int = 0,
        filtered: bool = False,
    ):
        """
        Display per-neuron drive values mapped back onto the tactile grid.

        Args:
            neuron_type: Population key (`'sa'`, `'ra'`, `'sa2'`).
            time_idx: Temporal index to visualise.
            filtered: When True, use filtered inputs instead of raw drives.

        Returns:
            plotly.graph_objects.Figure heatmap using millimetre axes when
                available.
        """
        keys_to_try: list[str]
        if filtered:
            keys_to_try = [
                f"{neuron_type}_filtered",
                f"{neuron_type}_filtered_noisy",
            ]
        else:
            keys_to_try = [f"{neuron_type}_inputs"]
        resp = None
        for key in keys_to_try:
            resp = self.results.get(key, None)
            if resp is not None:
                break
        if resp is None:
            raise ValueError(
                f"No sensory response found for {neuron_type}." f" Tried: {keys_to_try}"
            )
        if isinstance(resp, torch.Tensor):
            resp = resp.cpu().numpy()
        # Shape: (batch, time, neurons)
        if resp.ndim == 3:
            resp_frame = resp[0, time_idx]
        elif resp.ndim == 2:
            resp_frame = resp[time_idx]
        else:
            raise ValueError(f"Unexpected response shape: {resp.shape}")
        # Infer grid shape from response length if needed
        grid_shape = self.grid_shape
        if resp_frame.size != grid_shape[0] * grid_shape[1]:
            # Try to infer a square grid
            side = int(np.sqrt(resp_frame.size))
            if side * side == resp_frame.size:
                grid_shape = (side, side)
            else:
                raise ValueError(
                    "Cannot infer grid shape for response of length "
                    f"{resp_frame.size}"
                )
        resp_frame_2d = resp_frame.reshape(grid_shape)
        # Get physical coordinates if available
        x_coords = getattr(
            self.pipeline.grid_manager, "x_coords_mm", np.arange(grid_shape[1])
        )
        y_coords = getattr(
            self.pipeline.grid_manager, "y_coords_mm", np.arange(grid_shape[0])
        )
        fig = go.Figure(
            go.Heatmap(
                z=resp_frame_2d,
                x=x_coords,
                y=y_coords,
                coloraxis="coloraxis",
            )
        )
        title = (
            f"{neuron_type.upper()} {'Filtered' if filtered else 'Raw'} "
            f"Response (t={self.time_array[time_idx]:.1f} ms)"
        )
        fig.update_layout(
            title=title,
            coloraxis={"colorscale": "Viridis"},
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
        )
        return fig

    def plot_binned_spike_rate(
        self,
        neuron_type: str = "sa",
        bin_size: int = 10,
        time_idx: Optional[int] = None,
        time_ms: Optional[float] = None,
        time_range_ms: Optional[Tuple[float, float]] = None,
    ):
        """
        Aggregate spikes within a window and project firing rates spatially.

        Args:
            neuron_type: Population key whose spikes should be binned.
            bin_size: Samples per bin when no explicit window is provided.
            time_idx: Optional start index supplied in samples.
            time_ms: Optional start index supplied in milliseconds.
            time_range_ms: Optional `[start, end]` window in milliseconds.

        Returns:
            plotly.graph_objects.Figure heatmap of firing rate (spikes/sec).
        """
        spikes = self.results.get(f"{neuron_type}_spikes", None)
        if spikes is None:
            raise ValueError(f"Spikes for {neuron_type} not found.")
        if isinstance(spikes, torch.Tensor):
            spikes = spikes.cpu().numpy()
        # Shape: (neurons, time)
        if spikes.ndim == 3 and spikes.shape[0] == 1:
            spikes = spikes[0].T
        elif spikes.ndim == 3:
            spikes = spikes.transpose(2, 1, 0).squeeze(-1)
        if spikes.ndim == 1:
            spikes = spikes[np.newaxis, :]
        n_neurons, n_time = spikes.shape
        # Determine time indices
        dt = self.pipeline.config["temporal"]["dt"]
        if time_ms is not None:
            start_idx = int(time_ms / dt)
            end_idx = start_idx + bin_size
        elif time_range_ms is not None:
            start_idx = int(time_range_ms[0] / dt)
            end_idx = int(time_range_ms[1] / dt)
        elif time_idx is not None:
            start_idx = time_idx
            end_idx = start_idx + bin_size
        else:
            start_idx = 0
            end_idx = bin_size
        start_idx = max(0, min(start_idx, n_time - 1))
        end_idx = max(start_idx + 1, min(end_idx, n_time))
        # Compute binned spike rate per neuron
        window_width = max(end_idx - start_idx, 1)
        fr = np.sum(spikes[:, start_idx:end_idx], axis=1)
        fr = fr / (window_width * dt / 1000.0)  # spikes/sec
        # Map to grid
        grid_shape = self.grid_shape
        if n_neurons != grid_shape[0] * grid_shape[1]:
            # Try to reshape, else pad/crop
            if fr.size >= grid_shape[0] * grid_shape[1]:
                fr_grid = fr[: grid_shape[0] * grid_shape[1]].reshape(grid_shape)
            else:
                fr_grid = np.zeros(grid_shape)
                fr_grid.flat[: fr.size] = fr
        else:
            fr_grid = fr.reshape(grid_shape)
        # Use grid coordinates in mm if available
        x_coords = getattr(
            self.pipeline.grid_manager, "x_coords_mm", np.arange(grid_shape[1])
        )
        y_coords = getattr(
            self.pipeline.grid_manager, "y_coords_mm", np.arange(grid_shape[0])
        )
        fig = go.Figure(
            go.Heatmap(
                z=fr_grid,
                x=x_coords,
                y=y_coords,
                coloraxis="coloraxis",
            )
        )
        title = (
            f"{neuron_type.upper()} Binned Spike Rate Heatmap "
            f"({start_idx * dt:.1f}-{end_idx * dt:.1f} ms)"
        )
        fig.update_layout(title=title, coloraxis={"colorscale": "Viridis"})
        return fig

    def plot_all_summary(self):
        """
        Combine the most common diagnostic plots into a single 2x2 grid.

        Returns:
            plotly.graph_objects.Figure with stimulus, innervation, raster,
                and sensory heatmaps.
        """
        fig = psub.make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Stimulus Heatmap",
                "Innervation Heatmap",
                "Raster + PSTH",
                "Sensory Response Heatmap",
            ],
        )
        # Stimulus
        stim_fig = self.plot_stimulus_heatmap(time_idx=0)
        for trace in stim_fig.data:
            fig.add_trace(trace, row=1, col=1)
        # Extract base neuron type (e.g., 'sa' or 'ra') from the first entry.
        base_type = self.neuron_types[0]
        # Innervation
        inn_fig = self.plot_innervation_heatmap(neuron_type=base_type)
        for trace in inn_fig.data:
            fig.add_trace(trace, row=1, col=2)
        # Raster + PSTH
        raster_fig = self.plot_raster_and_stimulus(bin_size=10)
        for trace in raster_fig.data:
            fig.add_trace(trace, row=2, col=1)
        # Sensory response
        resp_fig = self.plot_sensory_response_heatmap(
            neuron_type=base_type,
            time_idx=0,
            filtered=True,
        )
        for trace in resp_fig.data:
            fig.add_trace(trace, row=2, col=2)
        fig.update_layout(
            height=900,
            width=1200,
            title="Tactile Pipeline Summary",
        )
        return fig

    def plot_filtered_response_overlay(self, neuron_type="sa", top_n=5):
        """
        Plot raw and filtered responses for the most active neurons.

        Args:
            neuron_type: `'sa'`, `'ra'`, or `'sa2'` population key.
            top_n: Number of high-energy neurons to overlay.

        Returns:
            plotly.graph_objects.Figure with paired raw/filtered traces.
        """
        # Get raw and filtered responses
        raw = self.results.get(f"{neuron_type}_inputs", None)
        filtered = self.results.get(f"{neuron_type}_filtered", None)
        if raw is None or filtered is None:
            raise ValueError(f"Raw or filtered response not found for {neuron_type}.")
        if isinstance(raw, torch.Tensor):
            raw = raw.cpu().numpy()
        if isinstance(filtered, torch.Tensor):
            filtered = filtered.cpu().numpy()
        # Shape: (batch, time, neurons)
        raw = raw[0] if raw.ndim == 3 else raw
        filtered = filtered[0] if filtered.ndim == 3 else filtered
        # Sum filtered response over time for each neuron
        neuron_sums = filtered.sum(axis=0)
        top_indices = np.argsort(neuron_sums)[-top_n:][::-1]
        fig = go.Figure()
        for idx in top_indices:
            # Alternate dot/solid styles to emphasise filtering effects.
            fig.add_trace(
                go.Scatter(
                    x=self.time_array,
                    y=raw[:, idx],
                    mode="lines",
                    name=f"Neuron {idx} Raw",
                    line=dict(dash="dot", color="gray"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.time_array,
                    y=filtered[:, idx],
                    mode="lines",
                    name=f"Neuron {idx} Filtered",
                    line=dict(color=None),
                )
            )
        fig.update_layout(
            title=(
                f"{neuron_type.upper()} Most Active Neurons: Raw vs "
                "Filtered Response"
            ),
            xaxis_title="Time (ms)",
            yaxis_title="Response Amplitude",
            legend_title="Neuron/Type",
            height=500,
            width=900,
        )
        return fig
