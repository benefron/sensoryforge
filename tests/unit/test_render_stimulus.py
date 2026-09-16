"""Tests for ``render_stimulus`` (Phase 2, Wave K, K1, F-052).

Before this module, ``GeneralizedTactileEncodingPipeline.generate_stimulus``
dispatched through a hard-coded if/elif chain that knew nine names;
``composite``, ``edge_grating`` and ``gabor`` were registered in
``STIMULUS_REGISTRY`` but unreachable from a config file, and a plugin
stimulus could be registered but never executed. These tests use only
built-ins already present before Wave K (``edge_grating``, ``gabor``,
``static``) plus a stimulus registered at test time, so they are independent
of the K2 ported stimuli (covered separately in ``test_tactile_stimuli.py``).
"""

from __future__ import annotations

import pytest
import torch

from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.register_components import register_all
from sensoryforge.registry import STIMULUS_REGISTRY
from sensoryforge.stimuli.base import BaseStimulus
from sensoryforge.stimuli.render import render_stimulus

register_all()


@pytest.fixture(scope="module")
def coords8():
    grid = ReceptorGrid(grid_size=(8, 8), spacing=0.15)
    return grid.get_coordinates()


class TestRegisteredDispatch:
    def test_edge_grating_runs_through_render_stimulus(self, coords8):
        xx, yy = coords8
        frames, time_ms = render_stimulus(
            "edge_grating", {}, xx, yy, dt_ms=1.0, duration_ms=10.0
        )
        assert frames.dim() == 3
        assert frames.shape[0] == time_ms.numel()
        assert tuple(frames.shape[1:]) == tuple(xx.shape)

    def test_gabor_runs_through_render_stimulus(self, coords8):
        xx, yy = coords8
        frames, time_ms = render_stimulus(
            "gabor", {}, xx, yy, dt_ms=1.0, duration_ms=10.0
        )
        assert frames.shape[0] == time_ms.numel()

    def test_single_frame_stimulus_is_expanded_with_envelope(self, coords8):
        """gaussian returns [H, W]; render_stimulus expands it to [T, H, W]
        using the temporal envelope. With an explicit plateau covering the
        whole duration and no ramps, every frame after t=0 holds at the same
        (full-amplitude) value."""
        xx, yy = coords8
        frames, time_ms = render_stimulus(
            "gaussian",
            {"amplitude": 2.0, "sigma": 0.3, "plateau_ms": 100.0},
            xx,
            yy,
            dt_ms=1.0,
            duration_ms=5.0,
        )
        assert frames.shape[0] == time_ms.numel()
        assert torch.allclose(frames[1], frames[-1])
        assert frames[1].max() > 0.0

    def test_unknown_stimulus_raises_value_error(self, coords8):
        xx, yy = coords8
        with pytest.raises(ValueError, match="totally_bogus_stimulus_name"):
            render_stimulus("totally_bogus_stimulus_name", {}, xx, yy, dt_ms=1.0)

    def test_plugin_stimulus_registered_only_at_test_time_runs(self, coords8):
        """The F-052 proof: a stimulus registered by a plugin (here, a
        two-line subclass registered only in this test) is runnable through
        render_stimulus without any change to render.py or the CLI."""

        class _PluginProbeStimulus(BaseStimulus):
            def forward(self, xx, yy):
                return torch.full_like(xx, 3.0)

            def reset_state(self):
                pass

        STIMULUS_REGISTRY.register("_plugin_probe_stimulus", _PluginProbeStimulus)
        xx, yy = coords8
        frames, time_ms = render_stimulus(
            "_plugin_probe_stimulus", {}, xx, yy, dt_ms=1.0, duration_ms=3.0
        )
        assert frames.shape[1:] == tuple(xx.shape)
        assert frames.shape[0] == time_ms.numel()


class TestLegacyFallback:
    @pytest.mark.parametrize("name", ["trapezoidal", "step", "ramp"])
    def test_legacy_names_still_reachable(self, coords8, name):
        xx, yy = coords8
        frames, time_ms = render_stimulus(name, {}, xx, yy, dt_ms=0.1, duration_ms=20.0)
        assert frames.dim() == 3
        assert frames.shape[0] == time_ms.numel()


class _FixedLengthStimulus(BaseStimulus):
    """A stimulus that always returns a fixed-length [T, H, W] sequence, used
    to test render_stimulus's truncate/pad behaviour independently of any
    real sequence stimulus's own parameters."""

    T = 20

    def forward(self, xx, yy):
        return torch.ones(self.T, *xx.shape, dtype=torch.float32)

    def reset_state(self):
        pass


class TestSequenceStimulusTruncationAndPadding:
    def test_sequence_stimulus_truncated_to_duration(self, coords8):
        xx, yy = coords8
        STIMULUS_REGISTRY.register("_fixed_length_probe", _FixedLengthStimulus)
        # dt_ms=1.0, duration_ms=5.0 -> round(5.0/1.0) = 5 samples (K9: a
        # duration, not a last-sample field), well under the stimulus's
        # native 20 -> truncated, not resampled.
        frames, time_ms = render_stimulus(
            "_fixed_length_probe", {}, xx, yy, dt_ms=1.0, duration_ms=5.0
        )
        assert frames.shape[0] == time_ms.numel() == 5
        assert torch.all(frames == 1.0)

    def test_sequence_stimulus_zero_padded_when_duration_longer(self, coords8):
        xx, yy = coords8
        STIMULUS_REGISTRY.register("_fixed_length_probe", _FixedLengthStimulus)
        frames, time_ms = render_stimulus(
            "_fixed_length_probe", {}, xx, yy, dt_ms=1.0, duration_ms=30.0
        )
        assert frames.shape[0] == time_ms.numel() == 30
        assert torch.all(frames[:20] == 1.0)
        assert torch.all(frames[20:] == 0.0)

    def test_sequence_stimulus_no_duration_keeps_native_length(self, coords8):
        xx, yy = coords8
        STIMULUS_REGISTRY.register("_fixed_length_probe", _FixedLengthStimulus)
        frames, time_ms = render_stimulus("_fixed_length_probe", {}, xx, yy, dt_ms=1.0)
        assert frames.shape[0] == 20
        assert time_ms.numel() == 20


class TestChannelAxis:
    """GridConfig.channels / StimulusConfig.channel (Phase 2, Wave L; added to
    render_stimulus after K6, docs/concepts/units_and_shapes.md)."""

    def test_none_or_single_channel_keeps_three_d_shape(self, coords8):
        xx, yy = coords8
        frames, _ = render_stimulus(
            "gaussian", {"amplitude": 1.0}, xx, yy, dt_ms=1.0, duration_ms=3.0
        )
        assert frames.dim() == 3

        frames, _ = render_stimulus(
            "gaussian",
            {"amplitude": 1.0},
            xx,
            yy,
            dt_ms=1.0,
            duration_ms=3.0,
            channels=["value"],
        )
        assert frames.dim() == 3

    def test_two_channels_fills_only_the_named_plane(self, coords8):
        xx, yy = coords8
        frames, time_ms = render_stimulus(
            "gaussian",
            {"amplitude": 5.0, "sigma": 0.3, "channel": "R"},
            xx,
            yy,
            dt_ms=1.0,
            duration_ms=3.0,
            channels=["R", "G"],
        )
        assert tuple(frames.shape) == (time_ms.numel(), 2, *xx.shape)
        assert frames[:, 0].abs().sum() > 0  # R plane has the stimulus
        assert torch.all(frames[:, 1] == 0.0)  # G plane stays zero

    def test_two_channels_default_targets_first_channel(self, coords8):
        xx, yy = coords8
        frames, _ = render_stimulus(
            "gaussian",
            {"amplitude": 5.0, "sigma": 0.3},  # no "channel" key
            xx,
            yy,
            dt_ms=1.0,
            duration_ms=3.0,
            channels=["R", "G"],
        )
        assert frames[:, 0].abs().sum() > 0
        assert torch.all(frames[:, 1] == 0.0)

    def test_unknown_channel_name_raises(self, coords8):
        xx, yy = coords8
        with pytest.raises(ValueError, match="not one of"):
            render_stimulus(
                "gaussian",
                {"amplitude": 1.0, "channel": "B"},
                xx,
                yy,
                dt_ms=1.0,
                duration_ms=3.0,
                channels=["R", "G"],
            )

    def test_two_calls_with_different_channels_compose_by_summing(self, coords8):
        """The caller's job (per docs/concepts/units_and_shapes.md): summing
        two render_stimulus calls, each targeting a different channel,
        composes into one multi-channel tensor since each call's
        non-target planes are zero."""
        xx, yy = coords8
        frames_r, _ = render_stimulus(
            "gaussian",
            {"amplitude": 5.0, "sigma": 0.3, "channel": "R"},
            xx,
            yy,
            dt_ms=1.0,
            duration_ms=3.0,
            channels=["R", "G"],
        )
        frames_g, _ = render_stimulus(
            "gaussian",
            {"amplitude": 7.0, "sigma": 0.3, "center_x": 0.1, "channel": "G"},
            xx,
            yy,
            dt_ms=1.0,
            duration_ms=3.0,
            channels=["R", "G"],
        )
        composed = frames_r + frames_g
        assert torch.equal(composed[:, 0], frames_r[:, 0])
        assert torch.equal(composed[:, 1], frames_g[:, 1])
        assert composed[:, 0].abs().sum() > 0
        assert composed[:, 1].abs().sum() > 0
