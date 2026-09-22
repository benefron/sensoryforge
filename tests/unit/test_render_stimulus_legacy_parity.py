"""render_stimulus reproduces the legacy pipeline's output for every name
both know, with no optional parameters supplied (Phase 2, Wave K, K8).

Before this fix, K1 routed five names that also exist in
``GeneralizedTactileEncodingPipeline``'s legacy dispatch chain --
``gaussian``, ``moving``, ``repeated_pattern``, ``texture``, ``timeline`` --
through the registered component's own constructor defaults, which are
independent values chosen for direct use of that class, not the legacy
generator's defaults. A config that omitted an optional parameter (very
common -- ``sigma`` in particular) silently got a different stimulus.
Measured for "gaussian" on a 40x40 grid at 0.15 mm, amplitude 10, no sigma:
32 receptors above 10% of peak and frame energy 111.7 with the registered
class's own default (sigma 0.2 mm), versus 648 receptors and energy 2777.6
with the legacy generator's default (sigma 1.0 mm) -- 20x narrower and 25x
weaker, silently, in every run that doesn't set sigma explicitly.

``sensoryforge.stimuli.render._LEGACY_DEFAULTS`` fixes this for the three
names where it is possible without deeper surgery (see below). For the
other two, this file documents the divergence with numbers and opens a
Finding rather than leaving it silent, per the Wave K report.
"""

from __future__ import annotations

import torch

from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.register_components import register_all
from sensoryforge.stimuli.render import render_stimulus

register_all()

_GRID_KW = dict(grid_size=(40, 40), spacing=0.15)
_LEGACY_CONFIG = {
    "pipeline": {
        "device": "cpu",
        "grid_size": (40, 40),
        "spacing": 0.15,
        "center": [0.0, 0.0],
    },
    "neurons": {"dt": 1.0},
    "temporal": {"dt": 1.0},
    "simulation": {"dt_ms": 1.0},
}
_DURATION_MS = 50.0


def _legacy_frames(stimulus_type: str) -> torch.Tensor:
    pipeline = GeneralizedTactileEncodingPipeline.from_config(_LEGACY_CONFIG)
    frames, _, _ = pipeline.generate_stimulus(
        stimulus_type=stimulus_type, duration=_DURATION_MS
    )
    return frames.squeeze(0)


def _registered_frames(stimulus_type: str) -> torch.Tensor:
    xx, yy = ReceptorGrid(**_GRID_KW).get_coordinates()
    # The legacy generator switched a still stimulus on as a step and held
    # it; the renderer now ramps it in and out by default. The spatial
    # defaults are what this parity is about, so compare under the legacy
    # step envelope.
    step = {"ramp_up_ms": 0.0, "ramp_down_ms": 0.0}
    frames, _ = render_stimulus(
        stimulus_type, step, xx, yy, dt_ms=1.0, duration_ms=_DURATION_MS
    )
    return frames


class TestBitIdenticalWithLegacyDefaults:
    """gaussian, texture, repeated_pattern: fully reproducible -- each is a
    single static [H, W] frame (or, for repeated_pattern, a sum of shifted
    copies of one), so the legacy generator's default-value gap is the
    *only* difference and _LEGACY_DEFAULTS closes it completely."""

    def test_gaussian_matches_legacy_with_no_params(self):
        assert torch.equal(_legacy_frames("gaussian"), _registered_frames("gaussian"))

    def test_texture_matches_legacy_with_no_params(self):
        assert torch.equal(_legacy_frames("texture"), _registered_frames("texture"))

    def test_repeated_pattern_matches_legacy_with_no_params(self):
        assert torch.equal(
            _legacy_frames("repeated_pattern"), _registered_frames("repeated_pattern")
        )


class TestTimelineFailsIdenticallyBothPaths:
    """timeline: not a default-value gap. TimelineStimulus.__init__ raises
    ValueError for an empty sub_stimuli list, and the legacy generator's own
    default is also an empty list (``params.get("sub_stimuli", [])``) -- so
    "no optional parameters" already raised in the legacy path before Wave
    K, and still does. Confirmed here rather than left as an assumption."""

    def test_legacy_timeline_with_no_params_raises(self):
        pipeline = GeneralizedTactileEncodingPipeline.from_config(_LEGACY_CONFIG)
        try:
            pipeline.generate_stimulus(stimulus_type="timeline", duration=_DURATION_MS)
        except ValueError as exc:
            assert "sub-stimulus" in str(exc)
        else:
            raise AssertionError("expected the legacy path to raise ValueError too")

    def test_registered_timeline_with_no_params_raises(self):
        xx, yy = ReceptorGrid(**_GRID_KW).get_coordinates()
        try:
            render_stimulus("timeline", {}, xx, yy, dt_ms=1.0, duration_ms=_DURATION_MS)
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("expected render_stimulus to raise too")


class TestMovingMatchesTheLegacyGenerator:
    """moving: fixed (F-057), and this class records what changed.

    This used to be ``TestMovingDivergesBeyondDefaults``, pinning the
    defect rather than the behaviour. The registered ``MovingStimulus``
    (``stimuli/builder.py``) returns the frame at its *current* step and
    advances on ``step()``, so ``render_stimulus``'s generic
    "call forward() once, multiply by a temporal envelope" path -- correct
    for a stateless stimulus -- returned the same static frame repeated
    over every time sample. Measured then, on this grid at 50 ms: frame 0
    matched the legacy generator exactly, mean absolute difference over all
    50 frames was 5.51 and the maximum 29.90, essentially the full peak
    amplitude.

    Two things fixed it. ``render_stimulus`` now drives a stepped stimulus
    frame by frame, and a caller using the legacy flat vocabulary
    (amplitude/sigma/start/end) is routed to the legacy generator, because
    the registered class takes a nested ``base_stimulus``/``motion_params``
    that a default-value map cannot translate into. See
    ``tests/unit/test_render_moving.py`` for the motion tests themselves.
    """

    def test_frames_match_the_legacy_generator(self):
        legacy = _legacy_frames("moving")
        registered = _registered_frames("moving")
        assert legacy.shape == registered.shape
        # Not bit-identical: the legacy render path rebuilds a GridManager
        # from the xx/yy it is handed, so the reconstructed coordinates
        # differ by float32 rounding -- about 1e-5 on values near 30.
        assert torch.allclose(
            legacy, registered, atol=1e-4, rtol=0
        ), f"max abs diff {float((legacy - registered).abs().max()):.3e}"

    def test_the_output_is_no_longer_a_repeated_static_frame(self):
        """The exact shape of the old defect, asserted to be gone."""
        registered = _registered_frames("moving")
        assert not torch.equal(
            registered, registered[0].expand_as(registered)
        ), "moving is still returning one frame repeated over the time axis"
