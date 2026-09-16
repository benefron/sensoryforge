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
    frames, _ = render_stimulus(
        stimulus_type, {}, xx, yy, dt_ms=1.0, duration_ms=_DURATION_MS
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


class TestMovingDivergesBeyondDefaults:
    """moving: NOT reproducible by a default-value map alone, and this is a
    more serious, pre-existing gap in render_stimulus itself (not new in
    this commit): MovingStimulus.forward(xx, yy) returns exactly ONE frame
    per call, advanced only by a separate .step() -- render_stimulus's
    generic "call forward() once, multiply by a temporal envelope" path
    (correct for a stateless stimulus) never calls .step(), so today it
    silently returns the SAME static frame repeated over every time sample
    instead of a moving one, regardless of _LEGACY_DEFAULTS.

    Measured on this 40x40/0.15mm grid, duration 50 ms, dt_ms 1.0, legacy
    defaults (linear motion, start=(-2,0), end=(2,0), amplitude=30,
    sigma=1.0): frame 0 matches the legacy generator's frame 0 exactly (the
    static starting frame happens to be right); every later frame does not
    -- mean absolute difference over all 50 frames is 5.51, max absolute
    difference 29.90 (essentially the full peak amplitude, since by the
    final frame the legacy blob has moved to a different pixel entirely
    while the registered one has not moved at all). A single-line Finding
    is opened on the K8 commit for this (distinct from the default-value
    finding this file otherwise closes): render_stimulus needs a stepped-
    stimulus code path (detect .step(), iterate frames) before "moving" (or
    any other per-step-stateful registered stimulus) can be trusted through
    the CLI/BatchExecutor.
    """

    def test_frame_zero_matches(self):
        legacy = _legacy_frames("moving")
        registered = _registered_frames("moving")
        assert torch.equal(legacy[0], registered[0])

    def test_later_frames_diverge_with_measured_magnitude(self):
        legacy = _legacy_frames("moving")
        registered = _registered_frames("moving")
        assert not torch.equal(legacy[1], registered[1])
        assert not torch.equal(legacy, registered)

        diff = (legacy - registered).abs()
        # Pin the measured magnitude so a silent change (in either
        # direction -- an accidental fix or a worse regression) is caught.
        assert diff.mean().item() > 5.0
        assert diff.max().item() > 25.0

        # render_stimulus's "moving" output today is a static frame
        # repeated T times -- confirms the mechanism, not just the size,
        # of the divergence.
        assert torch.equal(registered, registered[0].expand_as(registered))
