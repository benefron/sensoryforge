"""A rendered moving stimulus actually moves (F-057).

Wave K routed every registered stimulus name through ``STIMULUS_REGISTRY``.
That exposed a defect for ``moving``: the registered class
(``stimuli/builder.py``'s ``MovingStimulus``, not the same-named class in
``stimuli/moving.py``) returns the frame at its *current* step and advances
on ``step()``. Calling ``forward()`` once and broadcasting the result over
the time axis therefore produced a stimulus that never moved -- 20 identical
frames over a 20 ms run -- with no error anywhere.

The failure mode is what makes this worth pinning: a silent, plausible,
motionless "moving" stimulus feeding every downstream spike count.
"""

import pytest
import torch

from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.core.grid import create_grid_torch
from sensoryforge.stimuli import render as render_mod
from sensoryforge.stimuli.render import render_stimulus

ROWS = COLS = 40
SPACING = 0.15
DT_MS = 1.0

# The component's own parameter vocabulary, as opposed to the legacy
# generator's flat amplitude/sigma/start/end.
COMPONENT_PARAMS = {
    "base_stimulus": {
        "class": "StaticStimulus",
        "stim_type": "gaussian",
        "params": {"amplitude": 10.0, "sigma": 1.0, "center_x": 0.0, "center_y": 0.0},
    },
    "motion_type": "linear",
    "motion_params": {"start": (-2.0, 0.0), "end": (2.0, 0.0)},
}


@pytest.fixture
def coords():
    xx, yy, _, _ = create_grid_torch((ROWS, COLS), SPACING, (0.0, 0.0), "cpu")
    return xx, yy


def _legacy_frames(stimulus_type, duration_ms, **params):
    cfg = {
        "pipeline": {"device": "cpu", "grid_size": ROWS, "spacing": SPACING},
        "neurons": {"sa_neurons": 2, "ra_neurons": 2, "dt": DT_MS},
        "temporal": {"dt": DT_MS},
    }
    pipeline = GeneralizedTactileEncodingPipeline(config_dict=cfg)
    frames, _, _ = pipeline.generate_stimulus(
        stimulus_type=stimulus_type, duration=duration_ms, dt=DT_MS, **params
    )
    return frames[0]


def _motion_extent(frames):
    """How far the field changes between the first and last frame."""
    return float((frames[-1] - frames[0]).abs().max())


class TestItMoves:
    """The regression itself."""

    @pytest.mark.parametrize("duration_ms", [20.0, 50.0, 200.0])
    def test_frames_are_not_all_identical(self, coords, duration_ms):
        xx, yy = coords
        frames, _ = render_stimulus(
            "moving", {"amplitude": 10.0}, xx, yy, dt_ms=DT_MS, duration_ms=duration_ms
        )
        deviations = [
            float((frames[i] - frames[0]).abs().max()) for i in range(len(frames))
        ]
        assert max(deviations) > 1.0, (
            f"all {len(frames)} frames of a moving stimulus are the same to "
            f"within {max(deviations):.3e}; it is not moving"
        )

    def test_the_peak_travels(self, coords):
        """Not just "frames differ" -- the blob's peak has to move."""
        xx, yy = coords
        frames, _ = render_stimulus(
            "moving", {"amplitude": 10.0}, xx, yy, dt_ms=DT_MS, duration_ms=200.0
        )
        first = torch.nonzero(frames[0] == frames[0].max())[0]
        last = torch.nonzero(frames[-1] == frames[-1].max())[0]
        assert not torch.equal(first, last), (
            f"the peak sits at {first.tolist()} in the first frame and the "
            "same pixel in the last"
        )


class TestLegacyVocabularyIsPreserved:
    """A caller using the old flat parameters gets the old frames."""

    @pytest.mark.parametrize("duration_ms", [50.0, 200.0])
    def test_matches_the_legacy_generator(self, coords, duration_ms):
        xx, yy = coords
        new, _ = render_stimulus(
            "moving", {"amplitude": 10.0}, xx, yy, dt_ms=DT_MS, duration_ms=duration_ms
        )
        old = _legacy_frames("moving", duration_ms, amplitude=10.0)
        assert new.shape == old.shape
        # Not bit-identical: the legacy path rebuilds a GridManager from the
        # xx/yy it is handed, and the reconstructed coordinates differ from
        # the originals by float32 rounding. The gap is around 1e-5 on
        # values near 10, roughly one ulp, not a behavioural difference.
        assert torch.allclose(
            new, old, atol=1e-4, rtol=0
        ), f"max abs diff {float((new - old).abs().max()):.3e}"

    def test_routing_is_deliberate_not_accidental(self):
        assert render_mod._prefers_legacy("moving", {"amplitude": 10.0}) is True
        assert render_mod._prefers_legacy("moving", COMPONENT_PARAMS) is False
        assert render_mod._prefers_legacy("gaussian", {}) is False


class TestComponentVocabularyUsesTheComponent:
    """Passing the component's own keys reaches the component, stepped."""

    def test_component_params_move(self, coords):
        xx, yy = coords
        frames, _ = render_stimulus(
            "moving", dict(COMPONENT_PARAMS), xx, yy, dt_ms=DT_MS, duration_ms=50.0
        )
        assert frames.shape == (50, ROWS, COLS)
        assert _motion_extent(frames) > 1.0

    def test_rendering_twice_gives_the_same_answer(self, coords):
        """The stepped path must reset state, or the second call starts late."""
        xx, yy = coords
        first, _ = render_stimulus(
            "moving", dict(COMPONENT_PARAMS), xx, yy, dt_ms=DT_MS, duration_ms=30.0
        )
        second, _ = render_stimulus(
            "moving", dict(COMPONENT_PARAMS), xx, yy, dt_ms=DT_MS, duration_ms=30.0
        )
        assert torch.equal(first, second)

    def test_duration_longer_than_the_trajectory_holds_the_last_position(self, coords):
        xx, yy = coords
        params = dict(COMPONENT_PARAMS)
        params["motion_params"] = {**COMPONENT_PARAMS["motion_params"], "num_steps": 10}
        # A step envelope, so only the position is compared (by default the
        # stimulus now ramps down at the end).
        params.update(ramp_up_ms=0.0, ramp_down_ms=0.0)
        frames, _ = render_stimulus(
            "moving", params, xx, yy, dt_ms=DT_MS, duration_ms=20.0
        )
        assert frames.shape[0] == 20
        # Steps 10..19 all sit on the final trajectory entry.
        assert torch.equal(frames[10], frames[-1])


class TestSteppedDetection:
    """Only genuinely stepped stimuli take the stepped path."""

    def test_a_static_stimulus_is_not_stepped(self):
        from sensoryforge.registry import STIMULUS_REGISTRY

        cls = STIMULUS_REGISTRY.get_class("gaussian")
        instance = cls.from_config({"amplitude": 1.0, "sigma": 1.0})
        assert render_mod._is_stepped(instance) is False

    def test_a_moving_stimulus_is_stepped(self):
        from sensoryforge.registry import STIMULUS_REGISTRY

        cls = STIMULUS_REGISTRY.get_class("moving")
        instance = cls.from_config(dict(COMPONENT_PARAMS))
        assert render_mod._is_stepped(instance) is True
