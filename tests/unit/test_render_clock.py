"""A stimulus plays at the same speed whatever the run's time step."""

import pytest
import torch

from sensoryforge.config.schema import GridConfig, SensoryForgeConfig, StimulusConfig
from sensoryforge.stimuli.render import render_for_config


def _render(stim: StimulusConfig, dt_ms: float, duration_ms: float = 200.0):
    config = SensoryForgeConfig(grids=[GridConfig(name="g", rows=30, cols=30)])
    config.simulation.dt_ms = dt_ms
    config.stimulus = stim
    frames, time_ms, _, _ = render_for_config(
        config, duration_ms=duration_ms, dt_ms=dt_ms
    )
    return frames[0], time_ms


@pytest.mark.parametrize(
    "stim_type", ["moving_edge", "braille", "drifting_grating", "ramp_gaussian"]
)
def test_the_frame_at_a_given_time_does_not_depend_on_the_run_step(stim_type):
    coarse, t_coarse = _render(StimulusConfig(type=stim_type), 1.0)
    fine, t_fine = _render(StimulusConfig(type=stim_type), 0.5)
    for when in (50.0, 100.0, 150.0):
        a = coarse[int(torch.nonzero(t_coarse == when)[0])]
        b = fine[int(torch.nonzero(t_fine == when)[0])]
        # Motion matches exactly; a ramp envelope is sampled once per step, so
        # during a ramp the two steps differ by up to one step's resolution
        # (measured 0.0025 on a unit-amplitude grating at 50 ms). A speed
        # error is 100x that: the edge's position at 100 ms moved by 12 px.
        assert torch.allclose(a, b, atol=1e-2), (stim_type, when)


def _gaussian(x_mm: float) -> dict:
    return {
        "class": "StaticStimulus",
        "stim_type": "gaussian",
        "params": {"amplitude": 20.0, "sigma": 0.5, "center_x": x_mm, "center_y": 0.0},
    }


@pytest.mark.parametrize("dt_ms", [1.0, 0.5])
def test_a_timeline_advances_to_its_later_sub_stimuli(dt_ms):
    stim = StimulusConfig(type="timeline")
    stim.params["sub_stimuli"] = [
        {"stimulus": _gaussian(-1.0), "onset_ms": 0.0, "duration_ms": 50.0},
        {"stimulus": _gaussian(1.0), "onset_ms": 50.0, "duration_ms": 50.0},
    ]
    frames, time_ms = _render(stim, dt_ms, duration_ms=100.0)
    x_index = torch.arange(frames.shape[1], dtype=torch.float32)

    def centroid(when):
        frame = frames[int(torch.nonzero(time_ms == when)[0])]
        profile = frame.sum(dim=1)
        return float((profile * x_index).sum() / profile.sum())

    middle = (frames.shape[1] - 1) / 2
    assert centroid(10.0) < middle < centroid(70.0)


def test_a_stimulus_dt_other_than_the_run_step_is_refused():
    stim = StimulusConfig(type="moving_edge")
    stim.params["dt_ms"] = 0.5
    with pytest.raises(ValueError, match="wrong speed"):
        _render(stim, 1.0)
