"""No stimulus is a held step: each one moves, or ramps both in and out.

Neurons with dynamics respond to change. A still image switched on at t = 0
and held to the end gives an RA population one onset burst and no release,
so the renderer ramps still stimuli in and out by default (an eighth of the
run each), and stimuli with their own time course follow the run's length.
"""

import pytest
import torch

from sensoryforge.config.schema import GridConfig, SensoryForgeConfig, StimulusConfig
from sensoryforge.registry import STIMULUS_REGISTRY
from sensoryforge.stimuli.render import default_envelope, render_for_config

DURATION_MS = 400.0

#: Types that need sub-stimuli to render at all (covered in tests/gui_v2).
_NEEDS_PARTS = {"composite", "static", "timeline"}
#: pressure-simulation's ramp-and-hold probe, ported unchanged for parity:
#: it ramps in and holds on purpose (the sustained SA response).
_RAMP_AND_HOLD = {"ramp_gaussian"}


def _types():
    import sensoryforge.register_components as rc

    rc.register_all()
    return sorted(set(STIMULUS_REGISTRY.list_registered()) - _NEEDS_PARTS)


def _render(stim_type):
    config = SensoryForgeConfig(grids=[GridConfig(name="g", rows=30, cols=30)])
    config.stimulus = StimulusConfig(type=stim_type)
    frames, _, _, _ = render_for_config(config, duration_ms=DURATION_MS, dt_ms=1.0)
    return frames[0]


def _moves(frames) -> bool:
    active = frames.flatten(1).abs().amax(1) > 0.2 * frames.abs().max()
    shown = frames[active]
    return bool((shown - shown[:1]).abs().amax() > 0.2 * frames.abs().max())


@pytest.mark.parametrize("stim_type", _types())
def test_every_stimulus_moves_or_ramps_in_and_out(stim_type):
    frames = _render(stim_type)
    peak = frames.abs().max()
    amplitude = frames.flatten(1).abs().amax(1) / peak
    ramps_in = float(amplitude[0]) < 0.2
    ramps_out = float(amplitude[-1]) < 0.2 or stim_type in _RAMP_AND_HOLD
    assert _moves(frames) or (ramps_in and ramps_out), (
        stim_type,
        [round(float(amplitude[i]), 2) for i in (0, 50, 200, 350, -1)],
    )


def test_a_still_stimulus_ramps_over_an_eighth_of_the_run_each_way():
    amplitude = _render("gaussian").flatten(1).amax(1)
    amplitude = amplitude / amplitude.max()
    assert float(amplitude[25]) == pytest.approx(0.5, abs=0.02)  # 25 of 50 ms up
    assert float(amplitude[200]) == 1.0
    assert float(amplitude[375]) == pytest.approx(0.5, abs=0.02)  # 25 of 50 ms down


def test_explicit_ramps_win_and_zero_ramps_are_a_step():
    config = SensoryForgeConfig(grids=[GridConfig(name="g", rows=30, cols=30)])
    config.stimulus = StimulusConfig(type="gaussian")
    config.stimulus.ramp_up_ms = 0.0
    config.stimulus.ramp_down_ms = 0.0
    frames, _, _, _ = render_for_config(config, duration_ms=100.0, dt_ms=1.0)
    amplitude = frames[0].flatten(1).amax(1)
    assert torch.allclose(amplitude, amplitude[0].expand_as(amplitude))
    assert default_envelope(800.0) == (100.0, 600.0, 100.0)


def test_stimuli_with_their_own_length_follow_the_run():
    for stim_type in ("moving_edge", "drifting_grating", "braille"):
        long = _render(stim_type)
        assert long.shape[0] == int(DURATION_MS)
        # Still active near the end: not cut short, not left blank.
        late = long[int(0.87 * DURATION_MS) : int(0.95 * DURATION_MS)]
        assert float(late.abs().max()) > 0.05 * float(long.abs().max()), stim_type
