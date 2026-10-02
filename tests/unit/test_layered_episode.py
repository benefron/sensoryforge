"""Layered stimuli: slides, several contacts, modulation, braille cells, signed carriers."""

import pytest
import torch

from sensoryforge.stimuli.episode import contact_terms, pulse_modulation, span_progress
from sensoryforge.stimuli.layered import (
    MODULATIONS,
    TIMING_SPECS,
    default_layer,
    pattern_positions,
    render_layers,
)

H = 81
XS = torch.linspace(-8.0, 8.0, H)
XX, YY = torch.meshgrid(XS, XS, indexing="ij")


def _layer(shape, timing, motion=None, modulation=None, pattern=None):
    layer = default_layer(shape["kind"])
    layer["shape"].update(shape)
    layer["timing"] = timing
    if motion is not None:
        layer["motion"] = motion
    if modulation is not None:
        layer["modulation"] = modulation
    if pattern is not None:
        layer["pattern"] = pattern
    return layer


def _centroid_x(frame):
    return float((frame * XX).sum() / frame.sum())


def _centre(frame):
    return float(frame[40, 40])


def test_new_timing_fields_default_to_the_old_behaviour():
    names = {s.name: s.default for s in TIMING_SPECS}
    assert names["slide_ms"] == 0.0
    assert names["contacts"] == 1
    assert names["pause_ms"] == 0.0
    assert set(MODULATIONS) == {"none", "sine", "pulses"}


def test_slide_moves_only_after_the_hold():
    layer = _layer(
        {"kind": "gaussian", "sigma_mm": 0.3},
        {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": 20, "slide_ms": 20,
         "ramp_down_ms": 0},
        motion={"kind": "linear", "start": [0, 0], "end": [2, 0], "span": "slide"},
    )
    frames = render_layers([layer], XX, YY, dt_ms=1.0, total_ms=50.0)
    assert _centroid_x(frames[10]) == pytest.approx(0.0, abs=1e-4)
    assert _centroid_x(frames[30]) == pytest.approx(1.0, abs=1e-3)
    assert _centroid_x(frames[39]) == pytest.approx(1.9, abs=1e-3)
    assert float(frames[40].abs().max()) == 0.0  # contact over


def test_contacts_pause_and_retouch_where_the_last_ended():
    layer = _layer(
        {"kind": "gaussian", "sigma_mm": 0.3},
        {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": 10, "slide_ms": 10,
         "ramp_down_ms": 0, "contacts": 2, "pause_ms": 10},
        motion={"kind": "linear", "start": [0, 0], "end": [2, 0], "span": "slide"},
    )
    frames = render_layers([layer], XX, YY, dt_ms=1.0, total_ms=60.0)
    assert float(frames[20:30].abs().max()) == 0.0  # the pause is exactly zero
    assert _centroid_x(frames[19]) == pytest.approx(0.9, abs=1e-3)
    assert _centroid_x(frames[30]) == pytest.approx(1.0, abs=1e-3)  # re-touch
    assert _centroid_x(frames[49]) == pytest.approx(1.9, abs=1e-3)
    assert float(frames[50:].abs().max()) == 0.0


def test_contacts_need_an_explicit_hold():
    layer = _layer(
        {"kind": "gaussian"},
        {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": None, "ramp_down_ms": 0,
         "contacts": 2},
    )
    with pytest.raises(ValueError, match="contacts > 1 needs an explicit hold_ms"):
        render_layers([layer], XX, YY, dt_ms=1.0, total_ms=50.0)


def test_sine_modulation_swings_by_its_depth():
    layer = _layer(
        {"kind": "gaussian", "sigma_mm": 0.5},
        {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": 100, "ramp_down_ms": 0},
        modulation={"kind": "sine", "frequency_hz": 50.0, "depth": 0.5},
    )
    frames = render_layers([layer], XX, YY, dt_ms=1.0, total_ms=100.0)
    assert _centre(frames[0]) == pytest.approx(1.0, abs=1e-6)
    assert _centre(frames[10]) == pytest.approx(0.5, abs=1e-6)  # half a period
    assert _centre(frames[20]) == pytest.approx(1.0, abs=1e-5)


def test_pulses_tap_at_their_rate():
    layer = _layer(
        {"kind": "gaussian", "sigma_mm": 0.5},
        {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": 100, "ramp_down_ms": 0},
        modulation={"kind": "pulses", "rate_hz": 50.0, "duty": 0.5},
    )
    frames = render_layers([layer], XX, YY, dt_ms=1.0, total_ms=100.0)
    centre = torch.tensor([_centre(f) for f in frames])
    expected = ((torch.arange(100) % 20) < 10).float()
    assert torch.equal(centre, expected)


def test_pulse_edges_ramp_and_are_clamped_to_the_period():
    tc = torch.arange(0.0, 20.0, 1.0, dtype=torch.float64)
    one = torch.tensor(1.0, dtype=torch.float64)
    m = pulse_modulation(tc, 50.0 * one, 0.5 * one, 4.0 * one, one)
    assert m[0] == 0.0 and m[2] == pytest.approx(0.5) and m[4] == 1.0
    assert m[10] == 1.0 and m[12] == pytest.approx(0.5) and m[14] == 0.0
    wide = pulse_modulation(tc, 50.0 * one, 0.5 * one, 100.0 * one, one)
    assert float(wide.max()) <= 1.0 and float(wide.min()) >= 0.0


def test_contact_terms_and_progress_for_a_batch_of_parameters():
    t = torch.arange(0.0, 60.0, dtype=torch.float64).unsqueeze(0).expand(2, -1)
    col = lambda a, b: torch.tensor([[a], [b]], dtype=torch.float64)  # noqa: E731
    env, tau, k, local = contact_terms(
        t, col(0, 5), col(0, 0), col(10, 10), col(10, 0), col(0, 0), col(2, 1),
        col(10, 0),
    )
    assert float(env[0, 25]) == 0.0 and float(env[0, 30]) == 1.0
    assert float(env[1, 4]) == 0.0 and float(env[1, 5]) == 1.0
    progress = span_progress(tau, k, local, col(2, 1), col(10, 10), col(10, 0))
    assert float(progress[0, 15]) == pytest.approx(0.25)
    assert float(progress[0, 59]) == 1.0
    assert float(progress[1].abs().max()) == 0.0  # no slide, no motion


def test_braille_dots_give_the_same_cell_as_its_letter():
    by_text = pattern_positions({"kind": "braille", "text": "h"})
    by_dots = pattern_positions({"kind": "braille", "text": "z", "dots": "125"})
    assert by_dots == by_text


def test_braille_dots_reject_invalid_cells():
    with pytest.raises(ValueError, match="dot numbers 1-6"):
        pattern_positions({"kind": "braille", "dots": "127"})


def test_signed_grating_has_negative_lobes():
    timing = {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": None, "ramp_down_ms": 0}
    raised = render_layers(
        [_layer({"kind": "grating", "wavelength_mm": 0.8}, timing)],
        XX, YY, dt_ms=1.0, total_ms=2.0,
    )
    signed = render_layers(
        [_layer({"kind": "grating", "wavelength_mm": 0.8, "signed": True}, timing)],
        XX, YY, dt_ms=1.0, total_ms=2.0,
    )
    assert float(raised.min()) >= 0.0
    assert float(signed.min()) < -0.99 and float(signed.max()) > 0.99
    assert torch.allclose(signed, 2.0 * raised - 1.0, atol=1e-6)
