"""Tests for the four ported pressure-simulation stimuli (Phase 2, Wave K, K2).

Numerical parity against pressure-simulation itself is
``tests/integration/test_stimulus_parity.py``; this file covers registration,
shapes, round-trip completeness (F-045) and the parameter contract that
``tests/contract/test_component_contracts.py`` skips for these four (they
return ``[T, H, W]``, not the generic ``[H, W] -> [H, W]`` the sweep assumes).
"""

from __future__ import annotations

import pytest
import torch

from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.register_components import register_all
from sensoryforge.registry import STIMULUS_REGISTRY
from sensoryforge.stimuli.base import ParamSpec
from sensoryforge.stimuli.render import render_stimulus
from sensoryforge.stimuli.tactile import (
    BrailleStimulus,
    DriftingGratingStimulus,
    MovingEdgeStimulus,
    RampGaussianStimulus,
)

register_all()


@pytest.fixture(scope="module")
def coords8():
    grid = ReceptorGrid(grid_size=(8, 8), spacing=0.15)
    return grid.get_coordinates()


_CLASSES = {
    "ramp_gaussian": RampGaussianStimulus,
    "moving_edge": MovingEdgeStimulus,
    "braille": BrailleStimulus,
    "drifting_grating": DriftingGratingStimulus,
}


@pytest.mark.parametrize("name,cls", sorted(_CLASSES.items()))
def test_registered(name, cls):
    assert STIMULUS_REGISTRY.get_class(name) is cls


@pytest.mark.parametrize("name,cls", sorted(_CLASSES.items()))
def test_param_spec_is_list_of_paramspec(name, cls):
    spec = cls.get_param_spec()
    assert isinstance(spec, list) and len(spec) > 0
    assert all(isinstance(p, ParamSpec) for p in spec)


@pytest.mark.parametrize("name,cls", sorted(_CLASSES.items()))
def test_forward_returns_three_d(coords8, name, cls):
    xx, yy = coords8
    instance = cls()
    out = instance(xx, yy)
    assert out.dim() == 3
    assert tuple(out.shape[1:]) == tuple(xx.shape)


@pytest.mark.parametrize("name,cls", sorted(_CLASSES.items()))
def test_to_dict_round_trip_is_fixed_point(coords8, name, cls):
    instance = cls()
    d1 = instance.to_dict()
    reconstructed = cls.from_config(d1)
    d2 = reconstructed.to_dict()
    assert d1 == d2

    # Every __init__ parameter round-trips through to_dict() (F-045).
    import inspect

    sig = inspect.signature(cls.__init__)
    for pname in sig.parameters:
        if pname in ("self",):
            continue
        assert pname in d1, f"{cls.__name__}.to_dict() missing {pname!r}"


@pytest.mark.parametrize("name,cls", sorted(_CLASSES.items()))
def test_round_tripped_instance_matches_original_forward(coords8, name, cls):
    xx, yy = coords8
    instance = cls()
    reconstructed = cls.from_config(instance.to_dict())
    out1 = instance(xx, yy)
    out2 = reconstructed(xx, yy)
    assert torch.equal(out1, out2)


class TestRampGaussianStimulus:
    def test_default_shape(self, coords8):
        xx, yy = coords8
        out = RampGaussianStimulus()(xx, yy)
        assert tuple(out.shape) == (1100, 8, 8)

    def test_ramp_reaches_one(self, coords8):
        xx, yy = coords8
        out = RampGaussianStimulus(total_ms=10, ramp_ms=5, sigma_mm=1.0)(xx, yy)
        # After the ramp, amplitude scale is 1.0 -- peak value equals the
        # unscaled blob peak (center receptor, if present, else <=1).
        assert out[-1].max() <= 1.0 + 1e-6
        assert out[0].max() < out[-1].max()

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            RampGaussianStimulus(sigma_mm=0.0)


class TestMovingEdgeStimulus:
    def test_default_time_axis_length(self, coords8):
        xx, yy = coords8
        out = MovingEdgeStimulus()(xx, yy)
        # arange(0, 330 + 0.5, 1.0) -> 331 samples (Fact K-a half-step guard)
        assert out.shape[0] == 331

    def test_invalid_spread_raises(self):
        with pytest.raises(ValueError):
            MovingEdgeStimulus(spread=0.0)


class TestBrailleStimulus:
    def test_default_dots_are_letter_h(self):
        stim = BrailleStimulus()
        assert stim.dot_offsets == [(-1.5, -1.5), (1.5, -1.5), (1.5, 1.5)]

    def test_custom_dot_offsets_round_trip(self):
        stim = BrailleStimulus(dot_offsets=[(0.0, 0.0), (1.0, 1.0)])
        d = stim.to_dict()
        assert d["dot_offsets"] == [[0.0, 0.0], [1.0, 1.0]]
        reconstructed = BrailleStimulus.from_config(d)
        assert reconstructed.dot_offsets == [(0.0, 0.0), (1.0, 1.0)]

    def test_invalid_sigma_dot_raises(self):
        with pytest.raises(ValueError):
            BrailleStimulus(sigma_dot=-1.0)


class TestDriftingGratingStimulus:
    def test_output_in_unit_range(self, coords8):
        xx, yy = coords8
        out = DriftingGratingStimulus()(xx, yy)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-6

    def test_invalid_dt_raises(self):
        with pytest.raises(ValueError):
            DriftingGratingStimulus(dt_ms=0.0)


class TestRenderStimulusDispatch:
    """K1: render_stimulus reaches every registered stimulus, and these four
    specifically (fails on the Wave J merge commit, before K1/K2 existed)."""

    @pytest.mark.parametrize("name", sorted(_CLASSES))
    def test_render_stimulus_dispatches_to_registry(self, coords8, name):
        xx, yy = coords8
        frames, time_ms = render_stimulus(name, {}, xx, yy, dt_ms=1.0)
        assert frames.dim() == 3
        assert frames.shape[0] == time_ms.numel()
        assert tuple(frames.shape[1:]) == tuple(xx.shape)

    def test_unknown_stimulus_raises_value_error_listing_names(self, coords8):
        xx, yy = coords8
        with pytest.raises(ValueError, match="not_a_real_stimulus"):
            render_stimulus("not_a_real_stimulus", {}, xx, yy, dt_ms=1.0)

    def test_plugin_stimulus_registered_at_test_time_is_runnable(self, coords8):
        """A two-line subclass registered only here runs end to end through
        render_stimulus -- the F-052 proof that plugin stimuli are reachable."""
        from sensoryforge.stimuli.base import BaseStimulus

        class _TwoLineStimulus(BaseStimulus):
            def forward(self, xx, yy):
                return torch.ones_like(xx)

            def reset_state(self):
                pass

        STIMULUS_REGISTRY.register("_two_line_test_stimulus", _TwoLineStimulus)
        xx, yy = coords8
        frames, time_ms = render_stimulus(
            "_two_line_test_stimulus",
            {},
            xx,
            yy,
            dt_ms=1.0,
            duration_ms=5.0,
        )
        assert frames.shape[1:] == tuple(xx.shape)
        assert torch.all(frames[time_ms.numel() // 2] <= 1.0 + 1e-6)

    def test_legacy_name_still_works(self, coords8):
        xx, yy = coords8
        frames, time_ms = render_stimulus(
            "trapezoidal", {}, xx, yy, dt_ms=0.1, duration_ms=50.0
        )
        assert frames.dim() == 3
        assert frames.shape[0] == time_ms.numel()
