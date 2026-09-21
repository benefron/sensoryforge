"""`StimulusConfig.params` and `effective_defaults`: what is shown is what runs."""

import pytest
import torch

from sensoryforge.config.schema import GridConfig, SensoryForgeConfig, StimulusConfig
from sensoryforge.registry import STIMULUS_REGISTRY
from sensoryforge.stimuli.render import effective_defaults, render_for_config

# Types that cannot render with nothing set (they need sub-stimuli).
_NEEDS_CHILDREN = {"composite", "static", "timeline"}
_FIELDS = set(StimulusConfig.__dataclass_fields__)


def _render(stim: StimulusConfig) -> torch.Tensor:
    cfg = SensoryForgeConfig(grids=[GridConfig(name="g")])
    cfg.stimulus = stim
    frames, _, _, dropped = render_for_config(cfg, duration_ms=200, dt_ms=1.0)
    assert not dropped, dropped
    return frames


def _renderable_types():
    import sensoryforge.register_components as rc

    rc.register_all()
    return sorted(set(STIMULUS_REGISTRY.list_registered()) - _NEEDS_CHILDREN)


def test_gaussian_effective_default_is_the_amplitude_that_runs():
    eff = effective_defaults("gaussian")
    assert eff["amplitude"] == 30.0
    peak = float(_render(StimulusConfig(type="gaussian")).max())
    assert peak == pytest.approx(30.0, rel=0.02)


@pytest.mark.parametrize("stim_type", _renderable_types())
def test_setting_a_parameter_to_its_effective_default_changes_nothing(stim_type):
    base = _render(StimulusConfig(type=stim_type))
    for name, value in effective_defaults(stim_type).items():
        if value is None or isinstance(value, dict):
            continue
        stim = StimulusConfig(type=stim_type)
        if name in _FIELDS:
            setattr(stim, name, value)
        else:
            stim.params[name] = value
        out = _render(stim)
        assert out.shape == base.shape and torch.equal(out, base), (stim_type, name)


def test_params_reach_the_stimulus_and_change_the_frames():
    base = _render(StimulusConfig(type="edge_grating"))
    stim = StimulusConfig(type="edge_grating")
    stim.params["count"] = 2
    assert not torch.equal(_render(stim), base)


def test_params_round_trip_through_yaml_and_stay_explicit():
    cfg = SensoryForgeConfig(grids=[GridConfig(name="g")])
    cfg.stimulus = StimulusConfig(type="braille")
    cfg.stimulus.params["v_mms"] = 7.5
    again = SensoryForgeConfig.from_yaml(cfg.to_yaml())
    assert again.stimulus.params == {"v_mms": 7.5}
    assert "params" in again.stimulus.explicit_fields()
    assert torch.equal(_render(again.stimulus), _render(cfg.stimulus))


def test_empty_params_is_not_written():
    assert "params" not in StimulusConfig(type="gaussian").to_dict()


def test_params_may_not_shadow_a_named_field():
    with pytest.raises(ValueError, match="amplitude"):
        StimulusConfig(type="gaussian", params={"amplitude": 3.0})
