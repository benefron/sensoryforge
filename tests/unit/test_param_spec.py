"""Tests for D2 — get_param_spec() auto-discovery on stimulus classes."""

import pytest

from sensoryforge.stimuli.base import ParamSpec, BaseStimulus
from sensoryforge.stimuli.gaussian import GaussianStimulus
from sensoryforge.stimuli.texture import GaborTexture, EdgeGrating
from sensoryforge.register_components import register_all
from sensoryforge.registry import STIMULUS_REGISTRY

register_all()


# ---------------------------------------------------------------------------
# ParamSpec dataclass
# ---------------------------------------------------------------------------

class TestParamSpec:
    def test_defaults(self):
        p = ParamSpec("sigma")
        assert p.name == "sigma"
        assert p.dtype == "float"
        assert p.unit == ""
        assert p.tooltip == ""

    def test_label_auto_from_name(self):
        p = ParamSpec("center_x")
        assert p.label == "Center X"

    def test_explicit_label(self):
        p = ParamSpec("center_x", label="X pos")
        assert p.label == "X pos"

    def test_to_dict_roundtrip(self):
        p = ParamSpec("sigma", dtype="float", default=0.5, min_val=0.01,
                      max_val=10.0, step=0.05, unit="mm", tooltip="width")
        d = p.to_dict()
        assert d["name"] == "sigma"
        assert d["min_val"] == 0.01
        assert d["unit"] == "mm"


# ---------------------------------------------------------------------------
# BaseStimulus.get_param_spec default
# ---------------------------------------------------------------------------

class TestBaseStimulusDefault:
    def test_base_default_is_empty(self):
        # BaseStimulus.get_param_spec() must return [] by default
        # (can't instantiate abstract; call classmethod directly)
        assert BaseStimulus.get_param_spec() == []


# ---------------------------------------------------------------------------
# GaussianStimulus
# ---------------------------------------------------------------------------

class TestGaussianParamSpec:
    def test_returns_list(self):
        specs = GaussianStimulus.get_param_spec()
        assert isinstance(specs, list)

    def test_expected_names(self):
        names = [s.name for s in GaussianStimulus.get_param_spec()]
        assert "center_x" in names
        assert "center_y" in names
        assert "amplitude" in names
        assert "sigma" in names

    def test_all_are_param_spec(self):
        for s in GaussianStimulus.get_param_spec():
            assert isinstance(s, ParamSpec)

    def test_sigma_has_positive_min(self):
        sigma = next(s for s in GaussianStimulus.get_param_spec() if s.name == "sigma")
        assert sigma.min_val is not None and sigma.min_val > 0

    def test_amplitude_unit_is_ma(self):
        amp = next(s for s in GaussianStimulus.get_param_spec() if s.name == "amplitude")
        assert amp.unit == "mA"


# ---------------------------------------------------------------------------
# GaborTexture
# ---------------------------------------------------------------------------

class TestGaborParamSpec:
    def test_wavelength_present(self):
        names = [s.name for s in GaborTexture.get_param_spec()]
        assert "wavelength" in names
        assert "orientation" in names
        assert "phase" in names

    def test_wavelength_has_positive_min(self):
        spec = next(s for s in GaborTexture.get_param_spec() if s.name == "wavelength")
        assert spec.min_val is not None and spec.min_val > 0


# ---------------------------------------------------------------------------
# EdgeGrating
# ---------------------------------------------------------------------------

class TestEdgeGratingParamSpec:
    def test_count_is_int(self):
        spec = next(s for s in EdgeGrating.get_param_spec() if s.name == "count")
        assert spec.dtype == "int"

    def test_spacing_unit_is_mm(self):
        spec = next(s for s in EdgeGrating.get_param_spec() if s.name == "spacing")
        assert spec.unit == "mm"


# ---------------------------------------------------------------------------
# Registry.get_param_spec
# ---------------------------------------------------------------------------

class TestRegistryGetParamSpec:
    def test_gaussian_via_registry(self):
        specs = STIMULUS_REGISTRY.get_param_spec("gaussian")
        assert len(specs) > 0
        assert any(s.name == "sigma" for s in specs)

    def test_gabor_via_registry(self):
        specs = STIMULUS_REGISTRY.get_param_spec("gabor")
        assert any(s.name == "wavelength" for s in specs)

    def test_unimplemented_returns_empty(self):
        # composite does not implement get_param_spec
        specs = STIMULUS_REGISTRY.get_param_spec("composite")
        assert specs == []

    def test_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError):
            STIMULUS_REGISTRY.get_param_spec("nonexistent_stimulus_xyz")
