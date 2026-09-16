"""OnOffLayer: centre-surround ON/OFF processing (Phase 2, Wave M3).

Per the Wave M spec's anti-plausibility guardrail: an ON and an OFF plane
that are accidentally identical, or swapped, will look fine in any shape
test, so this drives the layer with a stimulus unambiguously asymmetric in
sign and asserts *which* plane responds.
"""

from __future__ import annotations

import pytest
import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    PopulationInput,
    RFBuilderConfig,
    SensoryForgeConfig,
)
from sensoryforge.core.processing import OnOffLayer, ProcessingPipeline
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.registry import PROCESSING_REGISTRY
from sensoryforge.register_components import register_all
from sensoryforge.stimuli.base import ParamSpec

register_all()


def _line_coords(n: int = 6, spacing: float = 0.2) -> torch.Tensor:
    xs = torch.arange(n, dtype=torch.float32) * spacing
    return torch.stack([xs, torch.zeros(n)], dim=-1)


class TestOnOffLayerRegistration:
    def test_registered_under_onoff(self):
        assert "onoff" in PROCESSING_REGISTRY.list_registered()
        assert PROCESSING_REGISTRY.get_class("onoff") is OnOffLayer

    def test_get_param_spec_returns_paramspecs(self):
        spec = OnOffLayer.get_param_spec()
        assert isinstance(spec, list)
        assert all(isinstance(p, ParamSpec) for p in spec)
        names = {p.name for p in spec}
        assert names == {"sigma_center_mm", "sigma_surround_mm"}


class TestOnOffLayerShapesAndRoundTrip:
    def test_forward_doubles_receptor_axis(self):
        coords = _line_coords()
        layer = OnOffLayer(coords)
        resp = torch.rand(2, 5, 6)
        out = layer(resp)
        assert out.shape == (2, 5, 12)

    def test_expand_receptor_coords_doubles_m(self):
        coords = _line_coords()
        expanded = OnOffLayer.expand_receptor_coords(coords)
        assert expanded.shape == (12, 2)
        assert torch.equal(expanded[:6], coords)
        assert torch.equal(expanded[6:], coords)

    def test_to_dict_from_config_round_trip(self):
        coords = _line_coords()
        layer = OnOffLayer(coords, sigma_center_mm=0.1, sigma_surround_mm=0.3)
        d = layer.to_dict()
        assert d == {
            "method": "onoff",
            "params": {"sigma_center_mm": 0.1, "sigma_surround_mm": 0.3},
        }
        reconstructed = OnOffLayer.from_config(d, receptor_coords=coords)
        assert torch.equal(layer.dog_kernel, reconstructed.dog_kernel)

    def test_rejects_bad_coords(self):
        with pytest.raises(ValueError):
            OnOffLayer(torch.rand(6, 3))

    def test_rejects_nonpositive_sigma(self):
        coords = _line_coords()
        with pytest.raises(ValueError):
            OnOffLayer(coords, sigma_center_mm=0.0)


class TestOnOffLayerAntiPlausibility:
    """The critical check: a positive (bright) input drives ON and not OFF;
    a negative (dark) input drives OFF and not ON, at the same receptor."""

    def test_bright_spot_drives_on_not_off(self):
        coords = _line_coords()
        layer = OnOffLayer(coords, sigma_center_mm=0.15, sigma_surround_mm=0.45)
        resp = torch.zeros(1, 1, 6)
        resp[0, 0, 3] = 5.0  # bright bump at receptor 3
        out = layer(resp)
        on, off = out[..., :6], out[..., 6:]
        assert on[0, 0, 3].item() > 0.0
        assert off[0, 0, 3].item() == 0.0

    def test_dark_spot_drives_off_not_on(self):
        coords = _line_coords()
        layer = OnOffLayer(coords, sigma_center_mm=0.15, sigma_surround_mm=0.45)
        resp = torch.zeros(1, 1, 6)
        resp[0, 0, 3] = -5.0  # dark bump at receptor 3
        out = layer(resp)
        on, off = out[..., :6], out[..., 6:]
        assert off[0, 0, 3].item() > 0.0
        assert on[0, 0, 3].item() == 0.0

    def test_on_and_off_planes_are_not_identical(self):
        """A regression this guards against directly: ON/OFF collapsing to
        the same plane (e.g. a copy-paste bug) would pass every shape
        check but fail this."""
        coords = _line_coords()
        layer = OnOffLayer(coords, sigma_center_mm=0.15, sigma_surround_mm=0.45)
        resp = torch.zeros(1, 1, 6)
        resp[0, 0, 2] = 3.0
        resp[0, 0, 4] = -3.0
        out = layer(resp)
        on, off = out[..., :6], out[..., 6:]
        assert not torch.equal(on, off)
        assert on[0, 0, 2].item() > 0.0 and off[0, 0, 2].item() == 0.0
        assert off[0, 0, 4].item() > 0.0 and on[0, 0, 4].item() == 0.0


class TestProcessingPipelineOnOff:
    def test_pipeline_from_config_builds_onoff(self):
        coords = _line_coords()
        pipeline = ProcessingPipeline.from_config(
            [
                {
                    "method": "onoff",
                    "params": {"sigma_center_mm": 0.1, "sigma_surround_mm": 0.3},
                }
            ],
            receptor_coords=coords,
        )
        resp = torch.rand(1, 2, 6)
        out = pipeline(resp)
        assert out.shape == (1, 2, 12)

    def test_pipeline_requires_receptor_coords_for_onoff(self):
        with pytest.raises(ValueError, match="requires receptor_coords"):
            ProcessingPipeline.from_config([{"method": "onoff"}])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown processing layer type"):
            ProcessingPipeline.from_config([{"method": "nonexistent"}])


class TestEngineWithOnOffInput:
    """M3's engine wiring: a population input with a processing pipeline
    builds its receptive-field bank on the post-processing (doubled)
    receptor axis, and run() sends real stimulus through it correctly."""

    @staticmethod
    def _config(processing) -> SensoryForgeConfig:
        grid = GridConfig(name="Grid", arrangement="grid", rows=6, cols=6, spacing=0.2)
        pop = PopulationConfig(
            name="onoff_pop",
            neuron_type="SA",
            neurons_per_row=3,
            inputs=[
                PopulationInput(
                    grid="Grid",
                    rf=RFBuilderConfig(method="one_to_one"),
                    processing=processing,
                )
            ],
        )
        return SensoryForgeConfig(grids=[grid], populations=[pop])

    def test_bank_built_on_doubled_receptor_axis(self):
        config = self._config([{"method": "onoff"}])
        engine = SimulationEngine(config)
        pop = engine.populations[0]
        ctx = pop["inputs"][0]
        # 6x6 = 36 raw receptors; the bank must be built on 72 (ON+OFF).
        assert ctx["receptor_coords"].shape[0] == 36
        assert ctx["bank"].num_receptors == 72

    def test_run_produces_finite_drive(self):
        config = self._config([{"method": "onoff"}])
        engine = SimulationEngine(config)
        torch.manual_seed(0)
        stimulus = torch.rand(1, 4, 6, 6)
        results = engine.run(stimulus, return_intermediates=True)
        drive = results["onoff_pop"]["drive"]
        assert torch.isfinite(drive).all()

    def test_no_processing_is_bit_identical_to_before(self):
        """Empty processing (the default) skips the pipeline entirely --
        same bank shape as a population with no processing at all."""
        config_none = self._config([])
        config_default = SensoryForgeConfig(
            grids=[
                GridConfig(name="Grid", arrangement="grid", rows=6, cols=6, spacing=0.2)
            ],
            populations=[
                PopulationConfig(
                    name="p",
                    neuron_type="SA",
                    neurons_per_row=3,
                    target_grid="Grid",
                    innervation_method="one_to_one",
                )
            ],
        )
        e1 = SimulationEngine(config_none)
        e2 = SimulationEngine(config_default)
        assert (
            e1.populations[0]["bank"].num_receptors
            == e2.populations[0]["bank"].num_receptors
            == 36
        )
