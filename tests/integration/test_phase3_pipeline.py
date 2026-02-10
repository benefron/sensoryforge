"""Integration tests for Phase 3 pipeline features.

Tests the GeneralizedTactileEncodingPipeline with:
- Composite grids (offset, color, flat innervation)
- Processing layers (identity, pipeline)
- Timeline stimulus generation
- Repeated-pattern stimulus generation
- Full forward pass through updated pipeline
"""

import pytest
import torch
from sensoryforge.core.generalized_pipeline import (
    GeneralizedTactileEncodingPipeline,
    create_generalized_pipeline,
)
from sensoryforge.core.innervation import FlatInnervationModule, InnervationModule
from sensoryforge.stimuli.builder import StaticStimulus


# ---------------------------------------------------------------------------
# Composite Grid + Flat Innervation
# ---------------------------------------------------------------------------

class TestCompositeGridPipeline:
    """Pipeline with composite grid and FlatInnervationModule."""

    @pytest.fixture
    def composite_config(self):
        return {
            "grid": {
                "type": "composite",
                "populations": {
                    "sa": {
                        "density": 30.0,
                        "arrangement": "grid",
                        "offset": [0.0, 0.0],
                        "color": [66, 135, 245, 255],
                    },
                    "ra": {
                        "density": 20.0,
                        "arrangement": "grid",
                        "offset": [0.05, 0.0],
                        "color": [245, 166, 66, 255],
                    },
                },
            },
            "innervation": {"method": "flat"},
        }

    def test_composite_pipeline_creates_flat_innervation(self, composite_config):
        """Flat innervation is used when composite grid + method=flat."""
        p = GeneralizedTactileEncodingPipeline(config_dict=composite_config)
        assert isinstance(p.sa_innervation, FlatInnervationModule)
        assert isinstance(p.ra_innervation, FlatInnervationModule)
        assert isinstance(p.sa2_innervation, FlatInnervationModule)

    def test_composite_pipeline_info_reports_flat(self, composite_config):
        """get_pipeline_info reports innervation_type='flat'."""
        p = GeneralizedTactileEncodingPipeline(config_dict=composite_config)
        info = p.get_pipeline_info()
        assert info["innervation_type"] == "flat"
        assert "composite_grid" in info
        assert set(info["composite_grid"]["layers"]) == {"sa", "ra"}
        assert info["composite_grid"]["total_receptors"] > 0

    def test_composite_pipeline_forward_runs(self, composite_config):
        """Full forward pass works with composite grid + flat innervation."""
        p = GeneralizedTactileEncodingPipeline(config_dict=composite_config)
        results = p.forward(stimulus_type="gaussian", duration=20)
        assert "sa_spikes" in results
        assert "ra_spikes" in results
        assert results["sa_spikes"].shape[0] == 1  # batch dim
        assert results["stimulus_sequence"].shape[0] == 1

    def test_composite_grid_without_flat_uses_grid_innervation(self):
        """Without method='flat', standard InnervationModule is used."""
        cfg = {
            "grid": {
                "type": "composite",
                "populations": {
                    "sa": {"density": 20.0, "arrangement": "grid"},
                },
            },
            # No innervation.method = "flat"
        }
        p = GeneralizedTactileEncodingPipeline(config_dict=cfg)
        assert isinstance(p.sa_innervation, InnervationModule)


# ---------------------------------------------------------------------------
# Processing Layers
# ---------------------------------------------------------------------------

class TestProcessingLayersPipeline:
    """Processing layers configuration in the pipeline."""

    def test_default_pipeline_has_identity_processing(self):
        """Default config uses IdentityLayer."""
        p = GeneralizedTactileEncodingPipeline()
        info = p.get_pipeline_info()
        assert info["processing_layers"] == [{"type": "identity"}]

    def test_explicit_identity_config(self):
        """Explicit identity layer config works."""
        cfg = {"processing_layers": [{"type": "identity"}]}
        p = GeneralizedTactileEncodingPipeline(config_dict=cfg)
        info = p.get_pipeline_info()
        assert info["processing_layers"] == [{"type": "identity"}]

    def test_processing_applied_in_forward(self):
        """Processing pipeline is invoked during forward pass."""
        p = GeneralizedTactileEncodingPipeline()
        results = p.forward(stimulus_type="gaussian", duration=10)
        # Mechanoreceptor responses should be present and pass through identity
        assert "mechanoreceptor_responses" in results

    def test_unknown_processing_layer_type_raises(self):
        """Unknown processing layer type raises ValueError."""
        cfg = {"processing_layers": [{"type": "nonexistent_layer"}]}
        with pytest.raises(ValueError, match="Unknown processing layer type"):
            GeneralizedTactileEncodingPipeline(config_dict=cfg)


# ---------------------------------------------------------------------------
# Timeline Stimulus
# ---------------------------------------------------------------------------

class TestTimelineStimulusPipeline:
    """Timeline stimulus generation through the pipeline."""

    def test_timeline_stimulus_shape(self):
        """Timeline stimulus has correct [1, T, H, W] shape."""
        p = GeneralizedTactileEncodingPipeline()
        s1 = StaticStimulus('gaussian', {
            'amplitude': 20.0, 'sigma': 0.3,
            'center_x': -1.0, 'center_y': 0.0,
        })
        seq, t, tp = p.generate_stimulus(
            "timeline",
            sub_stimuli=[
                {"stimulus": s1, "onset_ms": 0.0, "duration_ms": 50.0},
            ],
            duration=100.0,
        )
        assert seq.shape[0] == 1
        assert seq.shape[1] == 200  # 100ms / 0.5ms dt
        assert seq.shape[2] == 80
        assert seq.shape[3] == 80

    def test_timeline_two_overlapping_stimuli(self):
        """Two overlapping timeline sub-stimuli produce combined output."""
        p = GeneralizedTactileEncodingPipeline()
        s1 = StaticStimulus('gaussian', {
            'amplitude': 20.0, 'sigma': 0.3,
            'center_x': -1.0, 'center_y': 0.0,
        })
        s2 = StaticStimulus('gaussian', {
            'amplitude': 30.0, 'sigma': 0.3,
            'center_x': 1.0, 'center_y': 0.0,
        })
        seq, t, tp = p.generate_stimulus(
            "timeline",
            sub_stimuli=[
                {"stimulus": s1, "onset_ms": 0.0, "duration_ms": 60.0},
                {"stimulus": s2, "onset_ms": 20.0, "duration_ms": 60.0},
            ],
            duration=100.0,
        )
        # At t=0 only s1 is active, at t=30ms both are active
        t_zero = seq[0, 0].max().item()
        t_overlap_idx = int(30.0 / 0.5)  # step 60
        t_overlap = seq[0, t_overlap_idx].max().item()
        assert t_overlap > t_zero  # combined should be larger

    def test_timeline_full_forward_pass(self):
        """Timeline stimulus runs through full pipeline forward."""
        p = GeneralizedTactileEncodingPipeline()
        s1 = StaticStimulus('gaussian', {
            'amplitude': 30.0, 'sigma': 0.5,
        })
        results = p.forward(
            stimulus_type="timeline",
            sub_stimuli=[
                {"stimulus": s1, "onset_ms": 0.0, "duration_ms": 25.0},
            ],
            duration=50.0,
        )
        assert "sa_spikes" in results
        assert results["stimulus_sequence"].shape[1] == 100  # 50ms / 0.5


# ---------------------------------------------------------------------------
# Repeated Pattern Stimulus
# ---------------------------------------------------------------------------

class TestRepeatedPatternStimulusPipeline:
    """Repeated-pattern stimulus generation through the pipeline."""

    def test_repeated_pattern_shape(self):
        """Repeated pattern produces [1, T, H, W] output."""
        p = GeneralizedTactileEncodingPipeline()
        seq, t, tp = p.generate_stimulus(
            "repeated_pattern",
            copies_x=2,
            copies_y=2,
            spacing_x=1.0,
            spacing_y=1.0,
            duration=50.0,
        )
        assert seq.shape == (1, 100, 80, 80)

    def test_repeated_pattern_with_custom_base(self):
        """Repeated pattern accepts a custom base stimulus."""
        p = GeneralizedTactileEncodingPipeline()
        base = StaticStimulus('gaussian', {
            'amplitude': 15.0, 'sigma': 0.2,
        })
        seq, t, tp = p.generate_stimulus(
            "repeated_pattern",
            base_stimulus=base,
            copies_x=3,
            copies_y=3,
            spacing_x=0.5,
            spacing_y=0.5,
            duration=20.0,
        )
        assert seq.shape[0] == 1
        assert seq[0, 0].max().item() > 0  # non-zero stimulus

    def test_repeated_pattern_full_forward(self):
        """Repeated pattern runs through full pipeline forward."""
        p = GeneralizedTactileEncodingPipeline()
        results = p.forward(
            stimulus_type="repeated_pattern",
            copies_x=2,
            copies_y=2,
            duration=20.0,
        )
        assert "sa_spikes" in results
        assert "ra_spikes" in results


# ---------------------------------------------------------------------------
# Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Existing pipeline behaviour is preserved."""

    def test_default_pipeline_uses_grid_innervation(self):
        """Default (no composite grid) uses standard InnervationModule."""
        p = GeneralizedTactileEncodingPipeline()
        assert isinstance(p.sa_innervation, InnervationModule)
        info = p.get_pipeline_info()
        assert info["innervation_type"] == "grid"

    def test_existing_stimulus_types_still_work(self):
        """All pre-Phase-3 stimulus types remain functional."""
        p = GeneralizedTactileEncodingPipeline()
        for stype in ["trapezoidal", "gaussian", "step", "ramp"]:
            seq, t, tp = p.generate_stimulus(stype, duration=10)
            assert seq.shape[0] == 1
            assert seq.ndim == 4

    def test_factory_function_works(self):
        """create_generalized_pipeline() still works."""
        p = create_generalized_pipeline()
        assert p is not None
        info = p.get_pipeline_info()
        assert "processing_layers" in info

    def test_from_config_classmethod_works(self):
        """from_config() classmethod still works."""
        p = GeneralizedTactileEncodingPipeline.from_config({})
        assert p is not None
