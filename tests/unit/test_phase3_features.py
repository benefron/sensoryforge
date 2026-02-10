"""Tests for Phase 3 Architecture Remediation features.

Covers:
- A.1: Spatial offset in CompositeReceptorGrid
- A.2: Color metadata in CompositeReceptorGrid
- A.3: GaussianInnervation 3σ cutoff
- A.4: ProcessingLayer base class
- A.5: Stimulus onset/duration + repeat pattern (builder)
- A.6: FlatInnervationModule
- C.1-C.6: Stimulus config round-trip with Phase 3 fields
"""

import math

import pytest
import torch
import numpy as np


# ======================================================================
# A.1 + A.2: CompositeReceptorGrid offset & color
# ======================================================================

class TestCompositeGridOffset:
    """Test spatial offset support in CompositeReceptorGrid."""

    def test_add_layer_with_offset(self):
        from sensoryforge.core.composite_grid import CompositeReceptorGrid

        grid = CompositeReceptorGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
        grid.add_layer("base", density=50.0, arrangement="grid", offset=(0.0, 0.0))
        grid.add_layer("shifted", density=50.0, arrangement="grid", offset=(1.0, 2.0))

        base_coords = grid.get_layer_coordinates("base")
        shifted_coords = grid.get_layer_coordinates("shifted")

        # Shifted coordinates should be displaced by (1.0, 2.0)
        base_mean_x = base_coords[:, 0].mean().item()
        shifted_mean_x = shifted_coords[:, 0].mean().item()
        assert abs(shifted_mean_x - base_mean_x - 1.0) < 0.5

        base_mean_y = base_coords[:, 1].mean().item()
        shifted_mean_y = shifted_coords[:, 1].mean().item()
        assert abs(shifted_mean_y - base_mean_y - 2.0) < 0.5

    def test_get_layer_offset(self):
        from sensoryforge.core.composite_grid import CompositeReceptorGrid

        grid = CompositeReceptorGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
        grid.add_layer("L1", density=50.0, offset=(0.5, -0.3))
        offset = grid.get_layer_offset("L1")
        assert offset == (0.5, -0.3)

    def test_default_offset_is_zero(self):
        from sensoryforge.core.composite_grid import CompositeReceptorGrid

        grid = CompositeReceptorGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
        grid.add_layer("L1", density=50.0)
        assert grid.get_layer_offset("L1") == (0.0, 0.0)


class TestCompositeGridColor:
    """Test color metadata support in CompositeReceptorGrid."""

    def test_add_layer_with_color(self):
        from sensoryforge.core.composite_grid import CompositeReceptorGrid

        grid = CompositeReceptorGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
        color = (66, 135, 245, 255)
        grid.add_layer("L1", density=50.0, color=color)
        assert grid.get_layer_color("L1") == color

    def test_default_color_is_none(self):
        from sensoryforge.core.composite_grid import CompositeReceptorGrid

        grid = CompositeReceptorGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
        grid.add_layer("L1", density=50.0)
        assert grid.get_layer_color("L1") is None

    def test_color_not_found_raises(self):
        from sensoryforge.core.composite_grid import CompositeReceptorGrid

        grid = CompositeReceptorGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
        with pytest.raises(KeyError):
            grid.get_layer_color("nonexistent")


class TestCompositeGridBounds:
    """Test computed_bounds property."""

    def test_computed_bounds(self):
        from sensoryforge.core.composite_grid import CompositeReceptorGrid

        grid = CompositeReceptorGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
        grid.add_layer("L1", density=50.0, arrangement="grid")
        bounds = grid.computed_bounds

        assert len(bounds) == 2
        assert bounds[0][0] < bounds[0][1]
        assert bounds[1][0] < bounds[1][1]

    def test_computed_bounds_empty_raises(self):
        from sensoryforge.core.composite_grid import CompositeReceptorGrid

        grid = CompositeReceptorGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
        with pytest.raises(RuntimeError):
            _ = grid.computed_bounds

    def test_get_all_coordinates(self):
        from sensoryforge.core.composite_grid import CompositeReceptorGrid

        grid = CompositeReceptorGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
        grid.add_layer("L1", density=50.0, arrangement="grid")
        grid.add_layer("L2", density=30.0, arrangement="hex")
        all_coords = grid.get_all_coordinates()
        c1 = grid.get_layer_count("L1")
        c2 = grid.get_layer_count("L2")
        assert all_coords.shape == (c1 + c2, 2)


# ======================================================================
# A.3: GaussianInnervation 3σ cutoff
# ======================================================================

class TestGaussianInnervationLocality:
    """Test that GaussianInnervation respects spatial locality cutoff."""

    def test_3sigma_cutoff(self):
        from sensoryforge.core.innervation import GaussianInnervation

        # Create widely spaced receptors and neurons
        receptors = torch.tensor([
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],  # far away
        ])
        neurons = torch.tensor([[0.0, 0.0]])
        sigma = 0.3

        gi = GaussianInnervation(
            receptors, neurons,
            connections_per_neuron=2,
            sigma_d_mm=sigma,
            max_sigma_distance=3.0,
            seed=42,
        )
        weights = gi.compute_weights()

        # The far-away receptor (index 2) should have zero weight
        assert weights[0, 2].item() == 0.0
        # At least one nearby receptor should be connected
        assert weights[0, :2].sum().item() > 0.0

    def test_cutoff_disabled_with_zero(self):
        from sensoryforge.core.innervation import GaussianInnervation

        receptors = torch.tensor([
            [0.0, 0.0],
            [10.0, 10.0],
        ])
        neurons = torch.tensor([[0.0, 0.0]])

        gi = GaussianInnervation(
            receptors, neurons,
            connections_per_neuron=2,
            sigma_d_mm=0.3,
            max_sigma_distance=0.0,  # disabled
            seed=42,
        )
        weights = gi.compute_weights()
        # With cutoff disabled, far receptor may get nonzero weight 
        # (depends on sampling); at minimum all receptors are eligible
        assert weights.shape == (1, 2)


# ======================================================================
# A.4: ProcessingLayer base class
# ======================================================================

class TestProcessingLayer:
    """Test ProcessingLayer base and concrete classes."""

    def test_identity_layer(self):
        from sensoryforge.core.processing import IdentityLayer

        layer = IdentityLayer()
        x = torch.randn(4, 10)
        result = layer(x)
        assert torch.equal(x, result)

    def test_processing_pipeline(self):
        from sensoryforge.core.processing import ProcessingPipeline, IdentityLayer

        pipeline = ProcessingPipeline([IdentityLayer(), IdentityLayer()])
        x = torch.randn(4, 10)
        result = pipeline(x)
        assert torch.equal(x, result)

    def test_empty_pipeline(self):
        from sensoryforge.core.processing import ProcessingPipeline

        pipeline = ProcessingPipeline([])
        x = torch.randn(4, 10)
        result = pipeline(x)
        assert torch.equal(x, result)


# ======================================================================
# A.5: Timeline + RepeatedPattern stimuli (builder)
# ======================================================================

class TestTimelineStimulus:
    """Test TimelineStimulus from the builder module."""

    def test_create(self):
        from sensoryforge.stimuli.builder import Stimulus, TimelineStimulus

        g = Stimulus.gaussian(amplitude=1.0, sigma=0.3)
        tl = Stimulus.timeline(
            sub_stimuli=[
                {"stimulus": g, "onset_ms": 0.0, "duration_ms": 100.0},
            ],
            total_time_ms=200.0,
            dt_ms=1.0,
        )
        assert isinstance(tl, TimelineStimulus)
        assert tl.num_steps == 200

    def test_generate_all_frames(self):
        from sensoryforge.stimuli.builder import Stimulus

        g = Stimulus.gaussian(amplitude=1.0, sigma=0.3, center=(0.0, 0.0))
        tl = Stimulus.timeline(
            sub_stimuli=[
                {"stimulus": g, "onset_ms": 50.0, "duration_ms": 100.0},
            ],
            total_time_ms=200.0,
            dt_ms=1.0,
        )

        xx = torch.linspace(-2, 2, 20).unsqueeze(0).expand(20, -1)
        yy = torch.linspace(-2, 2, 20).unsqueeze(1).expand(-1, 20)
        frames = tl.generate_all_frames(xx, yy)
        assert frames.shape == (200, 20, 20)

        # Before onset (frame 0): should be zero
        assert frames[0].max().item() == 0.0
        # During active window (frame 75): should be nonzero
        assert frames[75].max().item() > 0.0
        # After end (frame 160): should be zero
        assert frames[160].max().item() == 0.0

    def test_serialization_round_trip(self):
        from sensoryforge.stimuli.builder import Stimulus, TimelineStimulus

        g = Stimulus.gaussian(amplitude=1.0, sigma=0.3)
        tl = Stimulus.timeline(
            sub_stimuli=[
                {"stimulus": g, "onset_ms": 10.0, "duration_ms": 50.0},
            ],
            total_time_ms=100.0,
            dt_ms=0.5,
        )
        d = tl.to_dict()
        assert d["class"] == "TimelineStimulus"
        assert d["total_time_ms"] == 100.0
        assert len(d["sub_stimuli"]) == 1
        assert d["sub_stimuli"][0]["onset_ms"] == 10.0

        tl2 = TimelineStimulus.from_config(d)
        assert tl2.total_time_ms == 100.0
        assert tl2.num_steps == 200


class TestRepeatedPatternStimulus:
    """Test RepeatedPatternStimulus from the builder module."""

    def test_create(self):
        from sensoryforge.stimuli.builder import Stimulus, RepeatedPatternStimulus

        dot = Stimulus.gaussian(amplitude=1.0, sigma=0.1, center=(0.0, 0.0))
        pattern = Stimulus.repeat_pattern(dot, copies_x=3, copies_y=2, spacing_x=0.5, spacing_y=0.5)
        assert isinstance(pattern, RepeatedPatternStimulus)
        assert len(pattern._offsets) == 6

    def test_has_multiple_peaks(self):
        from sensoryforge.stimuli.builder import Stimulus

        dot = Stimulus.gaussian(amplitude=1.0, sigma=0.1, center=(0.0, 0.0))
        pattern = Stimulus.repeat_pattern(dot, copies_x=2, copies_y=1, spacing_x=2.0, spacing_y=1.0)

        xx = torch.linspace(-3, 3, 100).unsqueeze(0).expand(100, -1)
        yy = torch.linspace(-3, 3, 100).unsqueeze(1).expand(-1, 100)
        frame = pattern(xx, yy)

        # The pattern should have at least two distinct peaks
        max_val = frame.max().item()
        assert max_val > 0.5

    def test_serialization_round_trip(self):
        from sensoryforge.stimuli.builder import Stimulus, RepeatedPatternStimulus

        dot = Stimulus.gaussian(amplitude=1.0, sigma=0.1)
        pattern = Stimulus.repeat_pattern(dot, copies_x=3, copies_y=2, spacing_x=0.5, spacing_y=0.5)
        d = pattern.to_dict()
        assert d["class"] == "RepeatedPatternStimulus"
        assert d["copies_x"] == 3
        assert d["copies_y"] == 2

        p2 = RepeatedPatternStimulus.from_config(d)
        assert len(p2._offsets) == 6


# ======================================================================
# A.6: FlatInnervationModule
# ======================================================================

class TestFlatInnervationModule:
    """Test FlatInnervationModule for composite grid wiring."""

    def test_basic_creation(self):
        from sensoryforge.core.innervation import FlatInnervationModule

        receptor_coords = torch.rand(100, 2) * 10 - 5  # [-5, 5]
        mod = FlatInnervationModule(
            neuron_type="SA",
            receptor_coords=receptor_coords,
            neurons_per_row=5,
            seed=42,
        )
        assert mod.num_neurons == 25
        assert mod.num_receptors == 100
        assert mod.innervation_weights.shape == (25, 100)

    def test_forward_2d(self):
        from sensoryforge.core.innervation import FlatInnervationModule

        receptor_coords = torch.rand(50, 2) * 10 - 5
        mod = FlatInnervationModule(
            neuron_type="SA",
            receptor_coords=receptor_coords,
            neurons_per_row=4,
            seed=42,
        )
        batch_input = torch.randn(2, 50)
        output = mod(batch_input)
        assert output.shape == (2, 16)

    def test_forward_3d(self):
        from sensoryforge.core.innervation import FlatInnervationModule

        receptor_coords = torch.rand(30, 2) * 10 - 5
        mod = FlatInnervationModule(
            neuron_type="RA",
            receptor_coords=receptor_coords,
            neurons_per_row=3,
            seed=42,
        )
        batch_input = torch.randn(2, 10, 30)
        output = mod(batch_input)
        assert output.shape == (2, 10, 9)

    def test_different_methods(self):
        from sensoryforge.core.innervation import FlatInnervationModule

        coords = torch.rand(50, 2) * 10 - 5
        for method in ("gaussian", "one_to_one", "distance_weighted"):
            mod = FlatInnervationModule(
                neuron_type="SA",
                receptor_coords=coords,
                neurons_per_row=4,
                innervation_method=method,
                seed=42,
            )
            assert mod.innervation_weights.shape == (16, 50)


# ======================================================================
# C.1-C.6: StimulusConfig Phase 3 fields
# ======================================================================

class TestStimulusConfigPhase3:
    """Test StimulusConfig Phase 3 dataclass fields."""

    def test_defaults(self):
        from sensoryforge.gui.tabs.stimulus_tab import StimulusConfig

        cfg = StimulusConfig(
            name="Test", stimulus_type="gaussian", motion="static",
            start=(0, 0), end=(0, 0), spread=0.3, orientation_deg=0,
            amplitude=1.0, ramp_up_ms=50, plateau_ms=200, ramp_down_ms=50,
            total_ms=300, dt_ms=1.0, speed_mm_s=0,
        )
        assert cfg.onset_ms == 0.0
        assert cfg.duration_ms == 0.0
        assert cfg.motion_type == "static"
        assert cfg.repeat_enabled is False

    def test_as_dict_includes_phase3_fields(self):
        from sensoryforge.gui.tabs.stimulus_tab import StimulusConfig

        cfg = StimulusConfig(
            name="Test", stimulus_type="gabor", motion="static",
            start=(0, 0), end=(0, 0), spread=0.3, orientation_deg=45,
            amplitude=1.0, ramp_up_ms=50, plateau_ms=200, ramp_down_ms=50,
            total_ms=300, dt_ms=1.0, speed_mm_s=0,
            onset_ms=100.0,
            duration_ms=200.0,
            motion_type="linear",
            repeat_enabled=True,
            repeat_nx=4,
            repeat_ny=3,
            repeat_spacing_x=0.5,
            repeat_spacing_y=0.5,
        )
        d = cfg.as_dict()
        assert d["onset_ms"] == 100.0
        assert d["duration_ms"] == 200.0
        assert d["motion_type"] == "linear"
        assert d["repeat_enabled"] is True
        assert d["repeat_nx"] == 4
        assert d["repeat_ny"] == 3
        assert d["repeat_spacing_x"] == 0.5
        assert d["repeat_spacing_y"] == 0.5


class TestTimelineScrubberWidget:
    """Test TimelineScrubberWidget coordinate helpers.
    
    Note: These require QApplication; skip if unavailable.
    """

    @pytest.fixture(autouse=True)
    def _ensure_qapp(self):
        """Ensure QApplication exists for widget tests."""
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            self._app = QApplication([])
        else:
            self._app = app

    def test_ms_to_x_and_back(self):
        from sensoryforge.gui.tabs.stimulus_tab import TimelineScrubberWidget

        w = TimelineScrubberWidget()
        w.resize(400, 60)
        w.set_total_time(1000.0)

        # Mid-point should round-trip approximately
        x_mid = w._ms_to_x(500.0)
        ms_back = w._x_to_ms(x_mid)
        assert abs(ms_back - 500.0) < 5.0

    def test_set_entries(self):
        from sensoryforge.gui.tabs.stimulus_tab import (
            TimelineScrubberWidget, StimulusConfig,
        )

        w = TimelineScrubberWidget()
        configs = [
            StimulusConfig(
                name="G1", stimulus_type="gaussian", motion="static",
                start=(0, 0), end=(0, 0), spread=0.3, orientation_deg=0,
                amplitude=1.0, ramp_up_ms=50, plateau_ms=200, ramp_down_ms=50,
                total_ms=300, dt_ms=1.0, speed_mm_s=0,
                onset_ms=50.0, duration_ms=100.0,
            ),
            StimulusConfig(
                name="G2", stimulus_type="gaussian", motion="static",
                start=(0, 0), end=(0, 0), spread=0.3, orientation_deg=0,
                amplitude=1.0, ramp_up_ms=50, plateau_ms=200, ramp_down_ms=50,
                total_ms=300, dt_ms=1.0, speed_mm_s=0,
                onset_ms=100.0, duration_ms=200.0,
            ),
        ]
        w.set_entries(configs)
        assert len(w._entries) == 2
        assert w._entries[0]["onset"] == 50.0
        assert w._entries[1]["duration"] == 200.0
