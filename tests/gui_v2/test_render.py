"""Tests for :mod:`sensoryforge.gui.execution.render`.

The GUI's renderer must produce exactly what ``render_stimulus`` produces on
the same canvas -- that is the whole point of having one renderer -- and must
say so when it silently discarded a stimulus setting the user had changed
(F-061).
"""

from __future__ import annotations

import pytest
import torch

from sensoryforge.config.schema import GridConfig, SensoryForgeConfig
from sensoryforge.gui.execution.render import render_for_config
from sensoryforge.stimuli.canvas import stimulus_canvas
from sensoryforge.stimuli.render import render_stimulus

PRESET = "sensoryforge/presets/tactile_sa1_ra1.yml"


def _small_config() -> SensoryForgeConfig:
    """The preset shrunk to a 20x20 grid, so a test renders in milliseconds."""
    config = SensoryForgeConfig.from_yaml_file(PRESET)
    config.grids[0].rows = 20
    config.grids[0].cols = 20
    return config


def test_rendered_tensor_equals_render_stimulus_on_the_canvas():
    config = _small_config()
    canvas = stimulus_canvas(config.grids[0], device="cpu")
    expected, expected_time = render_stimulus(
        "gaussian",
        {"amplitude": config.stimulus.amplitude, "sigma": config.stimulus.sigma},
        canvas.xx,
        canvas.yy,
        dt_ms=config.simulation.dt_ms,
        duration_ms=30.0,
        device="cpu",
    )

    rendered = render_for_config(config, duration_ms=30.0)

    assert rendered.stimulus.shape == (1, 30, 20, 20)
    assert torch.equal(rendered.stimulus[0], expected)
    assert rendered.time_ms.shape == expected_time.shape
    assert rendered.time_ms.tolist() == expected_time.tolist()


def test_dt_ms_defaults_to_the_config_and_can_be_overridden():
    config = _small_config()

    assert render_for_config(config, duration_ms=30.0).stimulus.shape[1] == 30
    assert (
        render_for_config(config, duration_ms=30.0, dt_ms=0.5).stimulus.shape[1] == 60
    )


def test_untouched_schema_fields_are_dropped_without_a_warning():
    # `StimulusConfig.to_dict()` hands the gaussian constructor a dozen fields
    # it knows nothing about. Every one of them is still at its default here,
    # so discarding them is housekeeping and must stay silent.
    rendered = render_for_config(_small_config(), duration_ms=10.0)

    assert rendered.dropped, "expected the schema's extra fields to be dropped"
    assert rendered.warning is None


def test_a_deliberately_set_dropped_field_is_named_in_the_warning():
    config = _small_config()
    config.stimulus.spread = 4.25  # not a `gaussian` constructor parameter

    rendered = render_for_config(config, duration_ms=10.0)

    assert rendered.warning is not None
    assert "spread=4.25" in rendered.warning
    assert "gaussian" in rendered.warning
    # Only the field the user set is named; the untouched ones are not.
    assert "wavelength" not in rendered.warning


def test_target_layer_picks_that_grid():
    config = _small_config()
    config.grids.append(GridConfig(name="Coarse", rows=6, cols=6, spacing=0.5))
    config.stimulus.target_layer = "Coarse"

    rendered = render_for_config(config, duration_ms=10.0)

    assert rendered.stimulus.shape[-2:] == (6, 6)


def test_unknown_target_layer_raises_naming_the_grids():
    config = _small_config()
    config.stimulus.target_layer = "Nope"

    with pytest.raises(ValueError, match="Nope"):
        render_for_config(config, duration_ms=10.0)


def test_a_config_with_no_grid_raises():
    config = SensoryForgeConfig()

    with pytest.raises(ValueError, match="no grids"):
        render_for_config(config, duration_ms=10.0)


@pytest.mark.parametrize("duration_ms,dt_ms", [(0.0, None), (10.0, 0.0)])
def test_non_positive_duration_or_dt_raises(duration_ms, dt_ms):
    with pytest.raises(ValueError):
        render_for_config(_small_config(), duration_ms=duration_ms, dt_ms=dt_ms)
