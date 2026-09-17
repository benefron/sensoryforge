"""Tests for :mod:`sensoryforge.gui.validation`.

Pure-function checks: no Qt objects are touched here.
"""

import pytest

pytestmark = pytest.mark.gui  # collected with the rest of tests/gui_v2

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.gui.validation import validate  # noqa: E402


def _basic_config() -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[GridConfig(name="skin", rows=4, cols=4, spacing=0.2)],
        populations=[
            PopulationConfig(name="SA", neuron_type="SA", target_grid="skin"),
        ],
    )


def test_empty_config_reports_at_root():
    errors = validate(SensoryForgeConfig())
    assert "" in errors
    assert "grid" in errors[""] or "population" in errors[""]


def test_clean_config_has_no_errors():
    assert validate(_basic_config()) == {}


def test_missing_grid_reference():
    config = _basic_config()
    config.populations[0].target_grid = "does-not-exist"
    errors = validate(config)
    assert "populations.0.target_grid" in errors
    assert "does-not-exist" in errors["populations.0.target_grid"]


def test_duplicate_population_names():
    config = _basic_config()
    config.populations.append(
        PopulationConfig(name="SA", neuron_type="RA", target_grid="skin")
    )
    errors = validate(config)
    assert "populations.1.name" in errors
    assert "SA" in errors["populations.1.name"]


def test_dt_ms_not_a_multiple_of_integrate_dt():
    config = _basic_config()
    # Bypass SimulationConfig.__post_init__ validation by mutating directly,
    # the way Session.set_by_path does.
    config.simulation = SimulationConfig()
    config.simulation.dt_ms = 1.0
    config.simulation.integrate_dt_ms = 0.3
    errors = validate(config)
    assert "simulation.dt_ms" in errors


def test_preset_validates_clean():
    config = SensoryForgeConfig.from_yaml_file(
        "sensoryforge/presets/tactile_sa1_ra1.yml"
    )
    assert validate(config) == {}
