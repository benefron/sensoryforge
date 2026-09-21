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


def _preset() -> SensoryForgeConfig:
    config = SensoryForgeConfig.from_yaml_file(
        "sensoryforge/presets/tactile_sa1_ra1.yml"
    )
    config.grids[0].rows = 12
    config.grids[0].cols = 12
    return config


def _two_input(combine: str) -> SensoryForgeConfig:
    data = _preset().to_dict()
    population = data["populations"][0]
    for key in (
        "target_grid",
        "innervation_method",
        "resolvable_distance_mm",
        "innervation_params",
    ):
        population.pop(key, None)
    population["inputs"] = [
        {
            "grid": "Main Grid",
            "rf": {"method": "template", "params": {"resolvable_distance_mm": 0.4}},
        },
        {"grid": "Main Grid", "rf": {"method": "gaussian", "params": {}}},
    ]
    population["combine"] = combine
    return SensoryForgeConfig.from_dict(data)


def test_the_shipped_preset_is_clean():
    assert validate(_preset()) == {}


def test_sum_combine_with_mismatched_neuron_counts_is_reported_f062():
    errors = validate(_two_input("sum"))
    assert set(errors) == {"populations.0.combine"}
    assert "same neuron count" in errors["populations.0.combine"]
    assert validate(_two_input("concat")) == {}


def test_filter_and_neuron_problems_land_on_their_own_stage():
    config = _preset()
    config.populations[0].filter_params = {"bogus": 1.0}
    config.populations[1].model_params = {"zz": 1.0}
    errors = validate(config)
    assert set(errors) == {"populations.0.filter", "populations.1.neuron"}


def test_a_stimulus_problem_does_not_hide_a_population_problem():
    config = _preset()
    config.stimulus.type = "no_such_stimulus"
    config.populations[0].filter_params = {"bogus": 1.0}
    errors = validate(config)
    assert "stimulus" in errors and "populations.0.filter" in errors


def test_grid_geometry_and_missing_coords_file_are_reported_on_the_grid():
    config = _preset()
    config.grids[0].spacing = -0.1
    assert "grids.0" in validate(config)
    config = _preset()
    config.grids[0].coords_file = "/no/such/coords.csv"
    assert "grids.0" in validate(config)


def test_validation_never_uses_the_run_device():
    config = _preset()
    config.simulation.device = "cuda"  # this machine may not have it
    assert validate(config) == {}


def test_every_stage_error_turns_its_own_strip_chip_red(qtbot):
    from sensoryforge.gui import theme
    from sensoryforge.gui.session import Session
    from sensoryforge.gui.widgets.pipeline_strip import PipelineStrip

    session = Session(_two_input("sum"))
    strip = PipelineStrip(session)
    qtbot.addWidget(strip)

    def dot_colour(stage):
        chip = strip._pop_rows[0].chips_by_stage[stage][0]
        return chip.status_dot.styleSheet()

    assert theme.PALETTE["error"] in dot_colour("combine")
    assert theme.PALETTE["error"] not in dot_colour("filter")

    session.set_by_path("populations.0.combine", "concat")
    assert theme.PALETTE["error"] not in dot_colour("combine")
    session.set_by_path("populations.0.filter_params", {"bogus": 1.0})
    assert theme.PALETTE["error"] in dot_colour("filter")


def test_each_screen_lists_only_its_own_problems(qtbot):
    from sensoryforge.gui.screens.populations import PopulationsScreen
    from sensoryforge.gui.screens.sensors import SensorsScreen
    from sensoryforge.gui.screens.stimulus import StimulusScreen
    from sensoryforge.gui.session import Session

    session = Session(_preset())
    screens = {
        "sensors": SensorsScreen(session),
        "stimulus": StimulusScreen(session),
        "populations": PopulationsScreen(session),
    }
    for screen in screens.values():
        qtbot.addWidget(screen)
        assert not screen.problems.messages()

    session.set_by_path("populations.0.filter_params", {"bogus": 1.0})
    assert any("bogus" in m for m in screens["populations"].problems.messages())
    assert not screens["sensors"].problems.messages()
    assert not screens["stimulus"].problems.messages()

    session.set_by_path("stimulus.type", "no_such_stimulus")
    assert any("no_such_stimulus" in m for m in screens["stimulus"].problems.messages())
    assert not screens["sensors"].problems.messages()
