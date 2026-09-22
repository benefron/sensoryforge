"""Tests for :class:`sensoryforge.gui.widgets.pipeline_strip.PipelineStrip`."""

import pytest

pytestmark = pytest.mark.gui

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    PopulationInput,
    RFBuilderConfig,
    SensoryForgeConfig,
)
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.gui.widgets.pipeline_strip import PipelineStrip  # noqa: E402


def _preset_config() -> SensoryForgeConfig:
    return SensoryForgeConfig.from_yaml_file("sensoryforge/presets/tactile_sa1_ra1.yml")


def test_preset_shows_two_population_rows_with_expected_chips(qtbot):
    session = Session(_preset_config())
    strip = PipelineStrip(session)
    qtbot.addWidget(strip)

    assert set(strip._pop_rows.keys()) == {0, 1}
    row0 = strip._pop_rows[0]
    stages = set(row0.chips_by_stage.keys())
    assert stages == {"sensor_array", "receptive_field", "filter", "neuron", "readout"}
    assert "combine" not in stages  # single input


def test_two_inputs_show_two_rf_chips_and_a_combine_chip(qtbot):
    config = SensoryForgeConfig(
        grids=[
            GridConfig(name="a", rows=4, cols=4),
            GridConfig(name="b", rows=4, cols=4),
        ],
        populations=[
            PopulationConfig(
                name="multi",
                inputs=[
                    PopulationInput(grid="a", rf=RFBuilderConfig(method="gaussian")),
                    PopulationInput(grid="b", rf=RFBuilderConfig(method="uniform")),
                ],
            )
        ],
    )
    session = Session(config)
    strip = PipelineStrip(session)
    qtbot.addWidget(strip)

    row = strip._pop_rows[0]
    assert len(row.chips_by_stage["receptive_field"]) == 2
    assert len(row.chips_by_stage["combine"]) == 1


def test_set_by_path_updates_only_the_affected_chip(qtbot):
    session = Session(_preset_config())
    strip = PipelineStrip(session)
    qtbot.addWidget(strip)

    other_row_before = strip._pop_rows[1]
    session.set_by_path("populations.0.filter_method", "none")

    row0 = strip._pop_rows[0]
    assert "none" in row0.chips_by_stage["filter"][0].text()
    # The untouched row was not rebuilt.
    assert strip._pop_rows[1] is other_row_before


def test_bad_target_grid_turns_dot_red_with_message_tooltip(qtbot):
    config = _preset_config()
    config.populations[0].target_grid = "nonexistent"
    session = Session(config)
    strip = PipelineStrip(session)
    qtbot.addWidget(strip)

    chip = strip._pop_rows[0].chips_by_stage["sensor_array"][0]
    assert "nonexistent" in chip.toolTip()


def test_clicking_a_chip_emits_chip_clicked(qtbot):
    session = Session(_preset_config())
    strip = PipelineStrip(session)
    qtbot.addWidget(strip)

    chip = strip._pop_rows[1].chips_by_stage["neuron"][0]
    with qtbot.waitSignal(strip.chipClicked, timeout=1000) as blocker:
        chip.click()
    assert blocker.args == ["neuron", 1]


def test_sensors_row_click_emits_sensors_stage(qtbot):
    session = Session(_preset_config())
    strip = PipelineStrip(session)
    qtbot.addWidget(strip)

    chip = strip._sensor_row.chips_by_stage["sensors"][0]
    with qtbot.waitSignal(strip.chipClicked, timeout=1000) as blocker:
        chip.click()
    assert blocker.args == ["sensors", 0]


def test_stale_label_follows_session_stale(qtbot):
    session = Session(_preset_config())
    strip = PipelineStrip(session)
    qtbot.addWidget(strip)

    assert strip._status_label.text() == "saved"
    session.notify("populations.0.filter_method")
    assert strip._status_label.text() == "● edited"

    session._set_stale(True)  # simulate results now stale
    assert strip._status_label.text() == "● edited since last run"


def test_rf_chip_shows_the_parameter_its_builder_uses(qtbot):
    from sensoryforge.config.schema import SensoryForgeConfig
    from sensoryforge.gui.session import Session
    from sensoryforge.gui.widgets.pipeline_strip import PipelineStrip

    config = SensoryForgeConfig.from_yaml_file(
        "sensoryforge/presets/tactile_sa1_ra1.yml"
    )
    session = Session(config)
    strip = PipelineStrip(session)
    qtbot.addWidget(strip)

    def rf_text():
        return strip._pop_rows[0].chips_by_stage["receptive_field"][0].text()

    assert rf_text().endswith("template d=0.4")
    # The preset's resolvable distance stays set; a gaussian ignores it.
    session.set_by_path("populations.0.innervation_method", "gaussian")
    session.set_by_path("populations.0.sigma_d_mm", 0.45)
    assert rf_text().endswith("gaussian σ=0.45")


def test_sensor_array_chip_names_the_grid_explicit_inputs_read(qtbot):
    from sensoryforge.config.schema import SensoryForgeConfig
    from sensoryforge.gui.session import Session
    from sensoryforge.gui.widgets.pipeline_strip import PipelineStrip

    data = SensoryForgeConfig.from_yaml_file(
        "sensoryforge/presets/tactile_sa1_ra1.yml"
    ).to_dict()
    population = data["populations"][0]
    for key in (
        "target_grid",
        "innervation_method",
        "resolvable_distance_mm",
        "innervation_params",
    ):
        population.pop(key, None)
    population["inputs"] = [
        {"grid": "Main Grid", "rf": {"method": "gaussian", "params": {}}},
        {"grid": "Main Grid", "rf": {"method": "gaussian", "params": {}}},
    ]
    strip = PipelineStrip(Session(SensoryForgeConfig.from_dict(data)))
    qtbot.addWidget(strip)
    text = strip._pop_rows[0].chips_by_stage["sensor_array"][0].text()
    assert "no grid" not in text and "80×80" in text
