"""Tests for :mod:`sensoryforge.gui.screens.sensors`.

Drives the real widgets (buttons, spin boxes, combo boxes, line edits) and
asserts on ``session.config`` and on the preview's actual data (receptor
counts from :func:`sensoryforge.core.simulation_engine.build_grid`), not
merely that the screen constructs.
"""

import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    PopulationInput,
    SensoryForgeConfig,
    grid_config_param_specs,
)
from sensoryforge.core.simulation_engine import build_grid  # noqa: E402
from sensoryforge.gui.screens.sensors import (  # noqa: E402
    SensorsScreen,
    _grids_in_use,
    _unique_name,
)
from sensoryforge.gui.session import Session  # noqa: E402

_DEBOUNCE_WAIT_MS = 400


def _wait_debounce(qtbot) -> None:
    qtbot.wait(_DEBOUNCE_WAIT_MS)


def _basic_config() -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[GridConfig(name="skin", rows=10, cols=10, spacing=0.15)],
        populations=[
            PopulationConfig(name="SA", neuron_type="SA", target_grid="skin"),
        ],
    )


@pytest.fixture
def session(qtbot) -> Session:
    return Session(_basic_config())


@pytest.fixture
def screen(qtbot, session) -> SensorsScreen:
    widget = SensorsScreen(session)
    qtbot.addWidget(widget)
    widget.show()
    return widget


class TestGridConfigParamSpecs:
    """The gap-closing GridConfig-shaped param specs (not GridArrangement's)."""

    def test_every_spec_name_is_a_real_field(self):
        for spec in grid_config_param_specs():
            assert spec.name in GridConfig.__dataclass_fields__

    def test_every_default_equals_the_dataclass_default(self):
        for spec in grid_config_param_specs():
            field = GridConfig.__dataclass_fields__[spec.name]
            assert spec.default == field.default, spec.name

    def test_center_and_seed_are_advanced(self):
        by_name = {s.name: s for s in grid_config_param_specs()}
        assert by_name["center_x"].advanced
        assert by_name["center_y"].advanced
        assert by_name["seed"].advanced

    def test_arrangement_has_the_registered_choices(self):
        by_name = {s.name: s for s in grid_config_param_specs()}
        assert set(by_name["arrangement"].choices) >= {
            "grid",
            "hex",
            "poisson",
            "jittered_grid",
            "blue_noise",
            "composite",
        }


class TestAddDuplicateRemove:
    def test_add_appends_unique_grid_and_selects_it(self, qtbot, screen, session):
        assert len(session.config.grids) == 1

        screen.btn_add.click()

        assert len(session.config.grids) == 2
        new_grid = session.config.grids[1]
        assert new_grid.name not in ("skin",)
        assert new_grid.name  # non-empty, unique by construction
        assert screen._selected_index == 1
        assert screen.name_edit.text() == new_grid.name

    def test_duplicate_inserts_after_selection_with_unique_name(
        self, qtbot, screen, session
    ):
        screen.grid_list.setCurrentRow(0)
        screen.btn_duplicate.click()

        assert len(session.config.grids) == 2
        assert session.config.grids[0].name == "skin"
        dup = session.config.grids[1]
        assert dup.name != "skin"
        assert dup.rows == session.config.grids[0].rows
        assert screen._selected_index == 1

    def test_remove_disabled_for_last_grid(self, qtbot, screen, session):
        assert len(session.config.grids) == 1
        assert not screen.btn_remove.isEnabled()

    def test_remove_refuses_while_population_targets_grid_then_works_after_retarget(
        self, qtbot, screen, session
    ):
        screen.btn_add.click()  # now two grids: "skin" (targeted) and a new one
        screen.grid_list.setCurrentRow(0)

        screen.btn_remove.click()

        assert len(session.config.grids) == 2
        assert screen.list_message.isVisible()
        assert "SA" in screen.list_message.text()

        session.config.populations[0].target_grid = session.config.grids[1].name
        screen.btn_remove.click()

        assert len(session.config.grids) == 1
        assert session.config.grids[0].name != "skin"

    def test_remove_refuses_for_multi_input_population(self, qtbot, session):
        session.config.populations[0].target_grid = None
        session.config.populations[0].inputs = [PopulationInput(grid="skin")]
        session.config.grids.append(GridConfig(name="extra"))
        widget = SensorsScreen(session)
        qtbot.addWidget(widget)
        widget.show()
        widget.grid_list.setCurrentRow(0)

        widget.btn_remove.click()

        assert len(session.config.grids) == 2
        assert widget.list_message.isVisible()


class TestRowsEditing:
    def test_editing_rows_updates_config_and_preview_after_debounce(
        self, qtbot, screen, session
    ):
        rows_widget = screen._param_form.widget_for("rows")
        cols_widget = screen._param_form.widget_for("cols")
        assert cols_widget.value() == 10

        rows_widget.setValue(15)
        rows_widget.editingFinished.emit()

        assert session.config.grids[0].rows == 15

        _wait_debounce(qtbot)

        assert screen.preview.receptor_count() == 15 * 10


class TestArrangementSwitch:
    def test_hex_is_sized_by_rows_and_cols_which_stay_editable(
        self, qtbot, screen, session
    ):
        arrangement_widget = screen._param_form.widget_for("arrangement")
        arrangement_widget.setCurrentIndex(arrangement_widget.findData("hex"))
        assert session.config.grids[0].arrangement == "hex"
        _wait_debounce(qtbot)
        before = screen.preview.receptor_count()

        rows_widget = screen._param_form.widget_for("rows")
        assert rows_widget.isEnabled()
        assert screen._param_form.widget_for("cols").isEnabled()
        rows_widget.setValue(12)
        rows_widget.editingFinished.emit()
        _wait_debounce(qtbot)

        expected = build_grid(
            session.config.grids[0], device="cpu"
        ).get_all_coordinates()
        assert screen.preview.receptor_count() == expected.shape[0]
        assert screen.preview.receptor_count() != before

    def test_density_disabled_for_grid_and_jittered_grid_enabled_otherwise(
        self, qtbot, screen, session
    ):
        # D-88b4b41 / F-081: density sizes poisson/hex/blue_noise but is an
        # error on grid/jittered_grid, where spacing already fixes the count.
        arrangement_widget = screen._param_form.widget_for("arrangement")
        density_widget = screen._param_form.widget_for("density")

        for name in ("grid", "jittered_grid"):
            arrangement_widget.setCurrentIndex(arrangement_widget.findData(name))
            assert not density_widget.isEnabled(), name

        for name in ("hex", "poisson", "blue_noise"):
            arrangement_widget.setCurrentIndex(arrangement_widget.findData(name))
            assert density_widget.isEnabled(), name

    def test_density_control_writes_through_and_scales_receptor_count(
        self, qtbot, screen, session
    ):
        arrangement_widget = screen._param_form.widget_for("arrangement")
        arrangement_widget.setCurrentIndex(arrangement_widget.findData("poisson"))
        _wait_debounce(qtbot)

        density_widget = screen._param_form.widget_for("density")
        assert density_widget.isEnabled()

        density_widget.setValue(5.0)
        density_widget.editingFinished.emit()
        assert session.config.grids[0].density == 5.0
        _wait_debounce(qtbot)
        low_count = screen.preview.receptor_count()

        density_widget.setValue(50.0)
        density_widget.editingFinished.emit()
        assert session.config.grids[0].density == 50.0
        _wait_debounce(qtbot)
        high_count = screen.preview.receptor_count()

        expected = build_grid(
            session.config.grids[0], device="cpu"
        ).get_all_coordinates()
        assert screen.preview.receptor_count() == expected.shape[0]
        assert high_count > low_count

    @pytest.mark.parametrize(
        "arrangement", ["grid", "hex", "poisson", "jittered_grid", "blue_noise"]
    )
    def test_every_enabled_geometry_control_changes_the_array(self, arrangement):
        import torch

        def coords(**kw):
            cfg = GridConfig(name="g", arrangement=arrangement, **kw)
            return build_grid(cfg, device="cpu").get_all_coordinates()

        base = dict(rows=10, cols=10, spacing=0.2, seed=1)
        ref = coords(**base)
        for name, value in (("rows", 12), ("cols", 12), ("spacing", 0.3)):
            out = coords(**{**base, name: value})
            assert out.shape != ref.shape or not torch.equal(out, ref), name


class TestChannels:
    def test_invalid_channels_shows_error_and_leaves_config_unchanged(
        self, qtbot, screen, session
    ):
        original = list(session.config.grids[0].channels)

        screen.channels_edit.setText("1bad, ok")
        screen.channels_edit.editingFinished.emit()

        assert screen.channels_error.isVisible()
        assert session.config.grids[0].channels == original

    def test_valid_channels_write_through(self, qtbot, screen, session):
        screen.channels_edit.setText("pressure, shear")
        screen.channels_edit.editingFinished.emit()

        assert not screen.channels_error.isVisible()
        assert session.config.grids[0].channels == ["pressure", "shear"]


class TestReplaceConfig:
    def test_replace_config_with_two_grids_rebuilds_list(self, qtbot, screen, session):
        new_config = SensoryForgeConfig(
            grids=[
                GridConfig(name="a", rows=5, cols=5),
                GridConfig(name="b", rows=6, cols=6),
            ]
        )

        session.replace_config(new_config)

        assert screen.grid_list.count() == 2
        assert screen._selected_index == 0
        _wait_debounce(qtbot)
        assert screen.preview.receptor_count() == 25 + 36


class TestNameEditing:
    def test_rename_updates_config_and_list_item(self, qtbot, screen, session):
        screen.name_edit.setText("dermis")
        screen.name_edit.editingFinished.emit()

        assert session.config.grids[0].name == "dermis"
        assert "dermis" in screen.grid_list.item(0).text()

    def test_duplicate_name_refused(self, qtbot, screen, session):
        screen.btn_add.click()  # second grid, auto-named
        second_index = screen._selected_index
        screen.name_edit.setText("skin")
        screen.name_edit.editingFinished.emit()

        assert session.config.grids[second_index].name != "skin"
        assert screen.list_message.isVisible()


class TestHelpers:
    def test_unique_name_increments(self):
        assert _unique_name("grid", []) == "grid"
        assert _unique_name("grid", ["grid"]) == "grid_2"
        assert _unique_name("grid", ["grid", "grid_2"]) == "grid_3"

    def test_grids_in_use_single_and_multi_input(self):
        config = SensoryForgeConfig(
            grids=[GridConfig(name="skin"), GridConfig(name="deep")],
            populations=[
                PopulationConfig(name="SA", target_grid="skin"),
                PopulationConfig(
                    name="Combo",
                    target_grid=None,
                    inputs=[PopulationInput(grid="skin"), PopulationInput(grid="deep")],
                ),
            ],
        )
        used = _grids_in_use(config)
        assert set(used["skin"]) == {"SA", "Combo"}
        assert set(used["deep"]) == {"Combo"}


class TestInvalidGridShowsErrorInCaption:
    def test_bad_coords_file_shows_error_and_keeps_last_preview(
        self, qtbot, screen, session
    ):
        _wait_debounce(qtbot)
        good_caption = screen.caption.text()
        assert "10" in good_caption or "receptors" in good_caption

        session.set_by_path("grids.0.coords_file", "/no/such/file.csv")
        _wait_debounce(qtbot)

        assert (
            "error" in screen.caption.styleSheet()
            or screen.caption.text() != good_caption
        )
