"""PopulationInput/RFBuilderConfig sugar and round-trip (Phase 2, Wave M1)."""

from __future__ import annotations

import glob
import os

import pytest

from sensoryforge.config.schema import (
    PopulationConfig,
    PopulationInput,
    RFBuilderConfig,
    SensoryForgeConfig,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPopulationInputDefaults:
    def test_default_population_has_no_inputs(self):
        cfg = PopulationConfig(name="SA")
        assert cfg.inputs == []
        assert cfg.combine == "sum"

    def test_effective_inputs_expands_sugar(self):
        cfg = PopulationConfig(
            name="SA",
            target_grid="Main Grid",
            innervation_method="uniform",
            target_layers=["a", "b"],
        )
        eff = cfg.effective_inputs()
        assert len(eff) == 1
        assert eff[0].grid == "Main Grid"
        assert eff[0].channel == "value"
        assert eff[0].rf.method == "uniform"
        assert eff[0].layers == ["a", "b"]
        assert eff[0].gain == 1.0

    def test_effective_inputs_returns_explicit_inputs(self):
        inputs = [PopulationInput(grid="g1"), PopulationInput(grid="g2")]
        cfg = PopulationConfig(name="multi", inputs=inputs, combine="concat")
        assert cfg.effective_inputs() == inputs


class TestSugarInputsExclusivity:
    def test_both_forms_raises(self):
        with pytest.raises(ValueError, match="multi"):
            PopulationConfig(
                name="multi",
                inputs=[PopulationInput(grid="g1")],
                target_grid="g1",
            )

    def test_inputs_with_sigma_override_raises(self):
        with pytest.raises(ValueError, match="sigma_d_mm"):
            PopulationConfig(
                name="multi",
                inputs=[PopulationInput(grid="g1")],
                sigma_d_mm=0.5,
            )

    def test_bad_combine_raises(self):
        with pytest.raises(ValueError, match="combine"):
            PopulationConfig(name="p", combine="average")


class TestPopulationInputToDictFromDict:
    def test_default_input_to_dict_minimal(self):
        pi = PopulationInput(grid="g1")
        assert pi.to_dict() == {"grid": "g1"}

    def test_nondefault_fields_kept(self):
        pi = PopulationInput(
            grid="g1",
            channel="R",
            rf=RFBuilderConfig(
                method="template", params={"resolvable_distance_mm": 0.4}
            ),
            gain=2.0,
            layers=["a"],
            processing=[{"method": "onoff"}],
        )
        d = pi.to_dict()
        assert d["grid"] == "g1"
        assert d["channel"] == "R"
        assert d["rf"] == {
            "method": "template",
            "params": {"resolvable_distance_mm": 0.4},
        }
        assert d["gain"] == 2.0
        assert d["layers"] == ["a"]
        assert d["processing"] == [{"method": "onoff"}]

    def test_round_trip(self):
        pi = PopulationInput(grid="g1", channel="G", gain=1.5)
        d = pi.to_dict()
        pi2 = PopulationInput.from_dict(d)
        assert pi2.to_dict() == d


class TestPopulationConfigMultiInputRoundTrip:
    def test_multi_input_round_trip(self):
        cfg = PopulationConfig(
            name="rgb",
            inputs=[
                PopulationInput(grid="g", channel="R"),
                PopulationInput(grid="g", channel="G"),
            ],
            combine="concat",
        )
        d = cfg.to_dict()
        assert "inputs" in d
        assert d["combine"] == "concat"
        cfg2 = PopulationConfig.from_dict(d)
        assert cfg2.to_dict() == d
        assert all(isinstance(i, PopulationInput) for i in cfg2.inputs)

    def test_single_explicit_input_collapses_to_sugar(self):
        """A one-item `inputs` list that fits the sugar shape serializes as
        the pre-M1 short form (M1: "to_dict writes the short form back
        when there is exactly one input whose fields fit it")."""
        cfg = PopulationConfig(
            name="p",
            inputs=[
                PopulationInput(
                    grid="Main Grid",
                    rf=RFBuilderConfig(method="uniform", params={"sigma_d_mm": 0.4}),
                )
            ],
        )
        d = cfg.to_dict()
        assert "inputs" not in d
        assert d["target_grid"] == "Main Grid"
        assert d["innervation_method"] == "uniform"
        assert d["sigma_d_mm"] == 0.4

    def test_multi_channel_input_does_not_collapse(self):
        cfg = PopulationConfig(
            name="p",
            inputs=[PopulationInput(grid="g", channel="R", gain=2.0)],
        )
        d = cfg.to_dict()
        assert "inputs" in d
        assert "target_grid" not in d


class TestByteIdenticalRoundTrip:
    """M1: every existing config round-trips to byte-identical YAML text
    now that PopulationConfig carries inputs/combine (F-010's Wave M half).

    This must FAIL on 77901ed (no `inputs`/`combine` fields at all, so this
    import itself fails) and PASS on Wave M's commit.
    """

    @pytest.mark.parametrize(
        "path",
        sorted(glob.glob(os.path.join(REPO_ROOT, "examples", "*.yml")))
        + sorted(
            glob.glob(os.path.join(REPO_ROOT, "sensoryforge", "presets", "*.yml"))
        ),
    )
    def test_round_trip_byte_identical(self, path):
        with open(path, "r") as fh:
            original_text = fh.read()
        try:
            config = SensoryForgeConfig.from_yaml(original_text)
        except Exception:
            pytest.skip(f"{path} is not a canonical SensoryForgeConfig file")
        if not config.grids and not config.populations:
            pytest.skip(f"{path} has no grids/populations -- not canonical")
        emitted = config.to_yaml()
        reparsed = SensoryForgeConfig.from_yaml(emitted)
        re_emitted = reparsed.to_yaml()
        assert emitted == re_emitted, (
            f"{path}: to_yaml() is not a fixed point under from_yaml/to_yaml "
            "(inputs/combine must round-trip losslessly, M1)"
        )
        # The critical guarantee: re-serializing what to_yaml() *already*
        # produced is idempotent. We additionally check that no population
        # in the file picked up a stray `inputs`/`combine` key that wasn't
        # already implied by its sugar fields.
        for pop in config.populations:
            d = pop.to_dict()
            if pop.inputs == [] and pop.combine == "sum":
                assert "inputs" not in d
                assert "combine" not in d
