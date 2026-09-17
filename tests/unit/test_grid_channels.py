"""GridConfig channels and coords_file (Phase 2, Wave L1)."""

from __future__ import annotations

import pytest

from sensoryforge.config.schema import GridConfig


class TestGridConfigChannels:
    def test_default_channels(self):
        cfg = GridConfig(name="g")
        assert cfg.channels == ["value"]
        assert cfg.coords_file is None

    def test_default_channels_omitted_from_to_dict(self):
        cfg = GridConfig(name="g")
        d = cfg.to_dict()
        assert "channels" not in d
        assert "coords_file" not in d
        assert "layers" not in d

    def test_nondefault_channels_kept_in_to_dict(self):
        cfg = GridConfig(name="g", channels=["pressure", "temperature"])
        d = cfg.to_dict()
        assert d["channels"] == ["pressure", "temperature"]

    def test_round_trip(self):
        cfg = GridConfig(name="g", channels=["a", "b"], coords_file="/tmp/coords.csv")
        d = cfg.to_dict()
        cfg2 = GridConfig.from_dict(d)
        assert cfg2.channels == ["a", "b"]
        assert cfg2.coords_file == "/tmp/coords.csv"

    def test_empty_channel_name_raises(self):
        with pytest.raises(ValueError, match="g"):
            GridConfig(name="g", channels=["a", ""])

    def test_duplicate_channel_name_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            GridConfig(name="g", channels=["a", "a"])

    def test_invalid_identifier_raises(self):
        with pytest.raises(ValueError, match="identifier"):
            GridConfig(name="g", channels=["not valid"])

    def test_byte_identical_default_grid_config(self):
        """A pre-Wave-L GridConfig's to_dict() is unaffected by the new fields."""
        cfg = GridConfig(name="Main Grid", arrangement="grid", rows=80, cols=80)
        d = cfg.to_dict()
        expected_keys = {
            "name",
            "arrangement",
            "rows",
            "cols",
            "spacing",
            "density",
            "center_x",
            "center_y",
            "color",
            "visible",
            "seed",
        }
        assert set(d.keys()) == expected_keys


class TestStimulusConfigChannel:
    def test_default_channel_is_none_and_omitted(self):
        from sensoryforge.config.schema import StimulusConfig

        cfg = StimulusConfig()
        assert cfg.channel is None
        assert "channel" not in cfg.to_dict()

    def test_named_channel_round_trips(self):
        from sensoryforge.config.schema import StimulusConfig

        cfg = StimulusConfig(channel="pressure")
        d = cfg.to_dict()
        assert d["channel"] == "pressure"
        cfg2 = StimulusConfig.from_dict(d)
        assert cfg2.channel == "pressure"


class TestPopulationConfigTargetLayers:
    def test_default_none_omitted(self):
        from sensoryforge.config.schema import PopulationConfig

        cfg = PopulationConfig(name="p")
        assert cfg.target_layers is None
        assert "target_layers" not in cfg.to_dict()

    def test_round_trips(self):
        from sensoryforge.config.schema import PopulationConfig

        cfg = PopulationConfig(name="p", target_layers=["a", "b"])
        d = cfg.to_dict()
        cfg2 = PopulationConfig.from_dict(d)
        assert cfg2.target_layers == ["a", "b"]
