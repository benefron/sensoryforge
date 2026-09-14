"""Unit tests for canonical configuration schema."""

import pytest
import yaml
from sensoryforge.config.schema import (
    SensoryForgeConfig,
    GridConfig,
    PopulationConfig,
    StimulusConfig,
    SimulationConfig,
)


class TestGridConfig:
    """Test GridConfig dataclass."""

    def test_from_dict(self):
        """Test creating GridConfig from dictionary."""
        config_dict = {
            "name": "Test Grid",
            "arrangement": "grid",
            "rows": 40,
            "cols": 40,
            "spacing": 0.15,
            "center_x": 0.0,
            "center_y": 0.0,
        }

        grid = GridConfig.from_dict(config_dict)
        assert grid.name == "Test Grid"
        assert grid.arrangement == "grid"
        assert grid.rows == 40
        assert grid.cols == 40
        assert grid.spacing == 0.15

    def test_to_dict(self):
        """Test serializing GridConfig to dictionary."""
        grid = GridConfig(
            name="Test Grid",
            arrangement="grid",
            rows=40,
            cols=40,
            spacing=0.15,
        )

        config_dict = grid.to_dict()
        assert config_dict["name"] == "Test Grid"
        assert config_dict["arrangement"] == "grid"
        assert config_dict["rows"] == 40


class TestPopulationConfig:
    """Test PopulationConfig dataclass."""

    def test_from_dict(self):
        """Test creating PopulationConfig from dictionary."""
        config_dict = {
            "name": "SA Population",
            "neuron_type": "SA",
            "neuron_model": "izhikevich",
            "filter_method": "sa",
            "innervation_method": "gaussian",
            "neurons_per_row": 10,
            "connections_per_neuron": 28,
            "sigma_d_mm": 0.3,
        }

        pop = PopulationConfig.from_dict(config_dict)
        assert pop.name == "SA Population"
        assert pop.neuron_type == "SA"
        assert pop.neuron_model == "izhikevich"
        assert pop.filter_method == "sa"
        assert pop.innervation_method == "gaussian"

    def test_to_dict(self):
        """Test serializing PopulationConfig to dictionary."""
        pop = PopulationConfig(
            name="SA Population",
            neuron_type="SA",
            neuron_model="izhikevich",
            filter_method="sa",
            innervation_method="gaussian",
            neurons_per_row=10,
            connections_per_neuron=28,
            sigma_d_mm=0.3,
        )

        config_dict = pop.to_dict()
        assert config_dict["name"] == "SA Population"
        assert config_dict["neuron_type"] == "SA"
        assert config_dict["neuron_model"] == "izhikevich"


class TestSensoryForgeConfig:
    """Test SensoryForgeConfig round-trip."""

    def test_config_round_trip(self):
        """Test saving and loading config maintains all fields."""
        # Create config
        config = SensoryForgeConfig(
            grids=[
                GridConfig(
                    name="Grid 1",
                    arrangement="grid",
                    rows=40,
                    cols=40,
                    spacing=0.15,
                )
            ],
            populations=[
                PopulationConfig(
                    name="SA Population",
                    neuron_type="SA",
                    neuron_model="izhikevich",
                    filter_method="sa",
                    innervation_method="gaussian",
                    neurons_per_row=10,
                    connections_per_neuron=28,
                    sigma_d_mm=0.3,
                )
            ],
            stimulus=StimulusConfig(
                type="gaussian",
                amplitude=10.0,
                sigma=1.0,
            ),
            simulation=SimulationConfig(
                device="cpu",
                dt=1.0,
            ),
        )

        # Serialize to dict
        config_dict = config.to_dict()

        # Deserialize from dict
        config2 = SensoryForgeConfig.from_dict(config_dict)

        # Verify round-trip
        assert len(config2.grids) == len(config.grids)
        assert config2.grids[0].name == config.grids[0].name
        assert len(config2.populations) == len(config.populations)
        assert config2.populations[0].name == config.populations[0].name
        assert config2.stimulus.type == config.stimulus.type
        assert config2.simulation.device == config.simulation.device

    def test_yaml_round_trip(self):
        """Test saving and loading config via YAML."""
        config = SensoryForgeConfig(
            grids=[
                GridConfig(
                    name="Grid 1",
                    arrangement="grid",
                    rows=40,
                    cols=40,
                    spacing=0.15,
                )
            ],
            populations=[
                PopulationConfig(
                    name="SA Population",
                    neuron_type="SA",
                    neuron_model="izhikevich",
                    filter_method="sa",
                    innervation_method="gaussian",
                    neurons_per_row=10,
                )
            ],
            stimulus=StimulusConfig(
                type="gaussian",
                amplitude=10.0,
            ),
            simulation=SimulationConfig(
                device="cpu",
                dt=1.0,
            ),
        )

        # Serialize to YAML
        yaml_str = config.to_yaml()

        # Deserialize from YAML
        config2 = SensoryForgeConfig.from_yaml(yaml_str)

        # Verify round-trip
        assert len(config2.grids) == len(config.grids)
        assert config2.grids[0].name == config.grids[0].name
        assert len(config2.populations) == len(config.populations)
        assert config2.populations[0].name == config.populations[0].name


class TestFromYaml:
    """Regression for F-027: from_yaml must not raise OSError on long inline text.

    ``from_yaml`` used to probe ``Path(yaml_str_or_path).is_file()`` before
    parsing. A one-line YAML/JSON string longer than the filesystem's max
    path length (typically 255 bytes) raised ``OSError: File name too long``
    instead of being parsed as YAML.
    """

    def _minimal_yaml(self, padding: int = 0) -> str:
        # Single line (no "\n") so the old code took the is_file() branch.
        pad = " " * padding
        return "{grids: [], populations: [], stimulus: {}, simulation: {}}" + pad

    def test_long_one_line_string_over_255_bytes(self):
        yaml_str = self._minimal_yaml(padding=300)
        assert len(yaml_str) > 255
        assert "\n" not in yaml_str
        config = SensoryForgeConfig.from_yaml(yaml_str)
        assert config.grids == []

    def test_multiline_yaml_string(self):
        yaml_str = "grids: []\npopulations: []\nstimulus: {}\nsimulation: {}\n"
        config = SensoryForgeConfig.from_yaml(yaml_str)
        assert config.grids == []

    def test_str_path(self, tmp_path):
        p = tmp_path / "config.yml"
        p.write_text("grids: []\npopulations: []\nstimulus: {}\nsimulation: {}\n")
        config = SensoryForgeConfig.from_yaml(str(p))
        assert config.grids == []

    def test_path_object(self, tmp_path):
        p = tmp_path / "config.yml"
        p.write_text("grids: []\npopulations: []\nstimulus: {}\nsimulation: {}\n")
        config = SensoryForgeConfig.from_yaml(p)
        assert config.grids == []

    def test_from_yaml_file(self, tmp_path):
        p = tmp_path / "config.yml"
        p.write_text("grids: []\npopulations: []\nstimulus: {}\nsimulation: {}\n")
        config = SensoryForgeConfig.from_yaml_file(p)
        assert config.grids == []
