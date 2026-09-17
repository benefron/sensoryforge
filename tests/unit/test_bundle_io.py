"""Unit tests for :mod:`sensoryforge.io.bundle` (Wave J, J1)."""

import json

import h5py
import pytest
import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.io.bundle import load_bundle, write_bundle


def _small_config():
    return SensoryForgeConfig(
        grids=[
            GridConfig(name="Main", arrangement="grid", rows=8, cols=8, spacing=0.15)
        ],
        populations=[
            PopulationConfig(
                name="SA Population",
                neuron_type="SA",
                neuron_model="izhikevich",
                filter_method="sa",
                innervation_method="gaussian",
                neurons_per_row=3,
                seed=7,
            ),
            PopulationConfig(
                name="RA Population",
                neuron_type="RA",
                neuron_model="izhikevich",
                filter_method="ra",
                innervation_method="gaussian",
                neurons_per_row=4,
                seed=7,
            ),
        ],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0),
    )


def _run(config, T=20):
    engine = SimulationEngine(config)
    stimulus = torch.rand(1, T, 8, 8)
    results = engine.run(stimulus, return_intermediates=True)
    return engine, stimulus, results


class TestWriteBundleLayout:
    def test_writes_expected_files(self, tmp_path):
        config = _small_config()
        engine, stimulus, results = _run(config)
        bundle_dir = write_bundle(
            tmp_path / "bundle", config, engine, results, stimulus, seed=7
        )
        assert (bundle_dir / "config.json").exists()
        assert (bundle_dir / "data.h5").exists()
        assert (bundle_dir / "stimuli" / "stimulus.json").exists()
        assert (bundle_dir / "population_01_SA_Population.pt").exists()
        assert (bundle_dir / "population_02_RA_Population.pt").exists()

    def test_config_json_fields(self, tmp_path):
        config = _small_config()
        engine, stimulus, results = _run(config)
        bundle_dir = write_bundle(
            tmp_path / "bundle", config, engine, results, stimulus
        )
        with open(bundle_dir / "config.json") as f:
            cfg = json.load(f)
        assert cfg["schema_version"] == "2.0.0"
        assert cfg["kind"] == "sensoryforge_bundle"
        assert cfg["grid"] == {
            "rows": 8,
            "cols": 8,
            "spacing_mm": 0.15,
            "center_mm": [0.0, 0.0],
            "device": "cpu",
        }
        assert len(cfg["populations"]) == 2
        pop0 = cfg["populations"][0]
        assert pop0["name"] == "SA Population"
        assert pop0["neuron_type"] == "SA"
        assert pop0["tensors"] == "population_01_SA_Population.pt"
        assert set(pop0["parameters"]) == {
            "neurons_per_row",
            "connections_per_neuron",
            "sigma_d_mm",
            "weight_min",
            "weight_max",
            "seed",
            "edge_offset",
        }
        assert "config" in cfg and "sensoryforge_version" in cfg

    def test_h5_dataset_names_shapes_dtypes(self, tmp_path):
        config = _small_config()
        engine, stimulus, results = _run(config, T=15)
        bundle_dir = write_bundle(
            tmp_path / "bundle", config, engine, results, stimulus
        )
        with h5py.File(bundle_dir / "data.h5", "r") as f:
            assert f["stimulus"]["frames"].shape == (15, 8, 8)
            assert f["time_ms"].shape == (15,)
            assert f.attrs["dt_ms"] == 1.0
            assert f.attrs["integrate_dt_ms"] == config.simulation.integrate_dt_ms
            for name in ("SA Population", "RA Population"):
                grp = f["populations"][name]
                n_neurons = grp["drive"].shape[-1]
                assert grp["drive"].shape == (15, n_neurons)
                assert grp["filtered"].shape == (15, n_neurons)
                assert grp["spikes"].shape == (15, n_neurons)
                assert grp["spikes"].dtype == "int16"
            assert "config_yaml" in f["meta"].attrs
            assert "provenance_json" in f["meta"].attrs


class TestRoundTrip:
    def test_two_population_round_trip_bit_identical(self, tmp_path):
        config = _small_config()
        engine, stimulus, results = _run(config, T=12)
        write_bundle(tmp_path / "bundle", config, engine, results, stimulus, seed=7)

        bundle = load_bundle(tmp_path / "bundle")

        assert bundle.config.simulation.dt_ms == config.simulation.dt_ms
        assert set(bundle.banks) == {"SA Population", "RA Population"}

        for pop in engine.populations:
            name = pop["name"]
            bank = pop["bank"]
            loaded_bank = bundle.banks[name]
            assert torch.equal(bank.weights.cpu(), loaded_bank.weights)
            assert torch.equal(bank.neuron_centers.cpu(), loaded_bank.neuron_centers)
            assert torch.equal(bank.receptor_coords.cpu(), loaded_bank.receptor_coords)

        assert bundle.stimulus is not None
        assert torch.equal(bundle.stimulus, stimulus[0].cpu())
        assert bundle.time_ms is not None
        assert bundle.time_ms.shape == (12,)

        for name, pop_results in results.items():
            loaded = bundle.populations[name]
            assert torch.equal(loaded["drive"], pop_results["drive"][0].cpu())
            assert torch.equal(loaded["filtered"], pop_results["filtered"][0].cpu())
            assert torch.equal(
                loaded["spikes"], pop_results["spikes"][0].cpu().to(torch.int16)
            )

        assert bundle.meta["seed"] == 7

    def test_channel_stimulus_round_trips(self, tmp_path):
        config = _small_config()
        engine, _, results = _run(config, T=10)
        stimulus_chw = torch.rand(1, 10, 2, 8, 8)
        write_bundle(tmp_path / "bundle", config, engine, results, stimulus_chw)
        bundle = load_bundle(tmp_path / "bundle")
        assert bundle.stimulus.shape == (10, 2, 8, 8)
        assert torch.equal(bundle.stimulus, stimulus_chw[0].cpu())


class TestSchemaVersionValidation:
    def test_missing_schema_version_raises(self, tmp_path):
        config = _small_config()
        engine, stimulus, results = _run(config)
        bundle_dir = write_bundle(
            tmp_path / "bundle", config, engine, results, stimulus
        )
        cfg_path = bundle_dir / "config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        del cfg["schema_version"]
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        with pytest.raises(ValueError, match="schema_version"):
            load_bundle(bundle_dir)

    def test_incompatible_major_version_raises(self, tmp_path):
        config = _small_config()
        engine, stimulus, results = _run(config)
        bundle_dir = write_bundle(
            tmp_path / "bundle", config, engine, results, stimulus
        )
        cfg_path = bundle_dir / "config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["schema_version"] = "1.0.0"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        with pytest.raises(ValueError, match="schema_version"):
            load_bundle(bundle_dir)


class TestOverwrite:
    def test_non_empty_dir_without_overwrite_raises(self, tmp_path):
        config = _small_config()
        engine, stimulus, results = _run(config)
        bundle_dir = tmp_path / "bundle"
        write_bundle(bundle_dir, config, engine, results, stimulus)
        with pytest.raises(FileExistsError):
            write_bundle(bundle_dir, config, engine, results, stimulus)
        # overwrite=True succeeds
        write_bundle(bundle_dir, config, engine, results, stimulus, overwrite=True)


class TestMissingIntermediates:
    def test_missing_drive_raises_value_error(self, tmp_path):
        config = _small_config()
        engine, stimulus, results = _run(config)
        bad_results = {
            name: {k: v for k, v in r.items() if k != "drive"}
            for name, r in results.items()
        }
        with pytest.raises(ValueError, match="drive"):
            write_bundle(tmp_path / "bundle", config, engine, bad_results, stimulus)
