"""Unit tests for :class:`sensoryforge.core.rf_bank.ReceptiveFieldBank` (I2)."""

import pytest
import torch

import sensoryforge
from sensoryforge.core.rf_bank import ReceptiveFieldBank


def _bank(n=3, m=4, provenance=None):
    weights = torch.arange(n * m, dtype=torch.float32).reshape(n, m) / 10.0
    centers = torch.tensor([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]][:n])
    coords = torch.stack([torch.arange(m, dtype=torch.float32), torch.zeros(m)], dim=1)
    return ReceptiveFieldBank(
        weights, centers, coords, provenance=provenance or {"builder": "test"}
    )


class TestConstruction:
    def test_buffers_and_counts(self):
        bank = _bank()
        assert bank.num_neurons == 3
        assert bank.num_receptors == 4
        assert bank.weights.dtype == torch.float32
        assert tuple(bank.weights.shape) == (3, 4)
        assert tuple(bank.neuron_centers.shape) == (3, 2)
        assert tuple(bank.receptor_coords.shape) == (4, 2)
        names = {name for name, _ in bank.named_buffers()}
        assert {"weights", "neuron_centers", "receptor_coords"} <= names

    def test_provenance_records_version_and_builder(self):
        bank = _bank(provenance={"builder": "gaussian", "seed": 3})
        assert bank.provenance["builder"] == "gaussian"
        assert bank.provenance["seed"] == 3
        assert bank.provenance["sensoryforge_version"] == sensoryforge.__version__

    def test_weights_cast_to_float32(self):
        bank = ReceptiveFieldBank(
            torch.ones(2, 3, dtype=torch.float64),
            torch.zeros(2, 2),
            torch.zeros(3, 2),
        )
        assert bank.weights.dtype == torch.float32

    def test_bad_weight_ndim_raises(self):
        with pytest.raises(ValueError, match=r"\[N, M\]"):
            ReceptiveFieldBank(
                torch.ones(2, 3, 4), torch.zeros(2, 2), torch.zeros(3, 2)
            )

    def test_centers_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="neuron_centers"):
            ReceptiveFieldBank(torch.ones(2, 3), torch.zeros(5, 2), torch.zeros(3, 2))

    def test_receptor_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="receptor_coords"):
            ReceptiveFieldBank(torch.ones(2, 3), torch.zeros(2, 2), torch.zeros(7, 2))

    def test_centers_not_xy_raises(self):
        with pytest.raises(ValueError, match=r"\[N, 2\]"):
            ReceptiveFieldBank(torch.ones(2, 3), torch.zeros(2, 3), torch.zeros(3, 2))


class TestForward:
    def test_batch_only_matches_matmul(self):
        bank = _bank()
        x = torch.randn(5, 4)
        out = bank(x)
        assert tuple(out.shape) == (5, 3)
        assert torch.allclose(out, x @ bank.weights.T)

    def test_batch_time_matches_matmul(self):
        bank = _bank()
        x = torch.randn(2, 7, 4)
        out = bank(x)
        assert tuple(out.shape) == (2, 7, 3)
        assert torch.allclose(out, x @ bank.weights.T)

    def test_hand_computed_values(self):
        weights = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        bank = ReceptiveFieldBank(weights, torch.zeros(2, 2), torch.zeros(3, 2))
        x = torch.tensor([[1.0, 10.0, 100.0]])
        out = bank(x)
        assert torch.equal(out, torch.tensor([[201.0, 30.0]]))

    def test_wrong_receptor_count_raises_with_shapes(self):
        bank = _bank()
        with pytest.raises(ValueError, match=r"4.*\[2, 5\]|\[2, 5\].*4"):
            bank(torch.randn(2, 5))

    def test_wrong_ndim_raises(self):
        bank = _bank()
        with pytest.raises(ValueError, match=r"\[batch, M\]"):
            bank(torch.randn(4))
        with pytest.raises(ValueError, match=r"\[batch, time, M\]"):
            bank(torch.randn(1, 1, 1, 4))


class TestSaveLoad:
    def test_round_trip_is_bit_identical(self, tmp_path):
        bank = _bank(provenance={"builder": "gaussian", "seed": 7, "params": {"k": 1}})
        path = tmp_path / "bank.pt"
        bank.save(path)
        raw = torch.load(path, weights_only=False)
        assert set(raw) >= {
            "innervation_weights",
            "neuron_centers",
            "receptor_coords",
            "provenance",
        }
        loaded = ReceptiveFieldBank.load(path)
        assert torch.equal(loaded.weights, bank.weights)
        assert torch.equal(loaded.neuron_centers, bank.neuron_centers)
        assert torch.equal(loaded.receptor_coords, bank.receptor_coords)
        assert loaded.provenance == bank.provenance

    def test_load_pressure_simulation_style_nhw(self, tmp_path):
        # pressure-simulation population files: innervation_weights [N, H, W]
        # and neuron_centers [N, 2]; no receptor_coords, so they are required.
        n, h, w = 2, 3, 4
        weights = torch.arange(n * h * w, dtype=torch.float32).reshape(n, h, w)
        centers = torch.zeros(n, 2)
        path = tmp_path / "population_01_SA.pt"
        torch.save({"innervation_weights": weights, "neuron_centers": centers}, path)
        coords = torch.zeros(h * w, 2)
        bank = ReceptiveFieldBank.load(path, receptor_coords=coords)
        assert tuple(bank.weights.shape) == (n, h * w)
        assert torch.equal(bank.weights, weights.reshape(n, h * w))
        assert bank.provenance == {"sensoryforge_version": sensoryforge.__version__}

    @pytest.mark.parametrize("key", ["weights", "W"])
    def test_load_accepts_alternate_weight_keys(self, tmp_path, key):
        weights = torch.rand(2, 6)
        path = tmp_path / "pop.pt"
        torch.save({key: weights, "neuron_centers": torch.zeros(2, 2)}, path)
        bank = ReceptiveFieldBank.load(path, receptor_coords=torch.zeros(6, 2))
        assert torch.equal(bank.weights, weights)

    def test_load_without_receptor_coords_raises(self, tmp_path):
        path = tmp_path / "pop.pt"
        torch.save(
            {
                "innervation_weights": torch.rand(2, 6),
                "neuron_centers": torch.zeros(2, 2),
            },
            path,
        )
        with pytest.raises(ValueError, match="receptor_coords"):
            ReceptiveFieldBank.load(path)

    def test_load_missing_weights_raises(self, tmp_path):
        path = tmp_path / "pop.pt"
        torch.save({"neuron_centers": torch.zeros(2, 2)}, path)
        with pytest.raises(ValueError, match="innervation_weights"):
            ReceptiveFieldBank.load(path, receptor_coords=torch.zeros(6, 2))

    def test_load_receptor_count_mismatch_raises(self, tmp_path):
        path = tmp_path / "pop.pt"
        torch.save(
            {
                "innervation_weights": torch.rand(2, 6),
                "neuron_centers": torch.zeros(2, 2),
            },
            path,
        )
        with pytest.raises(ValueError, match="receptor_coords"):
            ReceptiveFieldBank.load(path, receptor_coords=torch.zeros(5, 2))


class TestDevice:
    def test_to_moves_all_buffers(self):
        bank = _bank().to("cpu")
        assert bank.weights.device.type == "cpu"
        out = bank(torch.randn(2, 4))
        assert out.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_to_mps_and_back_is_lossless(self):
        bank = _bank()
        moved = _bank().to("mps")
        assert moved.weights.device.type == "mps"
        assert moved.neuron_centers.device.type == "mps"
        assert moved.receptor_coords.device.type == "mps"
        back = moved.to("cpu")
        assert torch.equal(back.weights, bank.weights)
