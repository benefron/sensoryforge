"""Sparse receptive-field weights in bank files and bundles (roadmap A11d).

pressure-simulation reads the same layout (its helper ``decoding/io/bank_weights.py``),
so the key names, dtypes and ordering here are a contract: see ``bundles.md``.
"""

import json

import pytest
import torch

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.rf_bank import (
    SPARSE_MAX_DENSITY,
    ReceptiveFieldBank,
    decode_weights,
    encode_weights,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.io.bundle import SPARSE_SCHEMA_VERSION, load_bundle, write_bundle
from sensoryforge.presets import load_preset

SPARSE_KEYS = {
    "innervation_weights_indices",
    "innervation_weights_values",
    "innervation_weights_shape",
}


def _sparse_matrix(n=6, m=40, seed=0):
    """A float32 ``[n, m]`` matrix with about 5% nonzero entries."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.rand(n, m, generator=gen)
    return torch.where(
        torch.rand(n, m, generator=gen) < 0.05, w + 0.1, torch.zeros(n, m)
    )


def _bank(weights):
    n, m = weights.shape
    centers = torch.zeros(n, 2)
    coords = torch.stack([torch.arange(m, dtype=torch.float32), torch.zeros(m)], dim=1)
    return ReceptiveFieldBank(weights, centers, coords, provenance={"builder": "test"})


class TestEncode:
    def test_layout_dtypes_and_order(self):
        w = _sparse_matrix()
        entries = encode_weights(w, "auto")
        assert set(entries) == SPARSE_KEYS
        idx, val = (
            entries["innervation_weights_indices"],
            entries["innervation_weights_values"],
        )
        assert idx.dtype == torch.int64 and tuple(idx.shape) == (2, val.shape[0])
        assert val.dtype == torch.float32 and val.ndim == 1
        assert entries["innervation_weights_shape"] == [6, 40]
        flat = idx[0] * 40 + idx[1]
        assert torch.equal(flat, torch.sort(flat).values)  # row-major order
        assert torch.unique(flat).numel() == flat.numel()  # no repeated pair

    def test_round_trip_is_bit_identical(self):
        w = _sparse_matrix()
        assert torch.equal(decode_weights(encode_weights(w, True)), w)

    def test_signed_zeros_survive_the_round_trip(self):
        # an imported bank may hold -0.0; decoding fills the gaps with +0.0
        w = _sparse_matrix()
        w[1, 3], w[4, 0] = -0.0, -0.0
        back = decode_weights(encode_weights(w, True))
        assert torch.equal(torch.signbit(back), torch.signbit(w))
        assert torch.equal(back, w)

    def test_auto_keeps_a_dense_matrix_dense(self):
        w = torch.rand(4, 5) + 0.1
        assert set(encode_weights(w, "auto")) == {"innervation_weights"}
        assert set(encode_weights(w, True)) == SPARSE_KEYS
        assert SPARSE_MAX_DENSITY == 0.10

    def test_false_is_always_dense(self):
        assert set(encode_weights(_sparse_matrix(), False)) == {"innervation_weights"}

    def test_all_zero_matrix(self):
        w = torch.zeros(3, 4)
        entries = encode_weights(w, "auto")
        assert tuple(entries["innervation_weights_indices"].shape) == (2, 0)
        assert torch.equal(decode_weights(entries), w)

    def test_other_dtypes(self):
        w = _sparse_matrix().double()
        assert set(encode_weights(w, "auto")) == {"innervation_weights"}
        with pytest.raises(ValueError, match="float32"):
            encode_weights(w, True)

    def test_argument_errors(self):
        with pytest.raises(ValueError, match="sparse must be"):
            encode_weights(_sparse_matrix(), "yes")
        with pytest.raises(ValueError, match="\\[N, M\\]"):
            encode_weights(torch.zeros(2, 3, 4), "auto")


class TestDecodeErrors:
    def _good(self):
        return dict(encode_weights(_sparse_matrix(), True))

    def test_neither_layout_is_none(self):
        assert decode_weights({"neuron_centers": torch.zeros(1, 2)}) is None

    def test_dense_layout(self):
        w = torch.rand(2, 3)
        assert torch.equal(decode_weights({"innervation_weights": w}), w)

    def test_partial_sparse_set(self):
        raw = self._good()
        del raw["innervation_weights_values"]
        with pytest.raises(ValueError, match="missing"):
            decode_weights(raw)

    def test_wrong_index_dtype(self):
        raw = self._good()
        raw["innervation_weights_indices"] = raw["innervation_weights_indices"].int()
        with pytest.raises(ValueError, match="int64"):
            decode_weights(raw)

    def test_values_length_mismatch(self):
        raw = self._good()
        raw["innervation_weights_values"] = raw["innervation_weights_values"][:-1]
        with pytest.raises(ValueError, match="nnz"):
            decode_weights(raw)

    def test_index_out_of_range(self):
        raw = self._good()
        raw["innervation_weights_indices"][1, 0] = 40
        with pytest.raises(ValueError, match="receptor index"):
            decode_weights(raw)
        raw = self._good()
        raw["innervation_weights_indices"][0, 0] = 6
        with pytest.raises(ValueError, match="neuron index"):
            decode_weights(raw)

    def test_repeated_pair(self):
        raw = self._good()
        raw["innervation_weights_indices"][:, 1] = raw["innervation_weights_indices"][
            :, 0
        ]
        with pytest.raises(ValueError, match="repeat"):
            decode_weights(raw)

    def test_bad_shape(self):
        raw = self._good()
        raw["innervation_weights_shape"] = [6]
        with pytest.raises(ValueError, match="shape"):
            decode_weights(raw)


class TestBankFile:
    def test_save_load_both_layouts_agree(self, tmp_path):
        bank = _bank(_sparse_matrix())
        bank.save(tmp_path / "dense.pt")
        bank.save(tmp_path / "sparse.pt", sparse="auto")
        raw = torch.load(
            tmp_path / "sparse.pt", weights_only=True
        )  # plain tensors only
        assert SPARSE_KEYS <= set(raw) and "innervation_weights" not in raw
        a = ReceptiveFieldBank.load(tmp_path / "dense.pt")
        b = ReceptiveFieldBank.load(tmp_path / "sparse.pt")
        assert torch.equal(a.weights, b.weights)
        assert (tmp_path / "sparse.pt").stat().st_size < (
            tmp_path / "dense.pt"
        ).stat().st_size

    def test_default_save_is_dense(self, tmp_path):
        _bank(_sparse_matrix()).save(tmp_path / "b.pt")
        assert "innervation_weights" in torch.load(tmp_path / "b.pt", weights_only=True)

    def test_load_error_names_both_layouts(self, tmp_path):
        torch.save({"neuron_centers": torch.zeros(1, 2)}, tmp_path / "x.pt")
        with pytest.raises(ValueError, match="innervation_weights_indices"):
            ReceptiveFieldBank.load(tmp_path / "x.pt")


@pytest.fixture(scope="module")
def recipe_run():
    """The 80x80 recipe run for a few frames (weights are 99.6% zeros)."""
    config = SensoryForgeConfig.from_dict(load_preset("tactile_sa1_ra1"))
    engine = SimulationEngine(config)
    stimulus = torch.rand(1, 5, 80, 80)
    return config, engine, stimulus, engine.run(stimulus, return_intermediates=True)


def _pop_bytes(bundle_dir):
    return sum(f.stat().st_size for f in bundle_dir.glob("population_*.pt"))


def _schema(bundle_dir):
    return json.loads((bundle_dir / "config.json").read_text())["schema_version"]


class TestBundle:
    def test_default_bundle_is_sparse_and_loads_to_the_same_banks(
        self, tmp_path, recipe_run
    ):
        config, engine, stimulus, results = recipe_run
        sparse = write_bundle(tmp_path / "s", config, engine, results, stimulus)
        dense = write_bundle(
            tmp_path / "d", config, engine, results, stimulus, sparse_weights=False
        )
        assert _schema(sparse) == SPARSE_SCHEMA_VERSION
        assert _schema(dense) == "2.3.0"  # all-dense bundles keep the older version
        a, b = load_bundle(sparse), load_bundle(dense)
        for name, bank in b.banks.items():
            assert torch.equal(a.banks[name].weights, bank.weights)
            assert torch.equal(bank.weights, engine_weights(engine, name))
        assert _pop_bytes(sparse) * 20 < _pop_bytes(dense)  # about 100x in practice


def engine_weights(engine, name):
    return next(p["bank"].weights for p in engine.populations if p["name"] == name)
