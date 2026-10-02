"""Counter-based hashing: reference values, independence from n, ranges."""

import hashlib

import numpy as np

from sensoryforge.world import rng


def test_splitmix64_matches_the_reference_sequence():
    # SplitMix64 seeded with 0: its first two outputs.
    first = rng._splitmix64(np.array([0], dtype=np.uint64))[0]
    second = rng._splitmix64(np.array([0x9E3779B97F4A7C15], dtype=np.uint64))[0]
    assert int(first) == 0xE220A8397B1DCDAF
    assert int(second) == 0x6E789E6AA1B965F4


def test_hashes_are_pinned():
    # Changing any of these changes every draw of every world, and with it
    # pressure-simulation's data sets (.claude/rules/world-engine.md). Values
    # computed with SF's env and bio-encoding before implementation: identical.
    assert rng.hash_parts(7, "class") == 18279675696851888426
    assert rng.seed53(20261002, "train", 0) == 7454758737064703
    seeds = rng.draw_seeds(7, [0, 1, 2])
    assert seeds.tolist() == [7125699848674262, 6996912338094675, 4294997795970717]
    assert rng.uniforms(seeds, "class").tolist() == [
        0.14982635123855137,
        0.23457615854478897,
        0.2137307826344046,
    ]


def test_string_keys_are_sha256_prefixes():
    digest = hashlib.sha256(b"sigma_mm").digest()[:8]
    assert rng.key("sigma_mm") == int.from_bytes(digest, "little")
    assert rng.key(-1) == 2**64 - 1


def test_draw_seeds_do_not_depend_on_how_many_are_asked_for():
    many = rng.draw_seeds(7, np.arange(1000))
    some = rng.draw_seeds(7, [3, 999])
    assert many[3] == some[0] and many[999] == some[1]
    assert int(many.max()) < rng.SEED_LIMIT
    assert len(set(many.tolist())) == 1000


def test_uniforms_are_in_the_unit_interval_and_differ_by_slot():
    seeds = rng.draw_seeds(1, np.arange(20000))
    a = rng.uniforms(seeds, "a")
    b = rng.uniforms(seeds, "b")
    assert a.dtype == np.float64 and 0.0 <= a.min() and a.max() < 1.0
    assert abs(a.mean() - 0.5) < 0.01
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.03
    assert np.array_equal(a, rng.uniforms(seeds, "a"))


def test_seed53_and_permutation():
    assert rng.seed53(1, "train", 0) != rng.seed53(1, "train", 1)
    assert rng.seed53(1, "train", 0) == rng.seed53(1, "train", 0)
    perm = rng.permutation(50, 9, "x")
    assert sorted(perm.tolist()) == list(range(50))
    assert perm.tolist() != list(range(50))
    assert np.array_equal(perm, rng.permutation(50, 9, "x"))
